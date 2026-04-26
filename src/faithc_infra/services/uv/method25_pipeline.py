from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import trimesh

from .field_projector import build_method25_projected_field
from .linear_solver import interpolate_sample_uv
from .method2_pipeline import Method2InternalState, run_method2_gradient_poisson, solve_method2_target_field_linear
from .method4_pipeline import run_method4_from_internal_state
from .quality import texture_reprojection_error


def _use_island_linear_init(cfg: Dict[str, Any], state: Method2InternalState) -> Optional[np.ndarray]:
    solve_per_island = bool(cfg.get("method2", {}).get("solve_per_island", True))
    if not solve_per_island:
        return None
    labels = np.asarray(state.solve_face_island_labels, dtype=np.int64)
    return labels if labels.shape[0] == int(len(state.solve_mesh.faces)) else None


def _build_method25_base_stats(
    *,
    method2_stats: Dict[str, Any],
    state_linear: Method2InternalState,
    linear_solve: Dict[str, Any],
    projected_field: Dict[str, Any],
    image,
) -> Dict[str, Any]:
    base_stats = dict(method2_stats)
    mapped_uv = np.asarray(state_linear.mapped_uv_init, dtype=np.float64)
    solve_mesh = state_linear.solve_mesh

    if state_linear.solve_sample_face_ids.size > 0:
        pred_uv = interpolate_sample_uv(
            np.asarray(solve_mesh.faces, dtype=np.int64),
            state_linear.solve_sample_face_ids,
            state_linear.solve_sample_bary,
            mapped_uv,
        )
        color_l1, color_l2 = texture_reprojection_error(image, state_linear.solve_target_uv, pred_uv)
    else:
        color_l1, color_l2 = None, None

    solve_meta = dict(linear_solve.get("solve_meta", {}))
    projector_meta = dict(projected_field.get("projector_meta", {}))
    base_stats.update(
        {
            "uv_mode_used": "method25_projected_jacobian_injective",
            "uv_method": "method25_projected_jacobian_injective",
            "uv_solver_stage": "m25_m2_init",
            "uv_color_reproj_l1": color_l1,
            "uv_color_reproj_l2": color_l2,
            "uv_m25_linear_init_color_reproj_l1": color_l1,
            "uv_m25_linear_init_color_reproj_l2": color_l2,
            "uv_m2_jacobian_valid_ratio": float(np.mean(np.asarray(state_linear.face_target_valid_mask, dtype=np.bool_))),
            "uv_m2_anchor_count_total": int(np.asarray(state_linear.anchor_vertex_ids, dtype=np.int64).size),
            "uv_solver_linear_backend_used": solve_meta.get("uv_solver_linear_backend_used"),
            "uv_solver_residual_u": solve_meta.get("uv_solver_residual_u"),
            "uv_solver_residual_v": solve_meta.get("uv_solver_residual_v"),
            "uv_m2_system_cond_proxy": solve_meta.get("uv_m2_system_cond_proxy"),
            "uv_m2_soft_anchor_count": solve_meta.get("uv_m2_soft_anchor_count"),
            "uv_m25_enabled": True,
            "uv_m25_field_source": projector_meta.get("field_source"),
            "uv_m25_lambda_decay": projector_meta.get("lambda_decay"),
            "uv_m25_lambda_curl": projector_meta.get("lambda_curl"),
            "uv_m25_projector_ridge_eps": projector_meta.get("ridge_eps"),
            "uv_m25_projector_matrix_meta": projector_meta.get("matrix"),
            "uv_m25_projector_solve_meta": projector_meta.get("solve"),
            "uv_m25_anchor_fill_meta": projected_field.get("anchor_fill_meta"),
            "uv_m25_samplefit_meta": projected_field.get("samplefit_meta"),
            "uv_m25_linear_init_solver_meta": solve_meta,
        }
    )
    return base_stats


def run_method25_projected_jacobian_injective(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    image,
    device: str,
    cfg: Dict[str, Any],
    nearest_mapper: Callable[[trimesh.Trimesh, trimesh.Trimesh, np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    barycentric_mapper: Callable[
        [trimesh.Trimesh, trimesh.Trimesh, np.ndarray, str, Dict[str, Any]],
        Tuple[np.ndarray, Dict[str, Any]],
    ],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    mapped_uv_m2, method2_stats, export_payload, internal = run_method2_gradient_poisson(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_uv=high_uv,
        image=image,
        device=device,
        cfg=cfg,
        nearest_mapper=nearest_mapper,
        barycentric_mapper=barycentric_mapper,
        return_internal=True,
    )
    if internal is None:
        out_stats = dict(method2_stats)
        out_stats["uv_mode_used"] = "method25_projected_jacobian_injective"
        out_stats["uv_method"] = "method25_projected_jacobian_injective"
        out_stats["uv_solver_stage"] = "m2"
        out_stats["uv_m25_enabled"] = True
        out_stats["uv_m25_status"] = "skipped_no_internal_state"
        return mapped_uv_m2, out_stats, export_payload

    projected_field = build_method25_projected_field(
        state=internal,
        high_mesh=high_mesh,
        high_uv=high_uv,
        cfg=cfg,
    )

    linear_solve = solve_method2_target_field_linear(
        solve_mesh=internal.solve_mesh,
        high_mesh=high_mesh,
        high_uv=high_uv,
        resolved_device=internal.resolved_device,
        cfg=cfg,
        face_jac=np.asarray(projected_field["face_jac"], dtype=np.float64),
        face_weights=np.asarray(projected_field["face_weights"], dtype=np.float64),
        face_valid_mask=np.asarray(projected_field["face_valid"], dtype=np.bool_),
        face_active_mask=np.asarray(internal.solve_face_active_mask, dtype=np.bool_),
        face_island_labels=_use_island_linear_init(cfg, internal),
        anchor_vertex_target_uv=np.asarray(projected_field["anchor_vertex_target_uv"], dtype=np.float64),
        anchor_vertex_confidence=np.asarray(internal.anchor_vertex_confidence, dtype=np.float64),
        face_smooth_alpha=np.asarray(internal.face_smooth_alpha, dtype=np.float64),
        disable_post_align=True,
    )

    internal_m25 = replace(
        internal,
        mapped_uv_init=np.asarray(linear_solve["mapped_uv"], dtype=np.float32),
        mapped_uv_pre_align=np.asarray(linear_solve["mapped_uv"], dtype=np.float32),
        face_target_jacobian=np.asarray(projected_field["face_jac"], dtype=np.float32),
        face_target_valid_mask=np.asarray(projected_field["face_valid"], dtype=np.bool_),
        face_target_weights=np.asarray(projected_field["face_weights"], dtype=np.float32),
        anchor_vertex_ids=np.asarray(linear_solve["anchor_ids"], dtype=np.int64),
        anchor_uv=np.asarray(linear_solve["anchor_uv"], dtype=np.float32),
        anchor_vertex_target_uv=np.asarray(projected_field["anchor_vertex_target_uv"], dtype=np.float32),
    )
    base_stats = _build_method25_base_stats(
        method2_stats=method2_stats,
        state_linear=internal_m25,
        linear_solve=linear_solve,
        projected_field=projected_field,
        image=image,
    )
    internal_m25 = replace(internal_m25, method_stats=base_stats)

    mapped_uv_m25, out_stats, export_payload = run_method4_from_internal_state(
        internal=internal_m25,
        base_stats=base_stats,
        export_payload=export_payload,
        image=image,
        cfg=cfg,
        success_solver_stage="m25_m4",
        fallback_solver_stage="m25_m2_fallback_after_m4",
        disabled_solver_stage="m25_m2_init",
    )
    out_stats["uv_mode_used"] = "method25_projected_jacobian_injective"
    out_stats["uv_method"] = "method25_projected_jacobian_injective"
    out_stats["uv_m25_enabled"] = True
    out_stats.setdefault("uv_m25_status", "ok")
    return mapped_uv_m25, out_stats, export_payload


__all__ = ["run_method25_projected_jacobian_injective"]
