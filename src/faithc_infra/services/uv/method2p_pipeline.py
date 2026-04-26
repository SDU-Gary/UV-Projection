from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import trimesh

from .field_projector import build_method25_projected_field
from .linear_solver import interpolate_sample_uv
from .method2_pipeline import run_method2_gradient_poisson, solve_method2_target_field_linear
from .quality import texture_reprojection_error


def _use_island_linear_init(cfg: Dict[str, Any], face_island_labels: np.ndarray, n_faces: int) -> Optional[np.ndarray]:
    solve_per_island = bool(cfg.get("method2", {}).get("solve_per_island", True))
    if not solve_per_island:
        return None
    labels = np.asarray(face_island_labels, dtype=np.int64)
    return labels if labels.shape[0] == int(n_faces) else None


def _build_method2p_stats(
    *,
    method2_stats: Dict[str, Any],
    solve_mesh: trimesh.Trimesh,
    sample_face_ids: np.ndarray,
    sample_bary: np.ndarray,
    target_uv: np.ndarray,
    mapped_uv: np.ndarray,
    linear_solve: Dict[str, Any],
    projected_field: Dict[str, Any],
    image,
) -> Dict[str, Any]:
    base_stats = dict(method2_stats)

    if sample_face_ids.size > 0:
        pred_uv = interpolate_sample_uv(
            np.asarray(solve_mesh.faces, dtype=np.int64),
            np.asarray(sample_face_ids, dtype=np.int64),
            np.asarray(sample_bary, dtype=np.float64),
            np.asarray(mapped_uv, dtype=np.float64),
        )
        color_l1, color_l2 = texture_reprojection_error(image, np.asarray(target_uv, dtype=np.float64), pred_uv)
    else:
        color_l1, color_l2 = None, None

    solve_meta = dict(linear_solve.get("solve_meta", {}))
    projector_meta = dict(projected_field.get("projector_meta", {}))
    projected_valid = np.asarray(projected_field["face_valid"], dtype=np.bool_)
    anchor_ids = np.asarray(linear_solve.get("anchor_ids", []), dtype=np.int64)

    base_stats.update(
        {
            "uv_mode_used": "method2p_projected_gradient_poisson",
            "uv_method": "method2p_projected_gradient_poisson",
            "uv_solver_stage": "m2p_m2_projected",
            "uv_color_reproj_l1": color_l1,
            "uv_color_reproj_l2": color_l2,
            "uv_m2_jacobian_valid_ratio": float(np.mean(projected_valid)) if projected_valid.size > 0 else 0.0,
            "uv_m2_anchor_count_total": int(anchor_ids.size),
            "uv_solver_linear_backend_used": solve_meta.get("uv_solver_linear_backend_used"),
            "uv_solver_residual_u": solve_meta.get("uv_solver_residual_u"),
            "uv_solver_residual_v": solve_meta.get("uv_solver_residual_v"),
            "uv_m2_system_cond_proxy": solve_meta.get("uv_m2_system_cond_proxy"),
            "uv_m2_soft_anchor_count": solve_meta.get("uv_m2_soft_anchor_count"),
            "uv_m2p_enabled": True,
            "uv_m2p_status": "ok",
            "uv_m2p_field_source": projector_meta.get("field_source"),
            "uv_m2p_lambda_decay": projector_meta.get("lambda_decay"),
            "uv_m2p_lambda_curl": projector_meta.get("lambda_curl"),
            "uv_m2p_projector_ridge_eps": projector_meta.get("ridge_eps"),
            "uv_m2p_projector_matrix_meta": projector_meta.get("matrix"),
            "uv_m2p_projector_solve_meta": projector_meta.get("solve"),
            "uv_m2p_anchor_fill_meta": projected_field.get("anchor_fill_meta"),
            "uv_m2p_samplefit_meta": projected_field.get("samplefit_meta"),
            "uv_m2p_linear_solver_meta": solve_meta,
            "uv_m2p_linear_init_color_reproj_l1": color_l1,
            "uv_m2p_linear_init_color_reproj_l2": color_l2,
        }
    )
    return base_stats


def run_method2p_projected_gradient_poisson(
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
        out_stats["uv_mode_used"] = "method2p_projected_gradient_poisson"
        out_stats["uv_method"] = "method2p_projected_gradient_poisson"
        out_stats["uv_solver_stage"] = "m2"
        out_stats["uv_m2p_enabled"] = True
        out_stats["uv_m2p_status"] = "skipped_no_internal_state"
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
        face_island_labels=_use_island_linear_init(
            cfg,
            np.asarray(internal.solve_face_island_labels, dtype=np.int64),
            len(internal.solve_mesh.faces),
        ),
        anchor_vertex_target_uv=np.asarray(projected_field["anchor_vertex_target_uv"], dtype=np.float64),
        anchor_vertex_confidence=np.asarray(internal.anchor_vertex_confidence, dtype=np.float64),
        face_smooth_alpha=np.asarray(internal.face_smooth_alpha, dtype=np.float64),
        disable_post_align=True,
    )

    mapped_uv_m2p = np.asarray(linear_solve["mapped_uv"], dtype=np.float64)
    out_stats = _build_method2p_stats(
        method2_stats=method2_stats,
        solve_mesh=internal.solve_mesh,
        sample_face_ids=np.asarray(internal.solve_sample_face_ids, dtype=np.int64),
        sample_bary=np.asarray(internal.solve_sample_bary, dtype=np.float64),
        target_uv=np.asarray(internal.solve_target_uv, dtype=np.float64),
        mapped_uv=mapped_uv_m2p,
        linear_solve=linear_solve,
        projected_field=projected_field,
        image=image,
    )
    return mapped_uv_m2p, out_stats, export_payload


__all__ = ["run_method2p_projected_gradient_poisson"]
