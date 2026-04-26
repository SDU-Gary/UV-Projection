from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import trimesh
from .island_pipeline import IslandPipelineResult, compute_cached_high_face_uv_islands, run_halfedge_island_pipeline
from .correspondence import (
    build_high_cuda_context,
    correspond_points_hybrid,
    detect_cross_seam_faces,
    major_face_island_labels,
)
from .linear_solver import interpolate_sample_uv, solve_global_uv
from .options import resolve_seam_validation_settings
from .quality import texture_gradient_weights, texture_reprojection_error
from .sampling import sample_low_mesh
from .texture_io import get_vertex_normals, resolve_device


def _build_guard_meta_template(
    *,
    island_guard_requested: bool,
    island_guard_mode_requested: str,
    island_guard_allow_unknown: bool,
    island_guard_conf_min: float,
    island_guard_fallback_policy: str,
    guard_mode_used: str,
    constrained_n: int,
    total_samples: int,
) -> Dict[str, Any]:
    constrained_ratio = float(constrained_n / max(1, total_samples)) if total_samples > 0 else 0.0
    return {
        "uv_island_guard_requested": island_guard_requested,
        "uv_island_guard_mode_requested": island_guard_mode_requested,
        "uv_island_guard_allow_unknown": island_guard_allow_unknown,
        "uv_island_guard_confidence_min": island_guard_conf_min,
        "uv_island_guard_fallback_policy": island_guard_fallback_policy,
        "uv_island_guard_enabled": constrained_n > 0,
        "uv_island_guard_mode_used": guard_mode_used,
        "uv_island_guard_constrained_points": int(constrained_n),
        "uv_island_guard_constrained_ratio": constrained_ratio,
        "uv_island_guard_reject_count": 0,
        "uv_island_guard_reject_ratio": 0.0,
        "uv_island_guard_fallback_success_ratio": 0.0,
        "uv_island_guard_invalid_after_guard_ratio": 0.0,
    }


def _prepare_high_islands(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    seam_cfg: Dict[str, Any],
    seam_strategy_requested: str,
    island_guard_requested: bool,
) -> Tuple[Optional[np.ndarray], Dict[str, Any], str, Dict[str, Any], Optional[str]]:
    seam_strategy_used = seam_strategy_requested
    seam_meta: Dict[str, Any] = {}
    high_meta: Dict[str, Any] = {}
    high_face_island: Optional[np.ndarray] = None
    guard_error: Optional[str] = None

    need_high_island = seam_strategy_requested == "halfedge_island" or island_guard_requested
    if not need_high_island:
        return high_face_island, high_meta, seam_strategy_used, seam_meta, guard_error

    try:
        high_face_island, high_meta, cache_hit = compute_cached_high_face_uv_islands(
            high_mesh=high_mesh,
            high_uv=high_uv,
            position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
            uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
            use_cache=bool(seam_cfg.get("perf_fast_island_cache", True)),
        )
        seam_meta["uv_m2_perf_high_island_cache_enabled"] = bool(seam_cfg.get("perf_fast_island_cache", True))
        seam_meta["uv_m2_perf_high_island_cache_hit"] = bool(cache_hit)
    except Exception as exc:
        high_face_island = None
        if seam_strategy_requested == "halfedge_island":
            seam_strategy_used = "fallback_legacy"
            seam_meta["uv_halfedge_island_error"] = str(exc)
        if island_guard_requested:
            guard_error = str(exc)

    if high_face_island is not None:
        seam_meta.update(
            {
                "uv_high_island_count": int(high_meta.get("high_island_count", 0)),
                "uv_high_seam_edges": int(high_meta.get("high_seam_edges", 0)),
                "uv_high_boundary_edges": int(high_meta.get("high_boundary_edges", 0)),
                "uv_high_nonmanifold_edges": int(high_meta.get("high_nonmanifold_edges", 0)),
            }
        )

    return high_face_island, high_meta, seam_strategy_used, seam_meta, guard_error


def _compute_pass_correspondence(
    *,
    sample_points: np.ndarray,
    sample_normals: np.ndarray,
    sample_face_ids: np.ndarray,
    n_faces: int,
    corr_cfg: Dict[str, Any],
    high_ctx: Dict[str, Any],
    high_face_island: Optional[np.ndarray],
    island_guard_requested: bool,
    island_guard_mode_requested: str,
    island_guard_mode_used: str,
    island_guard_allow_unknown: bool,
    island_guard_conf_min: float,
    island_guard_fallback_policy: str,
    expected_face_island: Optional[np.ndarray],
    expected_face_confidence: Optional[np.ndarray],
    fixed_face_island: Optional[np.ndarray],
    fixed_face_conflict: Optional[np.ndarray],
    fixed_face_confidence: Optional[np.ndarray],
    min_valid_samples_per_face: int,
    guard_error: Optional[str],
) -> Dict[str, Any]:
    correspondence = correspond_points_hybrid(
        points=sample_points,
        point_normals=sample_normals,
        corr_cfg=corr_cfg,
        high_ctx=high_ctx,
    )
    target_uv = correspondence["target_uv"]
    target_face_ids = correspondence["target_face_ids"]
    valid_mask = correspondence["valid_mask"]
    primary_mask = correspondence["primary_mask"]
    fallback_used_mask = correspondence["fallback_used_mask"]

    n_samples = int(sample_face_ids.shape[0])
    constrained_n = 0
    guard_mode = "off"
    if (
        island_guard_requested
        and high_face_island is not None
        and expected_face_island is not None
        and expected_face_confidence is not None
    ):
        sample_expected_island = expected_face_island[sample_face_ids]
        sample_expected_conf = expected_face_confidence[sample_face_ids]
        constrained_mask = (sample_expected_island >= 0) & (sample_expected_conf >= island_guard_conf_min)
        constrained_idx = np.where(constrained_mask)[0]
        constrained_n = int(constrained_idx.size)
        if constrained_n > 0:
            guard_mode = island_guard_mode_used
            guarded = correspond_points_hybrid(
                points=sample_points[constrained_idx],
                point_normals=sample_normals[constrained_idx],
                corr_cfg=corr_cfg,
                high_ctx=high_ctx,
                island_guard={
                    "enabled": True,
                    "mode": island_guard_mode_used,
                    "allow_unknown": island_guard_allow_unknown,
                    "high_face_island": high_face_island,
                    "expected_island": sample_expected_island[constrained_idx],
                },
            )
            target_uv[constrained_idx] = guarded["target_uv"]
            target_face_ids[constrained_idx] = guarded["target_face_ids"]
            valid_mask[constrained_idx] = guarded["valid_mask"]
            primary_mask[constrained_idx] = guarded["primary_mask"]
            fallback_used_mask[constrained_idx] = guarded["fallback_used_mask"]
            guard_stats = guarded.get("island_guard_stats", {})
        else:
            guard_stats = {
                "reject_count": 0,
                "fallback_success_count": 0,
                "invalid_after_guard_count": 0,
            }
    else:
        guard_stats = {
            "reject_count": 0,
            "fallback_success_count": 0,
            "invalid_after_guard_count": 0,
        }

    guard_meta = _build_guard_meta_template(
        island_guard_requested=island_guard_requested,
        island_guard_mode_requested=island_guard_mode_requested,
        island_guard_allow_unknown=island_guard_allow_unknown,
        island_guard_conf_min=island_guard_conf_min,
        island_guard_fallback_policy=island_guard_fallback_policy,
        guard_mode_used=guard_mode,
        constrained_n=constrained_n,
        total_samples=n_samples,
    )
    guard_meta["uv_island_guard_reject_count"] = int(guard_stats.get("reject_count", 0))
    guard_meta["uv_island_guard_reject_ratio"] = float(
        guard_meta["uv_island_guard_reject_count"] / max(1, constrained_n)
    )
    guard_meta["uv_island_guard_fallback_success_ratio"] = float(
        int(guard_stats.get("fallback_success_count", 0)) / max(1, constrained_n)
    )
    guard_meta["uv_island_guard_invalid_after_guard_ratio"] = float(
        int(guard_stats.get("invalid_after_guard_count", 0)) / max(1, constrained_n)
    )
    if guard_error:
        guard_meta["uv_island_guard_error"] = guard_error

    low_face_island = np.full((n_faces,), -1, dtype=np.int64)
    low_face_conflict = np.zeros((n_faces,), dtype=np.bool_)
    low_face_confidence = np.zeros((n_faces,), dtype=np.float32)
    if (
        fixed_face_island is not None
        and fixed_face_conflict is not None
        and fixed_face_confidence is not None
        and fixed_face_island.shape[0] == n_faces
        and fixed_face_conflict.shape[0] == n_faces
        and fixed_face_confidence.shape[0] == n_faces
    ):
        low_face_island = np.asarray(fixed_face_island, dtype=np.int64, copy=False)
        low_face_conflict = np.asarray(fixed_face_conflict, dtype=np.bool_, copy=False)
        low_face_confidence = np.asarray(fixed_face_confidence, dtype=np.float32, copy=False)
    elif high_face_island is not None:
        low_face_island, low_face_conflict, low_face_confidence = major_face_island_labels(
            sample_face_ids=sample_face_ids,
            target_face_ids=target_face_ids,
            valid_mask=valid_mask,
            high_face_island=high_face_island,
            n_low_faces=n_faces,
            min_samples=min_valid_samples_per_face,
        )

    return {
        "target_uv": target_uv,
        "target_face_ids": target_face_ids,
        "valid_mask": valid_mask,
        "primary_mask": primary_mask,
        "fallback_used_mask": fallback_used_mask,
        "guard_meta": guard_meta,
        "low_face_island": low_face_island,
        "low_face_conflict": low_face_conflict,
        "low_face_confidence": low_face_confidence,
    }


def _prepare_solver_inputs(
    *,
    low_mesh: trimesh.Trimesh,
    sample_face_ids: np.ndarray,
    target_uv: np.ndarray,
    target_face_ids: np.ndarray,
    valid_mask: np.ndarray,
    seam_cfg: Dict[str, Any],
    seam_strategy_used: str,
    high_face_island: Optional[np.ndarray],
    low_face_island: np.ndarray,
    low_face_conflict: np.ndarray,
    unknown_face_policy: str,
    seam_validation_strict: bool,
    seam_validation_require_closed: bool,
    seam_validation_require_pure_components: bool,
    precomputed_island_result: Optional[IslandPipelineResult] = None,
) -> Dict[str, Any]:
    n_faces = int(len(low_mesh.faces))
    cross_seam_face_mask = np.zeros(n_faces, dtype=np.bool_)
    seam_face_ids = np.zeros((0,), dtype=np.int64)
    seam_edges_out = np.zeros((0, 2), dtype=np.int64)
    solve_mesh = low_mesh
    smooth_face_mask: Optional[np.ndarray] = None
    solver_valid_mask = valid_mask.copy()

    seam_meta: Dict[str, Any] = {}
    if high_face_island is not None:
        seam_meta["uv_island_conflict_faces"] = int(np.count_nonzero(low_face_conflict))
        seam_meta["uv_island_unknown_faces"] = int(np.count_nonzero(low_face_island < 0))

    if seam_strategy_used == "halfedge_island" and high_face_island is not None:
        if precomputed_island_result is not None:
            seam_meta.update(precomputed_island_result.meta)
            if precomputed_island_result.seam is not None:
                seam_edges = np.asarray(precomputed_island_result.seam.seam_edges, dtype=np.int64, copy=False)
                seam_edges_out = seam_edges.astype(np.int64, copy=False)
            solve_mesh = precomputed_island_result.solve_mesh
            if seam_validation_strict and (not precomputed_island_result.validation_ok):
                raise RuntimeError(
                    "halfedge seam validation failed: "
                    + str(precomputed_island_result.validation_error or "validation failed")
                )
        seam_meta.setdefault("uv_halfedge_backend", "openmesh")
        seam_meta.setdefault("uv_seam_uv_span_threshold", float(seam_cfg.get("uv_span_threshold", 0.35)))

        sample_expected_island = low_face_island[sample_face_ids]
        sample_expected_conflict = low_face_conflict[sample_face_ids]
        sample_hit_island = np.full((len(target_face_ids),), -1, dtype=np.int64)
        ok_hit = (target_face_ids >= 0) & (target_face_ids < len(high_face_island))
        if np.any(ok_hit):
            sample_hit_island[ok_hit] = high_face_island[target_face_ids[ok_hit]]
        strict_keep = (
            (sample_expected_island >= 0)
            & (~sample_expected_conflict)
            & (sample_hit_island == sample_expected_island)
        )
        before_mask = solver_valid_mask.copy()
        solver_valid_mask &= strict_keep
        rejected = int(np.count_nonzero(before_mask & (~solver_valid_mask)))
        seam_meta["uv_island_strict_filter_enabled"] = True
        seam_meta["uv_island_strict_reject_count"] = int(rejected)
        seam_meta["uv_island_strict_reject_ratio"] = float(rejected / max(1, int(np.count_nonzero(before_mask))))
    else:
        cross_seam_face_mask = detect_cross_seam_faces(
            sample_face_ids=sample_face_ids,
            target_uv=target_uv,
            valid_mask=valid_mask,
            n_faces=n_faces,
            uv_span_threshold=float(seam_cfg.get("uv_span_threshold", 0.35)),
            min_valid_samples_per_face=int(seam_cfg.get("min_valid_samples_per_face", 2)),
        )
        if bool(seam_cfg.get("exclude_cross_seam_faces", True)):
            solver_valid_mask &= ~cross_seam_face_mask[sample_face_ids]
        smooth_face_mask = ~cross_seam_face_mask
        seam_face_ids = np.where(cross_seam_face_mask)[0].astype(np.int64)
        seam_meta["uv_cross_seam_faces"] = int(cross_seam_face_mask.sum())
        seam_meta["uv_cross_seam_face_ratio"] = (
            float(cross_seam_face_mask.mean()) if cross_seam_face_mask.size > 0 else 0.0
        )
        seam_meta["uv_seam_uv_span_threshold"] = float(seam_cfg.get("uv_span_threshold", 0.35))
        seam_meta["uv_halfedge_split_topology_applied"] = False
        seam_meta["uv_island_strict_filter_enabled"] = False
        seam_meta["uv_island_strict_reject_count"] = 0
        seam_meta["uv_island_strict_reject_ratio"] = 0.0

    if unknown_face_policy == "exclude" and high_face_island is not None:
        sample_face_island = low_face_island[sample_face_ids]
        solver_valid_mask &= sample_face_island >= 0

    return {
        "solve_mesh": solve_mesh,
        "solver_valid_mask": solver_valid_mask,
        "smooth_face_mask": smooth_face_mask,
        "cross_seam_face_mask": cross_seam_face_mask,
        "seam_face_ids": seam_face_ids,
        "seam_edges": seam_edges_out,
        "seam_meta": seam_meta,
    }


def _solve_pass(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    image,
    solve_mesh: trimesh.Trimesh,
    sample_face_ids: np.ndarray,
    sample_bary: np.ndarray,
    sample_area_weights: np.ndarray,
    target_uv: np.ndarray,
    fallback_used_mask: np.ndarray,
    solver_valid_mask: np.ndarray,
    smooth_face_mask: Optional[np.ndarray],
    corr_cfg: Dict[str, Any],
    solve_cfg: Dict[str, Any],
    tex_weight_cfg: Dict[str, Any],
    resolved: str,
) -> Dict[str, Any]:
    if not np.any(solver_valid_mask):
        return {
            "ok": False,
            "error": "no_valid_samples",
        }

    solve_face_ids = sample_face_ids[solver_valid_mask]
    solve_bary = sample_bary[solver_valid_mask]
    solve_target_uv = target_uv[solver_valid_mask]
    solve_area_w = sample_area_weights[solver_valid_mask]

    solve_corr_w = np.ones(len(solve_target_uv), dtype=np.float64)
    fallback_in_solver = fallback_used_mask[solver_valid_mask]
    solve_corr_w[fallback_in_solver] = float(corr_cfg.get("fallback_weight", 0.7))

    if bool(tex_weight_cfg.get("enabled", True)) and image is not None:
        tex_weights = texture_gradient_weights(
            image=image,
            uv=solve_target_uv,
            gamma=float(tex_weight_cfg.get("grad_weight_gamma", 1.0)),
            max_weight=float(tex_weight_cfg.get("max_weight", 5.0)),
        )
    else:
        tex_weights = np.ones(len(solve_target_uv), dtype=np.float64)

    weights = np.maximum(1e-12, solve_area_w * solve_corr_w * tex_weights)

    pcg_max_iter = int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000)))
    pcg_tol = float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6)))

    mapped_uv, solve_meta = solve_global_uv(
        low_mesh=solve_mesh,
        sample_face_ids=solve_face_ids,
        sample_bary=solve_bary,
        target_uv=solve_target_uv,
        sample_weights=weights,
        backend=str(solve_cfg.get("backend", "auto")),
        lambda_smooth=float(solve_cfg["lambda_smooth"]),
        pcg_max_iter=pcg_max_iter,
        pcg_tol=pcg_tol,
        pcg_check_every=int(solve_cfg.get("pcg_check_every", 25)),
        pcg_preconditioner=str(solve_cfg.get("pcg_preconditioner", "jacobi")),
        anchor_weight=float(solve_cfg["anchor_weight"]),
        ridge_eps=float(solve_cfg["ridge_eps"]),
        high_mesh=high_mesh,
        high_uv=high_uv,
        smooth_face_mask=smooth_face_mask,
        device=resolved,
    )

    pred_uv = interpolate_sample_uv(
        np.asarray(solve_mesh.faces, dtype=np.int64),
        solve_face_ids,
        solve_bary,
        mapped_uv,
    )
    color_l1, color_l2 = texture_reprojection_error(image, solve_target_uv, pred_uv)

    residual = pred_uv - solve_target_uv
    residual_sq = np.sum(residual * residual, axis=1)
    weight_sum = float(np.sum(weights))
    if weight_sum > 0.0:
        energy_data = float(np.sum(residual_sq * weights) / weight_sum)
    else:
        energy_data = float(np.mean(residual_sq)) if residual_sq.size > 0 else 0.0

    return {
        "ok": True,
        "mapped_uv": mapped_uv,
        "solve_meta": solve_meta,
        "solve_face_ids": solve_face_ids,
        "solve_bary": solve_bary,
        "solve_target_uv": solve_target_uv,
        "solver_valid_mask": solver_valid_mask,
        "weights": weights,
        "energy_data": energy_data,
        "color_l1": color_l1,
        "color_l2": color_l2,
    }


def _run_hybrid_iterative(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    image,
    device: str,
    cfg: Dict[str, Any],
    barycentric_mapper: Callable[
        [trimesh.Trimesh, trimesh.Trimesh, np.ndarray, str, Dict[str, Any]],
        Tuple[np.ndarray, Dict[str, Any]],
    ],
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    resolved = resolve_device(device)
    if resolved != "cuda":
        mapped_uv, stats = barycentric_mapper(high_mesh, low_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "barycentric_fallback"
        stats["uv_project_error"] = "CUDA unavailable for iterative hybrid optimization"
        stats["uv_iterative_enabled"] = True
        stats["uv_iter_count"] = 0
        stats["uv_iter_early_stop_reason"] = "cuda_unavailable"
        return mapped_uv, stats, {"local_vertex_split_applied": False}

    corr_cfg = cfg["correspondence"]
    seam_cfg = cfg.get("seam", {})
    solve_cfg = cfg["solve"]
    tex_weight_cfg = cfg["texture_weight"]
    iterative_cfg = cfg.get("iterative", {})

    seam_strategy_requested = str(seam_cfg.get("strategy", "legacy")).strip().lower()
    if seam_strategy_requested not in {"legacy", "halfedge_island"}:
        seam_strategy_requested = "legacy"

    island_guard_requested = bool(seam_cfg.get("uv_island_guard_enabled", True))
    island_guard_mode_requested = str(seam_cfg.get("uv_island_guard_mode", "soft")).strip().lower()
    if island_guard_mode_requested not in {"soft", "strict"}:
        island_guard_mode_requested = "soft"
    island_guard_conf_min = float(seam_cfg.get("uv_island_guard_confidence_min", 0.55))
    island_guard_allow_unknown = bool(seam_cfg.get("uv_island_guard_allow_unknown", False))
    island_guard_fallback_policy = str(
        seam_cfg.get("uv_island_guard_fallback", "nearest_same_island_then_udf")
    )
    validation = resolve_seam_validation_settings(seam_cfg)
    seam_validation_strict = bool(validation.strict)
    seam_validation_require_closed = bool(validation.require_closed_loops)
    seam_validation_require_pure = bool(validation.require_pure_components)

    max_iters = max(1, int(iterative_cfg.get("max_iters", 4)))
    min_iters = max(1, int(iterative_cfg.get("min_iters", 2)))
    min_iters = min(min_iters, max_iters)
    strict_mode_from_iter = int(iterative_cfg.get("strict_mode_from_iter", 2))
    label_change_tol = float(iterative_cfg.get("label_change_tol", 0.02))
    energy_rel_tol = float(iterative_cfg.get("energy_rel_tol", 1e-3))
    patience = max(1, int(iterative_cfg.get("patience", 1)))

    unknown_face_policy = str(iterative_cfg.get("unknown_face_policy", "exclude")).strip().lower()
    if unknown_face_policy not in {"exclude", "allow"}:
        unknown_face_policy = "exclude"

    high_ctx = build_high_cuda_context(
        high_mesh=high_mesh,
        high_uv=high_uv,
        device=resolved,
    )
    runtime_meta = dict(high_ctx.get("runtime_diag") or {})
    runtime_meta.pop("reason", None)

    work_low_mesh = low_mesh
    seam_meta_seed: Dict[str, Any] = {}
    seam_meta_seed.update(runtime_meta)
    seam_meta_seed["uv_island_validation_mode"] = validation.mode
    high_face_island: Optional[np.ndarray] = None
    guard_error: Optional[str] = None
    seam_strategy_used = seam_strategy_requested
    fixed_face_island: Optional[np.ndarray] = None
    fixed_face_conflict: Optional[np.ndarray] = None
    fixed_face_confidence: Optional[np.ndarray] = None
    island_result: Optional[IslandPipelineResult] = None
    if seam_strategy_requested == "halfedge_island":
        island_result = run_halfedge_island_pipeline(
            high_mesh=high_mesh,
            high_uv=high_uv,
            low_mesh=low_mesh,
            high_ctx=high_ctx,
            seam_cfg=seam_cfg,
            corr_cfg=corr_cfg,
            use_high_island_cache=bool(seam_cfg.get("perf_fast_island_cache", True)),
        )
        work_low_mesh = island_result.low_mesh
        seam_meta_seed.update(island_result.meta)
        seam_meta_seed["uv_halfedge_split_requested"] = True
        if island_result.high is not None:
            high_face_island = island_result.high.face_labels.astype(np.int64, copy=False)
        if island_result.semantic is not None:
            fixed_face_island = island_result.semantic.face_labels.astype(np.int64, copy=False)
            fixed_face_conflict = island_result.semantic.face_conflict.astype(np.bool_, copy=False)
            fixed_face_confidence = island_result.semantic.face_confidence.astype(np.float32, copy=False)
        if high_face_island is None:
            seam_meta_seed["uv_halfedge_split_fallback_to_legacy"] = True
            seam_strategy_used = "fallback_legacy"
            if seam_validation_strict:
                raise RuntimeError(
                    "halfedge_island failed: "
                    + str(island_result.validation_error or seam_meta_seed.get("uv_high_island_error", "unknown error"))
                )
        else:
            seam_strategy_used = "halfedge_island"
    else:
        seam_meta_seed["uv_halfedge_split_requested"] = False
        seam_meta_seed["uv_halfedge_split_topology_applied"] = False
        seam_meta_seed["uv_halfedge_split_fallback_to_legacy"] = False
    seam_meta_seed["uv_seam_validation_strict"] = bool(seam_validation_strict)
    seam_meta_seed["uv_seam_validation_require_closed_loops"] = bool(seam_validation_require_closed)
    seam_meta_seed["uv_seam_validation_require_pure_components"] = bool(seam_validation_require_pure)

    sample = sample_low_mesh(work_low_mesh, cfg["sample"])
    sample_points = sample["points"]
    sample_face_ids = sample["face_ids"]
    sample_bary = sample["bary"]
    sample_normals = sample["normals"]
    sample_area_weights = sample["area_weights"]

    n_faces = int(len(work_low_mesh.faces))
    if seam_strategy_requested != "halfedge_island":
        high_face_island, _, seam_strategy_used, seam_meta_high, guard_error = _prepare_high_islands(
            high_mesh=high_mesh,
            high_uv=high_uv,
            seam_cfg=seam_cfg,
            seam_strategy_requested=seam_strategy_requested,
            island_guard_requested=island_guard_requested,
        )
        seam_meta_seed.update(seam_meta_high)

    iter_energy_data: list[float] = []
    iter_label_change_ratio: list[float] = []
    iter_conflict_face_ratio: list[float] = []
    iter_unknown_face_ratio: list[float] = []
    iter_valid_sample_ratio: list[float] = []
    iter_guard_mode_used: list[str] = []

    best_state: Optional[Dict[str, Any]] = None
    best_energy = float("inf")
    best_iter = -1

    prev_face_island: Optional[np.ndarray] = None
    prev_face_confidence: Optional[np.ndarray] = None
    prev_energy: Optional[float] = None
    stable_count = 0
    early_stop_reason = "max_iters_reached"

    for iter_idx in range(max_iters):
        iter_num = iter_idx + 1
        guard_mode_used = island_guard_mode_requested
        if island_guard_mode_requested == "soft" and strict_mode_from_iter > 0 and iter_num >= strict_mode_from_iter:
            guard_mode_used = "strict"

        corr_pass = _compute_pass_correspondence(
            sample_points=sample_points,
            sample_normals=sample_normals,
            sample_face_ids=sample_face_ids,
            n_faces=n_faces,
            corr_cfg=corr_cfg,
            high_ctx=high_ctx,
            high_face_island=high_face_island,
            island_guard_requested=island_guard_requested,
            island_guard_mode_requested=island_guard_mode_requested,
            island_guard_mode_used=guard_mode_used,
            island_guard_allow_unknown=island_guard_allow_unknown,
            island_guard_conf_min=island_guard_conf_min,
            island_guard_fallback_policy=island_guard_fallback_policy,
            expected_face_island=fixed_face_island if fixed_face_island is not None else prev_face_island,
            expected_face_confidence=fixed_face_confidence if fixed_face_confidence is not None else prev_face_confidence,
            fixed_face_island=fixed_face_island,
            fixed_face_conflict=fixed_face_conflict,
            fixed_face_confidence=fixed_face_confidence,
            min_valid_samples_per_face=int(seam_cfg.get("min_valid_samples_per_face", 2)),
            guard_error=guard_error,
        )

        pass_inputs = _prepare_solver_inputs(
            low_mesh=work_low_mesh,
            sample_face_ids=sample_face_ids,
            target_uv=corr_pass["target_uv"],
            target_face_ids=corr_pass["target_face_ids"],
            valid_mask=corr_pass["valid_mask"],
            seam_cfg=seam_cfg,
            seam_strategy_used=seam_strategy_used,
            high_face_island=high_face_island,
            low_face_island=corr_pass["low_face_island"],
            low_face_conflict=corr_pass["low_face_conflict"],
            unknown_face_policy=unknown_face_policy,
            seam_validation_strict=seam_validation_strict,
            seam_validation_require_closed=seam_validation_require_closed,
            seam_validation_require_pure_components=seam_validation_require_pure,
            precomputed_island_result=island_result,
        )

        solve_pass = _solve_pass(
            high_mesh=high_mesh,
            high_uv=high_uv,
            image=image,
            solve_mesh=pass_inputs["solve_mesh"],
            sample_face_ids=sample_face_ids,
            sample_bary=sample_bary,
            sample_area_weights=sample_area_weights,
            target_uv=corr_pass["target_uv"],
            fallback_used_mask=corr_pass["fallback_used_mask"],
            solver_valid_mask=pass_inputs["solver_valid_mask"],
            smooth_face_mask=pass_inputs["smooth_face_mask"],
            corr_cfg=corr_cfg,
            solve_cfg=solve_cfg,
            tex_weight_cfg=tex_weight_cfg,
            resolved=resolved,
        )

        conflict_ratio = (
            float(np.count_nonzero(corr_pass["low_face_conflict"]) / max(1, n_faces)) if n_faces > 0 else 0.0
        )
        unknown_ratio = float(np.count_nonzero(corr_pass["low_face_island"] < 0) / max(1, n_faces)) if n_faces > 0 else 0.0
        valid_sample_ratio = (
            float(np.mean(pass_inputs["solver_valid_mask"])) if pass_inputs["solver_valid_mask"].size > 0 else 0.0
        )
        label_change = 1.0
        if prev_face_island is not None:
            label_change = float(np.mean(prev_face_island != corr_pass["low_face_island"]))

        iter_label_change_ratio.append(label_change)
        iter_conflict_face_ratio.append(conflict_ratio)
        iter_unknown_face_ratio.append(unknown_ratio)
        iter_valid_sample_ratio.append(valid_sample_ratio)
        iter_guard_mode_used.append(guard_mode_used if corr_pass["guard_meta"]["uv_island_guard_enabled"] else "off")

        prev_face_island = corr_pass["low_face_island"].copy()
        prev_face_confidence = corr_pass["low_face_confidence"].copy()

        if not solve_pass["ok"]:
            iter_energy_data.append(float("nan"))
            if best_state is not None and iter_num >= min_iters:
                early_stop_reason = "no_valid_samples_after_previous_success"
                break
            if best_state is None:
                early_stop_reason = "no_valid_samples"
                break
            continue

        iter_energy = float(solve_pass["energy_data"])
        iter_energy_data.append(iter_energy)

        if np.isfinite(iter_energy) and iter_energy < best_energy:
            best_energy = iter_energy
            best_iter = iter_idx
            best_state = {
                "corr": corr_pass,
                "inputs": pass_inputs,
                "solve": solve_pass,
            }

        rel_energy = float("inf")
        if prev_energy is not None and np.isfinite(prev_energy):
            rel_energy = abs(prev_energy - iter_energy) / max(abs(prev_energy), 1e-12)
        prev_energy = iter_energy

        stable = (label_change <= label_change_tol) and (rel_energy <= energy_rel_tol)
        if stable:
            stable_count += 1
        else:
            stable_count = 0

        if iter_num >= min_iters and stable_count >= patience:
            early_stop_reason = "converged"
            break

    if best_state is None:
        fallback_mesh = work_low_mesh if seam_strategy_requested == "halfedge_island" else low_mesh
        mapped_uv, stats = barycentric_mapper(high_mesh, fallback_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "barycentric_fallback_no_valid_samples"
        stats["uv_project_error"] = "Iterative hybrid correspondence had no valid samples after seam filtering"
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta_seed)
        stats["uv_iterative_enabled"] = True
        stats["uv_iter_count"] = int(len(iter_energy_data))
        stats["uv_iter_energy_data"] = [None if not np.isfinite(v) else float(v) for v in iter_energy_data]
        stats["uv_iter_label_change_ratio"] = [float(v) for v in iter_label_change_ratio]
        stats["uv_iter_conflict_face_ratio"] = [float(v) for v in iter_conflict_face_ratio]
        stats["uv_iter_unknown_face_ratio"] = [float(v) for v in iter_unknown_face_ratio]
        stats["uv_iter_valid_sample_ratio"] = [float(v) for v in iter_valid_sample_ratio]
        stats["uv_iter_guard_mode_used"] = iter_guard_mode_used
        stats["uv_iter_early_stop_reason"] = early_stop_reason
        stats["uv_iter_best_index"] = None
        return mapped_uv, stats, {"local_vertex_split_applied": False, "quality_mesh": fallback_mesh}

    final_corr = best_state["corr"]
    final_inputs = best_state["inputs"]
    final_solve = best_state["solve"]
    mapped_uv = final_solve["mapped_uv"]

    seam_face_ids = final_inputs["seam_face_ids"]
    seam_corner_uv = np.zeros((0, 3, 2), dtype=np.float32)
    local_split_applied = False

    if seam_strategy_used != "halfedge_island" and bool(seam_cfg.get("local_vertex_split", True)) and seam_face_ids.size > 0:
        low_faces = np.asarray(work_low_mesh.faces, dtype=np.int64)
        low_verts = np.asarray(work_low_mesh.vertices, dtype=np.float32)
        low_vn = get_vertex_normals(work_low_mesh)

        seam_corner_vid = low_faces[seam_face_ids]
        corner_points = low_verts[seam_corner_vid].reshape(-1, 3)
        corner_normals = low_vn[seam_corner_vid].reshape(-1, 3)
        corner_corr = correspond_points_hybrid(
            points=corner_points,
            point_normals=corner_normals,
            corr_cfg=corr_cfg,
            high_ctx=high_ctx,
        )
        corner_uv = corner_corr["target_uv"].reshape(-1, 3, 2)
        corner_valid = corner_corr["valid_mask"].reshape(-1, 3)
        fallback_corner_uv = mapped_uv[seam_corner_vid]
        seam_corner_uv = np.where(corner_valid[..., None], corner_uv, fallback_corner_uv).astype(np.float32)
        local_split_applied = True

    method_stats = {
        "uv_correspondence_primary_ratio": float(final_corr["primary_mask"].mean()) if final_corr["primary_mask"].size > 0 else 0.0,
        "uv_correspondence_success_ratio": float(final_corr["valid_mask"].mean()) if final_corr["valid_mask"].size > 0 else 0.0,
        "uv_correspondence_invalid_ratio": float((~final_corr["valid_mask"]).mean()) if final_corr["valid_mask"].size > 0 else 0.0,
        "uv_solve_num_samples": int(len(final_solve["solve_target_uv"])),
        "uv_solve_valid_sample_ratio": float(final_inputs["solver_valid_mask"].mean()) if final_inputs["solver_valid_mask"].size > 0 else 0.0,
        "uv_solve_num_faces": int(len(final_inputs["solve_mesh"].faces)),
        "uv_color_reproj_l1": final_solve["color_l1"],
        "uv_color_reproj_l2": final_solve["color_l2"],
        "uv_local_vertex_split_faces": int(len(seam_face_ids)) if local_split_applied else 0,
        "uv_seam_strategy_requested": seam_strategy_requested,
        "uv_seam_strategy_effective": seam_strategy_used,
        "uv_seam_strategy_used": seam_strategy_used,
        "uv_iterative_enabled": True,
        "uv_iter_count": int(len(iter_energy_data)),
        "uv_iter_energy_data": [None if not np.isfinite(v) else float(v) for v in iter_energy_data],
        "uv_iter_label_change_ratio": [float(v) for v in iter_label_change_ratio],
        "uv_iter_conflict_face_ratio": [float(v) for v in iter_conflict_face_ratio],
        "uv_iter_unknown_face_ratio": [float(v) for v in iter_unknown_face_ratio],
        "uv_iter_valid_sample_ratio": [float(v) for v in iter_valid_sample_ratio],
        "uv_iter_guard_mode_used": iter_guard_mode_used,
        "uv_iter_early_stop_reason": early_stop_reason,
        "uv_iter_best_index": int(best_iter + 1),
        **seam_meta_seed,
        **final_inputs["seam_meta"],
        **final_corr["guard_meta"],
        **final_solve["solve_meta"],
    }
    if bool(seam_cfg.get("emit_validation_sidecar_data", False)) and seam_strategy_used == "halfedge_island":
        method_stats["uv_low_face_semantic_labels"] = np.asarray(
            final_corr["low_face_island"], dtype=np.int64
        ).tolist()
        if island_result is not None and island_result.semantic is not None:
            method_stats["uv_low_face_semantic_confidence"] = np.asarray(
                island_result.semantic.face_confidence, dtype=np.float32
            ).tolist()
            method_stats["uv_low_face_semantic_conflict"] = np.asarray(
                island_result.semantic.face_conflict, dtype=np.bool_
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_labels"] = np.asarray(
                island_result.semantic.pre_bfs_labels, dtype=np.int64
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_confidence"] = np.asarray(
                island_result.semantic.pre_bfs_confidence, dtype=np.float32
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_state"] = np.asarray(
                island_result.semantic.pre_bfs_state, dtype=np.uint8
            ).tolist()
            if island_result.semantic.pre_cleanup_labels is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_labels"] = np.asarray(
                    island_result.semantic.pre_cleanup_labels, dtype=np.int64
                ).tolist()
            if island_result.semantic.pre_cleanup_confidence is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_confidence"] = np.asarray(
                    island_result.semantic.pre_cleanup_confidence, dtype=np.float32
                ).tolist()
            if island_result.semantic.pre_cleanup_conflict is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_conflict"] = np.asarray(
                    island_result.semantic.pre_cleanup_conflict, dtype=np.bool_
                ).tolist()
            if island_result.semantic.soft_top1_label is not None:
                method_stats["uv_low_face_semantic_soft_top1_label"] = np.asarray(
                    island_result.semantic.soft_top1_label, dtype=np.int64
                ).tolist()
            if island_result.semantic.soft_top1_prob is not None:
                method_stats["uv_low_face_semantic_soft_top1_prob"] = np.asarray(
                    island_result.semantic.soft_top1_prob, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_top2_label is not None:
                method_stats["uv_low_face_semantic_soft_top2_label"] = np.asarray(
                    island_result.semantic.soft_top2_label, dtype=np.int64
                ).tolist()
            if island_result.semantic.soft_top2_prob is not None:
                method_stats["uv_low_face_semantic_soft_top2_prob"] = np.asarray(
                    island_result.semantic.soft_top2_prob, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_entropy is not None:
                method_stats["uv_low_face_semantic_soft_entropy"] = np.asarray(
                    island_result.semantic.soft_entropy, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_candidate_count is not None:
                method_stats["uv_low_face_semantic_soft_candidate_count"] = np.asarray(
                    island_result.semantic.soft_candidate_count, dtype=np.int32
                ).tolist()
        method_stats["uv_low_seam_edges"] = np.asarray(final_inputs["seam_edges"], dtype=np.int64).tolist()

    if seam_strategy_used == "halfedge_island":
        export_payload = {
            "local_vertex_split_applied": False,
            "halfedge_split_topology": True,
            "split_vertices": np.asarray(final_inputs["solve_mesh"].vertices, dtype=np.float32),
            "split_faces": np.asarray(final_inputs["solve_mesh"].faces, dtype=np.int64),
            "quality_mesh": final_inputs["solve_mesh"],
        }
    else:
        export_payload = {
            "local_vertex_split_applied": local_split_applied,
            "seam_face_ids": seam_face_ids,
            "seam_corner_uv": seam_corner_uv,
            "quality_mesh": final_inputs["solve_mesh"],
        }

    return mapped_uv, method_stats, export_payload


def _run_hybrid_legacy_once(
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
    resolved = resolve_device(device)
    if resolved != "cuda":
        mapped_uv, stats = nearest_mapper(high_mesh, low_mesh, high_uv)
        stats["uv_mode_used"] = "nearest_vertex_fallback"
        stats["uv_project_error"] = "CUDA unavailable for hybrid optimization; fell back to nearest vertex"
        stats["uv_iterative_enabled"] = False
        stats["uv_iter_count"] = 1
        stats["uv_iter_early_stop_reason"] = "iterative_disabled"
        return mapped_uv, stats, {"local_vertex_split_applied": False}

    corr_cfg = cfg["correspondence"]
    seam_cfg = cfg.get("seam", {})
    solve_cfg = cfg["solve"]
    tex_weight_cfg = cfg["texture_weight"]
    seam_strategy_requested = str(seam_cfg.get("strategy", "legacy")).strip().lower()
    if seam_strategy_requested not in {"legacy", "halfedge_island"}:
        seam_strategy_requested = "legacy"
    island_guard_requested = bool(seam_cfg.get("uv_island_guard_enabled", True))
    island_guard_mode_requested = str(seam_cfg.get("uv_island_guard_mode", "soft")).strip().lower()
    if island_guard_mode_requested not in {"soft", "strict"}:
        island_guard_mode_requested = "soft"
    island_guard_conf_min = float(seam_cfg.get("uv_island_guard_confidence_min", 0.55))
    island_guard_allow_unknown = bool(seam_cfg.get("uv_island_guard_allow_unknown", False))
    island_guard_fallback_policy = str(
        seam_cfg.get("uv_island_guard_fallback", "nearest_same_island_then_udf")
    )
    validation = resolve_seam_validation_settings(seam_cfg)
    seam_validation_strict = bool(validation.strict)
    seam_validation_require_closed = bool(validation.require_closed_loops)
    seam_validation_require_pure = bool(validation.require_pure_components)

    high_ctx = build_high_cuda_context(
        high_mesh=high_mesh,
        high_uv=high_uv,
        device=resolved,
    )
    runtime_meta = dict(high_ctx.get("runtime_diag") or {})
    runtime_meta.pop("reason", None)
    work_low_mesh = low_mesh
    seam_meta_seed: Dict[str, Any] = {}
    seam_meta_seed.update(runtime_meta)
    seam_meta_seed["uv_island_validation_mode"] = validation.mode
    fixed_face_island: Optional[np.ndarray] = None
    fixed_face_conflict: Optional[np.ndarray] = None
    fixed_face_confidence: Optional[np.ndarray] = None
    high_face_island: Optional[np.ndarray] = None
    guard_error: Optional[str] = None
    seam_strategy_used = seam_strategy_requested
    island_result: Optional[IslandPipelineResult] = None
    if seam_strategy_requested == "halfedge_island":
        island_result = run_halfedge_island_pipeline(
            high_mesh=high_mesh,
            high_uv=high_uv,
            low_mesh=low_mesh,
            high_ctx=high_ctx,
            seam_cfg=seam_cfg,
            corr_cfg=corr_cfg,
            use_high_island_cache=bool(seam_cfg.get("perf_fast_island_cache", True)),
        )
        work_low_mesh = island_result.low_mesh
        seam_meta_seed.update(island_result.meta)
        seam_meta_seed["uv_halfedge_split_requested"] = True
        if island_result.high is not None:
            high_face_island = island_result.high.face_labels.astype(np.int64, copy=False)
        if island_result.semantic is not None:
            fixed_face_island = island_result.semantic.face_labels.astype(np.int64, copy=False)
            fixed_face_conflict = island_result.semantic.face_conflict.astype(np.bool_, copy=False)
            fixed_face_confidence = island_result.semantic.face_confidence.astype(np.float32, copy=False)
        if high_face_island is None:
            seam_meta_seed["uv_halfedge_split_fallback_to_legacy"] = True
            seam_strategy_used = "fallback_legacy"
            if seam_validation_strict:
                raise RuntimeError(
                    "halfedge_island failed: "
                    + str(island_result.validation_error or seam_meta_seed.get("uv_high_island_error", "unknown error"))
                )
        else:
            seam_strategy_used = "halfedge_island"
    else:
        seam_meta_seed["uv_halfedge_split_requested"] = False
        seam_meta_seed["uv_halfedge_split_topology_applied"] = False
        seam_meta_seed["uv_halfedge_split_fallback_to_legacy"] = False
        high_face_island, _, seam_strategy_used, seam_meta_high, guard_error = _prepare_high_islands(
            high_mesh=high_mesh,
            high_uv=high_uv,
            seam_cfg=seam_cfg,
            seam_strategy_requested=seam_strategy_requested,
            island_guard_requested=island_guard_requested,
        )
        seam_meta_seed.update(seam_meta_high)
    seam_meta_seed["uv_seam_validation_strict"] = bool(seam_validation_strict)
    seam_meta_seed["uv_seam_validation_require_closed_loops"] = bool(seam_validation_require_closed)
    seam_meta_seed["uv_seam_validation_require_pure_components"] = bool(seam_validation_require_pure)

    sample = sample_low_mesh(work_low_mesh, cfg["sample"])
    sample_points = sample["points"]
    sample_face_ids = sample["face_ids"]
    sample_bary = sample["bary"]
    sample_normals = sample["normals"]
    sample_area_weights = sample["area_weights"]
    n_faces = int(len(work_low_mesh.faces))

    corr_pass = _compute_pass_correspondence(
        sample_points=sample_points,
        sample_normals=sample_normals,
        sample_face_ids=sample_face_ids,
        n_faces=n_faces,
        corr_cfg=corr_cfg,
        high_ctx=high_ctx,
        high_face_island=high_face_island,
        island_guard_requested=island_guard_requested,
        island_guard_mode_requested=island_guard_mode_requested,
        island_guard_mode_used=island_guard_mode_requested,
        island_guard_allow_unknown=island_guard_allow_unknown,
        island_guard_conf_min=island_guard_conf_min,
        island_guard_fallback_policy=island_guard_fallback_policy,
        expected_face_island=fixed_face_island,
        expected_face_confidence=fixed_face_confidence,
        fixed_face_island=fixed_face_island,
        fixed_face_conflict=fixed_face_conflict,
        fixed_face_confidence=fixed_face_confidence,
        min_valid_samples_per_face=int(seam_cfg.get("min_valid_samples_per_face", 2)),
        guard_error=guard_error,
    )

    # Legacy non-iterative mode keeps one extra guarded pass for non-fixed labels.
    if (
        fixed_face_island is None
        and island_guard_requested
        and high_face_island is not None
        and np.any(corr_pass["low_face_island"] >= 0)
    ):
        corr_pass = _compute_pass_correspondence(
            sample_points=sample_points,
            sample_normals=sample_normals,
            sample_face_ids=sample_face_ids,
            n_faces=n_faces,
            corr_cfg=corr_cfg,
            high_ctx=high_ctx,
            high_face_island=high_face_island,
            island_guard_requested=island_guard_requested,
            island_guard_mode_requested=island_guard_mode_requested,
            island_guard_mode_used=island_guard_mode_requested,
            island_guard_allow_unknown=island_guard_allow_unknown,
            island_guard_conf_min=island_guard_conf_min,
            island_guard_fallback_policy=island_guard_fallback_policy,
            expected_face_island=corr_pass["low_face_island"],
            expected_face_confidence=corr_pass["low_face_confidence"],
            fixed_face_island=None,
            fixed_face_conflict=None,
            fixed_face_confidence=None,
            min_valid_samples_per_face=int(seam_cfg.get("min_valid_samples_per_face", 2)),
            guard_error=guard_error,
        )

    pass_inputs = _prepare_solver_inputs(
        low_mesh=work_low_mesh,
        sample_face_ids=sample_face_ids,
        target_uv=corr_pass["target_uv"],
        target_face_ids=corr_pass["target_face_ids"],
        valid_mask=corr_pass["valid_mask"],
        seam_cfg=seam_cfg,
        seam_strategy_used=seam_strategy_used,
        high_face_island=high_face_island,
        low_face_island=corr_pass["low_face_island"],
        low_face_conflict=corr_pass["low_face_conflict"],
        unknown_face_policy="exclude",
        seam_validation_strict=seam_validation_strict,
        seam_validation_require_closed=seam_validation_require_closed,
        seam_validation_require_pure_components=seam_validation_require_pure,
        precomputed_island_result=island_result,
    )

    if not np.any(pass_inputs["solver_valid_mask"]):
        fallback_mesh = work_low_mesh if seam_strategy_requested == "halfedge_island" else low_mesh
        mapped_uv, stats = barycentric_mapper(high_mesh, fallback_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "barycentric_fallback_no_valid_samples"
        stats["uv_project_error"] = "Hybrid correspondence had no valid samples after seam filtering"
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta_seed)
        stats.update(pass_inputs["seam_meta"])
        stats.update(corr_pass["guard_meta"])
        stats["uv_iterative_enabled"] = False
        stats["uv_iter_count"] = 1
        stats["uv_iter_early_stop_reason"] = "iterative_disabled"
        return mapped_uv, stats, {"local_vertex_split_applied": False, "quality_mesh": fallback_mesh}

    solve_pass = _solve_pass(
        high_mesh=high_mesh,
        high_uv=high_uv,
        image=image,
        solve_mesh=pass_inputs["solve_mesh"],
        sample_face_ids=sample_face_ids,
        sample_bary=sample_bary,
        sample_area_weights=sample_area_weights,
        target_uv=corr_pass["target_uv"],
        fallback_used_mask=corr_pass["fallback_used_mask"],
        solver_valid_mask=pass_inputs["solver_valid_mask"],
        smooth_face_mask=pass_inputs["smooth_face_mask"],
        corr_cfg=corr_cfg,
        solve_cfg=solve_cfg,
        tex_weight_cfg=tex_weight_cfg,
        resolved=resolved,
    )

    if not solve_pass["ok"]:
        fallback_mesh = work_low_mesh if seam_strategy_requested == "halfedge_island" else low_mesh
        mapped_uv, stats = barycentric_mapper(high_mesh, fallback_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "barycentric_fallback_solver_error"
        stats["uv_project_error"] = "Hybrid solve failed after seam preparation"
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta_seed)
        stats.update(pass_inputs["seam_meta"])
        stats.update(corr_pass["guard_meta"])
        stats["uv_iterative_enabled"] = False
        stats["uv_iter_count"] = 1
        stats["uv_iter_early_stop_reason"] = "iterative_disabled"
        return mapped_uv, stats, {"local_vertex_split_applied": False, "quality_mesh": fallback_mesh}

    mapped_uv = solve_pass["mapped_uv"]
    seam_face_ids = pass_inputs["seam_face_ids"]
    seam_corner_uv = np.zeros((0, 3, 2), dtype=np.float32)
    local_split_applied = False
    if seam_strategy_used != "halfedge_island" and bool(seam_cfg.get("local_vertex_split", True)) and seam_face_ids.size > 0:
        low_faces = np.asarray(work_low_mesh.faces, dtype=np.int64)
        low_verts = np.asarray(work_low_mesh.vertices, dtype=np.float32)
        low_vn = get_vertex_normals(work_low_mesh)

        seam_corner_vid = low_faces[seam_face_ids]
        corner_points = low_verts[seam_corner_vid].reshape(-1, 3)
        corner_normals = low_vn[seam_corner_vid].reshape(-1, 3)
        corner_corr = correspond_points_hybrid(
            points=corner_points,
            point_normals=corner_normals,
            corr_cfg=corr_cfg,
            high_ctx=high_ctx,
        )
        corner_uv = corner_corr["target_uv"].reshape(-1, 3, 2)
        corner_valid = corner_corr["valid_mask"].reshape(-1, 3)
        fallback_corner_uv = mapped_uv[seam_corner_vid]
        seam_corner_uv = np.where(corner_valid[..., None], corner_uv, fallback_corner_uv).astype(np.float32)
        local_split_applied = True

    method_stats = {
        "uv_correspondence_primary_ratio": float(corr_pass["primary_mask"].mean()) if corr_pass["primary_mask"].size > 0 else 0.0,
        "uv_correspondence_success_ratio": float(corr_pass["valid_mask"].mean()) if corr_pass["valid_mask"].size > 0 else 0.0,
        "uv_correspondence_invalid_ratio": float((~corr_pass["valid_mask"]).mean()) if corr_pass["valid_mask"].size > 0 else 0.0,
        "uv_solve_num_samples": int(len(solve_pass["solve_target_uv"])),
        "uv_solve_valid_sample_ratio": float(pass_inputs["solver_valid_mask"].mean()) if pass_inputs["solver_valid_mask"].size > 0 else 0.0,
        "uv_solve_num_faces": int(len(pass_inputs["solve_mesh"].faces)),
        "uv_color_reproj_l1": solve_pass["color_l1"],
        "uv_color_reproj_l2": solve_pass["color_l2"],
        "uv_local_vertex_split_faces": int(len(seam_face_ids)) if local_split_applied else 0,
        "uv_seam_strategy_requested": seam_strategy_requested,
        "uv_seam_strategy_effective": seam_strategy_used,
        "uv_seam_strategy_used": seam_strategy_used,
        "uv_iterative_enabled": False,
        "uv_iter_count": 1,
        "uv_iter_energy_data": [],
        "uv_iter_label_change_ratio": [],
        "uv_iter_conflict_face_ratio": [],
        "uv_iter_unknown_face_ratio": [],
        "uv_iter_valid_sample_ratio": [],
        "uv_iter_guard_mode_used": [],
        "uv_iter_early_stop_reason": "iterative_disabled",
        "uv_iter_best_index": 1,
        **seam_meta_seed,
        **pass_inputs["seam_meta"],
        **corr_pass["guard_meta"],
        **solve_pass["solve_meta"],
    }
    if bool(seam_cfg.get("emit_validation_sidecar_data", False)) and seam_strategy_used == "halfedge_island":
        method_stats["uv_low_face_semantic_labels"] = np.asarray(
            corr_pass["low_face_island"], dtype=np.int64
        ).tolist()
        if island_result is not None and island_result.semantic is not None:
            method_stats["uv_low_face_semantic_confidence"] = np.asarray(
                island_result.semantic.face_confidence, dtype=np.float32
            ).tolist()
            method_stats["uv_low_face_semantic_conflict"] = np.asarray(
                island_result.semantic.face_conflict, dtype=np.bool_
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_labels"] = np.asarray(
                island_result.semantic.pre_bfs_labels, dtype=np.int64
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_confidence"] = np.asarray(
                island_result.semantic.pre_bfs_confidence, dtype=np.float32
            ).tolist()
            method_stats["uv_low_face_semantic_pre_bfs_state"] = np.asarray(
                island_result.semantic.pre_bfs_state, dtype=np.uint8
            ).tolist()
            if island_result.semantic.pre_cleanup_labels is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_labels"] = np.asarray(
                    island_result.semantic.pre_cleanup_labels, dtype=np.int64
                ).tolist()
            if island_result.semantic.pre_cleanup_confidence is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_confidence"] = np.asarray(
                    island_result.semantic.pre_cleanup_confidence, dtype=np.float32
                ).tolist()
            if island_result.semantic.pre_cleanup_conflict is not None:
                method_stats["uv_low_face_semantic_pre_cleanup_conflict"] = np.asarray(
                    island_result.semantic.pre_cleanup_conflict, dtype=np.bool_
                ).tolist()
            if island_result.semantic.soft_top1_label is not None:
                method_stats["uv_low_face_semantic_soft_top1_label"] = np.asarray(
                    island_result.semantic.soft_top1_label, dtype=np.int64
                ).tolist()
            if island_result.semantic.soft_top1_prob is not None:
                method_stats["uv_low_face_semantic_soft_top1_prob"] = np.asarray(
                    island_result.semantic.soft_top1_prob, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_top2_label is not None:
                method_stats["uv_low_face_semantic_soft_top2_label"] = np.asarray(
                    island_result.semantic.soft_top2_label, dtype=np.int64
                ).tolist()
            if island_result.semantic.soft_top2_prob is not None:
                method_stats["uv_low_face_semantic_soft_top2_prob"] = np.asarray(
                    island_result.semantic.soft_top2_prob, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_entropy is not None:
                method_stats["uv_low_face_semantic_soft_entropy"] = np.asarray(
                    island_result.semantic.soft_entropy, dtype=np.float32
                ).tolist()
            if island_result.semantic.soft_candidate_count is not None:
                method_stats["uv_low_face_semantic_soft_candidate_count"] = np.asarray(
                    island_result.semantic.soft_candidate_count, dtype=np.int32
                ).tolist()
        method_stats["uv_low_seam_edges"] = np.asarray(pass_inputs["seam_edges"], dtype=np.int64).tolist()
    if seam_strategy_used == "halfedge_island":
        export_payload = {
            "local_vertex_split_applied": False,
            "halfedge_split_topology": True,
            "split_vertices": np.asarray(pass_inputs["solve_mesh"].vertices, dtype=np.float32),
            "split_faces": np.asarray(pass_inputs["solve_mesh"].faces, dtype=np.int64),
            "quality_mesh": pass_inputs["solve_mesh"],
        }
    else:
        export_payload = {
            "local_vertex_split_applied": local_split_applied,
            "seam_face_ids": seam_face_ids,
            "seam_corner_uv": seam_corner_uv,
            "quality_mesh": pass_inputs["solve_mesh"],
        }
    return mapped_uv, method_stats, export_payload


def run_hybrid_global_opt(
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
    iterative_cfg = cfg.get("iterative", {})
    iterative_enabled = bool(iterative_cfg.get("enabled", True))

    if iterative_enabled:
        return _run_hybrid_iterative(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            high_uv=high_uv,
            image=image,
            device=device,
            cfg=cfg,
            barycentric_mapper=barycentric_mapper,
        )

    return _run_hybrid_legacy_once(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_uv=high_uv,
        image=image,
        device=device,
        cfg=cfg,
        nearest_mapper=nearest_mapper,
        barycentric_mapper=barycentric_mapper,
    )


__all__ = ["run_hybrid_global_opt"]
