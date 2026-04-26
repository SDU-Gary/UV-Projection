#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix, diags

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.mesh_io import MeshIO
from faithc_infra.services.uv import DEFAULT_OPTIONS, deep_merge_dict
from faithc_infra.services.uv.island_pipeline import compute_cached_high_face_uv_islands
from faithc_infra.services.uv.linear_solver import (
    build_cuda_sparse_system,
    interpolate_sample_uv,
    mesh_laplacian,
    nearest_vertex_uv,
    solve_linear_cuda_pcg,
    solve_linear_robust,
)
from faithc_infra.services.uv.method2_pipeline import (
    Method2InternalState,
    _aggregate_face_target_jacobians,
    _build_gradient_constraint_system,
    _compute_face_geometry_pinv,
    _compute_high_face_jacobians,
    _solve_poisson_uv,
    _solve_poisson_uv_by_island,
    run_method2_gradient_poisson,
)
from faithc_infra.services.uv.quality import compute_uv_quality, face_stretch_anisotropy
from faithc_infra.services.uv.solve_constraints import summarize_uv_box_feasibility
from faithc_infra.services.uv.texture_io import extract_uv, resolve_basecolor_image
from faithc_infra.services.uv_projector import UVProjector


LARGE_STATS_KEYS = {
    "uv_low_face_semantic_labels",
    "uv_low_face_semantic_confidence",
    "uv_low_face_semantic_conflict",
    "uv_low_face_semantic_pre_bfs_labels",
    "uv_low_face_semantic_pre_bfs_confidence",
    "uv_low_face_semantic_pre_bfs_state",
    "uv_low_face_semantic_pre_cleanup_labels",
    "uv_low_face_semantic_pre_cleanup_confidence",
    "uv_low_face_semantic_pre_cleanup_conflict",
    "uv_low_face_semantic_soft_top1_label",
    "uv_low_face_semantic_soft_top1_prob",
    "uv_low_face_semantic_soft_top2_label",
    "uv_low_face_semantic_soft_top2_prob",
    "uv_low_face_semantic_soft_entropy",
    "uv_low_face_semantic_soft_candidate_count",
    "uv_low_seam_edges",
    "uv_m2_face_accepted_samples",
    "uv_m2_face_total_samples",
}


def _load_mesh(path: Path) -> trimesh.Trimesh:
    return MeshIO.load_mesh(path, process=False)


def _safe_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    uv = extract_uv(mesh)
    if uv is None:
        raise RuntimeError("mesh has no valid per-vertex UV coordinates")
    uv_np = np.asarray(uv, dtype=np.float64)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        raise RuntimeError("mesh UV shape invalid")
    if uv_np.shape[0] != int(len(mesh.vertices)):
        raise RuntimeError("mesh UV/vertex length mismatch")
    return uv_np


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _boundary_vertex_ids(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    ).astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(boundary_edges.reshape(-1)).astype(np.int64, copy=False)


def _face_neighbors(mesh: trimesh.Trimesh) -> List[List[int]]:
    n_faces = int(len(mesh.faces))
    neigh: List[List[int]] = [[] for _ in range(n_faces)]
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 2:
        return neigh
    for a, b in adj.tolist():
        ia = int(a)
        ib = int(b)
        if 0 <= ia < n_faces and 0 <= ib < n_faces and ia != ib:
            neigh[ia].append(ib)
            neigh[ib].append(ia)
    return neigh


def _connected_face_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    neighbors = _face_neighbors(mesh)
    labels = np.full((len(neighbors),), -1, dtype=np.int64)
    next_label = 0
    for fid in range(len(neighbors)):
        if labels[fid] >= 0:
            continue
        q: deque[int] = deque([int(fid)])
        labels[fid] = next_label
        while q:
            cur = q.popleft()
            for nb in neighbors[cur]:
                if labels[nb] < 0:
                    labels[nb] = next_label
                    q.append(int(nb))
        next_label += 1
    return labels


def _same_label_components(labels: np.ndarray, neighbors: List[List[int]]) -> List[Tuple[int, np.ndarray]]:
    labels_i64 = np.asarray(labels, dtype=np.int64).reshape(-1)
    seen = np.zeros((labels_i64.shape[0],), dtype=np.bool_)
    comps: List[Tuple[int, np.ndarray]] = []
    for fid in range(labels_i64.shape[0]):
        if seen[fid]:
            continue
        label = int(labels_i64[fid])
        q: deque[int] = deque([int(fid)])
        seen[fid] = True
        comp: List[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors[cur]:
                if not seen[nb] and int(labels_i64[nb]) == label:
                    seen[nb] = True
                    q.append(int(nb))
        comps.append((label, np.asarray(comp, dtype=np.int64)))
    return comps


def _weighted_quantile(values: np.ndarray, q: float) -> Optional[float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    qq = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(vals, qq))


def _signed_uv_area(mesh: trimesh.Trimesh, uv: np.ndarray) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = np.asarray(uv, dtype=np.float64)[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _stretch_only_bad_ratio(mesh: trimesh.Trimesh, uv: np.ndarray, stretch_factor: float = 1.5) -> float:
    stretch = face_stretch_anisotropy(mesh, uv)
    stretch = np.asarray(stretch, dtype=np.float64)
    stretch = stretch[np.isfinite(stretch)]
    if stretch.size == 0:
        return 0.0
    p95 = float(np.percentile(stretch, 95))
    thr = max(1.0, p95 * float(stretch_factor))
    return float(np.mean(stretch > thr))


def _quality_with_context(mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, Any]:
    uv_np = np.asarray(uv, dtype=np.float64)
    raw = dict(compute_uv_quality(mesh, uv_np))
    signed = _signed_uv_area(mesh, uv_np)
    abs_signed = np.abs(signed)
    stretch = face_stretch_anisotropy(mesh, uv_np)
    stretch = np.asarray(stretch, dtype=np.float64)
    finite_stretch = stretch[np.isfinite(stretch)]
    raw.update(
        {
            "uv_signed_negative_ratio": float(np.mean(signed < 0.0)) if signed.size > 0 else 0.0,
            "uv_signed_positive_ratio": float(np.mean(signed > 0.0)) if signed.size > 0 else 0.0,
            "uv_signed_near_zero_ratio": float(np.mean(abs_signed <= 1e-10)) if signed.size > 0 else 0.0,
            "uv_area_abs_p01": _weighted_quantile(abs_signed, 0.01),
            "uv_area_abs_p50": _weighted_quantile(abs_signed, 0.50),
            "uv_area_abs_p99": _weighted_quantile(abs_signed, 0.99),
            "uv_bad_tri_ratio_stretch_only": _stretch_only_bad_ratio(mesh, uv_np),
            "uv_stretch_p50": _weighted_quantile(finite_stretch, 0.50),
        }
    )
    return raw


def _compute_face_jacobians(face_geom_pinv: np.ndarray, mesh: trimesh.Trimesh, uv: np.ndarray) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_uv = np.asarray(uv, dtype=np.float64)[faces]
    du1 = tri_uv[:, 1] - tri_uv[:, 0]
    du2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_grad = np.stack([du1, du2], axis=2)
    return np.einsum("fij,fjk->fik", uv_grad, np.asarray(face_geom_pinv, dtype=np.float64), optimize=True)


def _jacobian_area_scale(jac: np.ndarray) -> np.ndarray:
    jj_t = np.einsum("fij,fkj->fik", jac, jac, optimize=True)
    det = np.linalg.det(jj_t)
    det = np.maximum(det, 0.0)
    return np.sqrt(det)


def _jacobian_diagnostics(
    target_jac: np.ndarray,
    solved_jac: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape(-1)
    tgt = np.asarray(target_jac, dtype=np.float64)
    sol = np.asarray(solved_jac, dtype=np.float64)
    if tgt.shape != sol.shape:
        raise RuntimeError("target/solved Jacobian shape mismatch")

    flat_t = tgt.reshape(tgt.shape[0], -1)
    flat_s = sol.reshape(sol.shape[0], -1)
    tgt_norm = np.linalg.norm(flat_t, axis=1)
    sol_norm = np.linalg.norm(flat_s, axis=1)
    diff = flat_s - flat_t
    diff_norm = np.linalg.norm(diff, axis=1)
    rel_err = diff_norm / np.maximum(tgt_norm, 1e-12)
    cosine = np.sum(flat_t * flat_s, axis=1) / np.maximum(tgt_norm * sol_norm, 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    tgt_area = _jacobian_area_scale(tgt)
    sol_area = _jacobian_area_scale(sol)
    area_ratio = sol_area / np.maximum(tgt_area, 1e-12)
    log_area_ratio = np.abs(np.log(np.maximum(area_ratio, 1e-12)))

    valid_idx = np.where(valid & np.isfinite(rel_err) & np.isfinite(cosine) & np.isfinite(log_area_ratio))[0]
    summary: Dict[str, Any] = {
        "valid_face_count": int(valid_idx.size),
        "total_face_count": int(valid.shape[0]),
    }
    if valid_idx.size == 0:
        summary.update(
            {
                "frob_rel_error_p50": None,
                "frob_rel_error_p95": None,
                "frob_rel_error_p99": None,
                "cosine_p05": None,
                "cosine_p50": None,
                "cosine_mean": None,
                "area_ratio_p50": None,
                "area_ratio_p95": None,
                "log_area_ratio_p50": None,
                "log_area_ratio_p95": None,
            }
        )
        return summary, rel_err, cosine, log_area_ratio

    rel_valid = rel_err[valid_idx]
    cos_valid = cosine[valid_idx]
    area_valid = area_ratio[valid_idx]
    log_area_valid = log_area_ratio[valid_idx]
    summary.update(
        {
            "frob_rel_error_p50": float(np.quantile(rel_valid, 0.50)),
            "frob_rel_error_p95": float(np.quantile(rel_valid, 0.95)),
            "frob_rel_error_p99": float(np.quantile(rel_valid, 0.99)),
            "cosine_p05": float(np.quantile(cos_valid, 0.05)),
            "cosine_p50": float(np.quantile(cos_valid, 0.50)),
            "cosine_mean": float(np.mean(cos_valid)),
            "area_ratio_p50": float(np.quantile(area_valid, 0.50)),
            "area_ratio_p95": float(np.quantile(area_valid, 0.95)),
            "log_area_ratio_p50": float(np.quantile(log_area_valid, 0.50)),
            "log_area_ratio_p95": float(np.quantile(log_area_valid, 0.95)),
        }
    )
    return summary, rel_err, cosine, log_area_ratio


def _sample_residual_summary(
    internal: Method2InternalState,
    mapped_uv: np.ndarray,
) -> Dict[str, Any]:
    sf = np.asarray(internal.solve_sample_face_ids, dtype=np.int64)
    sbary = np.asarray(internal.solve_sample_bary, dtype=np.float64)
    starget_uv = np.asarray(internal.solve_target_uv, dtype=np.float64)
    if sf.size == 0 or sbary.size == 0 or starget_uv.size == 0:
        return {
            "sample_count": 0,
            "residual_l2_mean": None,
            "residual_l2_p95": None,
            "residual_linf": None,
        }
    pred_uv = interpolate_sample_uv(
        np.asarray(internal.solve_mesh.faces, dtype=np.int64),
        sf,
        sbary,
        np.asarray(mapped_uv, dtype=np.float64),
    )
    residual = starget_uv - pred_uv
    l2 = np.linalg.norm(residual, axis=1)
    return {
        "sample_count": int(l2.shape[0]),
        "residual_l2_mean": float(np.mean(l2)),
        "residual_l2_p95": float(np.quantile(l2, 0.95)),
        "residual_linf": float(np.max(np.abs(residual))) if residual.size > 0 else None,
    }


def _coerce_face_count_array(raw: Any, n_faces: int) -> Optional[np.ndarray]:
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != int(n_faces):
        return None
    return arr.astype(np.int64, copy=False)


def _support_summary(
    *,
    face_valid: np.ndarray,
    face_accepted_samples: np.ndarray,
    face_total_samples: Optional[np.ndarray],
) -> Dict[str, Any]:
    valid = np.asarray(face_valid, dtype=np.bool_).reshape(-1)
    accepted = np.asarray(face_accepted_samples, dtype=np.int64).reshape(-1)
    total = None if face_total_samples is None else np.asarray(face_total_samples, dtype=np.int64).reshape(-1)
    accepted_pos = accepted > 0
    valid_count = int(np.count_nonzero(valid))
    accepted_valid = accepted_pos & valid
    unsupported_valid = (~accepted_pos) & valid
    summary: Dict[str, Any] = {
        "face_count_total": int(valid.shape[0]),
        "valid_face_count": int(valid_count),
        "valid_face_ratio": float(valid_count / max(1, valid.shape[0])),
        "accepted_face_count": int(np.count_nonzero(accepted_pos)),
        "accepted_face_ratio": float(np.count_nonzero(accepted_pos) / max(1, accepted.shape[0])),
        "unsupported_face_count": int(np.count_nonzero(~accepted_pos)),
        "unsupported_face_ratio": float(np.count_nonzero(~accepted_pos) / max(1, accepted.shape[0])),
        "accepted_valid_face_count": int(np.count_nonzero(accepted_valid)),
        "accepted_valid_face_ratio": float(np.count_nonzero(accepted_valid) / max(1, valid_count)),
        "unsupported_valid_face_count": int(np.count_nonzero(unsupported_valid)),
        "unsupported_valid_face_ratio": float(np.count_nonzero(unsupported_valid) / max(1, valid_count)),
        "accepted_samples_total": int(np.sum(accepted)),
        "accepted_samples_p50_per_face": float(np.quantile(accepted, 0.50)) if accepted.size > 0 else None,
        "accepted_samples_p95_per_face": float(np.quantile(accepted, 0.95)) if accepted.size > 0 else None,
    }
    if total is None:
        summary.update(
            {
                "total_samples_total": None,
                "total_samples_p50_per_face": None,
                "total_samples_p95_per_face": None,
                "sample_acceptance_ratio": None,
            }
        )
        return summary

    total_sum = int(np.sum(total))
    summary.update(
        {
            "total_samples_total": total_sum,
            "total_samples_p50_per_face": float(np.quantile(total, 0.50)) if total.size > 0 else None,
            "total_samples_p95_per_face": float(np.quantile(total, 0.95)) if total.size > 0 else None,
            "sample_acceptance_ratio": float(np.sum(accepted) / max(1, total_sum)),
        }
    )
    return summary


def _target_dispersion_summary(
    *,
    face_cov_trace: np.ndarray,
    face_valid: np.ndarray,
    face_smooth_alpha: Optional[np.ndarray],
    cov_scale_hint: Optional[float],
) -> Tuple[Dict[str, Any], np.ndarray]:
    cov_trace = np.asarray(face_cov_trace, dtype=np.float64).reshape(-1)
    valid = np.asarray(face_valid, dtype=np.bool_).reshape(-1)
    finite_valid = valid & np.isfinite(cov_trace)
    cov_valid = cov_trace[finite_valid]
    if cov_scale_hint is not None and math.isfinite(float(cov_scale_hint)) and float(cov_scale_hint) > 1e-12:
        cov_scale = float(cov_scale_hint)
    elif cov_valid.size > 0:
        cov_scale = max(float(np.quantile(cov_valid, 0.50)), 1e-12)
    else:
        cov_scale = 1.0
    cov_norm = cov_trace / cov_scale
    cov_norm[~np.isfinite(cov_norm)] = np.nan

    summary: Dict[str, Any] = {
        "cov_scale": float(cov_scale),
        "valid_face_count": int(np.count_nonzero(finite_valid)),
        "cov_trace_p50": float(np.quantile(cov_valid, 0.50)) if cov_valid.size > 0 else None,
        "cov_trace_p95": float(np.quantile(cov_valid, 0.95)) if cov_valid.size > 0 else None,
        "cov_trace_p99": float(np.quantile(cov_valid, 0.99)) if cov_valid.size > 0 else None,
    }
    cov_norm_valid = cov_norm[finite_valid & np.isfinite(cov_norm)]
    summary.update(
        {
            "cov_norm_p50": float(np.quantile(cov_norm_valid, 0.50)) if cov_norm_valid.size > 0 else None,
            "cov_norm_p95": float(np.quantile(cov_norm_valid, 0.95)) if cov_norm_valid.size > 0 else None,
            "cov_norm_p99": float(np.quantile(cov_norm_valid, 0.99)) if cov_norm_valid.size > 0 else None,
            "cov_norm_gt_2_ratio": float(np.mean(cov_norm_valid > 2.0)) if cov_norm_valid.size > 0 else 0.0,
            "cov_norm_gt_4_ratio": float(np.mean(cov_norm_valid > 4.0)) if cov_norm_valid.size > 0 else 0.0,
            "cov_norm_gt_8_ratio": float(np.mean(cov_norm_valid > 8.0)) if cov_norm_valid.size > 0 else 0.0,
            "high_dispersion_face_ratio": float(np.mean(cov_norm_valid > 4.0)) if cov_norm_valid.size > 0 else 0.0,
        }
    )
    if face_smooth_alpha is None:
        summary.update(
            {
                "smooth_alpha_p05": None,
                "smooth_alpha_p50": None,
                "smooth_alpha_p95": None,
            }
        )
        return summary, cov_norm

    alpha = np.asarray(face_smooth_alpha, dtype=np.float64).reshape(-1)
    alpha_valid = alpha[finite_valid & np.isfinite(alpha)]
    summary.update(
        {
            "smooth_alpha_p05": float(np.quantile(alpha_valid, 0.05)) if alpha_valid.size > 0 else None,
            "smooth_alpha_p50": float(np.quantile(alpha_valid, 0.50)) if alpha_valid.size > 0 else None,
            "smooth_alpha_p95": float(np.quantile(alpha_valid, 0.95)) if alpha_valid.size > 0 else None,
        }
    )
    return summary, cov_norm


def _strip_large_stats(method_stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in method_stats.items():
        if key in LARGE_STATS_KEYS:
            try:
                out[f"{key}_len"] = int(len(value))  # type: ignore[arg-type]
            except Exception:
                out[f"{key}_len"] = None
            continue
        out[key] = value
    return out


def _per_island_diagnostics(
    *,
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    island_labels: np.ndarray,
    island_label_source: str,
    anchor_ids: np.ndarray,
    face_valid: np.ndarray,
    face_rel_err: np.ndarray,
    face_cosine: np.ndarray,
    face_log_area_ratio: np.ndarray,
    face_accepted_samples: np.ndarray,
    face_total_samples: Optional[np.ndarray],
    face_target_cov_trace: np.ndarray,
    face_target_cov_norm: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    neighbors = _face_neighbors(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    boundary_ids = _boundary_vertex_ids(mesh)
    boundary_mask = np.zeros((len(mesh.vertices),), dtype=np.bool_)
    if boundary_ids.size > 0:
        boundary_mask[boundary_ids] = True
    signed = _signed_uv_area(mesh, uv)
    stretch = face_stretch_anisotropy(mesh, uv)
    comps = _same_label_components(np.asarray(island_labels, dtype=np.int64), neighbors)

    anchor_mask = np.zeros((len(mesh.vertices),), dtype=np.bool_)
    anchor_ids_i64 = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
    if anchor_ids_i64.size > 0:
        anchor_mask[np.clip(anchor_ids_i64, 0, len(mesh.vertices) - 1)] = True

    per_island: List[Dict[str, Any]] = []
    overlap_anchor_components = 0
    for label, comp_faces in comps:
        comp_faces_i64 = np.asarray(comp_faces, dtype=np.int64)
        comp_vertices = np.unique(faces[comp_faces_i64].reshape(-1)).astype(np.int64, copy=False)
        comp_anchor_mask = anchor_mask[comp_vertices]
        comp_anchor_vertices = comp_vertices[comp_anchor_mask]
        comp_boundary_anchor_vertices = comp_anchor_vertices[boundary_mask[comp_anchor_vertices]]
        if label >= 0:
            total_same_label_components = int(sum(1 for lb, _ in comps if int(lb) == int(label)))
        else:
            total_same_label_components = 1
        if comp_anchor_vertices.size > 0 and total_same_label_components > 1:
            overlap_anchor_components += 1

        comp_valid = np.asarray(face_valid[comp_faces_i64], dtype=np.bool_)
        comp_rel = np.asarray(face_rel_err[comp_faces_i64], dtype=np.float64)
        comp_cos = np.asarray(face_cosine[comp_faces_i64], dtype=np.float64)
        comp_log_area = np.asarray(face_log_area_ratio[comp_faces_i64], dtype=np.float64)
        comp_accepted_samples = np.asarray(face_accepted_samples[comp_faces_i64], dtype=np.int64)
        comp_total_samples = (
            None
            if face_total_samples is None
            else np.asarray(face_total_samples[comp_faces_i64], dtype=np.int64)
        )
        comp_signed = np.asarray(signed[comp_faces_i64], dtype=np.float64)
        comp_stretch = np.asarray(stretch[comp_faces_i64], dtype=np.float64)
        comp_cov_trace = np.asarray(face_target_cov_trace[comp_faces_i64], dtype=np.float64)
        comp_cov_norm = np.asarray(face_target_cov_norm[comp_faces_i64], dtype=np.float64)
        rel_valid = comp_rel[comp_valid & np.isfinite(comp_rel)]
        cos_valid = comp_cos[comp_valid & np.isfinite(comp_cos)]
        area_valid = comp_log_area[comp_valid & np.isfinite(comp_log_area)]
        stretch_valid = comp_stretch[np.isfinite(comp_stretch)]
        cov_trace_valid = comp_cov_trace[comp_valid & np.isfinite(comp_cov_trace)]
        cov_norm_valid = comp_cov_norm[comp_valid & np.isfinite(comp_cov_norm)]
        accepted_face_mask = comp_accepted_samples > 0

        entry = {
            "island_label": int(label),
            "component_faces": int(comp_faces_i64.size),
            "component_vertices": int(comp_vertices.size),
            "same_label_component_count": int(total_same_label_components),
            "valid_face_count": int(np.count_nonzero(comp_valid)),
            "valid_face_ratio": float(np.count_nonzero(comp_valid) / max(1, comp_faces_i64.size)),
            "anchor_count": int(comp_anchor_vertices.size),
            "boundary_anchor_count": int(comp_boundary_anchor_vertices.size),
            "anchor_per_1k_faces": float(1000.0 * comp_anchor_vertices.size / max(1, comp_faces_i64.size)),
            "anchor_per_1k_vertices": float(1000.0 * comp_anchor_vertices.size / max(1, comp_vertices.size)),
            "sample_count": int(np.sum(comp_accepted_samples)),
            "sample_per_face_mean": float(np.mean(comp_accepted_samples)) if comp_accepted_samples.size > 0 else 0.0,
            "accepted_face_ratio": float(np.mean(accepted_face_mask)) if accepted_face_mask.size > 0 else 0.0,
            "unsupported_face_ratio": float(np.mean(~accepted_face_mask)) if accepted_face_mask.size > 0 else 0.0,
            "accepted_samples_total": int(np.sum(comp_accepted_samples)),
            "accepted_samples_p50_per_face": float(np.quantile(comp_accepted_samples, 0.50))
            if comp_accepted_samples.size > 0
            else None,
            "accepted_samples_p95_per_face": float(np.quantile(comp_accepted_samples, 0.95))
            if comp_accepted_samples.size > 0
            else None,
            "total_samples_total": int(np.sum(comp_total_samples)) if comp_total_samples is not None else None,
            "total_samples_p50_per_face": float(np.quantile(comp_total_samples, 0.50))
            if comp_total_samples is not None and comp_total_samples.size > 0
            else None,
            "signed_negative_ratio": float(np.mean(comp_signed < 0.0)) if comp_signed.size > 0 else 0.0,
            "signed_near_zero_ratio": float(np.mean(np.abs(comp_signed) <= 1e-10)) if comp_signed.size > 0 else 0.0,
            "stretch_p95": float(np.quantile(stretch_valid, 0.95)) if stretch_valid.size > 0 else None,
            "stretch_p99": float(np.quantile(stretch_valid, 0.99)) if stretch_valid.size > 0 else None,
            "jacobian_rel_error_p95": float(np.quantile(rel_valid, 0.95)) if rel_valid.size > 0 else None,
            "jacobian_cosine_p05": float(np.quantile(cos_valid, 0.05)) if cos_valid.size > 0 else None,
            "jacobian_log_area_ratio_p95": float(np.quantile(area_valid, 0.95)) if area_valid.size > 0 else None,
            "target_cov_trace_p50": float(np.quantile(cov_trace_valid, 0.50)) if cov_trace_valid.size > 0 else None,
            "target_cov_trace_p95": float(np.quantile(cov_trace_valid, 0.95)) if cov_trace_valid.size > 0 else None,
            "target_cov_norm_p95": float(np.quantile(cov_norm_valid, 0.95)) if cov_norm_valid.size > 0 else None,
            "high_dispersion_face_ratio": float(np.mean(cov_norm_valid > 4.0)) if cov_norm_valid.size > 0 else 0.0,
            "is_tiny_component_le_4": bool(comp_faces_i64.size <= 4),
            "is_tiny_component_le_8": bool(comp_faces_i64.size <= 8),
            "is_tiny_component_le_16": bool(comp_faces_i64.size <= 16),
        }
        per_island.append(entry)

    per_island.sort(
        key=lambda x: (
            -float(x["stretch_p95"] if x["stretch_p95"] is not None else -1.0),
            -int(x["component_faces"]),
            int(x["island_label"]),
        )
    )
    summary = {
        "island_label_source": island_label_source,
        "island_component_count": int(len(per_island)),
        "island_nonnegative_label_count": int(len({int(x["island_label"]) for x in per_island if int(x["island_label"]) >= 0})),
        "island_unknown_component_count": int(sum(1 for x in per_island if int(x["island_label"]) < 0)),
        "island_component_anchor_overlap_candidates": int(overlap_anchor_components),
        "worst_island_stretch_p95": float(per_island[0]["stretch_p95"]) if per_island and per_island[0]["stretch_p95"] is not None else None,
    }
    return summary, per_island


def _component_size_per_face(labels: np.ndarray, neighbors: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    labels_i64 = np.asarray(labels, dtype=np.int64).reshape(-1)
    comp_ids = np.full((labels_i64.shape[0],), -1, dtype=np.int64)
    comp_sizes = np.zeros((labels_i64.shape[0],), dtype=np.int64)
    seen = np.zeros((labels_i64.shape[0],), dtype=np.bool_)
    next_comp = 0
    for fid in range(labels_i64.shape[0]):
        if seen[fid]:
            continue
        label = int(labels_i64[fid])
        q: deque[int] = deque([int(fid)])
        seen[fid] = True
        comp: List[int] = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in neighbors[cur]:
                if not seen[nb] and int(labels_i64[nb]) == label:
                    seen[nb] = True
                    q.append(int(nb))
        comp_np = np.asarray(comp, dtype=np.int64)
        comp_ids[comp_np] = next_comp
        comp_sizes[comp_np] = int(comp_np.size)
        next_comp += 1
    return comp_ids, comp_sizes


def _masked_quality_summary(mesh: trimesh.Trimesh, uv: np.ndarray, face_mask: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(face_mask, dtype=np.bool_).reshape(-1)
    if mask.shape[0] != int(len(mesh.faces)):
        raise RuntimeError("face_mask length mismatch for masked quality summary")
    if not np.any(mask):
        return {
            "face_count": 0,
            "vertex_count": 0,
            "uv_stretch_p50": None,
            "uv_stretch_p95": None,
            "uv_stretch_p99": None,
            "uv_bad_tri_ratio_stretch_only": None,
            "uv_out_of_bounds_ratio": None,
            "uv_signed_negative_ratio": None,
            "uv_signed_near_zero_ratio": None,
        }
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv_np = np.asarray(uv, dtype=np.float64)
    vert_ids = np.unique(faces[mask].reshape(-1)).astype(np.int64, copy=False)
    stretch_all = np.asarray(face_stretch_anisotropy(mesh, uv_np), dtype=np.float64)
    signed_all = _signed_uv_area(mesh, uv_np)
    stretch = stretch_all[mask & np.isfinite(stretch_all)]
    signed = np.asarray(signed_all[mask], dtype=np.float64)
    uv_sel = uv_np[vert_ids] if vert_ids.size > 0 else np.zeros((0, 2), dtype=np.float64)
    out_of_bounds = (
        float(np.mean(np.any((uv_sel < 0.0) | (uv_sel > 1.0), axis=1))) if uv_sel.size > 0 else None
    )
    if stretch.size > 0:
        p95 = float(np.quantile(stretch, 0.95))
        stretch_only_bad = float(np.mean(stretch > max(1.0, p95 * 1.5)))
    else:
        p95 = None
        stretch_only_bad = None
    return {
        "face_count": int(np.count_nonzero(mask)),
        "vertex_count": int(vert_ids.size),
        "uv_stretch_p50": _weighted_quantile(stretch, 0.50),
        "uv_stretch_p95": _weighted_quantile(stretch, 0.95),
        "uv_stretch_p99": _weighted_quantile(stretch, 0.99),
        "uv_bad_tri_ratio_stretch_only": stretch_only_bad,
        "uv_out_of_bounds_ratio": out_of_bounds,
        "uv_signed_negative_ratio": float(np.mean(signed < 0.0)) if signed.size > 0 else None,
        "uv_signed_near_zero_ratio": float(np.mean(np.abs(signed) <= 1e-10)) if signed.size > 0 else None,
    }


def _masked_jacobian_summary(
    face_rel_err: np.ndarray,
    face_cosine: np.ndarray,
    face_log_area_ratio: np.ndarray,
    face_mask: np.ndarray,
) -> Dict[str, Any]:
    mask = np.asarray(face_mask, dtype=np.bool_).reshape(-1)
    rel = np.asarray(face_rel_err, dtype=np.float64).reshape(-1)
    cos = np.asarray(face_cosine, dtype=np.float64).reshape(-1)
    loga = np.asarray(face_log_area_ratio, dtype=np.float64).reshape(-1)
    valid = mask & np.isfinite(rel) & np.isfinite(cos) & np.isfinite(loga)
    if not np.any(valid):
        return {
            "valid_face_count": 0,
            "frob_rel_error_p50": None,
            "frob_rel_error_p95": None,
            "frob_rel_error_p99": None,
            "cosine_p05": None,
            "log_area_ratio_p95": None,
        }
    rel_v = rel[valid]
    cos_v = cos[valid]
    loga_v = loga[valid]
    return {
        "valid_face_count": int(np.count_nonzero(valid)),
        "frob_rel_error_p50": float(np.quantile(rel_v, 0.50)),
        "frob_rel_error_p95": float(np.quantile(rel_v, 0.95)),
        "frob_rel_error_p99": float(np.quantile(rel_v, 0.99)),
        "cosine_p05": float(np.quantile(cos_v, 0.05)),
        "log_area_ratio_p95": float(np.quantile(loga_v, 0.95)),
    }


def _sample_residual_summary_from_arrays(
    *,
    mesh: trimesh.Trimesh,
    vertex_uv: np.ndarray,
    sample_face_ids: np.ndarray,
    sample_bary: np.ndarray,
    target_uv: np.ndarray,
) -> Dict[str, Any]:
    sf = np.asarray(sample_face_ids, dtype=np.int64).reshape(-1)
    sbary = np.asarray(sample_bary, dtype=np.float64)
    starget_uv = np.asarray(target_uv, dtype=np.float64)
    if sf.size == 0 or sbary.size == 0 or starget_uv.size == 0:
        return {
            "sample_count": 0,
            "residual_l2_mean": None,
            "residual_l2_p95": None,
            "residual_linf": None,
        }
    pred_uv = interpolate_sample_uv(
        np.asarray(mesh.faces, dtype=np.int64),
        sf,
        sbary,
        np.asarray(vertex_uv, dtype=np.float64),
    )
    residual = starget_uv - pred_uv
    l2 = np.linalg.norm(residual, axis=1)
    return {
        "sample_count": int(l2.shape[0]),
        "residual_l2_mean": float(np.mean(l2)),
        "residual_l2_p95": float(np.quantile(l2, 0.95)),
        "residual_linf": float(np.max(np.abs(residual))) if residual.size > 0 else None,
    }


def _face_adjacency_edges(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    edge_vid = np.asarray(getattr(mesh, "face_adjacency_edges", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 2 or edge_vid.ndim != 2 or edge_vid.shape[0] != adj.shape[0]:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return adj.astype(np.int64, copy=False), edge_vid.astype(np.int64, copy=False)


def _compute_edge_jump_data(
    mesh: trimesh.Trimesh,
    face_jac: np.ndarray,
    face_valid: np.ndarray,
) -> Dict[str, Any]:
    adj, edge_vid = _face_adjacency_edges(mesh)
    if adj.shape[0] == 0:
        return {
            "adjacency": adj,
            "edge_vertices": edge_vid,
            "valid_edge_mask": np.zeros((0,), dtype=np.bool_),
            "edge_jump_l2": np.zeros((0,), dtype=np.float64),
            "face_edge_jump_max": np.full((len(mesh.faces),), np.nan, dtype=np.float64),
        }
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edge_vec = verts[edge_vid[:, 1]] - verts[edge_vid[:, 0]]
    fj = np.asarray(face_jac, dtype=np.float64)
    fv = np.asarray(face_valid, dtype=np.bool_).reshape(-1)
    valid_edge = fv[adj[:, 0]] & fv[adj[:, 1]]
    delta_a = np.einsum("fij,fj->fi", fj[adj[:, 0]], edge_vec, optimize=True)
    delta_b = np.einsum("fij,fj->fi", fj[adj[:, 1]], edge_vec, optimize=True)
    jump = np.linalg.norm(delta_a - delta_b, axis=1)
    jump[~valid_edge] = np.nan
    face_jump_max = np.full((len(mesh.faces),), np.nan, dtype=np.float64)
    for edge_idx, (fa, fb) in enumerate(adj.tolist()):
        val = float(jump[edge_idx])
        if not math.isfinite(val):
            continue
        if not math.isfinite(face_jump_max[fa]) or val > face_jump_max[fa]:
            face_jump_max[fa] = val
        if not math.isfinite(face_jump_max[fb]) or val > face_jump_max[fb]:
            face_jump_max[fb] = val
    return {
        "adjacency": adj,
        "edge_vertices": edge_vid,
        "valid_edge_mask": valid_edge,
        "edge_jump_l2": jump,
        "face_edge_jump_max": face_jump_max,
    }


def _make_tangent_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    if not math.isfinite(norm) or norm <= 1e-12:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        n = n / norm
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t = np.cross(n, ref)
    t_norm = float(np.linalg.norm(t))
    if t_norm <= 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t = np.cross(n, ref)
        t_norm = float(np.linalg.norm(t))
    t = t / max(t_norm, 1e-12)
    b = np.cross(n, t)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return t, b


def _compute_vertex_cycle_residuals(
    mesh: trimesh.Trimesh,
    face_jac: np.ndarray,
    face_valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n_faces = int(len(faces))
    n_vertices = int(len(verts))
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    incident: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_vertices)]
    for fid, tri in enumerate(faces.tolist()):
        a, b, c = tri
        incident[a].append((b, c, fid))
        incident[b].append((c, a, fid))
        incident[c].append((a, b, fid))

    vertex_residuals = np.full((n_vertices,), np.nan, dtype=np.float64)
    face_cycle_residual = np.full((n_faces,), np.nan, dtype=np.float64)
    for vid, wedges in enumerate(incident):
        if len(wedges) < 3:
            continue
        neighbor_ids = sorted({int(a) for a, _, _ in wedges} | {int(b) for _, b, _ in wedges})
        if len(neighbor_ids) < 3:
            continue
        normal = vertex_normals[vid] if vid < vertex_normals.shape[0] else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t, b = _make_tangent_basis(normal)
        p0 = verts[vid]
        angle_pairs: List[Tuple[float, int]] = []
        for nb in neighbor_ids:
            vec = verts[nb] - p0
            angle = math.atan2(float(np.dot(vec, b)), float(np.dot(vec, t)))
            angle_pairs.append((angle, int(nb)))
        ordered = [nb for _, nb in sorted(angle_pairs, key=lambda x: x[0])]
        wedge_face: Dict[Tuple[int, int], int] = {}
        duplicate = False
        for a, c, fid in wedges:
            key = tuple(sorted((int(a), int(c))))
            if key in wedge_face:
                duplicate = True
                break
            wedge_face[key] = int(fid)
        if duplicate:
            continue
        total = np.zeros((2,), dtype=np.float64)
        used_faces: List[int] = []
        cycle_ok = True
        for idx in range(len(ordered)):
            va = int(ordered[idx])
            vb = int(ordered[(idx + 1) % len(ordered)])
            fid = wedge_face.get(tuple(sorted((va, vb))))
            if fid is None or not bool(face_valid[fid]):
                cycle_ok = False
                break
            total += np.asarray(face_jac[fid], dtype=np.float64) @ (verts[vb] - verts[va])
            used_faces.append(fid)
        if not cycle_ok:
            continue
        res = float(np.linalg.norm(total))
        vertex_residuals[vid] = res
        for fid in used_faces:
            if not math.isfinite(face_cycle_residual[fid]) or res > face_cycle_residual[fid]:
                face_cycle_residual[fid] = res
    return vertex_residuals, face_cycle_residual


def _component_labels_from_removed_adjacency(
    *,
    n_faces: int,
    adjacency: np.ndarray,
    face_valid_mask: np.ndarray,
    removed_edge_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    valid = np.asarray(face_valid_mask, dtype=np.bool_).reshape(-1)
    labels = np.full((n_faces,), -1, dtype=np.int64)
    if n_faces == 0:
        return labels
    removed = (
        np.zeros((adjacency.shape[0],), dtype=np.bool_)
        if removed_edge_mask is None
        else np.asarray(removed_edge_mask, dtype=np.bool_).reshape(-1)
    )
    neigh: List[List[int]] = [[] for _ in range(n_faces)]
    for edge_idx, (fa, fb) in enumerate(np.asarray(adjacency, dtype=np.int64).tolist()):
        if removed[edge_idx]:
            continue
        if not (valid[fa] and valid[fb]):
            continue
        neigh[fa].append(int(fb))
        neigh[fb].append(int(fa))
    next_label = 0
    for fid in range(n_faces):
        if not valid[fid] or labels[fid] >= 0:
            continue
        q: deque[int] = deque([int(fid)])
        labels[fid] = next_label
        while q:
            cur = q.popleft()
            for nb in neigh[cur]:
                if labels[nb] < 0:
                    labels[nb] = next_label
                    q.append(int(nb))
        next_label += 1
    return labels


def _solve_target_field(
    *,
    solve_mesh: trimesh.Trimesh,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    resolved_device: str,
    cfg: Dict[str, Any],
    face_jac: np.ndarray,
    face_weights: np.ndarray,
    face_valid_mask: np.ndarray,
    face_active_mask: np.ndarray,
    face_island_labels: Optional[np.ndarray],
    anchor_vertex_target_uv: np.ndarray,
    anchor_vertex_confidence: np.ndarray,
    face_smooth_alpha: np.ndarray,
) -> Dict[str, Any]:
    A, rhs_u, rhs_v, row_face_ids = _build_gradient_constraint_system(
        mesh=solve_mesh,
        face_jac=np.asarray(face_jac, dtype=np.float64),
        face_weights=np.asarray(face_weights, dtype=np.float64),
        face_valid_mask=np.asarray(face_valid_mask, dtype=np.bool_),
    )
    if A.shape[0] == 0:
        raise RuntimeError("no valid gradient constraints for target-field solve")
    solve_cfg = dict(cfg.get("solve", {}))
    m2_cfg = deep_merge_dict(cfg.get("method2", {}), {})
    m2_cfg["post_align_translation"] = False
    precomputed_anchor_uv = nearest_vertex_uv(solve_mesh, high_mesh, high_uv).astype(np.float64)
    if face_island_labels is not None and np.any(np.asarray(face_island_labels, dtype=np.int64) >= 0):
        mapped_uv, solve_meta, anchor_ids, anchor_uv = _solve_poisson_uv_by_island(
            mesh=solve_mesh,
            high_mesh=high_mesh,
            high_uv=high_uv,
            A=A,
            rhs_u=rhs_u,
            rhs_v=rhs_v,
            row_face_ids=row_face_ids,
            face_island_labels=np.asarray(face_island_labels, dtype=np.int64),
            solve_cfg=solve_cfg,
            m2_cfg=m2_cfg,
            resolved_device=resolved_device,
            anchor_mode=str(m2_cfg.get("anchor_mode", "component_minimal")),
            anchor_points_per_component=int(m2_cfg.get("anchor_points_per_component", 4)),
            anchor_vertex_target_uv=np.asarray(anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(anchor_vertex_confidence, dtype=np.float64),
            face_smooth_alpha=np.asarray(face_smooth_alpha, dtype=np.float64),
            precomputed_anchor_uv_all=precomputed_anchor_uv,
        )
    else:
        mapped_uv, solve_meta, anchor_ids, anchor_uv = _solve_poisson_uv(
            mesh=solve_mesh,
            high_mesh=high_mesh,
            high_uv=high_uv,
            A=A,
            rhs_u=rhs_u,
            rhs_v=rhs_v,
            solve_cfg=solve_cfg,
            m2_cfg=m2_cfg,
            face_active_mask=np.asarray(face_active_mask, dtype=np.bool_),
            resolved_device=resolved_device,
            anchor_mode=str(m2_cfg.get("anchor_mode", "component_minimal")),
            anchor_points_per_component=int(m2_cfg.get("anchor_points_per_component", 4)),
            anchor_vertex_target_uv=np.asarray(anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(anchor_vertex_confidence, dtype=np.float64),
            face_smooth_alpha=np.asarray(face_smooth_alpha, dtype=np.float64),
            precomputed_anchor_uv_all=precomputed_anchor_uv,
        )
    return {
        "mapped_uv": np.asarray(mapped_uv, dtype=np.float64),
        "solve_meta": solve_meta,
        "anchor_ids": np.asarray(anchor_ids, dtype=np.int64),
        "anchor_uv": np.asarray(anchor_uv, dtype=np.float64),
        "A": A,
        "rhs_u": np.asarray(rhs_u, dtype=np.float64),
        "rhs_v": np.asarray(rhs_v, dtype=np.float64),
        "row_face_ids": np.asarray(row_face_ids, dtype=np.int64),
    }


def _extract_submesh_for_faces(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    verts_all = np.asarray(mesh.vertices, dtype=np.float64)
    face_ids_i64 = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    tri = faces_all[face_ids_i64]
    global_vid, inverse = np.unique(tri.reshape(-1), return_inverse=True)
    tri_local = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    submesh = trimesh.Trimesh(vertices=verts_all[global_vid], faces=tri_local, process=False)
    return submesh, global_vid.astype(np.int64, copy=False), face_ids_i64


def _solve_spd_with_dirichlet(
    *,
    M: csr_matrix,
    rhs: np.ndarray,
    dirichlet_ids: np.ndarray,
    dirichlet_values: np.ndarray,
    resolved_device: str,
    solve_cfg: Dict[str, Any],
    channel_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = int(M.shape[0])
    rhs_np = np.asarray(rhs, dtype=np.float64).reshape(-1)
    dids = np.asarray(dirichlet_ids, dtype=np.int64).reshape(-1)
    dvals = np.asarray(dirichlet_values, dtype=np.float64).reshape(-1)
    if rhs_np.shape[0] != n:
        raise RuntimeError("rhs length mismatch for Dirichlet solve")
    if dids.size != dvals.size:
        raise RuntimeError("Dirichlet id/value length mismatch")
    if dids.size == 0:
        return solve_linear_robust(
            M=M,
            rhs=rhs_np,
            cg_max_iter=int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000))),
            cg_tol=float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6))),
            channel_name=channel_name,
        )
    fixed = np.zeros((n,), dtype=np.bool_)
    fixed[np.clip(dids, 0, n - 1)] = True
    free = ~fixed
    sol = np.zeros((n,), dtype=np.float64)
    sol[dids] = dvals
    if not np.any(free):
        return sol, {"backend": "dirichlet_only", "iters": 0, "residual": 0.0, "converged": True}
    M_free = M[free][:, free].tocsr()
    rhs_free = rhs_np[free] - M[free][:, dids] @ dvals
    backend_requested = str(solve_cfg.get("backend", "auto")).strip().lower()
    use_cuda_first = backend_requested == "cuda_pcg" or (
        backend_requested == "auto" and str(resolved_device).startswith("cuda")
    )
    if use_cuda_first:
        try:
            M_cuda, M_diag = build_cuda_sparse_system(M=M_free, device=resolved_device)
            sol_free, meta = solve_linear_cuda_pcg(
                M_cuda=M_cuda,
                M_diag_cuda=M_diag,
                rhs=rhs_free,
                pcg_max_iter=int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000))),
                pcg_tol=float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6))),
                pcg_check_every=int(solve_cfg.get("pcg_check_every", 25)),
                pcg_preconditioner=str(solve_cfg.get("pcg_preconditioner", "jacobi")),
                channel_name=channel_name,
            )
        except Exception as exc:
            sol_free, meta = solve_linear_robust(
                M=M_free,
                rhs=rhs_free,
                cg_max_iter=int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000))),
                cg_tol=float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6))),
                channel_name=channel_name,
            )
            meta["fallback_reason"] = str(exc)
    else:
        sol_free, meta = solve_linear_robust(
            M=M_free,
            rhs=rhs_free,
            cg_max_iter=int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000))),
            cg_tol=float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6))),
            channel_name=channel_name,
        )
    sol[free] = np.asarray(sol_free, dtype=np.float64)
    return sol, meta


def _patch_faces_from_seed(
    *,
    seed_face: int,
    target_size: int,
    neighbors: List[List[int]],
    face_valid_mask: np.ndarray,
    face_island_labels: np.ndarray,
) -> np.ndarray:
    valid = np.asarray(face_valid_mask, dtype=np.bool_).reshape(-1)
    labels = np.asarray(face_island_labels, dtype=np.int64).reshape(-1)
    if seed_face < 0 or seed_face >= valid.shape[0] or not bool(valid[seed_face]):
        return np.zeros((0,), dtype=np.int64)
    seed_label = int(labels[seed_face])
    q: deque[int] = deque([int(seed_face)])
    seen = np.zeros((valid.shape[0],), dtype=np.bool_)
    seen[seed_face] = True
    out: List[int] = []
    while q and len(out) < int(target_size):
        cur = q.popleft()
        out.append(cur)
        for nb in neighbors[cur]:
            if seen[nb] or not bool(valid[nb]):
                continue
            if int(labels[nb]) != seed_label:
                continue
            seen[nb] = True
            q.append(int(nb))
    return np.asarray(out, dtype=np.int64)


def _numeric_deltas(before: Dict[str, Any], after: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        a = before.get(key)
        b = after.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b)):
            out[key] = float(b) - float(a)
        else:
            out[key] = None
    return out


def _complete_anchor_vertex_uv(ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    anchor_uv = np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64).copy()
    solve_uv = np.asarray(ctx["solve_uv"], dtype=np.float64)
    finite = np.isfinite(anchor_uv).all(axis=1)
    fill_meta: Dict[str, Any] = {
        "vertex_count": int(anchor_uv.shape[0]),
        "anchor_finite_vertex_count_initial": int(np.count_nonzero(finite)),
        "anchor_missing_vertex_count_initial": int(np.count_nonzero(~finite)),
        "nearest_fill_count": 0,
        "solve_fill_count": 0,
        "remaining_missing_vertex_count": 0,
    }
    if np.any(~finite):
        nearest_uv = nearest_vertex_uv(ctx["solve_mesh"], ctx["high_mesh"], ctx["high_uv"]).astype(np.float64)
        nearest_ok = (~finite) & np.isfinite(nearest_uv).all(axis=1)
        if np.any(nearest_ok):
            anchor_uv[nearest_ok] = nearest_uv[nearest_ok]
            fill_meta["nearest_fill_count"] = int(np.count_nonzero(nearest_ok))
        finite = np.isfinite(anchor_uv).all(axis=1)
    if np.any(~finite):
        solve_ok = (~finite) & np.isfinite(solve_uv).all(axis=1)
        if np.any(solve_ok):
            anchor_uv[solve_ok] = solve_uv[solve_ok]
            fill_meta["solve_fill_count"] = int(np.count_nonzero(solve_ok))
        finite = np.isfinite(anchor_uv).all(axis=1)
    fill_meta["remaining_missing_vertex_count"] = int(np.count_nonzero(~finite))
    return anchor_uv, fill_meta


def _sample_mix_arrays(ctx: Dict[str, Any]) -> Dict[str, np.ndarray]:
    n_faces = int(len(ctx["solve_mesh"].faces))
    sample_face_ids = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64).reshape(-1)
    target_face_ids = np.asarray(ctx["internal"].solve_sample_target_face_ids, dtype=np.int64).reshape(-1)
    target_high_island = np.asarray(ctx["internal"].solve_sample_target_high_island, dtype=np.int64).reshape(-1)
    fallback_mask = np.asarray(ctx["internal"].solve_sample_fallback_mask, dtype=np.bool_).reshape(-1)
    fallback_ratio = np.full((n_faces,), np.nan, dtype=np.float64)
    unique_high_face_count = np.zeros((n_faces,), dtype=np.int64)
    unique_high_island_count = np.zeros((n_faces,), dtype=np.int64)
    if sample_face_ids.size == 0:
        return {
            "fallback_ratio": fallback_ratio,
            "unique_high_face_count": unique_high_face_count,
            "unique_high_island_count": unique_high_island_count,
        }
    order = np.argsort(sample_face_ids, kind="mergesort")
    sf = sample_face_ids[order]
    tf = target_face_ids[order]
    thi = target_high_island[order]
    fb = fallback_mask[order]
    split_idx = np.flatnonzero(np.diff(sf)) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [len(sf)]))
    for start, end in zip(starts.tolist(), ends.tolist()):
        fid = int(sf[start])
        if fid < 0 or fid >= n_faces:
            continue
        fallback_ratio[fid] = float(np.mean(fb[start:end]))
        unique_high_face_count[fid] = int(len(np.unique(tf[start:end])))
        valid_island = thi[start:end][thi[start:end] >= 0]
        unique_high_island_count[fid] = int(len(np.unique(valid_island))) if valid_island.size > 0 else 0
    return {
        "fallback_ratio": fallback_ratio,
        "unique_high_face_count": unique_high_face_count,
        "unique_high_island_count": unique_high_island_count,
    }


def _compare_jacobian_fields(
    reference_jac: np.ndarray,
    other_jac: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    ref_vs_other, rel_err, cosine, log_area = _jacobian_diagnostics(reference_jac, other_jac, valid_mask)
    other_vs_ref, _, _, _ = _jacobian_diagnostics(other_jac, reference_jac, valid_mask)
    return {
        "reference_to_other": ref_vs_other,
        "other_to_reference": other_vs_ref,
    }, rel_err, cosine, log_area


def _curl_global_summary(
    mesh: trimesh.Trimesh,
    face_jac: np.ndarray,
    face_valid: np.ndarray,
) -> Dict[str, Any]:
    edge_data = _compute_edge_jump_data(mesh, face_jac, face_valid)
    vertex_cycle_residual, face_cycle_residual = _compute_vertex_cycle_residuals(mesh, face_jac, face_valid)
    valid_edge_jump = edge_data["edge_jump_l2"][edge_data["valid_edge_mask"] & np.isfinite(edge_data["edge_jump_l2"])]
    valid_cycle = face_cycle_residual[np.isfinite(face_cycle_residual)]
    jump_thr = float(np.quantile(valid_edge_jump, 0.95)) if valid_edge_jump.size > 0 else None
    cycle_thr = float(np.quantile(valid_cycle, 0.95)) if valid_cycle.size > 0 else None
    return {
        "edge_jump_l2_p50": _weighted_quantile(valid_edge_jump, 0.50),
        "edge_jump_l2_p95": _weighted_quantile(valid_edge_jump, 0.95),
        "edge_jump_l2_p99": _weighted_quantile(valid_edge_jump, 0.99),
        "cycle_residual_l2_p50": _weighted_quantile(valid_cycle, 0.50),
        "cycle_residual_l2_p95": _weighted_quantile(valid_cycle, 0.95),
        "cycle_residual_l2_p99": _weighted_quantile(valid_cycle, 0.99),
        "high_jump_edge_ratio": float(np.mean(valid_edge_jump > jump_thr)) if valid_edge_jump.size > 0 and jump_thr is not None else 0.0,
        "high_cycle_face_ratio": float(np.mean(valid_cycle > cycle_thr)) if valid_cycle.size > 0 and cycle_thr is not None else 0.0,
        "vertex_cycle_count": int(np.count_nonzero(np.isfinite(vertex_cycle_residual))),
        "face_cycle_count": int(np.count_nonzero(np.isfinite(face_cycle_residual))),
    }


def _method2_aggregation_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m2_cfg = dict(cfg.get("method2", {}))
    seam_cfg = dict(cfg.get("seam", {}))
    return {
        "min_samples_per_face": int(m2_cfg.get("min_samples_per_face", seam_cfg.get("min_valid_samples_per_face", 2))),
        "outlier_sigma": float(m2_cfg.get("outlier_sigma", 4.0)),
        "outlier_quantile": float(m2_cfg.get("outlier_quantile", 0.95)),
        "face_weight_floor": float(m2_cfg.get("face_weight_floor", 1e-6)),
        "irls_iters": int(m2_cfg.get("irls_iters", 2)),
        "huber_delta": float(m2_cfg.get("huber_delta", 3.0)),
        "fast_mode": bool(m2_cfg.get("perf_fast_agg_vectorized", True)),
        "small_group_fast_threshold": int(m2_cfg.get("perf_fast_small_group_threshold", 6)),
        "small_group_skip_irls": bool(m2_cfg.get("perf_fast_small_group_skip_irls", True)),
        "small_group_skip_outlier": bool(m2_cfg.get("perf_fast_small_group_skip_outlier", True)),
    }


def _aggregate_face_field_variant(
    *,
    ctx: Dict[str, Any],
    sample_weights: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    sample_face_ids_all = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64).reshape(-1)
    target_face_ids_all = np.asarray(ctx["internal"].solve_sample_target_face_ids, dtype=np.int64).reshape(-1)
    if sample_mask is None:
        keep = np.ones((sample_face_ids_all.shape[0],), dtype=np.bool_)
    else:
        keep = np.asarray(sample_mask, dtype=np.bool_).reshape(-1)
        if keep.shape[0] != sample_face_ids_all.shape[0]:
            raise RuntimeError("sample_mask length mismatch for aggregation variant")
    weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != sample_face_ids_all.shape[0]:
        raise RuntimeError("sample_weights length mismatch for aggregation variant")
    keep &= np.isfinite(weights) & (weights > 0.0)
    params = dict(ctx["agg_params"])
    face_jac, face_weights, face_valid, face_cov_trace, meta = _aggregate_face_target_jacobians(
        n_low_faces=int(len(ctx["solve_mesh"].faces)),
        sample_face_ids=sample_face_ids_all[keep],
        target_face_ids=target_face_ids_all[keep],
        sample_weights=weights[keep],
        high_face_jac=ctx["high_face_jac"],
        min_samples_per_face=int(params["min_samples_per_face"]),
        outlier_sigma=float(params["outlier_sigma"]),
        outlier_quantile=float(params["outlier_quantile"]),
        face_weight_floor=float(params["face_weight_floor"]),
        irls_iters=int(params["irls_iters"]),
        huber_delta=float(params["huber_delta"]),
        fast_mode=bool(params["fast_mode"]),
        small_group_fast_threshold=int(params["small_group_fast_threshold"]),
        small_group_skip_irls=bool(params["small_group_skip_irls"]),
        small_group_skip_outlier=bool(params["small_group_skip_outlier"]),
    )
    face_valid &= np.asarray(ctx["face_active"], dtype=np.bool_)
    return {
        "face_jac": np.asarray(face_jac, dtype=np.float64),
        "face_weights": np.asarray(face_weights, dtype=np.float64),
        "face_valid": np.asarray(face_valid, dtype=np.bool_),
        "face_cov_trace": np.asarray(face_cov_trace, dtype=np.float64),
        "aggregate_meta": meta,
        "sample_keep_count": int(np.count_nonzero(keep)),
    }


def _solve_custom_field_summary(
    *,
    ctx: Dict[str, Any],
    variant_name: str,
    face_jac: np.ndarray,
    face_weights: np.ndarray,
    face_valid: np.ndarray,
    anchor_vertex_target_uv: np.ndarray,
    anchor_vertex_confidence: np.ndarray,
    face_island_labels: Optional[np.ndarray],
    compare_target_jac: Optional[np.ndarray] = None,
    compare_target_valid_mask: Optional[np.ndarray] = None,
    cfg_override: Optional[Dict[str, Any]] = None,
    include_mapped_uv: bool = False,
    solve_mesh_override: Optional[trimesh.Trimesh] = None,
    face_geom_pinv_override: Optional[np.ndarray] = None,
    face_active_override: Optional[np.ndarray] = None,
    face_smooth_alpha_override: Optional[np.ndarray] = None,
    sample_face_ids_override: Optional[np.ndarray] = None,
    sample_bary_override: Optional[np.ndarray] = None,
    target_uv_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cfg_local = deep_merge_dict(ctx["cfg"], {})
    if cfg_override:
        cfg_local = deep_merge_dict(cfg_local, cfg_override)
    solve_mesh_local = ctx["solve_mesh"] if solve_mesh_override is None else solve_mesh_override
    face_geom_pinv_local = (
        np.asarray(ctx["internal"].face_geom_pinv, dtype=np.float64)
        if face_geom_pinv_override is None
        else np.asarray(face_geom_pinv_override, dtype=np.float64)
    )
    face_active_local = (
        np.asarray(ctx["face_active"], dtype=np.bool_)
        if face_active_override is None
        else np.asarray(face_active_override, dtype=np.bool_)
    )
    face_smooth_alpha_local = (
        np.asarray(ctx["internal"].face_smooth_alpha, dtype=np.float64)
        if face_smooth_alpha_override is None
        else np.asarray(face_smooth_alpha_override, dtype=np.float64)
    )
    sample_face_ids_local = (
        np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64)
        if sample_face_ids_override is None
        else np.asarray(sample_face_ids_override, dtype=np.int64)
    )
    sample_bary_local = (
        np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64)
        if sample_bary_override is None
        else np.asarray(sample_bary_override, dtype=np.float64)
    )
    target_uv_local = (
        np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64)
        if target_uv_override is None
        else np.asarray(target_uv_override, dtype=np.float64)
    )
    solve_result = _solve_target_field(
        solve_mesh=solve_mesh_local,
        high_mesh=ctx["high_mesh"],
        high_uv=ctx["high_uv"],
        resolved_device=ctx["internal"].resolved_device,
        cfg=cfg_local,
        face_jac=np.asarray(face_jac, dtype=np.float64),
        face_weights=np.asarray(face_weights, dtype=np.float64),
        face_valid_mask=np.asarray(face_valid, dtype=np.bool_),
        face_active_mask=face_active_local,
        face_island_labels=None if face_island_labels is None else np.asarray(face_island_labels, dtype=np.int64),
        anchor_vertex_target_uv=np.asarray(anchor_vertex_target_uv, dtype=np.float64),
        anchor_vertex_confidence=np.asarray(anchor_vertex_confidence, dtype=np.float64),
        face_smooth_alpha=face_smooth_alpha_local,
    )
    mapped_uv = np.asarray(solve_result["mapped_uv"], dtype=np.float64)
    solved_jac = _compute_face_jacobians(face_geom_pinv_local, solve_mesh_local, mapped_uv)
    own_jac_summary, _, _, _ = _jacobian_diagnostics(
        np.asarray(face_jac, dtype=np.float64),
        solved_jac,
        np.asarray(face_valid, dtype=np.bool_),
    )
    against_target_summary = None
    if compare_target_jac is not None:
        compare_mask = (
            np.asarray(compare_target_valid_mask, dtype=np.bool_)
            if compare_target_valid_mask is not None
            else np.asarray(face_valid, dtype=np.bool_)
        )
        against_target_summary, _, _, _ = _jacobian_diagnostics(
            np.asarray(compare_target_jac, dtype=np.float64),
            solved_jac,
            compare_mask,
        )
    out = {
        "status": "ok",
        "variant": variant_name,
        "solve_per_island": bool(face_island_labels is not None),
        "solve_mesh_quality": {
            k: v
            for k, v in _quality_with_context(solve_mesh_local, mapped_uv).items()
            if k in {"uv_stretch_p95", "uv_stretch_p99", "uv_bad_tri_ratio_stretch_only", "uv_out_of_bounds_ratio"}
        },
        "sample_residual_summary": _sample_residual_summary_from_arrays(
            mesh=solve_mesh_local,
            vertex_uv=mapped_uv,
            sample_face_ids=sample_face_ids_local,
            sample_bary=sample_bary_local,
            target_uv=target_uv_local,
        ),
        "solve_vs_variant_target_jacobian": own_jac_summary,
        "solve_vs_current_target_jacobian": against_target_summary,
        "solve_meta": {
            "uv_solver_linear_backend_used": solve_result["solve_meta"].get("uv_solver_linear_backend_used"),
            "uv_m2_system_cond_proxy": solve_result["solve_meta"].get("uv_m2_system_cond_proxy"),
            "uv_solver_residual_u": solve_result["solve_meta"].get("uv_solver_residual_u"),
            "uv_solver_residual_v": solve_result["solve_meta"].get("uv_solver_residual_v"),
            "uv_m2_soft_anchor_count": solve_result["solve_meta"].get("uv_m2_soft_anchor_count"),
        },
        "solve_constraint_meta": {
            k: v
            for k, v in solve_result["solve_meta"].items()
            if str(k).startswith("uv_solver_constraint_")
        },
        "solve_feasibility_summary": summarize_uv_box_feasibility(solve_mesh_local, mapped_uv, margin=0.0),
    }
    if include_mapped_uv:
        out["mapped_uv"] = mapped_uv
    return out


def _default_variant_face_weights(ctx: Dict[str, Any], face_valid: np.ndarray) -> np.ndarray:
    base = np.asarray(ctx["face_weights"], dtype=np.float64).copy()
    valid = np.asarray(face_valid, dtype=np.bool_)
    pos = base[np.isfinite(base) & (base > 1e-12)]
    fill = float(np.quantile(pos, 0.50)) if pos.size > 0 else 1.0
    bad = ~np.isfinite(base) | (base <= 1e-12)
    base[bad & valid] = fill
    base[~valid] = 0.0
    return base


def _fit_face_sample_jacobian_field(
    *,
    ctx: Dict[str, Any],
    anchor_uv_prior: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
    min_samples: int = 3,
) -> Dict[str, Any]:
    faces = np.asarray(ctx["solve_mesh"].faces, dtype=np.int64)
    face_geom_pinv = np.asarray(ctx["internal"].face_geom_pinv, dtype=np.float64)
    sample_face_ids = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64).reshape(-1)
    sample_bary = np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64)
    target_uv = np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64)
    fallback_mask = np.asarray(ctx["internal"].solve_sample_fallback_mask, dtype=np.bool_).reshape(-1)
    n_faces = int(len(faces))
    face_jac = np.full((n_faces, 2, 3), np.nan, dtype=np.float64)
    face_valid = np.zeros((n_faces,), dtype=np.bool_)
    face_residual = np.full((n_faces,), np.nan, dtype=np.float64)
    face_sample_count = np.zeros((n_faces,), dtype=np.int64)
    face_fallback_ratio = np.full((n_faces,), np.nan, dtype=np.float64)
    face_rank = np.zeros((n_faces,), dtype=np.int64)
    if sample_face_ids.size == 0:
        return {
            "face_jac": face_jac,
            "face_valid": face_valid,
            "face_residual": face_residual,
            "face_sample_count": face_sample_count,
            "face_fallback_ratio": face_fallback_ratio,
            "face_rank": face_rank,
        }
    order = np.argsort(sample_face_ids, kind="mergesort")
    sf = sample_face_ids[order]
    sb = sample_bary[order]
    st = target_uv[order]
    fb = fallback_mask[order]
    split_idx = np.flatnonzero(np.diff(sf)) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [len(sf)]))
    lam = max(float(prior_weight), 0.0)
    sqrt_lam = math.sqrt(lam) if lam > 0.0 else 0.0
    for start, end in zip(starts.tolist(), ends.tolist()):
        fid = int(sf[start])
        if fid < 0 or fid >= n_faces:
            continue
        W = np.asarray(sb[start:end], dtype=np.float64)
        T = np.asarray(st[start:end], dtype=np.float64)
        face_sample_count[fid] = int(W.shape[0])
        face_fallback_ratio[fid] = float(np.mean(fb[start:end])) if W.shape[0] > 0 else np.nan
        if W.shape[0] < int(min_samples):
            continue
        rank = int(np.linalg.matrix_rank(W))
        face_rank[fid] = rank
        if rank < 3:
            continue
        try:
            if anchor_uv_prior is not None and lam > 0.0:
                U0 = np.asarray(anchor_uv_prior[faces[fid]], dtype=np.float64)
                A = np.vstack([W, sqrt_lam * np.eye(3, dtype=np.float64)])
                B = np.vstack([T, sqrt_lam * U0])
                U, *_ = np.linalg.lstsq(A, B, rcond=None)
            else:
                U, *_ = np.linalg.lstsq(W, T, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = W @ U
        face_residual[fid] = float(np.mean(np.linalg.norm(pred - T, axis=1)))
        du1 = U[1] - U[0]
        du2 = U[2] - U[0]
        uv_grad = np.stack([du1, du2], axis=1)
        J = uv_grad @ face_geom_pinv[fid]
        if not np.isfinite(J).all():
            continue
        face_jac[fid] = J
        face_valid[fid] = True
    return {
        "face_jac": face_jac,
        "face_valid": face_valid,
        "face_residual": face_residual,
        "face_sample_count": face_sample_count,
        "face_fallback_ratio": face_fallback_ratio,
        "face_rank": face_rank,
    }


def _build_method2_options(
    *,
    seam_strategy: str,
    sanitize_low: bool,
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    opts = deep_merge_dict(DEFAULT_OPTIONS, {})
    base = {
        "seam": {
            "strategy": str(seam_strategy),
            "sanitize_enabled": bool(sanitize_low),
            "emit_validation_sidecar_data": True,
            "validation_mode": "diagnostic",
            "validation_strict": False,
            "validation_require_closed_loops": False,
            "validation_require_pure_components": False,
            "validation_allow_open_on_boundary": True,
        },
        "method2": {
            "emit_face_sample_counts": True,
        },
    }
    opts = deep_merge_dict(opts, base)
    if overrides:
        opts = deep_merge_dict(opts, overrides)
    return opts


def run_method2_internal_audit_on_meshes(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    high_path: Optional[Path],
    low_path: Optional[Path],
    device: str,
    seam_strategy: str,
    sanitize_low: bool,
    options_overrides: Optional[Dict[str, Any]] = None,
    out_json: Optional[Path] = None,
    case_name: Optional[str] = None,
) -> Dict[str, Any]:
    high_uv = _safe_uv(high_mesh)
    image, image_source = resolve_basecolor_image(high_mesh, high_path)
    cfg = _build_method2_options(
        seam_strategy=seam_strategy,
        sanitize_low=sanitize_low,
        overrides=options_overrides,
    )
    projector = UVProjector()

    mapped_uv, method_stats, export_payload, internal = run_method2_gradient_poisson(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_uv=high_uv,
        image=image,
        device=device,
        cfg=cfg,
        nearest_mapper=projector._map_nearest_vertex,
        barycentric_mapper=projector._map_barycentric_closest,
        return_internal=True,
    )
    if internal is None:
        raise RuntimeError("method2 internal audit requires internal state, but pipeline returned None")

    solve_mesh = internal.solve_mesh
    export_mesh = projector.build_uv_mesh(
        low_mesh=low_mesh,
        mapped_uv=mapped_uv,
        image=image,
        export_payload=export_payload,
    )

    solve_uv = np.asarray(mapped_uv, dtype=np.float64)
    export_uv = _safe_uv(export_mesh)
    source_quality = _quality_with_context(high_mesh, high_uv)
    solve_quality = _quality_with_context(solve_mesh, solve_uv)
    export_quality = _quality_with_context(export_mesh, export_uv)

    solved_jac = _compute_face_jacobians(internal.face_geom_pinv, solve_mesh, solve_uv)
    jac_summary, face_rel_err, face_cosine, face_log_area_ratio = _jacobian_diagnostics(
        np.asarray(internal.face_target_jacobian, dtype=np.float64),
        solved_jac,
        np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
    )
    sample_summary = _sample_residual_summary(internal, solve_uv)

    face_accepted_samples = np.zeros((len(solve_mesh.faces),), dtype=np.int64)
    solve_sample_face_ids = np.asarray(internal.solve_sample_face_ids, dtype=np.int64).reshape(-1)
    if solve_sample_face_ids.size > 0:
        np.add.at(face_accepted_samples, solve_sample_face_ids, 1)
    face_accepted_samples_raw = _coerce_face_count_array(
        method_stats.get("uv_m2_face_accepted_samples"),
        len(solve_mesh.faces),
    )
    if face_accepted_samples_raw is not None:
        face_accepted_samples = face_accepted_samples_raw
    face_total_samples = _coerce_face_count_array(
        method_stats.get("uv_m2_face_total_samples"),
        len(solve_mesh.faces),
    )
    support_summary = _support_summary(
        face_valid=np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
        face_accepted_samples=face_accepted_samples,
        face_total_samples=face_total_samples,
    )
    target_dispersion_summary, face_target_cov_norm = _target_dispersion_summary(
        face_cov_trace=np.asarray(internal.face_target_cov_trace, dtype=np.float64),
        face_valid=np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
        face_smooth_alpha=np.asarray(internal.face_smooth_alpha, dtype=np.float64),
        cov_scale_hint=method_stats.get("uv_m2_face_cov_scale"),
    )

    semantic_labels_raw = method_stats.get("uv_low_face_semantic_labels", None)
    island_labels: np.ndarray
    island_label_source: str
    if semantic_labels_raw is not None:
        labels_np = np.asarray(semantic_labels_raw, dtype=np.int64).reshape(-1)
        if labels_np.shape[0] == int(len(solve_mesh.faces)):
            island_labels = labels_np
            island_label_source = "semantic_face_label"
        else:
            island_labels = _connected_face_labels(solve_mesh)
            island_label_source = "connected_face_component_fallback"
    else:
        island_labels = _connected_face_labels(solve_mesh)
        island_label_source = "connected_face_component"

    island_summary, per_island = _per_island_diagnostics(
        mesh=solve_mesh,
        uv=solve_uv,
        island_labels=island_labels,
        island_label_source=island_label_source,
        anchor_ids=np.asarray(internal.anchor_vertex_ids, dtype=np.int64),
        face_valid=np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
        face_rel_err=face_rel_err,
        face_cosine=face_cosine,
        face_log_area_ratio=face_log_area_ratio,
        face_accepted_samples=face_accepted_samples,
        face_total_samples=face_total_samples,
        face_target_cov_trace=np.asarray(internal.face_target_cov_trace, dtype=np.float64),
        face_target_cov_norm=face_target_cov_norm,
    )
    neighbors = _face_neighbors(solve_mesh)
    component_ids, component_sizes = _component_size_per_face(island_labels, neighbors)
    high_face_jac, _ = _compute_high_face_jacobians(high_mesh, high_uv)
    ctx = {
        "cfg": cfg,
        "method_stats": method_stats,
        "high_mesh": high_mesh,
        "high_uv": high_uv,
        "high_face_jac": high_face_jac,
        "image": image,
        "solve_mesh": solve_mesh,
        "solve_uv": solve_uv,
        "internal": internal,
        "target_jac": np.asarray(internal.face_target_jacobian, dtype=np.float64),
        "face_valid": np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
        "face_weights": np.asarray(internal.face_target_weights, dtype=np.float64),
        "face_active": np.asarray(internal.solve_face_active_mask, dtype=np.bool_),
        "face_rel_err": face_rel_err,
        "face_cosine": face_cosine,
        "face_log_area_ratio": face_log_area_ratio,
        "face_target_cov_norm": face_target_cov_norm,
        "face_accepted_samples": face_accepted_samples,
        "face_total_samples": face_total_samples,
        "island_labels": island_labels,
        "neighbors": neighbors,
        "component_ids": component_ids,
        "component_sizes": component_sizes,
        "agg_params": _method2_aggregation_params(cfg),
        "sample_mix_arrays": _sample_mix_arrays(
            {
                "solve_mesh": solve_mesh,
                "internal": internal,
            }
        ),
        "edge_data": _compute_edge_jump_data(
            solve_mesh,
            np.asarray(internal.face_target_jacobian, dtype=np.float64),
            np.asarray(internal.face_target_valid_mask, dtype=np.bool_),
        ),
    }
    from audit_method2_internal_experiments import _run_all_experiments

    experiments = _run_all_experiments(ctx)

    method_stats_stripped = _strip_large_stats(dict(method_stats))
    report: Dict[str, Any] = {
        "case_name": case_name,
        "inputs": {
            "high": str(high_path) if high_path is not None else None,
            "low": str(low_path) if low_path is not None else None,
            "device_requested": str(device),
            "seam_strategy": str(seam_strategy),
            "sanitize_low": bool(sanitize_low),
            "image_source": image_source,
        },
        "mesh_counts": {
            "high_vertices": int(len(high_mesh.vertices)),
            "high_faces": int(len(high_mesh.faces)),
            "low_vertices": int(len(low_mesh.vertices)),
            "low_faces": int(len(low_mesh.faces)),
            "solve_vertices": int(len(solve_mesh.vertices)),
            "solve_faces": int(len(solve_mesh.faces)),
            "export_vertices": int(len(export_mesh.vertices)),
            "export_faces": int(len(export_mesh.faces)),
        },
        "source_quality": source_quality,
        "solve_mesh_quality": solve_quality,
        "export_mesh_quality": export_quality,
        "sample_residual_summary": sample_summary,
        "jacobian_summary": jac_summary,
        "support_summary": support_summary,
        "target_dispersion_summary": target_dispersion_summary,
        "island_summary": island_summary,
        "per_island": per_island,
        "experiments": experiments,
        "method2_stats": method_stats_stripped,
        "selected_method2_stats": {
            "uv_mode_used": method_stats.get("uv_mode_used"),
            "uv_color_reproj_l1": method_stats.get("uv_color_reproj_l1"),
            "uv_color_reproj_l2": method_stats.get("uv_color_reproj_l2"),
            "uv_m2_jacobian_valid_ratio": method_stats.get("uv_m2_jacobian_valid_ratio"),
            "uv_m2_anchor_count_total": method_stats.get("uv_m2_anchor_count_total"),
            "uv_m2_boundary_anchor_count": method_stats.get("uv_m2_boundary_anchor_count"),
            "uv_m2_soft_anchor_count": method_stats.get("uv_m2_soft_anchor_count"),
            "uv_m2_island_count": method_stats.get("uv_m2_island_count"),
            "uv_m2_solve_per_island_enabled": method_stats.get("uv_m2_solve_per_island_enabled"),
            "uv_m2_post_align_applied": method_stats.get("uv_m2_post_align_applied"),
            "uv_m2_post_align_shift_norm": method_stats.get("uv_m2_post_align_shift_norm"),
            "uv_m2_face_cov_scale": method_stats.get("uv_m2_face_cov_scale"),
            "uv_m2_face_jacobian_cov_p95": method_stats.get("uv_m2_face_jacobian_cov_p95"),
            "uv_m2_adaptive_smooth_alpha_mean": method_stats.get("uv_m2_adaptive_smooth_alpha_mean"),
            "uv_m2_adaptive_smooth_alpha_min": method_stats.get("uv_m2_adaptive_smooth_alpha_min"),
            "uv_m2_adaptive_smooth_alpha_max": method_stats.get("uv_m2_adaptive_smooth_alpha_max"),
            "uv_m2_constraint_relaxation_used": method_stats.get("uv_m2_constraint_relaxation_used"),
            "uv_m2_poisson_residual_u": method_stats.get("uv_m2_poisson_residual_u"),
            "uv_m2_poisson_residual_v": method_stats.get("uv_m2_poisson_residual_v"),
            "uv_m2_system_cond_proxy": method_stats.get("uv_m2_system_cond_proxy"),
            "uv_solver_linear_backend_used": method_stats.get("uv_solver_linear_backend_used"),
            "uv_export_vertices": method_stats.get("uv_export_vertices"),
            "uv_export_faces": method_stats.get("uv_export_faces"),
        },
        "config_effective": {
            "solve": cfg.get("solve", {}),
            "seam": {
                "strategy": cfg.get("seam", {}).get("strategy"),
                "sanitize_enabled": cfg.get("seam", {}).get("sanitize_enabled"),
                "transfer_sampling_mode": cfg.get("seam", {}).get("transfer_sampling_mode"),
                "validation_mode": cfg.get("seam", {}).get("validation_mode"),
            },
            "method2": {
                "anchor_mode": cfg.get("method2", {}).get("anchor_mode"),
                "anchor_points_per_component": cfg.get("method2", {}).get("anchor_points_per_component"),
                "anchor_min_points_per_component": cfg.get("method2", {}).get("anchor_min_points_per_component"),
                "anchor_max_points_per_component": cfg.get("method2", {}).get("anchor_max_points_per_component"),
                "anchor_target_vertices_per_anchor": cfg.get("method2", {}).get("anchor_target_vertices_per_anchor"),
                "post_align_translation": cfg.get("method2", {}).get("post_align_translation"),
                "post_align_max_shift": cfg.get("method2", {}).get("post_align_max_shift"),
                "solve_per_island": cfg.get("method2", {}).get("solve_per_island"),
                "outlier_sigma": cfg.get("method2", {}).get("outlier_sigma"),
                "outlier_quantile": cfg.get("method2", {}).get("outlier_quantile"),
            },
        },
    }
    report = _sanitize_json(report)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Method2 internal Jacobian / anchor / island diagnostics")
    p.add_argument("--high", type=Path, required=True, help="High mesh with UV")
    p.add_argument("--low", type=Path, required=True, help="Low mesh to audit")
    p.add_argument("--out-json", type=Path, default=None, help="Optional output JSON report")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--uv-seam-strategy",
        type=str,
        default="halfedge_island",
        choices=["legacy", "halfedge_island"],
        help="Method2 seam strategy used during audit run",
    )
    p.add_argument(
        "--sanitize-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-mesh sanitization in halfedge island path",
    )
    p.add_argument(
        "--solve-per-island",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Method2 solve_per_island",
    )
    p.add_argument(
        "--post-align",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Method2 post_align_translation",
    )
    p.add_argument("--anchor-weight", type=float, default=None, help="Override solve.anchor_weight")
    p.add_argument("--anchor-points-per-component", type=int, default=None, help="Override method2 anchor density")
    p.add_argument("--anchor-min-points-per-component", type=int, default=None)
    p.add_argument("--anchor-max-points-per-component", type=int, default=None)
    p.add_argument("--anchor-target-vertices-per-anchor", type=int, default=None)
    p.add_argument("--case-name", type=str, default=None, help="Optional case label")
    return p.parse_args()


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if args.anchor_weight is not None:
        out.setdefault("solve", {})["anchor_weight"] = float(args.anchor_weight)
    method2: Dict[str, Any] = {}
    if args.solve_per_island is not None:
        method2["solve_per_island"] = bool(args.solve_per_island)
    if args.post_align is not None:
        method2["post_align_translation"] = bool(args.post_align)
    if args.anchor_points_per_component is not None:
        method2["anchor_points_per_component"] = int(args.anchor_points_per_component)
    if args.anchor_min_points_per_component is not None:
        method2["anchor_min_points_per_component"] = int(args.anchor_min_points_per_component)
    if args.anchor_max_points_per_component is not None:
        method2["anchor_max_points_per_component"] = int(args.anchor_max_points_per_component)
    if args.anchor_target_vertices_per_anchor is not None:
        method2["anchor_target_vertices_per_anchor"] = int(args.anchor_target_vertices_per_anchor)
    if method2:
        out["method2"] = method2
    return out


def main() -> None:
    args = parse_args()
    report = run_method2_internal_audit_on_meshes(
        high_mesh=_load_mesh(args.high.resolve()),
        low_mesh=_load_mesh(args.low.resolve()),
        high_path=args.high.resolve(),
        low_path=args.low.resolve(),
        device=str(args.device),
        seam_strategy=str(args.uv_seam_strategy),
        sanitize_low=bool(args.sanitize_low),
        options_overrides=_cli_overrides(args),
        out_json=args.out_json.resolve() if args.out_json is not None else None,
        case_name=args.case_name,
    )
    sel = report["selected_method2_stats"]
    sq = report["solve_mesh_quality"]
    jq = report["jacobian_summary"]
    print(
        "[method2_internal_audit] "
        f"case={report.get('case_name') or '-'}, "
        f"solve_stretch_p95={sq.get('uv_stretch_p95')}, "
        f"stretch_only_bad={sq.get('uv_bad_tri_ratio_stretch_only')}, "
        f"jac_rel_p95={jq.get('frob_rel_error_p95')}, "
        f"reproj_l1={sel.get('uv_color_reproj_l1')}, "
        f"anchors={sel.get('uv_m2_anchor_count_total')}, "
        f"islands={sel.get('uv_m2_island_count')}"
    )
    if args.out_json is not None:
        print(f"report_json={args.out_json.resolve()}")


if __name__ == "__main__":
    main()
