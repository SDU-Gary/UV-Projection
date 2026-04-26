from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import trimesh


def _finite_quantile(values: np.ndarray, q: float, fallback: float = 1.0) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback)
    scale = float(np.quantile(vals, float(q)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(vals)) if vals.size > 0 else float(fallback)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(fallback)
    return float(scale)


def _quantile_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "p50": float(np.quantile(vals, 0.50)),
        "p95": float(np.quantile(vals, 0.95)),
        "p99": float(np.quantile(vals, 0.99)),
        "max": float(np.max(vals)),
    }


def build_interior_edge_table(
    *,
    mesh: trimesh.Trimesh,
    face_valid_mask: np.ndarray,
    existing_seam_edges: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    edge_vid = np.asarray(getattr(mesh, "face_adjacency_edges", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    n_faces = int(len(mesh.faces))
    if adj.ndim != 2 or adj.shape[1] != 2 or edge_vid.ndim != 2 or edge_vid.shape[0] != adj.shape[0]:
        return {
            "edge_ids": np.zeros((0,), dtype=np.int64),
            "adjacency": np.zeros((0, 2), dtype=np.int64),
            "edge_vertices": np.zeros((0, 2), dtype=np.int64),
            "edge_length": np.zeros((0,), dtype=np.float64),
            "eligible_mask_full": np.zeros((0,), dtype=np.bool_),
            "summary": {
                "face_count": int(n_faces),
                "adjacency_edge_count": 0,
                "eligible_interior_edge_count": 0,
                "existing_seam_suppressed_count": 0,
                "invalid_face_suppressed_count": 0,
                "degenerate_edge_suppressed_count": 0,
                "eligible_edge_length_total": 0.0,
            },
        }

    faces_ok = np.asarray(face_valid_mask, dtype=np.bool_).reshape(-1)
    if faces_ok.shape[0] != n_faces:
        raise RuntimeError("face_valid_mask length mismatch for build_interior_edge_table")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edge_vertices_sorted = np.sort(edge_vid, axis=1)
    edge_length = np.linalg.norm(verts[edge_vertices_sorted[:, 1]] - verts[edge_vertices_sorted[:, 0]], axis=1)
    valid_faces_edge = faces_ok[adj[:, 0]] & faces_ok[adj[:, 1]]
    nondeg = np.isfinite(edge_length) & (edge_length > 1e-12)

    seam_hit = np.zeros((adj.shape[0],), dtype=np.bool_)
    if existing_seam_edges is not None:
        seam_np = np.asarray(existing_seam_edges, dtype=np.int64)
        if seam_np.ndim == 2 and seam_np.shape[1] == 2 and seam_np.shape[0] > 0:
            seam_keys = {tuple(map(int, row)) for row in np.unique(np.sort(seam_np, axis=1), axis=0).tolist()}
            if seam_keys:
                seam_hit = np.asarray([tuple(map(int, row)) in seam_keys for row in edge_vertices_sorted.tolist()], dtype=np.bool_)

    eligible = valid_faces_edge & nondeg & (~seam_hit)
    edge_ids = np.where(eligible)[0].astype(np.int64, copy=False)
    return {
        "edge_ids": edge_ids,
        "adjacency": adj[edge_ids].astype(np.int64, copy=False),
        "edge_vertices": edge_vertices_sorted[edge_ids].astype(np.int64, copy=False),
        "edge_length": edge_length[edge_ids].astype(np.float64, copy=False),
        "eligible_mask_full": eligible.astype(np.bool_, copy=False),
        "summary": {
            "face_count": int(n_faces),
            "adjacency_edge_count": int(adj.shape[0]),
            "eligible_interior_edge_count": int(edge_ids.size),
            "existing_seam_suppressed_count": int(np.count_nonzero(seam_hit)),
            "invalid_face_suppressed_count": int(np.count_nonzero(~valid_faces_edge)),
            "degenerate_edge_suppressed_count": int(np.count_nonzero(valid_faces_edge & (~nondeg))),
            "eligible_edge_length_total": float(np.sum(edge_length[edge_ids])) if edge_ids.size > 0 else 0.0,
        },
    }


def score_route_c_cut_edges(
    *,
    edge_table: Dict[str, Any],
    edge_jump_l2: np.ndarray,
    face_cycle_residual: np.ndarray,
    face_stretch: np.ndarray,
    face_divergence: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    edge_ids = np.asarray(edge_table.get("edge_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    adj = np.asarray(edge_table.get("adjacency", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    edge_vertices = np.asarray(edge_table.get("edge_vertices", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    edge_length = np.asarray(edge_table.get("edge_length", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    n_edges = int(edge_ids.size)
    if n_edges == 0:
        zero = np.zeros((0,), dtype=np.float64)
        return {
            "edge_ids": edge_ids,
            "adjacency": adj,
            "edge_vertices": edge_vertices,
            "edge_length": edge_length,
            "score": zero,
            "jump": zero,
            "cycle": zero,
            "stretch": zero,
            "divergence": zero,
            "norm_jump": zero,
            "norm_cycle": zero,
            "norm_stretch": zero,
            "norm_divergence": zero,
            "scales": {
                "jump_p95": None,
                "cycle_p95": None,
                "stretch_p95": None,
                "divergence_p95": None,
            },
            "summary": {
                "eligible_edge_count": 0,
                "score": _quantile_summary(zero),
            },
        }

    edge_jump = np.asarray(edge_jump_l2, dtype=np.float64).reshape(-1)[edge_ids]
    face_cycle = np.asarray(face_cycle_residual, dtype=np.float64).reshape(-1)
    face_stretch_np = np.asarray(face_stretch, dtype=np.float64).reshape(-1)
    face_div_np = np.asarray(face_divergence, dtype=np.float64).reshape(-1)

    cycle_edge = np.maximum(face_cycle[adj[:, 0]], face_cycle[adj[:, 1]])
    stretch_edge = np.maximum(face_stretch_np[adj[:, 0]], face_stretch_np[adj[:, 1]])
    div_edge = np.maximum(face_div_np[adj[:, 0]], face_div_np[adj[:, 1]])

    jump_scale = _finite_quantile(edge_jump, 0.95)
    cycle_scale = _finite_quantile(cycle_edge, 0.95)
    stretch_scale = _finite_quantile(stretch_edge, 0.95)
    div_scale = _finite_quantile(div_edge, 0.95)

    jump_norm = np.clip(np.nan_to_num(edge_jump / jump_scale, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    cycle_norm = np.clip(np.nan_to_num(cycle_edge / cycle_scale, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    stretch_norm = np.clip(np.nan_to_num(stretch_edge / stretch_scale, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)
    div_norm = np.clip(np.nan_to_num(div_edge / div_scale, nan=0.0, posinf=5.0, neginf=0.0), 0.0, 5.0)

    w = {
        "jump": 1.0,
        "cycle": 0.75,
        "stretch": 0.5,
        "divergence": 0.25,
    }
    if weights:
        w.update({str(k): float(v) for k, v in weights.items()})
    score = (
        w["jump"] * jump_norm
        + w["cycle"] * cycle_norm
        + w["stretch"] * stretch_norm
        + w["divergence"] * div_norm
    ).astype(np.float64, copy=False)

    return {
        "edge_ids": edge_ids,
        "adjacency": adj,
        "edge_vertices": edge_vertices,
        "edge_length": edge_length,
        "score": score,
        "jump": edge_jump,
        "cycle": cycle_edge,
        "stretch": stretch_edge,
        "divergence": div_edge,
        "norm_jump": jump_norm,
        "norm_cycle": cycle_norm,
        "norm_stretch": stretch_norm,
        "norm_divergence": div_norm,
        "scales": {
            "jump_p95": float(jump_scale),
            "cycle_p95": float(cycle_scale),
            "stretch_p95": float(stretch_scale),
            "divergence_p95": float(div_scale),
        },
        "summary": {
            "eligible_edge_count": int(n_edges),
            "score": _quantile_summary(score),
        },
    }


def select_budgeted_cut_edges(
    *,
    scored_edges: Dict[str, Any],
    fraction: float,
) -> Dict[str, Any]:
    edge_vertices = np.asarray(scored_edges.get("edge_vertices", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
    score = np.asarray(scored_edges.get("score", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    edge_length = np.asarray(scored_edges.get("edge_length", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    eligible_count = int(score.size)
    frac = float(np.clip(fraction, 0.0, 1.0))
    if eligible_count == 0 or frac <= 0.0:
        zero = np.zeros((0,), dtype=np.float64)
        return {
            "fraction": frac,
            "selected_count": 0,
            "selected_ratio": 0.0,
            "selected_edge_indices": np.zeros((0,), dtype=np.int64),
            "selected_edges": np.zeros((0, 2), dtype=np.int64),
            "selected_edge_length": 0.0,
            "selected_edge_length_ratio": 0.0,
            "selected_score_summary": _quantile_summary(zero),
            "all_score_summary": _quantile_summary(score),
        }

    target_count = int(np.ceil(frac * eligible_count))
    target_count = max(1, min(target_count, eligible_count))
    order = np.argsort(-score, kind="mergesort")
    sel = np.sort(order[:target_count]).astype(np.int64, copy=False)
    total_length = float(np.sum(edge_length)) if edge_length.size > 0 else 0.0
    sel_length = float(np.sum(edge_length[sel])) if sel.size > 0 else 0.0
    return {
        "fraction": frac,
        "selected_count": int(sel.size),
        "selected_ratio": float(sel.size / max(1, eligible_count)),
        "selected_edge_indices": sel,
        "selected_edges": edge_vertices[sel].astype(np.int64, copy=False),
        "selected_edge_length": sel_length,
        "selected_edge_length_ratio": float(sel_length / max(total_length, 1e-12)) if sel.size > 0 else 0.0,
        "selected_score_summary": _quantile_summary(score[sel]),
        "all_score_summary": _quantile_summary(score),
    }

