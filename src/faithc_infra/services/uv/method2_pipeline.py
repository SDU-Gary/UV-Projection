from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import time
import trimesh
from scipy.sparse import coo_matrix, csr_matrix, diags

from ..halfedge_topology import (
    compute_high_face_uv_islands,
    detect_cut_edges_from_face_labels,
    split_vertices_along_cut_edges,
)
from .correspondence import (
    build_high_cuda_context,
    correspond_points_hybrid,
    detect_cross_seam_faces,
    major_face_island_labels,
)
from .mesh_sanitizer import ensure_halfedge_external_dependencies, sanitize_mesh_for_halfedge
from .openmesh_seams import extract_seam_edges_openmesh, validate_face_partition_by_seams
from .semantic_transfer import transfer_face_semantics_by_projection
from .linear_solver import (
    build_cuda_sparse_system,
    connected_components_labels,
    interpolate_sample_uv,
    mesh_laplacian,
    nearest_vertex_uv,
    solve_linear_cuda_pcg,
    solve_linear_robust,
)
from .quality import texture_gradient_weights, texture_reprojection_error
from .sampling import sample_low_mesh
from .texture_io import resolve_device

_HIGH_ISLAND_CACHE: Dict[Tuple[int, int, int, int, int, float, float], Tuple[np.ndarray, Dict[str, int]]] = {}
_HIGH_ISLAND_CACHE_MAX = 8


@dataclass
class Method2InternalState:
    solve_mesh: trimesh.Trimesh
    mapped_uv_init: np.ndarray
    face_target_jacobian: np.ndarray
    face_target_valid_mask: np.ndarray
    face_target_weights: np.ndarray
    face_geom_pinv: np.ndarray
    solve_sample_face_ids: np.ndarray
    solve_sample_bary: np.ndarray
    solve_target_uv: np.ndarray
    anchor_vertex_ids: np.ndarray
    anchor_uv: np.ndarray
    resolved_device: str
    export_payload: Dict[str, Any]
    method_stats: Dict[str, Any]


def _cached_high_face_uv_islands(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    position_eps: float,
    uv_eps: float,
    use_cache: bool,
) -> Tuple[np.ndarray, Dict[str, int], bool]:
    pos_eps = float(position_eps)
    uv_eps_f = float(uv_eps)
    if not use_cache:
        labels, meta = compute_high_face_uv_islands(
            vertices=np.asarray(high_mesh.vertices, dtype=np.float64),
            faces=np.asarray(high_mesh.faces, dtype=np.int64),
            uv=np.asarray(high_uv, dtype=np.float64),
            position_eps=pos_eps,
            uv_eps=uv_eps_f,
        )
        return labels, meta, False

    key = (
        id(high_mesh.vertices),
        id(high_mesh.faces),
        id(high_uv),
        int(len(high_mesh.vertices)),
        int(len(high_mesh.faces)),
        round(pos_eps, 12),
        round(uv_eps_f, 12),
    )
    cached = _HIGH_ISLAND_CACHE.get(key)
    if cached is not None:
        labels_c, meta_c = cached
        return labels_c, dict(meta_c), True

    labels, meta = compute_high_face_uv_islands(
        vertices=np.asarray(high_mesh.vertices, dtype=np.float64),
        faces=np.asarray(high_mesh.faces, dtype=np.int64),
        uv=np.asarray(high_uv, dtype=np.float64),
        position_eps=pos_eps,
        uv_eps=uv_eps_f,
    )
    _HIGH_ISLAND_CACHE[key] = (labels, dict(meta))
    if len(_HIGH_ISLAND_CACHE) > _HIGH_ISLAND_CACHE_MAX:
        first_key = next(iter(_HIGH_ISLAND_CACHE.keys()))
        _HIGH_ISLAND_CACHE.pop(first_key, None)
    return labels, meta, False


def _compute_face_geometry_pinv(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tri = verts[faces]

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]

    g11 = np.sum(e1 * e1, axis=1)
    g12 = np.sum(e1 * e2, axis=1)
    g22 = np.sum(e2 * e2, axis=1)

    det = g11 * g22 - g12 * g12
    valid = np.isfinite(det) & (det > 1e-18)

    pinv = np.zeros((len(faces), 2, 3), dtype=np.float64)
    if np.any(valid):
        inv_det = 1.0 / det[valid]
        m00 = g22[valid] * inv_det
        m01 = -g12[valid] * inv_det
        m10 = -g12[valid] * inv_det
        m11 = g11[valid] * inv_det

        e1v = e1[valid]
        e2v = e2[valid]
        pinv[valid, 0] = m00[:, None] * e1v + m01[:, None] * e2v
        pinv[valid, 1] = m10[:, None] * e1v + m11[:, None] * e2v
    return pinv, valid


def _compute_high_face_jacobians(high_mesh: trimesh.Trimesh, high_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(high_mesh.faces, dtype=np.int64)
    tri_uv = np.asarray(high_uv, dtype=np.float64)[faces]
    du1 = tri_uv[:, 1] - tri_uv[:, 0]
    du2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_grad = np.stack([du1, du2], axis=2)  # [F, 2, 2]

    pinv, valid = _compute_face_geometry_pinv(high_mesh)
    jac = np.zeros((len(faces), 2, 3), dtype=np.float64)
    if np.any(valid):
        jac[valid] = np.einsum("fij,fjk->fik", uv_grad[valid], pinv[valid], optimize=True)
    return jac, valid


def _print_local_frame_length_diagnostics(mesh: trimesh.Trimesh, *, tag: str = "mesh") -> None:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if faces.size == 0 or verts.size == 0:
        print(f"[method2][diag] {tag} local frame: empty mesh")
        return

    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    eps = 1e-20

    t_norm = np.linalg.norm(e1, axis=1)
    tangent = np.zeros_like(e1)
    t_ok = t_norm > eps
    tangent[t_ok] = e1[t_ok] / t_norm[t_ok, None]

    n_raw = np.cross(e1, e2)
    n_norm = np.linalg.norm(n_raw, axis=1)
    normal = np.zeros_like(n_raw)
    n_ok = n_norm > eps
    normal[n_ok] = n_raw[n_ok] / n_norm[n_ok, None]

    b_raw = np.cross(normal, tangent)
    b_norm = np.linalg.norm(b_raw, axis=1)
    bitangent = np.zeros_like(b_raw)
    b_ok = b_norm > eps
    bitangent[b_ok] = b_raw[b_ok] / b_norm[b_ok, None]

    valid = t_ok & n_ok & b_ok
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        print(f"[method2][diag] {tag} local frame: no valid tangent/bitangent")
        return

    tangent_len = np.linalg.norm(tangent[valid], axis=1)
    bitangent_len = np.linalg.norm(bitangent[valid], axis=1)
    tol = 1e-6
    tangent_exact = int(np.count_nonzero(tangent_len == 1.0))
    bitangent_exact = int(np.count_nonzero(bitangent_len == 1.0))
    tangent_close = int(np.count_nonzero(np.abs(tangent_len - 1.0) <= tol))
    bitangent_close = int(np.count_nonzero(np.abs(bitangent_len - 1.0) <= tol))

    print(
        f"[method2][diag] {tag} tangent_norm min/mean/max="
        f"{float(np.min(tangent_len)):.9f}/{float(np.mean(tangent_len)):.9f}/{float(np.max(tangent_len)):.9f}, "
        f"exact_1={tangent_exact}/{valid_count}, tol({tol})={tangent_close}/{valid_count}"
    )
    print(
        f"[method2][diag] {tag} bitangent_norm min/mean/max="
        f"{float(np.min(bitangent_len)):.9f}/{float(np.mean(bitangent_len)):.9f}/{float(np.max(bitangent_len)):.9f}, "
        f"exact_1={bitangent_exact}/{valid_count}, tol({tol})={bitangent_close}/{valid_count}"
    )


def _aggregate_jacobian_weighted_mean_2x3(jac: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    jac_np = np.asarray(jac, dtype=np.float64)
    if jac_np.size == 0 or jac_np.ndim != 3 or jac_np.shape[1:] != (2, 3):
        return None
    w_np = np.asarray(weights, dtype=np.float64)
    if w_np.ndim != 1 or w_np.shape[0] != jac_np.shape[0]:
        return None
    valid = np.isfinite(jac_np).all(axis=(1, 2)) & np.isfinite(w_np)
    if not np.any(valid):
        return None
    jv = jac_np[valid]
    wv = np.maximum(w_np[valid], 1e-12)
    ws = float(np.sum(wv))
    if ws <= 0.0:
        return None
    mu = np.sum(jv * wv[:, None, None], axis=0) / ws
    if not np.isfinite(mu).all():
        return None
    return mu


def _aggregate_face_target_jacobians(
    *,
    n_low_faces: int,
    sample_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    sample_weights: np.ndarray,
    high_face_jac: np.ndarray,
    min_samples_per_face: int,
    outlier_sigma: float,
    outlier_quantile: float,
    face_weight_floor: float,
    irls_iters: int,
    huber_delta: float,
    fast_mode: bool = True,
    small_group_fast_threshold: int = 6,
    small_group_skip_irls: bool = True,
    small_group_skip_outlier: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    face_jac = np.zeros((n_low_faces, 2, 3), dtype=np.float64)
    face_weights = np.zeros((n_low_faces,), dtype=np.float64)
    face_valid = np.zeros((n_low_faces,), dtype=np.bool_)
    face_cov_trace = np.zeros((n_low_faces,), dtype=np.float64)

    if sample_face_ids.size == 0:
        return face_jac, face_weights, face_valid, face_cov_trace, {
            "uv_m2_outlier_reject_ratio": 0.0,
            "uv_m2_outlier_total_candidates": 0,
            "uv_m2_outlier_total_rejected": 0,
            "uv_m2_face_jacobian_cov_p95": 0.0,
            "uv_m2_irls_iters_used": 0,
            "uv_m2_jacobian_negative_det_drop_ratio": 0.0,
        }

    order = np.argsort(sample_face_ids, kind="mergesort")
    sf = sample_face_ids[order]
    tf = target_face_ids[order]
    sw = sample_weights[order]

    total_candidates = 0
    total_rejected = 0
    n = int(len(sf))
    if n == 0:
        return face_jac, face_weights, face_valid, face_cov_trace, {
            "uv_m2_outlier_reject_ratio": 0.0,
            "uv_m2_outlier_total_candidates": 0,
            "uv_m2_outlier_total_rejected": 0,
            "uv_m2_face_jacobian_cov_p95": 0.0,
            "uv_m2_irls_iters_used": 0,
            "uv_m2_jacobian_negative_det_drop_ratio": 0.0,
        }

    split_idx = np.flatnonzero(np.diff(sf)) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [n]))

    q = float(np.clip(outlier_quantile, 0.50, 1.0))
    min_samples = max(1, int(min_samples_per_face))
    small_fast_th = max(min_samples, int(small_group_fast_threshold))
    sigma = max(0.0, float(outlier_sigma))
    w_floor = max(float(face_weight_floor), 1e-12)
    irls_n = max(1, int(irls_iters))
    hub_delta = max(1e-6, float(huber_delta))
    det_drop = 0
    det_total = 0
    scale_repair_count = 0
    scale_repair_ratio_sum = 0.0
    scale_repair_ratio_min = np.inf
    scale_repair_ratio_max = 0.0

    def _resuscitate_scale(j_agg: np.ndarray, j_samples: np.ndarray, w_samples: np.ndarray) -> Tuple[np.ndarray, float]:
        # Keep aggregated direction but restore magnitude to sampled Jacobian scale.
        agg_norm = float(np.linalg.norm(j_agg))
        if agg_norm <= 1e-12:
            return j_agg, 1.0
        sample_norm = np.linalg.norm(j_samples, axis=(1, 2))
        w_sum = float(np.sum(w_samples))
        if w_sum <= 1e-12:
            return j_agg, 1.0
        mean_norm = float(np.sum(sample_norm * w_samples) / w_sum)
        if not np.isfinite(mean_norm) or mean_norm <= 1e-12:
            return j_agg, 1.0
        ratio = mean_norm / agg_norm
        if not np.isfinite(ratio) or ratio <= 0.0:
            return j_agg, 1.0
        return j_agg * ratio, float(ratio)

    for i, j in zip(starts.tolist(), ends.tolist()):
        fi = int(sf[i])
        count = j - i
        if fi < 0 or fi >= n_low_faces or count < min_samples:
            continue

        tf_group = tf[i:j]
        w_group = np.asarray(sw[i:j], dtype=np.float64, copy=False)
        jac_group3 = high_face_jac[tf_group]
        finite_mask = np.isfinite(jac_group3).all(axis=(1, 2)) & np.isfinite(w_group)
        det_total += int(finite_mask.size)
        det_drop += int(np.count_nonzero(~finite_mask))
        if not np.any(finite_mask):
            continue

        jac_group3 = jac_group3[finite_mask]
        w_group = w_group[finite_mask]
        count_eff = int(jac_group3.shape[0])
        if count_eff < min_samples:
            continue
        total_candidates += int(count_eff)

        base_w = np.maximum(w_group, w_floor)
        base_w_sum = float(np.sum(base_w))
        if base_w_sum <= 0.0:
            continue

        # Fast path for tiny face-sample groups: robust stats become overhead-dominated
        # and provide little signal with only a few samples.
        if fast_mode and count_eff <= small_fast_th:
            mu_fast = np.sum(jac_group3 * base_w[:, None, None], axis=0) / base_w_sum
            if small_group_skip_irls and small_group_skip_outlier:
                mu_fast_fixed, ratio_fast = _resuscitate_scale(mu_fast, jac_group3, base_w)
                face_jac[fi] = mu_fast_fixed
                face_weights[fi] = base_w_sum
                face_valid[fi] = True
                scale_repair_count += 1
                scale_repair_ratio_sum += float(ratio_fast)
                scale_repair_ratio_min = float(min(scale_repair_ratio_min, ratio_fast))
                scale_repair_ratio_max = float(max(scale_repair_ratio_max, ratio_fast))
                jflat = jac_group3.reshape(count_eff, -1)
                mu_flat = np.sum(jflat * base_w[:, None], axis=0) / base_w_sum
                diff = jflat - mu_flat[None, :]
                face_cov_trace[fi] = float(np.sum((base_w[:, None] * diff) * diff) / max(base_w_sum, 1e-12))
                continue

        mu3 = mu_fast if (fast_mode and count_eff <= small_fast_th) else (
            np.sum(jac_group3 * base_w[:, None, None], axis=0) / base_w_sum
        )

        irls_local = 0 if small_group_skip_irls and (fast_mode and count_eff <= small_fast_th) else irls_n
        for _ in range(irls_local):
            diff3 = jac_group3 - mu3[None, :, :]
            res = np.sqrt(np.maximum(np.sum(diff3 * diff3, axis=(1, 2)), 0.0))
            scale = max(float(np.median(res)), 1e-8)
            r = res / scale
            hub_w = np.ones_like(r)
            over = r > hub_delta
            hub_w[over] = hub_delta / np.maximum(r[over], 1e-12)
            w_tot = np.maximum(base_w * hub_w, w_floor)
            w_tot_sum = float(np.sum(w_tot))
            if w_tot_sum <= 0.0:
                break
            mu3 = np.sum(jac_group3 * w_tot[:, None, None], axis=0) / w_tot_sum

        diff3 = jac_group3 - mu3[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff3 * diff3, axis=(1, 2)), 0.0))

        if small_group_skip_outlier and (fast_mode and count_eff <= small_fast_th):
            keep = np.ones((count_eff,), dtype=np.bool_)
        elif count_eff >= 3:
            d_med = float(np.median(dist))
            d_mad = float(np.median(np.abs(dist - d_med)))
            thr_mad = d_med + sigma * max(d_mad, 1e-8)
            if fast_mode and q < 1.0:
                q_idx = min(count_eff - 1, max(0, int(np.floor(q * (count_eff - 1)))))
                thr_q = float(np.partition(dist, q_idx)[q_idx])
            else:
                thr_q = float(np.quantile(dist, q))
            thr = max(thr_mad, thr_q)
            keep = dist <= thr
        else:
            keep = np.ones((count_eff,), dtype=np.bool_)

        kept = int(np.count_nonzero(keep))
        if kept < min_samples:
            keep = np.zeros((count_eff,), dtype=np.bool_)
            k = min(min_samples, count_eff)
            if k >= count_eff:
                keep[:] = True
            else:
                keep[np.argpartition(dist, k - 1)[:k]] = True
            kept = int(np.count_nonzero(keep))
        total_rejected += int(count_eff - kept)

        if kept <= 0:
            continue

        wk = np.maximum(w_group[keep], w_floor)
        jk3 = jac_group3[keep]
        wsum = float(np.sum(wk))
        if wsum <= 0.0:
            continue
        mu_keep = np.sum(jk3 * wk[:, None, None], axis=0) / wsum
        if not np.isfinite(mu_keep).all():
            continue
        mu_keep, ratio_keep = _resuscitate_scale(mu_keep, jk3, wk)

        face_jac[fi] = mu_keep
        face_weights[fi] = wsum
        face_valid[fi] = True
        scale_repair_count += 1
        scale_repair_ratio_sum += float(ratio_keep)
        scale_repair_ratio_min = float(min(scale_repair_ratio_min, ratio_keep))
        scale_repair_ratio_max = float(max(scale_repair_ratio_max, ratio_keep))

        jflat = jk3.reshape(kept, -1)
        mu_flat = np.sum(jflat * wk[:, None], axis=0) / wsum
        diff = jflat - mu_flat[None, :]
        face_cov_trace[fi] = float(np.sum((wk[:, None] * diff) * diff) / max(wsum, 1e-12))

    reject_ratio = float(total_rejected / max(1, total_candidates))
    cov_valid = face_cov_trace[face_valid]
    invalid_drop_ratio = float(det_drop / max(1, det_total))
    return face_jac, face_weights, face_valid, face_cov_trace, {
        "uv_m2_outlier_reject_ratio": reject_ratio,
        "uv_m2_outlier_total_candidates": int(total_candidates),
        "uv_m2_outlier_total_rejected": int(total_rejected),
        "uv_m2_face_jacobian_cov_p95": float(np.percentile(cov_valid, 95)) if cov_valid.size > 0 else 0.0,
        "uv_m2_irls_iters_used": int(irls_n),
        "uv_m2_jacobian_scale_resuscitation_enabled": True,
        "uv_m2_jacobian_scale_resuscitation_count": int(scale_repair_count),
        "uv_m2_jacobian_scale_resuscitation_ratio_mean": float(scale_repair_ratio_sum / max(1, scale_repair_count)),
        "uv_m2_jacobian_scale_resuscitation_ratio_min": (
            float(scale_repair_ratio_min) if np.isfinite(scale_repair_ratio_min) else 1.0
        ),
        "uv_m2_jacobian_scale_resuscitation_ratio_max": float(scale_repair_ratio_max),
        # Backward-compatible key kept for existing report/preview tooling.
        "uv_m2_jacobian_negative_det_drop_ratio": invalid_drop_ratio,
        "uv_m2_jacobian_invalid_drop_ratio": invalid_drop_ratio,
    }


def _weighted_edge_laplacian_from_face_alpha(
    *,
    faces: np.ndarray,
    n_vertices: int,
    face_alpha: np.ndarray,
    face_mask: Optional[np.ndarray],
) -> csr_matrix:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.size == 0 or n_vertices <= 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float64)
    if face_mask is not None:
        fm = np.asarray(face_mask, dtype=np.bool_)
        if fm.shape[0] == tri.shape[0]:
            tri = tri[fm]
            face_alpha = np.asarray(face_alpha, dtype=np.float64)[fm]
    if tri.size == 0:
        return csr_matrix((n_vertices, n_vertices), dtype=np.float64)

    face_alpha = np.asarray(face_alpha, dtype=np.float64)
    face_alpha = np.maximum(face_alpha, 1e-12)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).astype(np.int64, copy=False)
    edge_w = np.repeat(face_alpha, 3).astype(np.float64, copy=False)
    edges_sorted = np.sort(edges, axis=1)
    uniq, inv = np.unique(edges_sorted, axis=0, return_inverse=True)
    uniq_w = np.bincount(inv, weights=edge_w, minlength=uniq.shape[0]).astype(np.float64, copy=False)
    i = uniq[:, 0].astype(np.int64, copy=False)
    j = uniq[:, 1].astype(np.int64, copy=False)
    w = np.maximum(uniq_w, 0.0)

    rows = np.concatenate([i, j, i, j]).astype(np.int64, copy=False)
    cols = np.concatenate([i, j, j, i]).astype(np.int64, copy=False)
    data = np.concatenate([w, w, -w, -w]).astype(np.float64, copy=False)
    return coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices), dtype=np.float64).tocsr()


def _vertex_curvature_proxy(mesh: trimesh.Trimesh) -> np.ndarray:
    n_vertices = int(len(mesh.vertices))
    if n_vertices <= 0:
        return np.zeros((0,), dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((n_vertices,), dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != n_vertices:
        return np.zeros((n_vertices,), dtype=np.float64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    if edges.size == 0:
        return np.zeros((n_vertices,), dtype=np.float64)
    n0 = normals[edges[:, 0]]
    n1 = normals[edges[:, 1]]
    dot = np.sum(n0 * n1, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    edge_k = 1.0 - dot
    acc = np.zeros((n_vertices,), dtype=np.float64)
    deg = np.zeros((n_vertices,), dtype=np.float64)
    np.add.at(acc, edges[:, 0], edge_k)
    np.add.at(acc, edges[:, 1], edge_k)
    np.add.at(deg, edges[:, 0], 1.0)
    np.add.at(deg, edges[:, 1], 1.0)
    curv = acc / np.maximum(deg, 1.0)
    pos = curv[curv > 0]
    if pos.size > 0:
        s = float(np.percentile(pos, 95))
        if s > 1e-12:
            curv = np.clip(curv / s, 0.0, 1.0)
    return curv


def _build_gradient_constraint_system(
    *,
    mesh: trimesh.Trimesh,
    face_jac: np.ndarray,
    face_weights: np.ndarray,
    face_valid_mask: np.ndarray,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = int(len(verts))

    valid_faces = np.where(face_valid_mask)[0]
    if valid_faces.size == 0:
        A = csr_matrix((0, n_vertices), dtype=np.float64)
        return A, np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    tri = faces[valid_faces]  # [Fv,3]
    p = verts[tri]  # [Fv,3,3]
    jac = np.asarray(face_jac[valid_faces], dtype=np.float64)  # [Fv,2,3]
    w_face = np.sqrt(np.maximum(np.asarray(face_weights[valid_faces], dtype=np.float64), 1e-12))  # [Fv]

    a_idx = np.asarray([0, 0, 1], dtype=np.int64)
    b_idx = np.asarray([1, 2, 2], dtype=np.int64)

    e = p[:, b_idx, :] - p[:, a_idx, :]  # [Fv,3,3]
    delta = np.einsum("fij,fkj->fki", jac, e, optimize=True)  # [Fv,3,2]

    w_row = np.repeat(w_face, 3).astype(np.float64, copy=False)  # [3*Fv]
    rhs_u_np = (w_row * delta[:, :, 0].reshape(-1)).astype(np.float64, copy=False)
    rhs_v_np = (w_row * delta[:, :, 1].reshape(-1)).astype(np.float64, copy=False)
    delta_u = delta[:, :, 0].reshape(-1)
    delta_v = delta[:, :, 1].reshape(-1)
    max_delta_u = float(np.max(np.abs(delta_u))) if delta_u.size > 0 else 0.0
    max_delta_v = float(np.max(np.abs(delta_v))) if delta_v.size > 0 else 0.0
    max_rhs_u = float(np.max(np.abs(rhs_u_np))) if rhs_u_np.size > 0 else 0.0
    max_rhs_v = float(np.max(np.abs(rhs_v_np))) if rhs_v_np.size > 0 else 0.0
    max_target_delta = float(max(max_delta_u, max_delta_v))
    print(f"Max target gradient/delta: {max_target_delta}")
    print(
        f"[method2][diag] max |delta_u|={max_delta_u:.9e}, max |delta_v|={max_delta_v:.9e}, "
        f"max |rhs_u|={max_rhs_u:.9e}, max |rhs_v|={max_rhs_v:.9e}"
    )
    row_face_ids_np = np.repeat(valid_faces, 3).astype(np.int64, copy=False)

    n_rows = int(row_face_ids_np.size)
    rows_np = np.repeat(np.arange(n_rows, dtype=np.int64), 2)
    va = tri[:, a_idx].reshape(-1).astype(np.int64, copy=False)
    vb = tri[:, b_idx].reshape(-1).astype(np.int64, copy=False)
    cols_np = np.empty((2 * n_rows,), dtype=np.int64)
    cols_np[0::2] = va
    cols_np[1::2] = vb
    data_np = np.empty((2 * n_rows,), dtype=np.float64)
    data_np[0::2] = -w_row
    data_np[1::2] = w_row

    A = coo_matrix(
        (data_np, (rows_np, cols_np)),
        shape=(n_rows, n_vertices),
        dtype=np.float64,
    ).tocsr()
    return (
        A,
        rhs_u_np,
        rhs_v_np,
        row_face_ids_np,
    )


def _boundary_vertex_ids(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq[counts == 1]
    if boundary_edges.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(boundary_edges.reshape(-1)).astype(np.int64)


def _count_cut_edges_from_face_labels_fast(faces: np.ndarray, face_labels: np.ndarray) -> int:
    tri = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    if tri.size == 0 or labels.size == 0:
        return 0
    if tri.ndim != 2 or tri.shape[1] != 3 or labels.shape[0] != tri.shape[0]:
        return 0

    edge_dir = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).astype(np.int64, copy=False)
    edge_face = np.repeat(np.arange(tri.shape[0], dtype=np.int64), 3)
    edge_key = np.sort(edge_dir, axis=1)

    _, inv, counts = np.unique(edge_key, axis=0, return_inverse=True, return_counts=True)
    interior_mask = counts[inv] == 2
    if not np.any(interior_mask):
        return 0

    edge_ids = np.arange(edge_key.shape[0], dtype=np.int64)[interior_mask]
    edge_grp = inv[edge_ids]
    order = np.argsort(edge_grp, kind="mergesort")
    pair = edge_ids[order]
    if pair.size < 2:
        return 0
    if pair.size % 2 != 0:
        pair = pair[: pair.size - 1]
        if pair.size < 2:
            return 0

    i0 = pair[0::2]
    i1 = pair[1::2]
    f0 = edge_face[i0]
    f1 = edge_face[i1]
    l0 = labels[f0]
    l1 = labels[f1]
    cut = (f0 != f1) & (l0 >= 0) & (l1 >= 0) & (l0 != l1)
    return int(np.count_nonzero(cut))


def _component_minimal_anchor_ids(
    *,
    labels: np.ndarray,
    vertices: np.ndarray,
    points_per_component: int,
    anchor_scores: Optional[np.ndarray] = None,
    adaptive: bool = False,
    min_points_per_component: int = 3,
    max_points_per_component: int = 5,
    target_vertices_per_anchor: int = 8000,
    score_power: float = 1.0,
) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0,), dtype=np.int64)

    base_k = max(1, int(points_per_component))
    min_k = max(1, int(min_points_per_component))
    max_k = max(min_k, int(max_points_per_component))
    target_size = max(1, int(target_vertices_per_anchor))
    score_pow = max(0.0, float(score_power))
    score_all = None
    if anchor_scores is not None:
        score_all = np.asarray(anchor_scores, dtype=np.float64)
        if score_all.shape[0] != labels.shape[0]:
            score_all = None

    anchor_ids: list[int] = []
    for comp in np.unique(labels).tolist():
        comp_vid = np.where(labels == comp)[0]
        if comp_vid.size == 0:
            continue
        if adaptive:
            k = int(np.ceil(float(comp_vid.size) / float(target_size)))
            k = max(min_k, min(max_k, k))
        else:
            k = base_k
        if comp_vid.size <= k:
            anchor_ids.extend(comp_vid.tolist())
            continue

        comp_pos = vertices[comp_vid]
        if score_all is not None:
            comp_scores = np.maximum(score_all[comp_vid], 1e-8)
            selected_local = [int(np.argmax(comp_scores))]
        else:
            comp_scores = None
            selected_local = [0]
        while len(selected_local) < k:
            selected_pos = comp_pos[np.asarray(selected_local, dtype=np.int64)]
            diff = comp_pos[:, None, :] - selected_pos[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            min_dist2 = np.min(dist2, axis=1)
            min_dist2[np.asarray(selected_local, dtype=np.int64)] = -1.0
            if comp_scores is not None:
                score_term = np.power(comp_scores, score_pow)
                next_local = int(np.argmax(min_dist2 * score_term))
            else:
                next_local = int(np.argmax(min_dist2))
            if next_local in selected_local:
                break
            selected_local.append(next_local)
        anchor_ids.extend(comp_vid[np.asarray(selected_local, dtype=np.int64)].tolist())

    if len(anchor_ids) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(set(anchor_ids)), dtype=np.int64)


def _solve_poisson_uv(
    *,
    mesh: trimesh.Trimesh,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    A: csr_matrix,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    solve_cfg: Dict[str, Any],
    m2_cfg: Dict[str, Any],
    face_active_mask: Optional[np.ndarray],
    resolved_device: str,
    anchor_mode: str,
    anchor_points_per_component: int,
    anchor_vertex_target_uv: Optional[np.ndarray],
    anchor_vertex_confidence: Optional[np.ndarray],
    face_smooth_alpha: Optional[np.ndarray],
    precomputed_anchor_uv_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    n_vertices = int(len(mesh.vertices))
    if A.shape[0] == 0 or n_vertices == 0:
        raise RuntimeError("method2_gradient_poisson has no valid gradient constraints")

    M = (A.T @ A).tocsr()
    lambda_smooth = float(solve_cfg.get("lambda_smooth", 1e-3))
    lap_mode = str(m2_cfg.get("laplacian_mode", "cotan")).strip().lower()
    if lap_mode not in {"uniform", "cotan"}:
        lap_mode = "cotan"
    adaptive_smooth_enabled = bool(m2_cfg.get("adaptive_smooth_enabled", True))
    if lambda_smooth > 0.0:
        if adaptive_smooth_enabled and face_smooth_alpha is not None:
            L = _weighted_edge_laplacian_from_face_alpha(
                faces=np.asarray(mesh.faces, dtype=np.int64),
                n_vertices=n_vertices,
                face_alpha=np.asarray(face_smooth_alpha, dtype=np.float64),
                face_mask=face_active_mask,
            )
            solve_laplacian_used = "adaptive_face_weighted"
        else:
            L = mesh_laplacian(
                np.asarray(mesh.faces, dtype=np.int64),
                n_vertices,
                face_mask=face_active_mask,
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                mode=lap_mode,
            )
            solve_laplacian_used = lap_mode
        M = (M + lambda_smooth * L).tocsr()
    else:
        solve_laplacian_used = "off"
    ridge_eps = max(float(solve_cfg.get("ridge_eps", 1e-8)), 1e-8)
    M = (M + ridge_eps * diags(np.ones((n_vertices,), dtype=np.float64))).tocsr()

    rhs_u_full = A.T @ rhs_u
    rhs_v_full = A.T @ rhs_v

    anchor_weight = max(float(solve_cfg.get("anchor_weight", 1e2)), 0.0)
    if precomputed_anchor_uv_all is not None:
        pre_uv = np.asarray(precomputed_anchor_uv_all, dtype=np.float64)
        if pre_uv.shape == (n_vertices, 2):
            anchor_uv_all = pre_uv
        else:
            anchor_uv_all = nearest_vertex_uv(mesh, high_mesh, high_uv).astype(np.float64)
    else:
        anchor_uv_all = nearest_vertex_uv(mesh, high_mesh, high_uv).astype(np.float64)
    anchor_uv_target = anchor_uv_all.copy()
    if anchor_vertex_target_uv is not None:
        tgt = np.asarray(anchor_vertex_target_uv, dtype=np.float64)
        if tgt.shape == anchor_uv_target.shape:
            valid_tgt = np.isfinite(tgt).all(axis=1)
            anchor_uv_target[valid_tgt] = tgt[valid_tgt]

    conf_all = np.ones((n_vertices,), dtype=np.float64)
    if anchor_vertex_confidence is not None:
        c = np.asarray(anchor_vertex_confidence, dtype=np.float64)
        if c.shape[0] == n_vertices:
            conf_all = np.clip(c, 0.0, 1.0)

    adaptive_anchor_enabled = bool(m2_cfg.get("adaptive_anchor_enabled", True))
    anchor_conf_floor = float(m2_cfg.get("anchor_confidence_floor", 0.2))
    anchor_conf_power = float(m2_cfg.get("anchor_confidence_power", 1.0))
    anchor_boundary_boost = float(m2_cfg.get("anchor_boundary_boost", 0.5))
    anchor_curvature_boost = float(m2_cfg.get("anchor_curvature_boost", 0.5))
    anchor_target_vertices_per_anchor = int(m2_cfg.get("anchor_target_vertices_per_anchor", 8000))
    anchor_min_points = int(m2_cfg.get("anchor_min_points_per_component", 3))
    anchor_max_points = int(m2_cfg.get("anchor_max_points_per_component", 5))

    score_all = np.power(np.maximum(conf_all, 1e-8), max(anchor_conf_power, 0.0))
    labels, _ = connected_components_labels(np.asarray(mesh.faces, dtype=np.int64), n_vertices)
    vertices_np = np.asarray(mesh.vertices, dtype=np.float64)

    anchor_mode_norm = str(anchor_mode or "component_minimal").strip().lower()
    if anchor_mode_norm not in {"component_minimal", "boundary", "none"}:
        anchor_mode_norm = "component_minimal"
    points_per_comp = max(1, int(anchor_points_per_component))

    boundary_ids = _boundary_vertex_ids(mesh)
    if anchor_mode_norm == "boundary":
        anchor_ids_np = boundary_ids
    elif anchor_mode_norm == "none":
        anchor_ids_np = _component_minimal_anchor_ids(
            labels=labels,
            vertices=vertices_np,
            points_per_component=1,
            anchor_scores=score_all,
        )
    else:
        if adaptive_anchor_enabled:
            curv = _vertex_curvature_proxy(mesh)
            if curv.shape[0] == n_vertices:
                score_all = score_all * (1.0 + anchor_curvature_boost * np.clip(curv, 0.0, 1.0))
            if boundary_ids.size > 0 and anchor_boundary_boost > 0.0:
                bmask = np.zeros((n_vertices,), dtype=np.float64)
                bmask[boundary_ids] = 1.0
                score_all = score_all * (1.0 + anchor_boundary_boost * bmask)
        anchor_ids_np = _component_minimal_anchor_ids(
            labels=labels,
            vertices=vertices_np,
            points_per_component=points_per_comp,
            anchor_scores=score_all,
            adaptive=adaptive_anchor_enabled,
            min_points_per_component=anchor_min_points,
            max_points_per_component=anchor_max_points,
            target_vertices_per_anchor=anchor_target_vertices_per_anchor,
            score_power=anchor_conf_power,
        )

    if anchor_ids_np.size == 0:
        anchor_ids_np = _component_minimal_anchor_ids(
            labels=labels,
            vertices=vertices_np,
            points_per_component=1,
        )

    # Use strong soft anchors only; avoid hard Dirichlet elimination to reduce stress concentration.
    soft_anchor_ids = anchor_ids_np
    if anchor_weight > 0.0 and soft_anchor_ids.size > 0:
        M = M.tolil(copy=False)
        conf_sel = np.clip(conf_all[soft_anchor_ids], 0.0, 1.0)
        conf_sel = anchor_conf_floor + (1.0 - anchor_conf_floor) * conf_sel
        conf_sel = np.maximum(conf_sel, 1e-6)
        for local_idx, vid in enumerate(soft_anchor_ids.tolist()):
            wv = anchor_weight * float(conf_sel[local_idx])
            M[vid, vid] = float(M[vid, vid]) + wv
            rhs_u_full[vid] += wv * float(anchor_uv_target[vid, 0])
            rhs_v_full[vid] += wv * float(anchor_uv_target[vid, 1])
        M = M.tocsr()

    backend_requested = str(solve_cfg.get("backend", "auto")).strip().lower()
    if backend_requested not in {"auto", "cuda_pcg", "cpu_scipy"}:
        backend_requested = "auto"
    use_cuda_first = backend_requested == "cuda_pcg" or (backend_requested == "auto" and resolved_device.startswith("cuda"))
    pcg_max_iter = int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000)))
    pcg_tol = float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6)))
    pcg_check_every = int(solve_cfg.get("pcg_check_every", 25))
    pcg_preconditioner = str(solve_cfg.get("pcg_preconditioner", "jacobi"))

    backend_used = "cpu_scipy"
    fallback_reason = None
    if use_cuda_first:
        try:
            M_cuda, M_diag_cuda = build_cuda_sparse_system(M=M, device=resolved_device)
            sol_u, meta_u = solve_linear_cuda_pcg(
                M_cuda=M_cuda,
                M_diag_cuda=M_diag_cuda,
                rhs=np.asarray(rhs_u_full, dtype=np.float64),
                pcg_max_iter=pcg_max_iter,
                pcg_tol=pcg_tol,
                pcg_check_every=pcg_check_every,
                pcg_preconditioner=pcg_preconditioner,
                channel_name="u",
            )
            sol_v, meta_v = solve_linear_cuda_pcg(
                M_cuda=M_cuda,
                M_diag_cuda=M_diag_cuda,
                rhs=np.asarray(rhs_v_full, dtype=np.float64),
                pcg_max_iter=pcg_max_iter,
                pcg_tol=pcg_tol,
                pcg_check_every=pcg_check_every,
                pcg_preconditioner=pcg_preconditioner,
                channel_name="v",
            )
            backend_used = "cuda_pcg"
        except Exception as exc:
            fallback_reason = f"cuda_pcg_failed: {exc}"
            sol_u, meta_u = solve_linear_robust(
                M=M,
                rhs=np.asarray(rhs_u_full, dtype=np.float64),
                cg_max_iter=pcg_max_iter,
                cg_tol=pcg_tol,
                channel_name="u",
            )
            sol_v, meta_v = solve_linear_robust(
                M=M,
                rhs=np.asarray(rhs_v_full, dtype=np.float64),
                cg_max_iter=pcg_max_iter,
                cg_tol=pcg_tol,
                channel_name="v",
            )
            backend_used = "cpu_scipy"
    else:
        sol_u, meta_u = solve_linear_robust(
            M=M,
            rhs=np.asarray(rhs_u_full, dtype=np.float64),
            cg_max_iter=pcg_max_iter,
            cg_tol=pcg_tol,
            channel_name="u",
        )
        sol_v, meta_v = solve_linear_robust(
            M=M,
            rhs=np.asarray(rhs_v_full, dtype=np.float64),
            cg_max_iter=pcg_max_iter,
            cg_tol=pcg_tol,
            channel_name="v",
        )
        backend_used = "cpu_scipy"

    mapped_uv = np.stack([sol_u, sol_v], axis=1).astype(np.float32)
    solve_meta: Dict[str, Any] = {
        "uv_solver_backend_requested": backend_requested,
        "uv_solver_backend_used": backend_used,
        "uv_solver_linear_backend_requested": backend_requested,
        "uv_solver_linear_backend_used": meta_u["backend"],
        "uv_solver_backend_u": meta_u["backend"],
        "uv_solver_backend_v": meta_v["backend"],
        "uv_solver_iters_u": int(meta_u.get("iters", -1)),
        "uv_solver_iters_v": int(meta_v.get("iters", -1)),
        "uv_solver_residual_u": float(meta_u.get("residual", float("nan"))),
        "uv_solver_residual_v": float(meta_v.get("residual", float("nan"))),
        "uv_solver_converged_u": bool(meta_u.get("converged", False)),
        "uv_solver_converged_v": bool(meta_v.get("converged", False)),
        "uv_solver_cg_info_u": int(meta_u.get("cg_info", 0)),
        "uv_solver_cg_info_v": int(meta_v.get("cg_info", 0)),
        "uv_solver_refine_cg_info_u": int(meta_u.get("cg2_info", 0)),
        "uv_solver_refine_cg_info_v": int(meta_v.get("cg2_info", 0)),
    }
    if fallback_reason:
        solve_meta["uv_solver_fallback_reason"] = fallback_reason

    cond_mode = str(m2_cfg.get("system_cond_estimate", "diag_ratio")).strip().lower()
    cond_proxy = None
    if cond_mode == "diag_ratio":
        diag_abs = np.abs(M.diagonal())
        pos = diag_abs[diag_abs > 1e-12]
        if pos.size > 0:
            cond_proxy = float(np.max(pos) / max(np.min(pos), 1e-12))
    elif cond_mode == "eigsh":
        try:
            from scipy.sparse.linalg import eigsh

            lm = float(eigsh(M, k=1, which="LM", return_eigenvectors=False)[0])
            sm = float(eigsh(M, k=1, which="SM", return_eigenvectors=False)[0])
            if np.isfinite(lm) and np.isfinite(sm) and sm > 1e-12:
                cond_proxy = float(abs(lm) / abs(sm))
        except Exception:
            cond_proxy = None
    if cond_proxy is not None and np.isfinite(cond_proxy):
        solve_meta["uv_m2_system_cond_proxy"] = float(cond_proxy)
    solve_meta["uv_m2_laplacian_mode"] = solve_laplacian_used
    solve_meta["uv_m2_anchor_mode_used"] = anchor_mode_norm
    solve_meta["uv_m2_anchor_points_per_component"] = int(points_per_comp)
    solve_meta["uv_m2_anchor_boundary_candidates"] = int(boundary_ids.size)
    solve_meta["uv_m2_adaptive_anchor_enabled"] = bool(adaptive_anchor_enabled)
    solve_meta["uv_m2_anchor_min_points_per_component"] = int(anchor_min_points)
    solve_meta["uv_m2_anchor_max_points_per_component"] = int(anchor_max_points)
    solve_meta["uv_m2_anchor_target_vertices_per_anchor"] = int(anchor_target_vertices_per_anchor)
    solve_meta["uv_m2_adaptive_smooth_enabled"] = bool(adaptive_smooth_enabled)
    solve_meta["uv_m2_anchor_weight"] = float(anchor_weight)
    solve_meta["uv_m2_hard_anchor_enabled"] = False
    solve_meta["uv_m2_hard_anchor_conf_min"] = float(m2_cfg.get("hard_anchor_conf_min", 0.85))
    solve_meta["uv_m2_hard_anchor_count"] = 0
    solve_meta["uv_m2_soft_anchor_count"] = int(soft_anchor_ids.size)

    anchor_uv = anchor_uv_target[anchor_ids_np] if anchor_ids_np.size > 0 else np.zeros((0, 2), dtype=np.float64)
    return mapped_uv, solve_meta, anchor_ids_np, anchor_uv.astype(np.float32)


def _extract_submesh_for_faces(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    faces_all = np.asarray(mesh.faces, dtype=np.int64)
    verts_all = np.asarray(mesh.vertices, dtype=np.float64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    if face_ids.size == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3), dtype=np.float64), faces=np.zeros((0, 3), dtype=np.int64), process=False), np.zeros((0,), dtype=np.int64)
    tri = faces_all[face_ids]
    global_vid, inverse = np.unique(tri.reshape(-1), return_inverse=True)
    tri_local = inverse.reshape(-1, 3).astype(np.int64, copy=False)
    submesh = trimesh.Trimesh(vertices=verts_all[global_vid], faces=tri_local, process=False)
    return submesh, global_vid.astype(np.int64, copy=False)


def _solve_poisson_uv_by_island(
    *,
    mesh: trimesh.Trimesh,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    A: csr_matrix,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    row_face_ids: np.ndarray,
    face_island_labels: np.ndarray,
    solve_cfg: Dict[str, Any],
    m2_cfg: Dict[str, Any],
    resolved_device: str,
    anchor_mode: str,
    anchor_points_per_component: int,
    anchor_vertex_target_uv: Optional[np.ndarray],
    anchor_vertex_confidence: Optional[np.ndarray],
    face_smooth_alpha: Optional[np.ndarray],
    precomputed_anchor_uv_all: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    n_vertices = int(len(mesh.vertices))
    if n_vertices == 0:
        raise RuntimeError("method2_gradient_poisson has empty mesh for island solve")
    if A.shape[0] == 0 or row_face_ids.size == 0:
        raise RuntimeError("method2_gradient_poisson has no constraints for island solve")

    face_island = np.asarray(face_island_labels, dtype=np.int64)
    if face_island.shape[0] != int(len(mesh.faces)):
        raise RuntimeError("method2 island solve got mismatched face island labels")

    row_face_ids = np.asarray(row_face_ids, dtype=np.int64)
    row_island = face_island[np.clip(row_face_ids, 0, len(face_island) - 1)]
    islands = np.unique(row_island[row_island >= 0])
    if islands.size == 0:
        raise RuntimeError("method2 island solve has no known-island constraints")

    if precomputed_anchor_uv_all is not None:
        pre_uv = np.asarray(precomputed_anchor_uv_all, dtype=np.float32)
        if pre_uv.shape == (n_vertices, 2):
            mapped_uv = pre_uv.copy()
        else:
            mapped_uv = nearest_vertex_uv(mesh, high_mesh, high_uv).astype(np.float32)
    else:
        mapped_uv = nearest_vertex_uv(mesh, high_mesh, high_uv).astype(np.float32)
    assigned = np.zeros((n_vertices,), dtype=np.bool_)
    overlap_vertices = 0
    solved_islands = 0

    backend_used_list: list[str] = []
    linear_backend_list: list[str] = []
    backend_u_list: list[str] = []
    backend_v_list: list[str] = []
    iters_u: list[int] = []
    iters_v: list[int] = []
    residual_u: list[float] = []
    residual_v: list[float] = []
    converged_u: list[bool] = []
    converged_v: list[bool] = []
    cg_info_u: list[int] = []
    cg_info_v: list[int] = []
    cg2_info_u: list[int] = []
    cg2_info_v: list[int] = []
    cond_proxy: list[float] = []
    lap_used: list[str] = []

    anchor_map: Dict[int, np.ndarray] = {}

    for island in islands.tolist():
        row_sel = row_island == int(island)
        if not np.any(row_sel):
            continue
        island_face_ids = np.unique(row_face_ids[row_sel]).astype(np.int64, copy=False)
        if island_face_ids.size == 0:
            continue

        submesh, global_vid = _extract_submesh_for_faces(mesh, island_face_ids)
        if global_vid.size == 0:
            continue

        A_rows = A[row_sel].tocoo()
        col_map = np.full((n_vertices,), -1, dtype=np.int64)
        col_map[global_vid] = np.arange(global_vid.size, dtype=np.int64)
        local_cols = col_map[A_rows.col]
        keep = local_cols >= 0
        if not np.any(keep):
            continue
        A_local = coo_matrix(
            (
                A_rows.data[keep].astype(np.float64, copy=False),
                (
                    A_rows.row[keep].astype(np.int64, copy=False),
                    local_cols[keep].astype(np.int64, copy=False),
                ),
            ),
            shape=(A_rows.shape[0], int(global_vid.size)),
            dtype=np.float64,
        ).tocsr()
        rhs_u_local = np.asarray(rhs_u[row_sel], dtype=np.float64)
        rhs_v_local = np.asarray(rhs_v[row_sel], dtype=np.float64)

        local_anchor_uv = None
        if anchor_vertex_target_uv is not None:
            local_anchor_uv = np.asarray(anchor_vertex_target_uv[global_vid], dtype=np.float64)
        local_anchor_conf = None
        if anchor_vertex_confidence is not None:
            local_anchor_conf = np.asarray(anchor_vertex_confidence[global_vid], dtype=np.float64)
        local_face_alpha = None
        if face_smooth_alpha is not None:
            local_face_alpha = np.asarray(face_smooth_alpha[island_face_ids], dtype=np.float64)

        mapped_local, meta_local, anchor_ids_local, anchor_uv_local = _solve_poisson_uv(
            mesh=submesh,
            high_mesh=high_mesh,
            high_uv=high_uv,
            A=A_local,
            rhs_u=rhs_u_local,
            rhs_v=rhs_v_local,
            solve_cfg=solve_cfg,
            m2_cfg=m2_cfg,
            face_active_mask=None,
            resolved_device=resolved_device,
            anchor_mode=anchor_mode,
            anchor_points_per_component=anchor_points_per_component,
            anchor_vertex_target_uv=local_anchor_uv,
            anchor_vertex_confidence=local_anchor_conf,
            face_smooth_alpha=local_face_alpha,
            precomputed_anchor_uv_all=(
                np.asarray(precomputed_anchor_uv_all[global_vid], dtype=np.float64)
                if (precomputed_anchor_uv_all is not None and precomputed_anchor_uv_all.shape[0] == n_vertices)
                else None
            ),
        )

        pre_assigned = assigned[global_vid]
        overlap_vertices += int(np.count_nonzero(pre_assigned))
        write_mask = ~pre_assigned
        if np.any(write_mask):
            mapped_uv[global_vid[write_mask]] = mapped_local[write_mask]
        assigned[global_vid] = True

        if anchor_ids_local.size > 0:
            g_anchor = global_vid[anchor_ids_local]
            for k, vid in enumerate(g_anchor.tolist()):
                if vid not in anchor_map:
                    anchor_map[int(vid)] = np.asarray(anchor_uv_local[k], dtype=np.float32)

        solved_islands += 1
        backend_used_list.append(str(meta_local.get("uv_solver_backend_used", "unknown")))
        linear_backend_list.append(str(meta_local.get("uv_solver_linear_backend_used", "unknown")))
        backend_u_list.append(str(meta_local.get("uv_solver_backend_u", "unknown")))
        backend_v_list.append(str(meta_local.get("uv_solver_backend_v", "unknown")))
        iters_u.append(int(meta_local.get("uv_solver_iters_u", -1)))
        iters_v.append(int(meta_local.get("uv_solver_iters_v", -1)))
        ru = float(meta_local.get("uv_solver_residual_u", float("nan")))
        rv = float(meta_local.get("uv_solver_residual_v", float("nan")))
        if np.isfinite(ru):
            residual_u.append(ru)
        if np.isfinite(rv):
            residual_v.append(rv)
        converged_u.append(bool(meta_local.get("uv_solver_converged_u", False)))
        converged_v.append(bool(meta_local.get("uv_solver_converged_v", False)))
        cg_info_u.append(int(meta_local.get("uv_solver_cg_info_u", 0)))
        cg_info_v.append(int(meta_local.get("uv_solver_cg_info_v", 0)))
        cg2_info_u.append(int(meta_local.get("uv_solver_refine_cg_info_u", 0)))
        cg2_info_v.append(int(meta_local.get("uv_solver_refine_cg_info_v", 0)))
        cp = meta_local.get("uv_m2_system_cond_proxy", None)
        if cp is not None:
            cp_f = float(cp)
            if np.isfinite(cp_f):
                cond_proxy.append(cp_f)
        lap_used.append(str(meta_local.get("uv_m2_laplacian_mode", "unknown")))

    if solved_islands == 0:
        raise RuntimeError("method2 island solve failed: no island sub-problem solved")

    def _mode_or_mixed(vals: list[str]) -> str:
        if len(vals) == 0:
            return "unknown"
        uniq = sorted(set(vals))
        if len(uniq) == 1:
            return uniq[0]
        return "mixed"

    anchor_ids = np.asarray(sorted(anchor_map.keys()), dtype=np.int64) if anchor_map else np.zeros((0,), dtype=np.int64)
    anchor_uv = (
        np.asarray([anchor_map[int(v)] for v in anchor_ids.tolist()], dtype=np.float32)
        if anchor_ids.size > 0
        else np.zeros((0, 2), dtype=np.float32)
    )

    solve_meta: Dict[str, Any] = {
        "uv_solver_backend_requested": str(solve_cfg.get("backend", "auto")),
        "uv_solver_backend_used": _mode_or_mixed(backend_used_list),
        "uv_solver_linear_backend_requested": str(solve_cfg.get("backend", "auto")),
        "uv_solver_linear_backend_used": _mode_or_mixed(linear_backend_list),
        "uv_solver_backend_u": _mode_or_mixed(backend_u_list),
        "uv_solver_backend_v": _mode_or_mixed(backend_v_list),
        "uv_solver_iters_u": int(max(iters_u)) if len(iters_u) > 0 else -1,
        "uv_solver_iters_v": int(max(iters_v)) if len(iters_v) > 0 else -1,
        "uv_solver_residual_u": float(np.mean(residual_u)) if len(residual_u) > 0 else float("nan"),
        "uv_solver_residual_v": float(np.mean(residual_v)) if len(residual_v) > 0 else float("nan"),
        "uv_solver_converged_u": bool(all(converged_u)) if len(converged_u) > 0 else False,
        "uv_solver_converged_v": bool(all(converged_v)) if len(converged_v) > 0 else False,
        "uv_solver_cg_info_u": int(max(cg_info_u)) if len(cg_info_u) > 0 else 0,
        "uv_solver_cg_info_v": int(max(cg_info_v)) if len(cg_info_v) > 0 else 0,
        "uv_solver_refine_cg_info_u": int(max(cg2_info_u)) if len(cg2_info_u) > 0 else 0,
        "uv_solver_refine_cg_info_v": int(max(cg2_info_v)) if len(cg2_info_v) > 0 else 0,
        "uv_m2_island_solver_enabled": True,
        "uv_m2_island_count": int(islands.size),
        "uv_m2_island_solved_count": int(solved_islands),
        "uv_m2_island_unsolved_count": int(islands.size - solved_islands),
        "uv_m2_island_vertex_overlap_count": int(overlap_vertices),
        "uv_m2_island_unassigned_vertices": int(np.count_nonzero(~assigned)),
        "uv_m2_laplacian_mode": _mode_or_mixed(lap_used),
        "uv_m2_anchor_mode_used": str(anchor_mode or "component_minimal"),
        "uv_m2_anchor_points_per_component": int(max(1, int(anchor_points_per_component))),
        "uv_m2_anchor_weight": float(solve_cfg.get("anchor_weight", 1e2)),
        "uv_m2_hard_anchor_enabled": False,
        "uv_m2_hard_anchor_conf_min": float(m2_cfg.get("hard_anchor_conf_min", 0.85)),
        "uv_m2_hard_anchor_count": 0,
        "uv_m2_soft_anchor_count": int(anchor_ids.size),
    }
    if len(cond_proxy) > 0:
        solve_meta["uv_m2_system_cond_proxy"] = float(np.mean(cond_proxy))
    return mapped_uv.astype(np.float32, copy=False), solve_meta, anchor_ids, anchor_uv


def _package_method2_result(
    *,
    mapped_uv: np.ndarray,
    stats: Dict[str, Any],
    quality_mesh: trimesh.Trimesh,
    return_internal: bool,
):
    payload: Dict[str, Any] = {
        "local_vertex_split_applied": False,
        "quality_mesh": quality_mesh,
    }
    if return_internal:
        return mapped_uv, stats, payload, None
    return mapped_uv, stats, payload


def run_method2_gradient_poisson(
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
    return_internal: bool = False,
):
    corr_cfg = cfg["correspondence"]
    seam_cfg = cfg.get("seam", {})
    solve_cfg = cfg["solve"]
    tex_weight_cfg = cfg.get("texture_weight", {})
    m2_cfg = cfg.get("method2", {})
    seam_strategy_requested = str(seam_cfg.get("strategy", "legacy")).strip().lower()
    if seam_strategy_requested not in {"legacy", "halfedge_island"}:
        seam_strategy_requested = "legacy"
    emit_validation_sidecar_data = bool(seam_cfg.get("emit_validation_sidecar_data", False))
    seam_validation_strict = bool(seam_cfg.get("validation_strict", True))
    seam_require_closed = bool(seam_cfg.get("validation_require_closed_loops", True))
    seam_require_partition_pure = bool(seam_cfg.get("validation_require_pure_components", True))

    resolved = resolve_device(device)
    if resolved != "cuda":
        mapped_uv, stats = barycentric_mapper(high_mesh, low_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "barycentric_fallback"
        stats["uv_project_error"] = "CUDA unavailable for method2 gradient-poisson pipeline"
        stats["uv_m2_jacobian_valid_ratio"] = 0.0
        stats["uv_m2_poisson_residual_u"] = None
        stats["uv_m2_poisson_residual_v"] = None
        stats["uv_m2_boundary_anchor_count"] = 0
        stats["uv_m2_outlier_reject_ratio"] = 0.0
        return _package_method2_result(
            mapped_uv=mapped_uv,
            stats=stats,
            quality_mesh=low_mesh,
            return_internal=return_internal,
        )

    work_low_mesh = low_mesh
    pre_seam_meta: Dict[str, Any] = {}
    if seam_strategy_requested == "halfedge_island":
        ensure_halfedge_external_dependencies()
        work_low_mesh, sanitize_meta = sanitize_mesh_for_halfedge(low_mesh=low_mesh, seam_cfg=seam_cfg)
        pre_seam_meta.update(sanitize_meta)

    sample = sample_low_mesh(work_low_mesh, cfg["sample"])
    sample_points = sample["points"]
    sample_face_ids = sample["face_ids"]
    sample_bary = sample["bary"]
    sample_normals = sample["normals"]
    sample_area_weights = sample["area_weights"]

    high_ctx = build_high_cuda_context(high_mesh=high_mesh, high_uv=high_uv, device=resolved)
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

    n_faces = int(len(work_low_mesh.faces))
    solve_mesh = work_low_mesh
    face_active_mask = np.ones((n_faces,), dtype=np.bool_)
    solver_valid_mask = valid_mask.copy()
    seam_strategy_effective = seam_strategy_requested
    seam_strategy_used = "method2_ideal_noguard"
    split_vertices_out: Optional[np.ndarray] = None
    split_faces_out: Optional[np.ndarray] = None

    t_island_start = time.perf_counter()
    high_face_island = None
    low_face_island = np.full((n_faces,), -1, dtype=np.int64)
    low_face_conflict = np.zeros((n_faces,), dtype=np.bool_)
    low_face_confidence = np.zeros((n_faces,), dtype=np.float32)
    low_face_expected_high = np.full((n_faces,), -1, dtype=np.int64)
    seam_edges_for_export = np.zeros((0, 2), dtype=np.int64)
    seam_meta: Dict[str, Any] = {}
    seam_meta.update(pre_seam_meta)
    seam_meta["uv_island_conflict_policy"] = "drop_conflict_and_unknown_samples"
    seam_meta["uv_halfedge_split_requested"] = bool(seam_strategy_requested == "halfedge_island")
    seam_meta["uv_halfedge_split_topology_applied"] = False
    seam_meta["uv_halfedge_split_fallback_to_legacy"] = False
    seam_meta["uv_seam_validation_strict"] = bool(seam_validation_strict)
    seam_meta["uv_seam_validation_require_closed_loops"] = bool(seam_require_closed)
    seam_meta["uv_seam_validation_require_pure_components"] = bool(seam_require_partition_pure)
    island_cache_enabled = bool(m2_cfg.get("perf_fast_island_cache", True))
    seam_meta["uv_m2_perf_high_island_cache_enabled"] = island_cache_enabled
    seam_meta["uv_m2_perf_high_island_cache_hit"] = False
    try:
        high_face_island, high_meta, cache_hit = _cached_high_face_uv_islands(
            high_mesh=high_mesh,
            high_uv=high_uv,
            position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
            uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
            use_cache=island_cache_enabled,
        )
        seam_meta["uv_m2_perf_high_island_cache_hit"] = bool(cache_hit)
        seam_meta["uv_high_island_count"] = int(high_meta.get("high_island_count", 0))
        seam_meta["uv_high_seam_edges"] = int(high_meta.get("high_seam_edges", 0))
        seam_meta["uv_high_boundary_edges"] = int(high_meta.get("high_boundary_edges", 0))
        seam_meta["uv_high_nonmanifold_edges"] = int(high_meta.get("high_nonmanifold_edges", 0))
    except Exception as exc:
        seam_meta["uv_high_island_error"] = str(exc)

    if high_face_island is None and seam_strategy_requested == "halfedge_island":
        if seam_validation_strict:
            raise RuntimeError("halfedge_island failed: unable to compute high-face UV islands")
        mapped_uv, stats = barycentric_mapper(high_mesh, work_low_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "method2_fallback_halfedge_island_error"
        stats["uv_project_error"] = "halfedge_island failed: unable to compute high-face UV islands"
        stats["uv_seam_strategy_requested"] = seam_strategy_requested
        stats["uv_seam_strategy_effective"] = "halfedge_island_failed"
        stats["uv_seam_strategy_used"] = "method2_halfedge_failed"
        stats.update(seam_meta)
        return _package_method2_result(
            mapped_uv=mapped_uv,
            stats=stats,
            quality_mesh=work_low_mesh,
            return_internal=return_internal,
        )

    if high_face_island is not None and seam_strategy_requested == "halfedge_island":
        seam_strategy_effective = "halfedge_island"
        semantic = transfer_face_semantics_by_projection(
            high_ctx=high_ctx,
            high_face_island=high_face_island,
            low_mesh=work_low_mesh,
            seam_cfg=seam_cfg,
            corr_cfg=corr_cfg,
        )
        low_face_island = np.asarray(semantic["low_face_island"], dtype=np.int64, copy=False)
        low_face_conflict = np.asarray(semantic["low_face_conflict"], dtype=np.bool_, copy=False)
        low_face_confidence = np.asarray(semantic["low_face_confidence"], dtype=np.float32, copy=False)
        low_face_expected_high = low_face_island.astype(np.int64, copy=False)
        seam_meta.update(dict(semantic.get("meta", {})))
        seam_meta["uv_halfedge_backend"] = "openmesh"

        seam_result = extract_seam_edges_openmesh(
            low_mesh=work_low_mesh,
            face_labels=low_face_island,
            include_boundary_as_seam=bool(seam_cfg.get("include_boundary_as_seam", False)),
        )
        seam_edges = np.asarray(seam_result.seam_edges, dtype=np.int64, copy=False)
        seam_edges_for_export = seam_edges.astype(np.int64, copy=False)
        seam_meta.update(dict(seam_result.meta))
        partition_meta = validate_face_partition_by_seams(
            low_mesh=work_low_mesh,
            face_labels=low_face_island,
            seam_edges=seam_edges,
        )
        seam_meta.update(partition_meta)
        seam_topology_ok = bool(seam_result.meta.get("uv_seam_topology_valid", False)) or (not seam_require_closed)
        seam_partition_ok = bool(partition_meta.get("uv_seam_partition_is_valid", False)) or (
            not seam_require_partition_pure
        )
        seam_validation_ok = bool(seam_topology_ok and seam_partition_ok)
        seam_meta["uv_seam_validation_ok"] = bool(seam_validation_ok)
        if not seam_validation_ok:
            seam_meta["uv_seam_validation_error"] = (
                f"topology_ok={seam_topology_ok}, partition_ok={seam_partition_ok}, "
                f"components_open={int(seam_result.meta.get('uv_seam_components_open', 0))}, "
                f"mixed_components={int(partition_meta.get('uv_seam_partition_mixed_components', 0))}"
            )
            if seam_validation_strict:
                raise RuntimeError(f"halfedge seam validation failed: {seam_meta['uv_seam_validation_error']}")

        split_vertices, split_faces, split_meta = split_vertices_along_cut_edges(
            vertices=np.asarray(work_low_mesh.vertices, dtype=np.float32),
            faces=np.asarray(work_low_mesh.faces, dtype=np.int64),
            cut_edges=seam_edges,
        )
        seam_meta["uv_low_cut_edges"] = int(split_meta.get("cut_edges", seam_edges.shape[0]))
        seam_meta["uv_low_split_vertices"] = int(split_meta.get("split_vertices_added", 0))
        seam_meta["uv_low_split_faces"] = int(split_meta.get("split_faces", n_faces))
        seam_meta["uv_island_conflict_faces"] = int(np.count_nonzero(low_face_conflict))
        seam_meta["uv_island_unknown_faces"] = int(np.count_nonzero(low_face_island < 0))
        seam_meta["uv_island_conflict_faces_excluded"] = 0
        if int(split_vertices.shape[0]) > int(len(work_low_mesh.vertices)):
            solve_mesh = trimesh.Trimesh(vertices=split_vertices, faces=split_faces, process=False)
            split_vertices_out = split_vertices
            split_faces_out = split_faces
            seam_meta["uv_halfedge_split_topology_applied"] = True
        seam_strategy_used = "method2_halfedge_semantic_openmesh"
    elif high_face_island is not None:
        low_face_island, low_face_conflict, low_face_confidence = major_face_island_labels(
            sample_face_ids=sample_face_ids,
            target_face_ids=target_face_ids,
            valid_mask=valid_mask,
            high_face_island=high_face_island,
            n_low_faces=n_faces,
            min_samples=int(seam_cfg.get("min_valid_samples_per_face", 2)),
        )
        low_face_expected_high = low_face_island.astype(np.int64, copy=False)
        seam_meta["uv_island_conflict_faces"] = int(np.count_nonzero(low_face_conflict))
        seam_meta["uv_island_unknown_faces"] = int(np.count_nonzero(low_face_island < 0))
        seam_meta["uv_island_conflict_faces_excluded"] = 0
        fast_cut_diag = bool(m2_cfg.get("perf_fast_cut_edge_count", True))
        seam_meta["uv_m2_perf_fast_cut_edge_count"] = bool(fast_cut_diag)
        if fast_cut_diag:
            seam_meta["uv_low_cut_edges"] = int(
                _count_cut_edges_from_face_labels_fast(
                    np.asarray(work_low_mesh.faces, dtype=np.int64),
                    low_face_island,
                )
            )
        else:
            try:
                cut_edges = detect_cut_edges_from_face_labels(
                    np.asarray(work_low_mesh.faces, dtype=np.int64),
                    low_face_island,
                )
                seam_meta["uv_low_cut_edges"] = int(len(cut_edges))
            except Exception:
                seam_meta["uv_low_cut_edges"] = 0
        seam_meta["uv_low_split_vertices"] = 0
        seam_meta["uv_low_split_faces"] = int(n_faces)
        seam_strategy_used = "method2_ideal_noguard"
    else:
        seam_meta.setdefault("uv_island_conflict_faces", 0)
        seam_meta.setdefault("uv_island_unknown_faces", int(n_faces))
        seam_meta.setdefault("uv_island_conflict_faces_excluded", 0)
        seam_meta.setdefault("uv_low_cut_edges", 0)
        seam_meta.setdefault("uv_low_split_vertices", 0)
        seam_meta.setdefault("uv_low_split_faces", int(n_faces))
        low_face_expected_high = low_face_island.astype(np.int64, copy=False)

    if seam_strategy_effective == "halfedge_island":
        cross_seam_face_mask = np.zeros((n_faces,), dtype=np.bool_)
        seam_meta["uv_cross_seam_faces"] = 0
        seam_meta["uv_cross_seam_face_ratio"] = 0.0
        seam_meta["uv_seam_uv_span_threshold"] = float(seam_cfg.get("uv_span_threshold", 0.35))
        seam_meta["uv_cross_seam_faces_excluded"] = 0
    else:
        cross_seam_face_mask = detect_cross_seam_faces(
            sample_face_ids=sample_face_ids,
            target_uv=target_uv,
            valid_mask=valid_mask,
            n_faces=n_faces,
            uv_span_threshold=float(seam_cfg.get("uv_span_threshold", 0.35)),
            min_valid_samples_per_face=int(seam_cfg.get("min_valid_samples_per_face", 2)),
        )
        seam_meta["uv_cross_seam_faces"] = int(np.count_nonzero(cross_seam_face_mask))
        seam_meta["uv_cross_seam_face_ratio"] = float(np.mean(cross_seam_face_mask)) if n_faces > 0 else 0.0
        seam_meta["uv_seam_uv_span_threshold"] = float(seam_cfg.get("uv_span_threshold", 0.35))
        excluded_cross = 0
        if bool(seam_cfg.get("exclude_cross_seam_faces", True)):
            before_mask = solver_valid_mask.copy()
            solver_valid_mask &= ~cross_seam_face_mask[sample_face_ids]
            excluded_cross = int(np.count_nonzero(before_mask & (~solver_valid_mask)))
        seam_meta["uv_cross_seam_faces_excluded"] = int(excluded_cross)

    guard_requested = bool(m2_cfg.get("use_island_guard", False))
    guard_allow_unknown = bool(seam_cfg.get("uv_island_guard_allow_unknown", False))
    seam_meta["uv_island_guard_requested"] = guard_requested
    seam_meta["uv_island_guard_allow_unknown"] = guard_allow_unknown
    seam_meta["uv_island_guard_mode_requested"] = str(seam_cfg.get("uv_island_guard_mode", "soft"))
    seam_meta["uv_island_guard_confidence_min"] = float(seam_cfg.get("uv_island_guard_confidence_min", 0.55))
    seam_meta["uv_island_guard_fallback_policy"] = str(
        seam_cfg.get("uv_island_guard_fallback", "nearest_same_island_then_udf")
    )
    seam_meta["uv_island_guard_enabled"] = False
    seam_meta["uv_island_guard_mode_used"] = "off"
    seam_meta["uv_island_guard_constrained_points"] = 0
    seam_meta["uv_island_guard_constrained_ratio"] = 0.0
    seam_meta["uv_island_guard_reject_count"] = 0
    seam_meta["uv_island_guard_reject_ratio"] = 0.0
    seam_meta["uv_island_guard_fallback_success_ratio"] = 0.0
    seam_meta["uv_island_guard_invalid_after_guard_ratio"] = 0.0

    if guard_requested and high_face_island is not None:
        sample_face_island = low_face_island[sample_face_ids]
        sample_face_conflict = low_face_conflict[sample_face_ids]
        guard_constrained = sample_face_island >= 0

        keep = ~sample_face_conflict
        if not guard_allow_unknown:
            keep &= guard_constrained

        before_mask = solver_valid_mask.copy()
        solver_valid_mask &= keep
        rejected = int(np.count_nonzero(before_mask & (~solver_valid_mask)))
        rejected_conflict = int(np.count_nonzero(before_mask & sample_face_conflict))

        constrained_points = int(np.count_nonzero(guard_constrained))
        seam_meta["uv_island_guard_enabled"] = True
        seam_meta["uv_island_guard_mode_used"] = "face_label_filter"
        seam_meta["uv_island_guard_constrained_points"] = constrained_points
        seam_meta["uv_island_guard_constrained_ratio"] = float(constrained_points / max(1, len(sample_face_ids)))
        seam_meta["uv_island_guard_reject_count"] = rejected
        seam_meta["uv_island_guard_reject_ratio"] = float(rejected / max(1, constrained_points))
        seam_meta["uv_island_conflict_faces_excluded"] = int(rejected_conflict)

    high_face_jac, high_face_jac_valid = _compute_high_face_jacobians(high_mesh, high_uv)
    J_high = high_face_jac[high_face_jac_valid] if np.any(high_face_jac_valid) else high_face_jac
    print(np.linalg.norm(J_high))
    _print_local_frame_length_diagnostics(high_mesh, tag="high_mesh")
    target_face_ok = (target_face_ids >= 0) & (target_face_ids < len(high_face_jac_valid))
    solver_valid_mask &= target_face_ok
    if np.any(target_face_ok):
        solver_valid_mask &= high_face_jac_valid[np.clip(target_face_ids, 0, len(high_face_jac_valid) - 1)]

    seam_meta["uv_island_strict_filter_enabled"] = False
    seam_meta["uv_island_strict_reject_count"] = 0
    seam_meta["uv_island_strict_reject_ratio"] = 0.0
    seam_meta["uv_island_strict_unknown_faces_excluded"] = 0
    seam_meta["uv_island_strict_conflict_faces_excluded"] = 0
    if high_face_island is not None:
        # Enforce strict single-island consistency: each low face can only consume samples
        # from its routed/mapped high-island label.
        sample_face_major_island = low_face_expected_high[sample_face_ids]
        sample_face_conflict = low_face_conflict[sample_face_ids]
        sample_hit_island = np.full((len(target_face_ids),), -1, dtype=np.int64)
        ok_hit = (target_face_ids >= 0) & (target_face_ids < len(high_face_island))
        if np.any(ok_hit):
            sample_hit_island[ok_hit] = high_face_island[target_face_ids[ok_hit]]
        strict_keep = (
            (sample_face_major_island >= 0)
            & (~sample_face_conflict)
            & (sample_hit_island == sample_face_major_island)
        )
        before_mask = solver_valid_mask.copy()
        solver_valid_mask &= strict_keep
        rejected = int(np.count_nonzero(before_mask & (~solver_valid_mask)))
        seam_meta["uv_island_strict_filter_enabled"] = True
        seam_meta["uv_island_strict_reject_count"] = int(rejected)
        seam_meta["uv_island_strict_reject_ratio"] = float(rejected / max(1, int(np.count_nonzero(before_mask))))
        seam_meta["uv_island_strict_unknown_faces_excluded"] = int(np.count_nonzero(low_face_expected_high < 0))
        seam_meta["uv_island_strict_conflict_faces_excluded"] = int(np.count_nonzero(low_face_conflict))
        face_active_mask &= (low_face_island >= 0) & (~low_face_conflict)
    island_stage_seconds = float(time.perf_counter() - t_island_start)
    seam_meta["uv_m2_time_island_seconds"] = island_stage_seconds

    emit_face_sample_counts = bool(m2_cfg.get("emit_face_sample_counts", False))
    face_total_samples = None
    face_accepted_samples = None
    if emit_face_sample_counts and n_faces > 0 and sample_face_ids.size > 0:
        sf_all = np.asarray(sample_face_ids, dtype=np.int64, copy=False)
        valid_for_total = valid_mask & (sf_all >= 0) & (sf_all < n_faces)
        valid_for_accept = solver_valid_mask & (sf_all >= 0) & (sf_all < n_faces)
        face_total_samples = np.bincount(sf_all[valid_for_total], minlength=n_faces).astype(np.int32, copy=False)
        face_accepted_samples = np.bincount(sf_all[valid_for_accept], minlength=n_faces).astype(np.int32, copy=False)

    if not np.any(solver_valid_mask):
        mapped_uv, stats = nearest_mapper(high_mesh, work_low_mesh, high_uv)
        stats["uv_mode_used"] = "nearest_fallback_no_valid_samples"
        stats["uv_project_error"] = "method2_gradient_poisson has no valid correspondence samples"
        stats["uv_m2_jacobian_valid_ratio"] = 0.0
        stats["uv_m2_poisson_residual_u"] = None
        stats["uv_m2_poisson_residual_v"] = None
        stats["uv_m2_boundary_anchor_count"] = 0
        stats["uv_m2_outlier_reject_ratio"] = 0.0
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta)
        if emit_face_sample_counts and face_accepted_samples is not None:
            stats["uv_m2_face_accepted_samples"] = face_accepted_samples.tolist()
            if face_total_samples is not None:
                stats["uv_m2_face_total_samples"] = face_total_samples.tolist()
        return _package_method2_result(
            mapped_uv=mapped_uv,
            stats=stats,
            quality_mesh=work_low_mesh,
            return_internal=return_internal,
        )

    valid_ids = np.where(solver_valid_mask)[0]
    sf = sample_face_ids[valid_ids].astype(np.int64, copy=False)
    tf = target_face_ids[valid_ids].astype(np.int64, copy=False)
    sbary = sample_bary[valid_ids].astype(np.float64, copy=False)
    starget_uv = target_uv[valid_ids].astype(np.float64, copy=False)
    sarea_w = sample_area_weights[valid_ids].astype(np.float64, copy=False)
    sfallback = fallback_used_mask[valid_ids]
    if tf.size > 0:
        J_high = high_face_jac[tf]
        print(np.linalg.norm(J_high))

    corr_w = np.ones((len(valid_ids),), dtype=np.float64)
    corr_w[sfallback] = float(corr_cfg.get("fallback_weight", 0.7))
    if bool(tex_weight_cfg.get("enabled", True)) and image is not None and starget_uv.size > 0:
        tex_w = texture_gradient_weights(
            image=image,
            uv=starget_uv,
            gamma=float(tex_weight_cfg.get("grad_weight_gamma", 1.0)),
            max_weight=float(tex_weight_cfg.get("max_weight", 5.0)),
        )
    else:
        tex_w = np.ones((len(valid_ids),), dtype=np.float64)
    sample_weights = np.maximum(1e-12, sarea_w * corr_w * tex_w)

    m2_min_samples = int(m2_cfg.get("min_samples_per_face", seam_cfg.get("min_valid_samples_per_face", 2)))
    m2_outlier_sigma = float(m2_cfg.get("outlier_sigma", 4.0))
    m2_outlier_quantile = float(m2_cfg.get("outlier_quantile", 0.95))
    m2_face_weight_floor = float(m2_cfg.get("face_weight_floor", 1e-6))
    m2_irls_iters = int(m2_cfg.get("irls_iters", 2))
    m2_huber_delta = float(m2_cfg.get("huber_delta", 3.0))
    fast_agg_vectorized = bool(m2_cfg.get("perf_fast_agg_vectorized", True))
    small_group_fast_threshold = int(m2_cfg.get("perf_fast_small_group_threshold", 6))
    small_group_skip_irls = bool(m2_cfg.get("perf_fast_small_group_skip_irls", True))
    small_group_skip_outlier = bool(m2_cfg.get("perf_fast_small_group_skip_outlier", True))

    t_agg_start = time.perf_counter()
    face_jac, face_weights, face_valid, face_cov_trace, outlier_meta = _aggregate_face_target_jacobians(
        n_low_faces=n_faces,
        sample_face_ids=sf,
        target_face_ids=tf,
        sample_weights=sample_weights,
        high_face_jac=high_face_jac,
        min_samples_per_face=m2_min_samples,
        outlier_sigma=m2_outlier_sigma,
        outlier_quantile=m2_outlier_quantile,
        face_weight_floor=m2_face_weight_floor,
        irls_iters=m2_irls_iters,
        huber_delta=m2_huber_delta,
        fast_mode=fast_agg_vectorized,
        small_group_fast_threshold=small_group_fast_threshold,
        small_group_skip_irls=small_group_skip_irls,
        small_group_skip_outlier=small_group_skip_outlier,
    )
    face_valid &= face_active_mask
    relaxed_constraints = False
    if np.count_nonzero(face_valid) == 0 and sf.size > 0:
        face_jac, face_weights, face_valid, face_cov_trace, outlier_meta_relaxed = _aggregate_face_target_jacobians(
            n_low_faces=n_faces,
            sample_face_ids=sf,
            target_face_ids=tf,
            sample_weights=sample_weights,
            high_face_jac=high_face_jac,
            min_samples_per_face=1,
            outlier_sigma=0.0,
            outlier_quantile=1.0,
            face_weight_floor=m2_face_weight_floor,
            irls_iters=1,
            huber_delta=max(m2_huber_delta, 1.0),
            fast_mode=fast_agg_vectorized,
            small_group_fast_threshold=small_group_fast_threshold,
            small_group_skip_irls=small_group_skip_irls,
            small_group_skip_outlier=small_group_skip_outlier,
        )
        face_valid &= face_active_mask
        relaxed_constraints = True
        outlier_meta.update(
            {
                "uv_m2_constraint_relaxation_used": True,
                "uv_m2_constraint_relaxation_min_samples": 1,
                "uv_m2_constraint_relaxation_outlier_sigma": 0.0,
                "uv_m2_constraint_relaxation_outlier_quantile": 1.0,
                "uv_m2_constraint_relaxation_faces_valid": int(np.count_nonzero(face_valid)),
                "uv_m2_constraint_relaxation_outlier_reject_ratio": outlier_meta_relaxed.get(
                    "uv_m2_outlier_reject_ratio", 0.0
                ),
            }
        )
    else:
        outlier_meta["uv_m2_constraint_relaxation_used"] = False
    aggregate_stage_seconds = float(time.perf_counter() - t_agg_start)
    outlier_meta["uv_m2_time_aggregate_seconds"] = aggregate_stage_seconds
    outlier_meta["uv_m2_perf_fast_agg_vectorized"] = bool(fast_agg_vectorized)
    outlier_meta["uv_m2_perf_fast_small_group_threshold"] = int(small_group_fast_threshold)
    outlier_meta["uv_m2_perf_fast_small_group_skip_irls"] = bool(small_group_skip_irls)
    outlier_meta["uv_m2_perf_fast_small_group_skip_outlier"] = bool(small_group_skip_outlier)

    adaptive_smooth_enabled = bool(m2_cfg.get("adaptive_smooth_enabled", True))
    adaptive_smooth_beta = max(0.0, float(m2_cfg.get("adaptive_smooth_beta", 2.0)))
    adaptive_smooth_min_alpha = max(1e-3, float(m2_cfg.get("adaptive_smooth_min_alpha", 0.25)))
    adaptive_smooth_max_alpha = max(adaptive_smooth_min_alpha, float(m2_cfg.get("adaptive_smooth_max_alpha", 1.5)))
    if adaptive_smooth_enabled:
        face_alpha = np.ones((n_faces,), dtype=np.float64)
        cov_valid = face_cov_trace[face_valid]
        cov_scale = float(np.percentile(cov_valid, 50)) if cov_valid.size > 0 else 0.0
        cov_scale = max(cov_scale, 1e-12)
        cov_norm = face_cov_trace / cov_scale
        face_alpha = 1.0 / (1.0 + adaptive_smooth_beta * np.maximum(cov_norm, 0.0))
        face_alpha = np.clip(face_alpha, adaptive_smooth_min_alpha, adaptive_smooth_max_alpha)
    else:
        face_alpha = np.ones((n_faces,), dtype=np.float64)

    n_vertices_solve = int(len(solve_mesh.vertices))
    anchor_vertex_conf = np.zeros((n_vertices_solve,), dtype=np.float64)
    anchor_vertex_uv_acc = np.zeros((n_vertices_solve, 2), dtype=np.float64)
    anchor_vertex_uv_w = np.zeros((n_vertices_solve,), dtype=np.float64)
    if len(sf) > 0 and n_vertices_solve > 0:
        tri_vid = np.asarray(solve_mesh.faces, dtype=np.int64)[sf]
        fallback_w = float(corr_cfg.get("fallback_weight", 0.7))
        sample_corr_conf = np.where(sfallback, fallback_w, 1.0).astype(np.float64, copy=False)
        anchor_sample_w = np.maximum(1e-12, sarea_w * sample_corr_conf)
        for c in range(3):
            vids = tri_vid[:, c]
            wb = anchor_sample_w * sbary[:, c]
            np.add.at(anchor_vertex_conf, vids, wb)
            np.add.at(anchor_vertex_uv_w, vids, wb)
            np.add.at(anchor_vertex_uv_acc[:, 0], vids, wb * starget_uv[:, 0])
            np.add.at(anchor_vertex_uv_acc[:, 1], vids, wb * starget_uv[:, 1])

    anchor_vertex_target_uv = np.full((n_vertices_solve, 2), np.nan, dtype=np.float64)
    valid_anchor_tgt = anchor_vertex_uv_w > 1e-12
    if np.any(valid_anchor_tgt):
        anchor_vertex_target_uv[valid_anchor_tgt] = (
            anchor_vertex_uv_acc[valid_anchor_tgt] / anchor_vertex_uv_w[valid_anchor_tgt, None]
        )
    conf_pos = anchor_vertex_conf[anchor_vertex_conf > 0]
    conf_scale = float(np.percentile(conf_pos, 95)) if conf_pos.size > 0 else 1.0
    conf_scale = max(conf_scale, 1e-12)
    anchor_vertex_conf_norm = np.clip(anchor_vertex_conf / conf_scale, 0.0, 1.0)

    A, rhs_u, rhs_v, row_face_ids = _build_gradient_constraint_system(
        mesh=solve_mesh,
        face_jac=face_jac,
        face_weights=face_weights,
        face_valid_mask=face_valid,
    )

    if np.count_nonzero(face_valid) == 0 or A.shape[0] == 0:
        mapped_uv, stats = barycentric_mapper(high_mesh, solve_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "method2_fallback_no_gradient_constraints"
        stats["uv_project_error"] = (
            "method2_gradient_poisson has no valid gradient constraints"
            + (" (relaxed)" if relaxed_constraints else "")
        )
        stats["uv_m2_jacobian_valid_ratio"] = 0.0
        stats["uv_m2_poisson_residual_u"] = None
        stats["uv_m2_poisson_residual_v"] = None
        stats["uv_m2_boundary_anchor_count"] = 0
        stats["uv_m2_anchor_count_total"] = 0
        stats["uv_m2_outlier_reject_ratio"] = float(outlier_meta.get("uv_m2_outlier_reject_ratio", 0.0))
        stats["uv_m2_constraint_relaxation_used"] = bool(outlier_meta.get("uv_m2_constraint_relaxation_used", False))
        stats["uv_solve_num_constraints"] = int(A.shape[0])
        stats["uv_solve_num_samples"] = int(len(sf))
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta)
        if emit_face_sample_counts and face_accepted_samples is not None:
            stats["uv_m2_face_accepted_samples"] = face_accepted_samples.tolist()
            if face_total_samples is not None:
                stats["uv_m2_face_total_samples"] = face_total_samples.tolist()
        return _package_method2_result(
            mapped_uv=mapped_uv,
            stats=stats,
            quality_mesh=solve_mesh,
            return_internal=return_internal,
        )

    precomputed_anchor_uv_all = nearest_vertex_uv(solve_mesh, high_mesh, high_uv).astype(np.float64)
    t_solve_start = time.perf_counter()
    try:
        solve_per_island = bool(m2_cfg.get("solve_per_island", True))
        solve_face_island = np.full((len(solve_mesh.faces),), -1, dtype=np.int64)
        if low_face_island.shape[0] == solve_face_island.shape[0]:
            solve_face_island = low_face_island.astype(np.int64, copy=False)

        if solve_per_island and np.any(solve_face_island >= 0):
            mapped_uv, solve_meta, anchor_ids, anchor_uv = _solve_poisson_uv_by_island(
                mesh=solve_mesh,
                high_mesh=high_mesh,
                high_uv=high_uv,
                A=A,
                rhs_u=rhs_u,
                rhs_v=rhs_v,
                row_face_ids=row_face_ids,
                face_island_labels=solve_face_island,
                solve_cfg=solve_cfg,
                m2_cfg=m2_cfg,
                resolved_device=resolved,
                anchor_mode=str(m2_cfg.get("anchor_mode", "component_minimal")),
                anchor_points_per_component=int(m2_cfg.get("anchor_points_per_component", 4)),
                anchor_vertex_target_uv=anchor_vertex_target_uv,
                anchor_vertex_confidence=anchor_vertex_conf_norm,
                face_smooth_alpha=face_alpha,
                precomputed_anchor_uv_all=precomputed_anchor_uv_all,
            )
            solve_meta["uv_m2_solve_per_island_enabled"] = True
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
                face_active_mask=face_active_mask,
                resolved_device=resolved,
                anchor_mode=str(m2_cfg.get("anchor_mode", "component_minimal")),
                anchor_points_per_component=int(m2_cfg.get("anchor_points_per_component", 4)),
                anchor_vertex_target_uv=anchor_vertex_target_uv,
                anchor_vertex_confidence=anchor_vertex_conf_norm,
                face_smooth_alpha=face_alpha,
                precomputed_anchor_uv_all=precomputed_anchor_uv_all,
            )
            solve_meta["uv_m2_solve_per_island_enabled"] = False
    except Exception as exc:
        mapped_uv, stats = barycentric_mapper(high_mesh, solve_mesh, high_uv, resolved, cfg)
        stats["uv_mode_used"] = "method2_fallback_solver_error"
        stats["uv_project_error"] = f"method2 solver failed: {exc}"
        stats["uv_m2_jacobian_valid_ratio"] = float(np.count_nonzero(face_valid) / max(1, n_faces))
        stats["uv_m2_poisson_residual_u"] = None
        stats["uv_m2_poisson_residual_v"] = None
        stats["uv_m2_boundary_anchor_count"] = 0
        stats["uv_m2_anchor_count_total"] = 0
        stats["uv_solve_num_constraints"] = int(A.shape[0])
        stats["uv_solve_num_samples"] = int(len(sf))
        stats["uv_seam_strategy_used"] = seam_strategy_used
        stats.update(seam_meta)
        stats.update(outlier_meta)
        if emit_face_sample_counts and face_accepted_samples is not None:
            stats["uv_m2_face_accepted_samples"] = face_accepted_samples.tolist()
            if face_total_samples is not None:
                stats["uv_m2_face_total_samples"] = face_total_samples.tolist()
        return _package_method2_result(
            mapped_uv=mapped_uv,
            stats=stats,
            quality_mesh=solve_mesh,
            return_internal=return_internal,
        )
    solve_stage_seconds = float(time.perf_counter() - t_solve_start)
    solve_meta["uv_m2_time_solve_seconds"] = solve_stage_seconds

    pred_uv_pre = interpolate_sample_uv(
        np.asarray(solve_mesh.faces, dtype=np.int64),
        sf,
        sbary,
        mapped_uv,
    )
    residual_pre = starget_uv - pred_uv_pre
    pre_shift_mean = np.mean(residual_pre, axis=0) if residual_pre.size > 0 else np.zeros((2,), dtype=np.float64)
    pre_shift_median = np.median(residual_pre, axis=0) if residual_pre.size > 0 else np.zeros((2,), dtype=np.float64)
    post_align_enabled = bool(m2_cfg.get("post_align_translation", True))
    post_align_min_samples = max(1, int(m2_cfg.get("post_align_min_samples", 64)))
    post_align_max_shift = max(0.0, float(m2_cfg.get("post_align_max_shift", 0.25)))
    post_align_applied = False
    post_align_shift = np.zeros((2,), dtype=np.float64)
    if post_align_enabled and residual_pre.shape[0] >= post_align_min_samples:
        shift = pre_shift_median.astype(np.float64, copy=True)
        shift_norm = float(np.linalg.norm(shift))
        if post_align_max_shift > 0.0 and shift_norm > post_align_max_shift:
            shift *= post_align_max_shift / max(shift_norm, 1e-12)
        mapped_uv = (mapped_uv.astype(np.float64) + shift[None, :]).astype(np.float32, copy=False)
        post_align_applied = True
        post_align_shift = shift

    pred_uv = interpolate_sample_uv(
        np.asarray(solve_mesh.faces, dtype=np.int64),
        sf,
        sbary,
        mapped_uv,
    )
    residual_post = starget_uv - pred_uv
    color_l1, color_l2 = texture_reprojection_error(image, starget_uv, pred_uv)

    jacobian_valid_ratio = float(np.count_nonzero(face_valid) / max(1, n_faces))
    boundary_ids = _boundary_vertex_ids(solve_mesh)

    method_stats: Dict[str, Any] = {
        "uv_correspondence_primary_ratio": float(np.mean(primary_mask)) if primary_mask.size > 0 else 0.0,
        "uv_correspondence_success_ratio": float(np.mean(valid_mask)) if valid_mask.size > 0 else 0.0,
        "uv_correspondence_invalid_ratio": float(np.mean(~valid_mask)) if valid_mask.size > 0 else 0.0,
        "uv_solve_num_samples": int(len(sf)),
        "uv_solve_num_constraints": int(A.shape[0]),
        "uv_solve_valid_sample_ratio": float(np.mean(solver_valid_mask)) if solver_valid_mask.size > 0 else 0.0,
        "uv_solve_num_faces": int(len(solve_mesh.faces)),
        "uv_solver_stage": "m2",
        "uv_color_reproj_l1": color_l1,
        "uv_color_reproj_l2": color_l2,
        "uv_seam_strategy_requested": seam_strategy_requested,
        "uv_seam_strategy_effective": seam_strategy_effective,
        "uv_seam_strategy_used": seam_strategy_used,
        "uv_iterative_enabled": False,
        "uv_iter_count": 1,
        "uv_iter_energy_data": [],
        "uv_iter_label_change_ratio": [],
        "uv_iter_conflict_face_ratio": [],
        "uv_iter_unknown_face_ratio": [],
        "uv_iter_valid_sample_ratio": [],
        "uv_iter_guard_mode_used": [],
        "uv_iter_early_stop_reason": "method2_single_pass",
        "uv_iter_best_index": 1,
        "uv_m2_jacobian_valid_ratio": jacobian_valid_ratio,
        "uv_m2_jacobian_valid_faces": int(np.count_nonzero(face_valid)),
        "uv_m2_jacobian_total_faces": int(n_faces),
        "uv_m2_face_cov_scale": float(np.percentile(face_cov_trace[face_valid], 50))
        if np.count_nonzero(face_valid) > 0
        else 0.0,
        "uv_m2_adaptive_smooth_alpha_mean": float(np.mean(face_alpha[face_valid]))
        if np.count_nonzero(face_valid) > 0
        else 1.0,
        "uv_m2_adaptive_smooth_alpha_min": float(np.min(face_alpha[face_valid]))
        if np.count_nonzero(face_valid) > 0
        else 1.0,
        "uv_m2_adaptive_smooth_alpha_max": float(np.max(face_alpha[face_valid]))
        if np.count_nonzero(face_valid) > 0
        else 1.0,
        "uv_m2_boundary_anchor_count": int(np.count_nonzero(np.isin(anchor_ids, boundary_ids))),
        "uv_m2_anchor_count_total": int(len(anchor_ids)),
        "uv_m2_use_island_guard": bool(guard_requested),
        "uv_m2_post_align_enabled": bool(post_align_enabled),
        "uv_m2_post_align_applied": bool(post_align_applied),
        "uv_m2_post_align_shift_u": float(post_align_shift[0]),
        "uv_m2_post_align_shift_v": float(post_align_shift[1]),
        "uv_m2_post_align_shift_norm": float(np.linalg.norm(post_align_shift)),
        "uv_m2_pre_align_shift_u_mean": float(pre_shift_mean[0]),
        "uv_m2_pre_align_shift_v_mean": float(pre_shift_mean[1]),
        "uv_m2_pre_align_shift_u_median": float(pre_shift_median[0]),
        "uv_m2_pre_align_shift_v_median": float(pre_shift_median[1]),
        "uv_m2_pre_align_residual_l2_mean": float(np.mean(np.linalg.norm(residual_pre, axis=1)))
        if residual_pre.size > 0
        else 0.0,
        "uv_m2_post_align_residual_l2_mean": float(np.mean(np.linalg.norm(residual_post, axis=1)))
        if residual_post.size > 0
        else 0.0,
        "uv_m2_poisson_residual_u": float(solve_meta.get("uv_solver_residual_u", float("nan"))),
        "uv_m2_poisson_residual_v": float(solve_meta.get("uv_solver_residual_v", float("nan"))),
        "uv_m2_system_cond_proxy": float(solve_meta.get("uv_m2_system_cond_proxy", float("nan")))
        if solve_meta.get("uv_m2_system_cond_proxy", None) is not None
        else None,
        **outlier_meta,
        **seam_meta,
        **solve_meta,
    }
    if emit_face_sample_counts and face_accepted_samples is not None:
        method_stats["uv_m2_face_accepted_samples"] = face_accepted_samples.tolist()
        if face_total_samples is not None:
            method_stats["uv_m2_face_total_samples"] = face_total_samples.tolist()
    if emit_validation_sidecar_data and seam_strategy_effective == "halfedge_island":
        method_stats["uv_low_face_semantic_labels"] = low_face_island.astype(np.int64, copy=False).tolist()
        method_stats["uv_low_seam_edges"] = seam_edges_for_export.astype(np.int64, copy=False).tolist()

    export_payload: Dict[str, Any] = {
        "local_vertex_split_applied": False,
        "quality_mesh": solve_mesh,
    }
    if split_vertices_out is not None and split_faces_out is not None:
        export_payload["halfedge_split_topology"] = True
        export_payload["split_vertices"] = split_vertices_out.astype(np.float32, copy=False)
        export_payload["split_faces"] = split_faces_out.astype(np.int64, copy=False)
    elif (
        seam_strategy_effective == "halfedge_island"
        and (
            int(len(solve_mesh.vertices)) != int(len(low_mesh.vertices))
            or int(len(solve_mesh.faces)) != int(len(low_mesh.faces))
        )
    ):
        # Sanitization can already alter topology (e.g., split non-manifold vertices).
        # Export the actual solve mesh to keep UV/vertex indexing consistent.
        export_payload["halfedge_split_topology"] = True
        export_payload["split_vertices"] = np.asarray(solve_mesh.vertices, dtype=np.float32)
        export_payload["split_faces"] = np.asarray(solve_mesh.faces, dtype=np.int64)

    if not return_internal:
        return mapped_uv, method_stats, export_payload

    face_geom_pinv, _ = _compute_face_geometry_pinv(solve_mesh)
    internal = Method2InternalState(
        solve_mesh=solve_mesh,
        mapped_uv_init=mapped_uv.copy(),
        face_target_jacobian=face_jac.astype(np.float32),
        face_target_valid_mask=face_valid.astype(np.bool_),
        face_target_weights=face_weights.astype(np.float32),
        face_geom_pinv=face_geom_pinv.astype(np.float32),
        solve_sample_face_ids=sf.astype(np.int64),
        solve_sample_bary=sbary.astype(np.float32),
        solve_target_uv=starget_uv.astype(np.float32),
        anchor_vertex_ids=anchor_ids.astype(np.int64),
        anchor_uv=anchor_uv.astype(np.float32),
        resolved_device=resolved,
        export_payload=export_payload,
        method_stats=method_stats,
    )
    return mapped_uv, method_stats, export_payload, internal


__all__ = [
    "Method2InternalState",
    "run_method2_gradient_poisson",
]
