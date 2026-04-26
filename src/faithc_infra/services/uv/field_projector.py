from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix, diags

from .linear_solver import build_cuda_sparse_system, nearest_vertex_uv, solve_linear_cuda_pcg, solve_linear_robust
from .method2_pipeline import Method2InternalState


def _compute_face_jacobians(face_geom_pinv: np.ndarray, mesh: trimesh.Trimesh, uv: np.ndarray) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri_uv = np.asarray(uv, dtype=np.float64)[faces]
    du1 = tri_uv[:, 1] - tri_uv[:, 0]
    du2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_grad = np.stack([du1, du2], axis=2)
    return np.einsum("fij,fjk->fik", uv_grad, np.asarray(face_geom_pinv, dtype=np.float64), optimize=True)


def _face_adjacency_edges(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    edge_vid = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 2:
        adj = np.zeros((0, 2), dtype=np.int64)
    if edge_vid.ndim != 2 or edge_vid.shape[1] != 2:
        edge_vid = np.zeros((0, 2), dtype=np.int64)
    if adj.shape[0] != edge_vid.shape[0]:
        n = min(adj.shape[0], edge_vid.shape[0])
        adj = adj[:n]
        edge_vid = edge_vid[:n]
    return adj, edge_vid


def _complete_anchor_vertex_uv(
    *,
    state: Method2InternalState,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    anchor_uv = np.asarray(state.anchor_vertex_target_uv, dtype=np.float64).copy()
    solve_uv = np.asarray(state.mapped_uv_init, dtype=np.float64)
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
        nearest_uv = nearest_vertex_uv(state.solve_mesh, high_mesh, high_uv).astype(np.float64)
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


def _fit_face_sample_jacobian_field(
    *,
    state: Method2InternalState,
    min_samples: int,
) -> Dict[str, Any]:
    faces = np.asarray(state.solve_mesh.faces, dtype=np.int64)
    face_geom_pinv = np.asarray(state.face_geom_pinv, dtype=np.float64)
    sample_face_ids = np.asarray(state.solve_sample_face_ids, dtype=np.int64).reshape(-1)
    sample_bary = np.asarray(state.solve_sample_bary, dtype=np.float64)
    target_uv = np.asarray(state.solve_target_uv, dtype=np.float64)
    fallback_mask = np.asarray(state.solve_sample_fallback_mask, dtype=np.bool_).reshape(-1)
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


def _covariance_norm_from_state(state: Method2InternalState) -> np.ndarray:
    cov_trace = np.asarray(state.face_target_cov_trace, dtype=np.float64)
    face_valid = np.asarray(state.face_target_valid_mask, dtype=np.bool_)
    cov_valid = cov_trace[face_valid & np.isfinite(cov_trace)]
    cov_scale = float(np.percentile(cov_valid, 50)) if cov_valid.size > 0 else 1.0
    cov_scale = max(cov_scale, 1e-12)
    cov_norm = cov_trace / cov_scale
    cov_norm[~np.isfinite(cov_norm)] = np.nan
    return cov_norm


def _projection_confidence_samplefit(
    *,
    state: Method2InternalState,
    samplefit: Dict[str, Any],
    strict_gate: bool,
) -> np.ndarray:
    valid = np.asarray(samplefit["face_valid"], dtype=np.bool_)
    sample_count = np.asarray(samplefit["face_sample_count"], dtype=np.float64)
    fallback_ratio = np.asarray(samplefit["face_fallback_ratio"], dtype=np.float64)
    face_residual = np.asarray(samplefit["face_residual"], dtype=np.float64)
    cov_norm = _covariance_norm_from_state(state)

    count_score = np.clip((sample_count - 2.0) / 2.0, 0.0, 1.0)
    fallback_score = 1.0 - np.clip(np.nan_to_num(fallback_ratio, nan=1.0), 0.0, 1.0)
    cov_score = 1.0 / (1.0 + np.maximum(np.nan_to_num(cov_norm, nan=1e6), 0.0))

    res_valid = valid & np.isfinite(face_residual) & (face_residual >= 0.0)
    res_scale = float(np.quantile(face_residual[res_valid], 0.75)) if np.any(res_valid) else 1e-4
    res_scale = max(res_scale, 1e-8)
    residual_score = 1.0 / (1.0 + np.maximum(np.nan_to_num(face_residual, nan=1e6), 0.0) / res_scale)

    conf = count_score * fallback_score * cov_score * residual_score
    if strict_gate:
        strict_mask = (sample_count >= 4.0) & (np.nan_to_num(fallback_ratio, nan=1.0) <= 0.25)
        conf = np.where(strict_mask, conf, 0.0)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~valid] = 0.0
    return conf


def _build_field_projection_matrix(
    *,
    mesh: trimesh.Trimesh,
    local_face_ids: np.ndarray,
    lambda_curl: float,
    ridge_eps: float,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, Any]]:
    local_faces = np.asarray(local_face_ids, dtype=np.int64).reshape(-1)
    n_local_faces = int(local_faces.size)
    if n_local_faces == 0:
        M0 = csr_matrix((0, 0), dtype=np.float64)
        return M0, np.zeros((0,), dtype=np.float64), {
            "local_face_count": 0,
            "interior_edge_count": 0,
            "edge_scale_sq": None,
            "matrix_nnz": 0,
        }

    local_index = np.full((len(mesh.faces),), -1, dtype=np.int64)
    local_index[local_faces] = np.arange(n_local_faces, dtype=np.int64)
    dim = 3 * n_local_faces

    diag_rows = np.arange(dim, dtype=np.int64)
    diag_cols = diag_rows.copy()
    diag_data = np.full((dim,), max(float(ridge_eps), 1e-8), dtype=np.float64)

    adj, edge_vid = _face_adjacency_edges(mesh)
    if adj.shape[0] == 0 or float(lambda_curl) <= 0.0:
        M = coo_matrix((diag_data, (diag_rows, diag_cols)), shape=(dim, dim), dtype=np.float64).tocsr()
        return M, local_index, {
            "local_face_count": int(n_local_faces),
            "interior_edge_count": 0,
            "edge_scale_sq": None,
            "matrix_nnz": int(M.nnz),
        }

    keep = (local_index[adj[:, 0]] >= 0) & (local_index[adj[:, 1]] >= 0)
    if not np.any(keep):
        M = coo_matrix((diag_data, (diag_rows, diag_cols)), shape=(dim, dim), dtype=np.float64).tocsr()
        return M, local_index, {
            "local_face_count": int(n_local_faces),
            "interior_edge_count": 0,
            "edge_scale_sq": None,
            "matrix_nnz": int(M.nnz),
        }

    fa = local_index[adj[keep, 0]]
    fb = local_index[adj[keep, 1]]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edge_vec = verts[edge_vid[keep, 1]] - verts[edge_vid[keep, 0]]
    edge_len_sq = np.sum(edge_vec * edge_vec, axis=1)
    edge_scale_sq = float(np.quantile(edge_len_sq[np.isfinite(edge_len_sq)], 0.50)) if np.any(np.isfinite(edge_len_sq)) else 1.0
    edge_scale_sq = max(edge_scale_sq, 1e-12)
    Q = (float(lambda_curl) / edge_scale_sq) * np.einsum("ei,ej->eij", edge_vec, edge_vec, optimize=True)

    pq = np.indices((3, 3)).reshape(2, -1)
    p = pq[0].astype(np.int64, copy=False)
    q = pq[1].astype(np.int64, copy=False)
    q_flat = Q.reshape(Q.shape[0], 9)

    row_aa = (3 * fa[:, None] + p[None, :]).reshape(-1)
    col_aa = (3 * fa[:, None] + q[None, :]).reshape(-1)
    row_bb = (3 * fb[:, None] + p[None, :]).reshape(-1)
    col_bb = (3 * fb[:, None] + q[None, :]).reshape(-1)
    row_ab = row_aa.copy()
    col_ab = col_bb.copy()
    row_ba = row_bb.copy()
    col_ba = col_aa.copy()

    data_aa = q_flat.reshape(-1)
    data_bb = q_flat.reshape(-1)
    data_ab = (-q_flat).reshape(-1)
    data_ba = (-q_flat).reshape(-1)

    rows = np.concatenate([diag_rows, row_aa, row_bb, row_ab, row_ba]).astype(np.int64, copy=False)
    cols = np.concatenate([diag_cols, col_aa, col_bb, col_ab, col_ba]).astype(np.int64, copy=False)
    data = np.concatenate([diag_data, data_aa, data_bb, data_ab, data_ba]).astype(np.float64, copy=False)
    M = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64).tocsr()
    return M, local_index, {
        "local_face_count": int(n_local_faces),
        "interior_edge_count": int(fa.shape[0]),
        "edge_scale_sq": float(edge_scale_sq),
        "matrix_nnz": int(M.nnz),
    }


def _solve_field_projection_rows(
    *,
    M: csr_matrix,
    rhs_rows: Sequence[np.ndarray],
    resolved_device: str,
    solve_cfg: Dict[str, Any],
    channel_prefix: str,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    backend_requested = str(solve_cfg.get("backend", "auto")).strip().lower()
    if backend_requested not in {"auto", "cuda_pcg", "cpu_scipy"}:
        backend_requested = "auto"
    use_cuda_first = backend_requested == "cuda_pcg" or (
        backend_requested == "auto" and str(resolved_device).startswith("cuda")
    )
    pcg_max_iter = max(6000, int(solve_cfg.get("pcg_max_iter", solve_cfg.get("cg_max_iter", 2000))))
    pcg_tol = max(1e-5, float(solve_cfg.get("pcg_tol", solve_cfg.get("cg_tol", 1e-6))))
    pcg_check_every = max(25, int(solve_cfg.get("pcg_check_every", 25)))
    pcg_preconditioner = str(solve_cfg.get("pcg_preconditioner", "jacobi"))

    sols: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []
    fallback_reason: Optional[str] = None
    if use_cuda_first and M.shape[0] > 0:
        try:
            M_cuda, M_diag_cuda = build_cuda_sparse_system(M=M, device=resolved_device)
            for idx, rhs in enumerate(rhs_rows):
                sol_i, meta_i = solve_linear_cuda_pcg(
                    M_cuda=M_cuda,
                    M_diag_cuda=M_diag_cuda,
                    rhs=np.asarray(rhs, dtype=np.float64),
                    pcg_max_iter=pcg_max_iter,
                    pcg_tol=pcg_tol,
                    pcg_check_every=pcg_check_every,
                    pcg_preconditioner=pcg_preconditioner,
                    channel_name=f"{channel_prefix}_{idx}",
                )
                sols.append(np.asarray(sol_i, dtype=np.float64))
                meta_rows.append(meta_i)
        except Exception as exc:
            fallback_reason = f"cuda_pcg_failed: {exc}"
            sols = []
            meta_rows = []

    if len(sols) != len(rhs_rows):
        sols = []
        meta_rows = []
        cpu_error: Optional[str] = None
        try:
            for idx, rhs in enumerate(rhs_rows):
                sol_i, meta_i = solve_linear_robust(
                    M=M,
                    rhs=np.asarray(rhs, dtype=np.float64),
                    cg_max_iter=pcg_max_iter,
                    cg_tol=pcg_tol,
                    channel_name=f"{channel_prefix}_{idx}",
                )
                sols.append(np.asarray(sol_i, dtype=np.float64))
                meta_rows.append(meta_i)
        except Exception as exc:
            cpu_error = str(exc)
            if fallback_reason is not None:
                raise RuntimeError(f"{fallback_reason}; cpu_fallback_failed: {cpu_error}") from exc
            raise

    residuals = [
        float(meta_i.get("residual", float("nan")))
        for meta_i in meta_rows
        if meta_i.get("residual", None) is not None and math.isfinite(float(meta_i.get("residual")))
    ]
    out_meta: Dict[str, Any] = {
        "backend_requested": backend_requested,
        "backend_used": str(meta_rows[0].get("backend", "unknown")) if meta_rows else "unknown",
        "residual_mean": float(np.mean(residuals)) if residuals else None,
        "residual_max": float(np.max(residuals)) if residuals else None,
        "row_count": int(len(rhs_rows)),
        "pcg_max_iter": int(pcg_max_iter),
        "pcg_tol": float(pcg_tol),
    }
    if fallback_reason is not None:
        out_meta["fallback_reason"] = fallback_reason
    return sols, out_meta


def _project_face_jacobian_field(
    *,
    state: Method2InternalState,
    solve_cfg: Dict[str, Any],
    data_jac: np.ndarray,
    data_valid: np.ndarray,
    base_jac: np.ndarray,
    base_valid: np.ndarray,
    face_confidence: np.ndarray,
    lambda_base: float,
    lambda_curl: float,
    ridge_eps: float = 1e-8,
) -> Dict[str, Any]:
    data = np.asarray(data_jac, dtype=np.float64)
    data_ok = np.asarray(data_valid, dtype=np.bool_) & np.isfinite(data).all(axis=(1, 2))
    base = np.asarray(base_jac, dtype=np.float64)
    base_ok = np.asarray(base_valid, dtype=np.bool_) & np.isfinite(base).all(axis=(1, 2))
    face_active = np.asarray(state.solve_face_active_mask, dtype=np.bool_)
    conf = np.clip(np.asarray(face_confidence, dtype=np.float64), 0.0, 1.0)
    conf[~data_ok] = 0.0

    project_mask = face_active & base_ok
    local_faces = np.where(project_mask)[0].astype(np.int64, copy=False)
    wd = np.zeros((len(project_mask),), dtype=np.float64)
    wb = np.zeros((len(project_mask),), dtype=np.float64)
    wd[project_mask] = conf[project_mask]
    wb[project_mask] = float(lambda_base) * (1.0 - conf[project_mask])
    wb[project_mask & ~data_ok] = float(lambda_base)

    M, _, matrix_meta = _build_field_projection_matrix(
        mesh=state.solve_mesh,
        local_face_ids=local_faces,
        lambda_curl=float(lambda_curl),
        ridge_eps=float(ridge_eps),
    )
    if M.shape[0] == 0:
        raise RuntimeError("field projector has no active faces to solve")

    local_wd = wd[local_faces]
    local_wb = wb[local_faces]
    diag_boost = np.repeat(local_wd + local_wb, 3).astype(np.float64, copy=False)
    if np.any(diag_boost > 0.0):
        M = (M + diags(diag_boost, offsets=0, shape=M.shape, dtype=np.float64)).tocsr()

    rhs_rows: List[np.ndarray] = []
    data_local = np.nan_to_num(data[local_faces], nan=0.0, posinf=0.0, neginf=0.0)
    base_local = np.nan_to_num(base[local_faces], nan=0.0, posinf=0.0, neginf=0.0)
    for row_idx in range(2):
        rhs_row = (
            local_wd[:, None] * data_local[:, row_idx, :]
            + local_wb[:, None] * base_local[:, row_idx, :]
        ).reshape(-1)
        rhs_rows.append(rhs_row.astype(np.float64, copy=False))

    sols, solve_meta = _solve_field_projection_rows(
        M=M,
        rhs_rows=rhs_rows,
        resolved_device=state.resolved_device,
        solve_cfg=solve_cfg,
        channel_prefix="field_projector",
    )

    projected_jac = np.full_like(base, np.nan, dtype=np.float64)
    for row_idx in range(2):
        projected_jac[local_faces, row_idx, :] = np.asarray(sols[row_idx], dtype=np.float64).reshape(-1, 3)
    projected_valid = np.zeros((len(project_mask),), dtype=np.bool_)
    projected_valid[local_faces] = np.isfinite(projected_jac[local_faces]).all(axis=(1, 2))

    return {
        "face_jac": projected_jac,
        "face_valid": projected_valid,
        "face_confidence": conf,
        "weight_data": wd,
        "weight_base": wb,
        "matrix_meta": matrix_meta,
        "solve_meta": solve_meta,
    }


def _default_projected_face_weights(current_weights: np.ndarray, projected_valid: np.ndarray) -> np.ndarray:
    base = np.asarray(current_weights, dtype=np.float64).copy()
    valid = np.asarray(projected_valid, dtype=np.bool_)
    pos = base[np.isfinite(base) & (base > 1e-12)]
    fill = float(np.quantile(pos, 0.50)) if pos.size > 0 else 1.0
    bad = ~np.isfinite(base) | (base <= 1e-12)
    base[bad & valid] = fill
    base[~valid] = 0.0
    return base


def build_method25_projected_field(
    *,
    state: Method2InternalState,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    m25_cfg = dict(cfg.get("method25", {}))
    strict_gate = bool(m25_cfg.get("strict_gate", False))
    min_samples = max(3, int(m25_cfg.get("samplefit_min_samples", 3)))
    lambda_decay = float(m25_cfg.get("lambda_decay", 1.0))
    lambda_curl = float(m25_cfg.get("lambda_curl", 20.0))
    ridge_eps = float(m25_cfg.get("ridge_eps", 1e-8))

    anchor_uv, anchor_fill_meta = _complete_anchor_vertex_uv(
        state=state,
        high_mesh=high_mesh,
        high_uv=high_uv,
    )
    anchor_jac = _compute_face_jacobians(state.face_geom_pinv, state.solve_mesh, anchor_uv)
    base_valid = np.asarray(state.solve_face_active_mask, dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))

    samplefit_raw = _fit_face_sample_jacobian_field(state=state, min_samples=min_samples)
    confidence = _projection_confidence_samplefit(
        state=state,
        samplefit=samplefit_raw,
        strict_gate=strict_gate,
    )

    data = np.asarray(samplefit_raw["face_jac"], dtype=np.float64)
    data_ok = np.asarray(samplefit_raw["face_valid"], dtype=np.bool_) & np.isfinite(data).all(axis=(1, 2))
    residual_data = np.full_like(anchor_jac, np.nan, dtype=np.float64)
    residual_data_valid = data_ok & base_valid
    if np.any(residual_data_valid):
        residual_data[residual_data_valid] = data[residual_data_valid] - anchor_jac[residual_data_valid]

    projected_residual = _project_face_jacobian_field(
        state=state,
        solve_cfg=dict(cfg.get("solve", {})),
        data_jac=residual_data,
        data_valid=residual_data_valid,
        base_jac=np.zeros_like(anchor_jac, dtype=np.float64),
        base_valid=base_valid,
        face_confidence=confidence,
        lambda_base=lambda_decay,
        lambda_curl=lambda_curl,
        ridge_eps=ridge_eps,
    )

    residual_jac = np.asarray(projected_residual["face_jac"], dtype=np.float64)
    residual_valid = np.asarray(projected_residual["face_valid"], dtype=np.bool_)
    final_jac = np.full_like(anchor_jac, np.nan, dtype=np.float64)
    final_valid = base_valid & residual_valid
    if np.any(final_valid):
        final_jac[final_valid] = anchor_jac[final_valid] + residual_jac[final_valid]

    face_weights = _default_projected_face_weights(state.face_target_weights, final_valid)
    return {
        "face_jac": final_jac,
        "face_valid": final_valid,
        "face_weights": face_weights,
        "anchor_vertex_target_uv": anchor_uv,
        "anchor_fill_meta": anchor_fill_meta,
        "samplefit_meta": {
            "valid_face_count": int(np.count_nonzero(np.asarray(samplefit_raw["face_valid"], dtype=np.bool_))),
            "strict_gate": strict_gate,
            "min_samples": int(min_samples),
        },
        "projector_meta": {
            "field_source": "residual_samplefit",
            "lambda_decay": float(lambda_decay),
            "lambda_curl": float(lambda_curl),
            "ridge_eps": float(ridge_eps),
            "matrix": projected_residual["matrix_meta"],
            "solve": projected_residual["solve_meta"],
        },
    }


__all__ = ["build_method25_projected_field"]
