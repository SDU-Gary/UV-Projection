#!/usr/bin/env python3
from __future__ import annotations

# Internal split from audit_method2_internal_core.py: experiment matrix and field-projector probes.

import audit_method2_internal_core as _core
from audit_method2_internal_core import *  # noqa: F401,F403
from faithc_infra.services.halfedge_topology import split_vertices_along_cut_edges
from faithc_infra.services.uv.seam_optimization import (
    build_interior_edge_table,
    score_route_c_cut_edges,
    select_budgeted_cut_edges,
)
from faithc_infra.services.uv.solve_constraints import compute_uv_box_feasibility_arrays, summarize_uv_box_feasibility

globals().update({name: getattr(_core, name) for name in dir(_core) if not name.startswith("__")})

def _blend_face_fields(
    base_jac: np.ndarray,
    base_valid: np.ndarray,
    override_jac: np.ndarray,
    override_valid: np.ndarray,
    override_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    out_jac = np.asarray(base_jac, dtype=np.float64).copy()
    out_valid = np.asarray(base_valid, dtype=np.bool_).copy()
    mask = np.asarray(override_mask, dtype=np.bool_) & np.asarray(override_valid, dtype=np.bool_)
    if np.any(mask):
        out_jac[mask] = np.asarray(override_jac, dtype=np.float64)[mask]
        out_valid[mask] = True
    return out_jac, out_valid


def _projection_confidence_current(
    ctx: Dict[str, Any],
    *,
    strict_gate: bool = False,
) -> np.ndarray:
    current_valid = np.asarray(ctx["face_valid"], dtype=np.bool_)
    accepted = np.asarray(ctx["face_accepted_samples"], dtype=np.float64)
    fallback_ratio = np.asarray(ctx["sample_mix_arrays"]["fallback_ratio"], dtype=np.float64)
    cov_norm = np.asarray(ctx["face_target_cov_norm"], dtype=np.float64)
    count_score = np.clip(accepted / 4.0, 0.0, 1.0)
    fallback_score = 1.0 - np.clip(np.nan_to_num(fallback_ratio, nan=1.0), 0.0, 1.0)
    cov_score = 1.0 / (1.0 + np.maximum(np.nan_to_num(cov_norm, nan=1e6), 0.0))
    conf = count_score * fallback_score * cov_score
    if strict_gate:
        strict_mask = (accepted >= 4.0) & (np.nan_to_num(fallback_ratio, nan=1.0) <= 0.25)
        conf = np.where(strict_mask, conf, 0.0)
    conf = np.clip(conf, 0.0, 1.0)
    conf[~current_valid] = 0.0
    return conf


def _projection_confidence_samplefit(
    ctx: Dict[str, Any],
    samplefit: Dict[str, Any],
    *,
    strict_gate: bool,
) -> np.ndarray:
    valid = np.asarray(samplefit["face_valid"], dtype=np.bool_)
    sample_count = np.asarray(samplefit["face_sample_count"], dtype=np.float64)
    fallback_ratio = np.asarray(samplefit["face_fallback_ratio"], dtype=np.float64)
    face_residual = np.asarray(samplefit["face_residual"], dtype=np.float64)
    cov_norm = np.asarray(ctx["face_target_cov_norm"], dtype=np.float64)

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
    use_cuda_first = backend_requested == "cuda_pcg" or (backend_requested == "auto" and str(resolved_device).startswith("cuda"))
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
    ctx: Dict[str, Any],
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
    face_active = np.asarray(ctx["face_active"], dtype=np.bool_)
    conf = np.clip(np.asarray(face_confidence, dtype=np.float64), 0.0, 1.0)
    conf[~data_ok] = 0.0

    project_mask = face_active & base_ok
    local_faces = np.where(project_mask)[0].astype(np.int64, copy=False)
    wd = np.zeros((len(project_mask),), dtype=np.float64)
    wb = np.zeros((len(project_mask),), dtype=np.float64)
    wd[project_mask] = conf[project_mask]
    wb[project_mask] = float(lambda_base) * (1.0 - conf[project_mask])
    wb[project_mask & ~data_ok] = float(lambda_base)

    M, local_index, matrix_meta = _build_field_projection_matrix(
        mesh=ctx["solve_mesh"],
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
        resolved_device=ctx["internal"].resolved_device,
        solve_cfg=dict(ctx["cfg"].get("solve", {})),
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


def _project_face_jacobian_residual_field(
    *,
    ctx: Dict[str, Any],
    data_jac: np.ndarray,
    data_valid: np.ndarray,
    base_jac: np.ndarray,
    base_valid: np.ndarray,
    face_confidence: np.ndarray,
    lambda_decay: float,
    lambda_curl: float,
    ridge_eps: float = 1e-8,
) -> Dict[str, Any]:
    data = np.asarray(data_jac, dtype=np.float64)
    data_ok = np.asarray(data_valid, dtype=np.bool_) & np.isfinite(data).all(axis=(1, 2))
    base = np.asarray(base_jac, dtype=np.float64)
    base_ok = np.asarray(base_valid, dtype=np.bool_) & np.isfinite(base).all(axis=(1, 2))

    residual_data = np.full_like(base, np.nan, dtype=np.float64)
    residual_data_valid = data_ok & base_ok
    if np.any(residual_data_valid):
        residual_data[residual_data_valid] = data[residual_data_valid] - base[residual_data_valid]

    projected_residual = _project_face_jacobian_field(
        ctx=ctx,
        data_jac=residual_data,
        data_valid=residual_data_valid,
        base_jac=np.zeros_like(base, dtype=np.float64),
        base_valid=base_ok,
        face_confidence=np.asarray(face_confidence, dtype=np.float64),
        lambda_base=float(lambda_decay),
        lambda_curl=float(lambda_curl),
        ridge_eps=float(ridge_eps),
    )

    residual_jac = np.asarray(projected_residual["face_jac"], dtype=np.float64)
    residual_valid = np.asarray(projected_residual["face_valid"], dtype=np.bool_)
    final_jac = np.full_like(base, np.nan, dtype=np.float64)
    final_valid = base_ok & residual_valid
    if np.any(final_valid):
        final_jac[final_valid] = base[final_valid] + residual_jac[final_valid]

    return {
        "face_jac": final_jac,
        "face_valid": final_valid,
        "face_confidence": np.asarray(projected_residual["face_confidence"], dtype=np.float64),
        "weight_data": np.asarray(projected_residual["weight_data"], dtype=np.float64),
        "weight_decay": np.asarray(projected_residual["weight_base"], dtype=np.float64),
        "matrix_meta": projected_residual["matrix_meta"],
        "solve_meta": projected_residual["solve_meta"],
        "residual_data_jac": residual_data,
        "residual_data_valid": residual_data_valid,
        "residual_jac": residual_jac,
        "residual_valid": residual_valid,
    }


def _field_sample_explain_summary(ctx: Dict[str, Any], face_jac: np.ndarray) -> Dict[str, Any]:
    faces = np.asarray(ctx["solve_mesh"].faces, dtype=np.int64)
    verts = np.asarray(ctx["solve_mesh"].vertices, dtype=np.float64)
    sample_face_ids = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64).reshape(-1)
    sample_bary = np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64)
    target_uv = np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64)
    if sample_face_ids.size == 0:
        return {
            "face_count": 0,
            "residual_mean_p50": None,
            "residual_mean_p95": None,
            "residual_mean_p99": None,
        }
    order = np.argsort(sample_face_ids, kind="mergesort")
    sf = sample_face_ids[order]
    sb = sample_bary[order]
    st = target_uv[order]
    split_idx = np.flatnonzero(np.diff(sf)) + 1
    starts = np.concatenate(([0], split_idx))
    ends = np.concatenate((split_idx, [len(sf)]))
    per_face_err: List[float] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        fid = int(sf[start])
        J = np.asarray(face_jac[fid], dtype=np.float64)
        if fid < 0 or fid >= len(faces) or not np.isfinite(J).all():
            continue
        W = np.asarray(sb[start:end], dtype=np.float64)
        T = np.asarray(st[start:end], dtype=np.float64)
        p = verts[faces[fid]]
        x = W @ p
        xref = x[0]
        pred_delta = (x - xref) @ J.T
        pred_uv = T[0] + pred_delta
        per_face_err.append(float(np.mean(np.linalg.norm(pred_uv - T, axis=1))))
    vals = np.asarray(per_face_err, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {
        "face_count": int(vals.size),
        "residual_mean_p50": float(np.quantile(vals, 0.50)) if vals.size > 0 else None,
        "residual_mean_p95": float(np.quantile(vals, 0.95)) if vals.size > 0 else None,
        "residual_mean_p99": float(np.quantile(vals, 0.99)) if vals.size > 0 else None,
    }


def _field_divergence_proxy(
    *,
    mesh: trimesh.Trimesh,
    face_jac: np.ndarray,
    face_weights: np.ndarray,
    face_valid: np.ndarray,
) -> Dict[str, Any]:
    valid = np.asarray(face_valid, dtype=np.bool_).reshape(-1)
    weights = np.asarray(face_weights, dtype=np.float64).reshape(-1)
    n_faces = int(len(mesh.faces))
    n_vertices = int(len(mesh.vertices))
    if n_faces == 0 or n_vertices == 0 or not np.any(valid):
        return {
            "vertex_rhs_norm": np.zeros((n_vertices,), dtype=np.float64),
            "face_rhs_norm": np.full((n_faces,), np.nan, dtype=np.float64),
            "summary": {
                "valid_face_count": int(np.count_nonzero(valid)),
                "total_face_count": n_faces,
                "vertex_rhs_norm_p50": None,
                "vertex_rhs_norm_p95": None,
                "vertex_rhs_norm_p99": None,
                "face_rhs_norm_p50": None,
                "face_rhs_norm_p95": None,
                "face_rhs_norm_p99": None,
                "face_rhs_norm_mean": None,
                "face_rhs_norm_max": None,
                "high_div_face_ratio": 0.0,
                "high_div_threshold_p95": None,
                "constraint_row_count": 0,
            },
        }

    A, rhs_u, rhs_v, _ = _build_gradient_constraint_system(
        mesh=mesh,
        face_jac=np.asarray(face_jac, dtype=np.float64),
        face_weights=weights,
        face_valid_mask=valid,
    )
    vertex_rhs_u = np.asarray(A.T @ np.asarray(rhs_u, dtype=np.float64), dtype=np.float64).reshape(-1)
    vertex_rhs_v = np.asarray(A.T @ np.asarray(rhs_v, dtype=np.float64), dtype=np.float64).reshape(-1)
    vertex_rhs_norm = np.sqrt(vertex_rhs_u * vertex_rhs_u + vertex_rhs_v * vertex_rhs_v)
    face_rhs_norm = np.full((n_faces,), np.nan, dtype=np.float64)
    tri = np.asarray(mesh.faces, dtype=np.int64)
    if np.any(valid):
        face_rhs_norm[valid] = np.mean(vertex_rhs_norm[tri[valid]], axis=1)

    vvals = vertex_rhs_norm[np.isfinite(vertex_rhs_norm)]
    fvals = face_rhs_norm[np.isfinite(face_rhs_norm)]
    fthr = float(np.quantile(fvals, 0.95)) if fvals.size > 0 else None
    summary = {
        "valid_face_count": int(np.count_nonzero(np.isfinite(face_rhs_norm))),
        "total_face_count": n_faces,
        "vertex_rhs_norm_p50": float(np.quantile(vvals, 0.50)) if vvals.size > 0 else None,
        "vertex_rhs_norm_p95": float(np.quantile(vvals, 0.95)) if vvals.size > 0 else None,
        "vertex_rhs_norm_p99": float(np.quantile(vvals, 0.99)) if vvals.size > 0 else None,
        "face_rhs_norm_p50": float(np.quantile(fvals, 0.50)) if fvals.size > 0 else None,
        "face_rhs_norm_p95": float(np.quantile(fvals, 0.95)) if fvals.size > 0 else None,
        "face_rhs_norm_p99": float(np.quantile(fvals, 0.99)) if fvals.size > 0 else None,
        "face_rhs_norm_mean": float(np.mean(fvals)) if fvals.size > 0 else None,
        "face_rhs_norm_max": float(np.max(fvals)) if fvals.size > 0 else None,
        "high_div_face_ratio": float(np.mean(fvals >= fthr)) if fvals.size > 0 and fthr is not None else 0.0,
        "high_div_threshold_p95": fthr,
        "constraint_row_count": int(A.shape[0]),
    }
    return {
        "vertex_rhs_norm": vertex_rhs_norm,
        "face_rhs_norm": face_rhs_norm,
        "summary": summary,
    }


def _face_out_of_bounds_mask(mesh: trimesh.Trimesh, vertex_uv: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    tri_uv = np.asarray(vertex_uv, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    low = tri_uv < (0.0 - float(tol))
    high = tri_uv > (1.0 + float(tol))
    return np.any(low | high, axis=(1, 2))


def _divergence_shift_summary(
    reference_face_rhs_norm: np.ndarray,
    other_face_rhs_norm: np.ndarray,
    face_mask: np.ndarray,
) -> Dict[str, Any]:
    ref = np.asarray(reference_face_rhs_norm, dtype=np.float64).reshape(-1)
    oth = np.asarray(other_face_rhs_norm, dtype=np.float64).reshape(-1)
    mask = np.asarray(face_mask, dtype=np.bool_).reshape(-1)
    valid = mask & np.isfinite(ref) & np.isfinite(oth)
    if not np.any(valid):
        return {
            "face_count": 0,
            "signed_mean": None,
            "abs_delta_p50": None,
            "abs_delta_p95": None,
            "abs_delta_p99": None,
            "ratio_p50": None,
            "ratio_p95": None,
        }
    delta = oth[valid] - ref[valid]
    abs_delta = np.abs(delta)
    ratio = oth[valid] / np.maximum(ref[valid], 1e-12)
    return {
        "face_count": int(np.count_nonzero(valid)),
        "signed_mean": float(np.mean(delta)),
        "abs_delta_p50": float(np.quantile(abs_delta, 0.50)),
        "abs_delta_p95": float(np.quantile(abs_delta, 0.95)),
        "abs_delta_p99": float(np.quantile(abs_delta, 0.99)),
        "ratio_p50": float(np.quantile(ratio, 0.50)),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
    }


def _high_oob_high_divergence_overlap(
    *,
    mesh: trimesh.Trimesh,
    vertex_uv: np.ndarray,
    face_rhs_norm: np.ndarray,
    face_mask: np.ndarray,
) -> Dict[str, Any]:
    face_div = np.asarray(face_rhs_norm, dtype=np.float64).reshape(-1)
    mask = np.asarray(face_mask, dtype=np.bool_).reshape(-1) & np.isfinite(face_div)
    if not np.any(mask):
        return {
            "face_count": 0,
            "oob_face_ratio": 0.0,
            "high_div_face_ratio": 0.0,
            "overlap_face_ratio": 0.0,
            "overlap_over_oob": 0.0,
            "overlap_over_high_div": 0.0,
            "divergence_oob_corr": None,
            "high_div_threshold_p95": None,
        }
    oob_mask = _face_out_of_bounds_mask(mesh, np.asarray(vertex_uv, dtype=np.float64))
    div_vals = face_div[mask]
    div_thr = float(np.quantile(div_vals, 0.95))
    high_div = np.zeros_like(mask, dtype=np.bool_)
    high_div[mask] = face_div[mask] >= div_thr
    overlap = high_div & oob_mask & mask
    corr = None
    if np.count_nonzero(mask) >= 8 and np.any(oob_mask[mask]) and not np.all(oob_mask[mask]):
        corr = float(np.corrcoef(face_div[mask], oob_mask[mask].astype(np.float64))[0, 1])
    return {
        "face_count": int(np.count_nonzero(mask)),
        "oob_face_ratio": float(np.mean(oob_mask[mask])),
        "high_div_face_ratio": float(np.mean(high_div[mask])),
        "overlap_face_ratio": float(np.mean(overlap[mask])),
        "overlap_over_oob": float(np.count_nonzero(overlap) / max(1, np.count_nonzero(oob_mask & mask))),
        "overlap_over_high_div": float(np.count_nonzero(overlap) / max(1, np.count_nonzero(high_div & mask))),
        "divergence_oob_corr": corr,
        "high_div_threshold_p95": div_thr,
    }


def _norm_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
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


def _route_b_feasibility_summary(
    *,
    mesh: trimesh.Trimesh,
    reference_uv: np.ndarray,
    candidate_uv: np.ndarray,
    face_island_labels: np.ndarray,
    box_margin: float,
) -> Dict[str, Any]:
    ref_uv = np.asarray(reference_uv, dtype=np.float64)
    cand_uv = np.asarray(candidate_uv, dtype=np.float64)
    labels = np.asarray(face_island_labels, dtype=np.int64).reshape(-1)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    ref_feas = summarize_uv_box_feasibility(mesh, ref_uv, margin=box_margin)
    cand_feas = summarize_uv_box_feasibility(mesh, cand_uv, margin=box_margin)
    cand_arrays = compute_uv_box_feasibility_arrays(mesh, cand_uv, margin=box_margin)

    delta = cand_uv - ref_uv
    delta_norm = np.linalg.norm(delta, axis=1)
    boundary_ids = _boundary_vertex_ids(mesh)
    boundary_delta = delta_norm[boundary_ids] if boundary_ids.size > 0 else np.zeros((0,), dtype=np.float64)
    barrier_vid = np.where(np.asarray(cand_arrays["vertex_barrier_mask"], dtype=np.bool_))[0].astype(np.int64, copy=False)
    barrier_delta = delta_norm[barrier_vid] if barrier_vid.size > 0 else np.zeros((0,), dtype=np.float64)

    centroid_shift: List[float] = []
    bbox_area_ratio: List[float] = []
    bbox_area_log_abs: List[float] = []
    for label in np.unique(labels[labels >= 0]).tolist():
        face_mask = labels == int(label)
        if not np.any(face_mask):
            continue
        vids = np.unique(faces[face_mask].reshape(-1)).astype(np.int64, copy=False)
        if vids.size == 0:
            continue
        ref_sel = ref_uv[vids]
        cand_sel = cand_uv[vids]
        centroid_shift.append(float(np.linalg.norm(np.mean(cand_sel, axis=0) - np.mean(ref_sel, axis=0))))
        ref_extent = np.maximum(np.max(ref_sel, axis=0) - np.min(ref_sel, axis=0), 1e-12)
        cand_extent = np.maximum(np.max(cand_sel, axis=0) - np.min(cand_sel, axis=0), 1e-12)
        area_ratio = float(np.prod(cand_extent) / np.prod(ref_extent))
        bbox_area_ratio.append(area_ratio)
        bbox_area_log_abs.append(float(abs(np.log(max(area_ratio, 1e-12)))))

    return {
        "reference": ref_feas,
        "candidate": cand_feas,
        "delta": {
            "vertex_oob_ratio": float(cand_feas["vertex_oob_ratio"] - ref_feas["vertex_oob_ratio"]),
            "face_oob_ratio": float(cand_feas["face_oob_ratio"] - ref_feas["face_oob_ratio"]),
            "max_oob_overshoot": float(cand_feas["max_oob_overshoot"] - ref_feas["max_oob_overshoot"]),
            "barrier_active_vertex_ratio": float(
                cand_feas["barrier_active_vertex_ratio"] - ref_feas["barrier_active_vertex_ratio"]
            ),
            "barrier_active_face_ratio": float(
                cand_feas["barrier_active_face_ratio"] - ref_feas["barrier_active_face_ratio"]
            ),
            "max_barrier_overshoot": float(
                cand_feas["max_barrier_overshoot"] - ref_feas["max_barrier_overshoot"]
            ),
        },
        "all_vertex_displacement": _norm_summary(delta_norm),
        "boundary_vertex_displacement": _norm_summary(boundary_delta),
        "barrier_active_vertex_displacement": _norm_summary(barrier_delta),
        "island_drift": {
            "island_count": int(len(centroid_shift)),
            "centroid_shift": _norm_summary(np.asarray(centroid_shift, dtype=np.float64)),
            "bbox_area_ratio_p50": _weighted_quantile(np.asarray(bbox_area_ratio, dtype=np.float64), 0.50),
            "bbox_area_ratio_p95": _weighted_quantile(np.asarray(bbox_area_ratio, dtype=np.float64), 0.95),
            "bbox_area_log_abs_p50": _weighted_quantile(np.asarray(bbox_area_log_abs, dtype=np.float64), 0.50),
            "bbox_area_log_abs_p95": _weighted_quantile(np.asarray(bbox_area_log_abs, dtype=np.float64), 0.95),
            "bbox_area_log_abs_max": float(np.max(bbox_area_log_abs)) if bbox_area_log_abs else None,
        },
    }


def _route_c_build_field_sources(ctx: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    anchor_uv, _ = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    base_valid = np.asarray(ctx["face_active"], dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))
    samplefit_raw = _fit_face_sample_jacobian_field(ctx=ctx, anchor_uv_prior=None, prior_weight=0.0, min_samples=3)
    conf_samplefit = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=False)
    exp13_best = _project_face_jacobian_residual_field(
        ctx=ctx,
        data_jac=np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
        data_valid=np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
        base_jac=anchor_jac,
        base_valid=base_valid,
        face_confidence=conf_samplefit,
        lambda_decay=1.0,
        lambda_curl=20.0,
    )
    return (
        {
            "field_source": "current_target",
            "note": "current Method2 face Jacobian target",
            "face_jac": np.asarray(ctx["target_jac"], dtype=np.float64),
            "face_valid": np.asarray(ctx["face_valid"], dtype=np.bool_),
        },
        {
            "field_source": "exp13_residual_samplefit_soft_curl20",
            "note": "Exp13 best residual-space projector output",
            "face_jac": np.asarray(exp13_best["face_jac"], dtype=np.float64),
            "face_valid": np.asarray(exp13_best["face_valid"], dtype=np.bool_),
        },
    )


def _infer_split_vertex_parent_map(
    *,
    original_faces: np.ndarray,
    split_faces: np.ndarray,
    original_vertex_count: int,
    split_vertex_count: Optional[int] = None,
) -> np.ndarray:
    faces_ref = np.asarray(original_faces, dtype=np.int64)
    faces_split = np.asarray(split_faces, dtype=np.int64)
    if faces_ref.shape != faces_split.shape:
        raise RuntimeError("split face topology shape mismatch")
    if split_vertex_count is None:
        n_split_vertices = int(np.max(faces_split)) + 1 if faces_split.size > 0 else int(original_vertex_count)
    else:
        n_split_vertices = int(split_vertex_count)
    new_to_old = np.full((n_split_vertices,), -1, dtype=np.int64)
    for fid in range(faces_ref.shape[0]):
        for corner in range(3):
            old_vid = int(faces_ref[fid, corner])
            new_vid = int(faces_split[fid, corner])
            prev = int(new_to_old[new_vid])
            if prev < 0:
                new_to_old[new_vid] = old_vid
            elif prev != old_vid:
                raise RuntimeError("inconsistent split vertex parent mapping")
    for vid in range(min(int(original_vertex_count), int(new_to_old.shape[0]))):
        if int(new_to_old[vid]) < 0:
            new_to_old[vid] = int(vid)
    if np.any(new_to_old < 0):
        raise RuntimeError("failed to infer split vertex parents for all vertices")
    return new_to_old.astype(np.int64, copy=False)


def _remap_vertex_array_to_split(values: np.ndarray, new_to_old: np.ndarray) -> np.ndarray:
    src = np.asarray(values)
    parent = np.asarray(new_to_old, dtype=np.int64).reshape(-1)
    if src.shape[0] == 0 and parent.size == 0:
        return src.copy()
    if np.any(parent < 0) or np.any(parent >= int(src.shape[0])):
        raise RuntimeError("split parent map out of range for vertex remap")
    return np.asarray(src[parent]).copy()


def _route_c_topology_delta_summary(
    *,
    reference_mesh: trimesh.Trimesh,
    reference_uv: np.ndarray,
    candidate_mesh: trimesh.Trimesh,
    candidate_uv: np.ndarray,
    candidate_face_labels: np.ndarray,
    candidate_new_to_old: np.ndarray,
) -> Dict[str, Any]:
    ref_uv = np.asarray(reference_uv, dtype=np.float64)
    cand_uv = np.asarray(candidate_uv, dtype=np.float64)
    labels = np.asarray(candidate_face_labels, dtype=np.int64).reshape(-1)
    new_to_old = np.asarray(candidate_new_to_old, dtype=np.int64).reshape(-1)
    if cand_uv.shape[0] != new_to_old.shape[0]:
        raise RuntimeError("candidate UV / parent map length mismatch")

    ref_feas = summarize_uv_box_feasibility(reference_mesh, ref_uv, margin=0.0)
    cand_feas = summarize_uv_box_feasibility(candidate_mesh, cand_uv, margin=0.0)

    parent_ref_uv = ref_uv[new_to_old]
    delta = cand_uv - parent_ref_uv
    delta_norm = np.linalg.norm(delta, axis=1)
    cand_boundary = _boundary_vertex_ids(candidate_mesh)
    boundary_delta = delta_norm[cand_boundary] if cand_boundary.size > 0 else np.zeros((0,), dtype=np.float64)
    dup_vid = np.arange(cand_uv.shape[0], dtype=np.int64) >= int(len(reference_mesh.vertices))
    dup_delta = delta_norm[dup_vid] if np.any(dup_vid) else np.zeros((0,), dtype=np.float64)

    centroid_shift: List[float] = []
    bbox_area_ratio: List[float] = []
    bbox_area_log_abs: List[float] = []
    cand_faces = np.asarray(candidate_mesh.faces, dtype=np.int64)
    for label in np.unique(labels[labels >= 0]).tolist():
        face_mask = labels == int(label)
        if not np.any(face_mask):
            continue
        cand_vid = np.unique(cand_faces[face_mask].reshape(-1)).astype(np.int64, copy=False)
        parent_vid = np.unique(new_to_old[cand_vid]).astype(np.int64, copy=False)
        if cand_vid.size == 0 or parent_vid.size == 0:
            continue
        ref_sel = ref_uv[parent_vid]
        cand_sel = cand_uv[cand_vid]
        centroid_shift.append(float(np.linalg.norm(np.mean(cand_sel, axis=0) - np.mean(ref_sel, axis=0))))
        ref_extent = np.maximum(np.max(ref_sel, axis=0) - np.min(ref_sel, axis=0), 1e-12)
        cand_extent = np.maximum(np.max(cand_sel, axis=0) - np.min(cand_sel, axis=0), 1e-12)
        area_ratio = float(np.prod(cand_extent) / np.prod(ref_extent))
        bbox_area_ratio.append(area_ratio)
        bbox_area_log_abs.append(float(abs(np.log(max(area_ratio, 1e-12)))))

    return {
        "reference_feasibility": ref_feas,
        "candidate_feasibility": cand_feas,
        "feasibility_delta": {
            "vertex_oob_ratio": float(cand_feas["vertex_oob_ratio"] - ref_feas["vertex_oob_ratio"]),
            "face_oob_ratio": float(cand_feas["face_oob_ratio"] - ref_feas["face_oob_ratio"]),
            "max_oob_overshoot": float(cand_feas["max_oob_overshoot"] - ref_feas["max_oob_overshoot"]),
            "barrier_active_vertex_ratio": float(
                cand_feas["barrier_active_vertex_ratio"] - ref_feas["barrier_active_vertex_ratio"]
            ),
            "barrier_active_face_ratio": float(
                cand_feas["barrier_active_face_ratio"] - ref_feas["barrier_active_face_ratio"]
            ),
            "max_barrier_overshoot": float(
                cand_feas["max_barrier_overshoot"] - ref_feas["max_barrier_overshoot"]
            ),
        },
        "all_vertex_displacement_vs_parent": _norm_summary(delta_norm),
        "boundary_vertex_displacement_vs_parent": _norm_summary(boundary_delta),
        "duplicated_vertex_displacement_vs_parent": _norm_summary(dup_delta),
        "island_drift_vs_parent": {
            "island_count": int(len(centroid_shift)),
            "centroid_shift": _norm_summary(np.asarray(centroid_shift, dtype=np.float64)),
            "bbox_area_ratio_p50": _weighted_quantile(np.asarray(bbox_area_ratio, dtype=np.float64), 0.50),
            "bbox_area_ratio_p95": _weighted_quantile(np.asarray(bbox_area_ratio, dtype=np.float64), 0.95),
            "bbox_area_log_abs_p50": _weighted_quantile(np.asarray(bbox_area_log_abs, dtype=np.float64), 0.50),
            "bbox_area_log_abs_p95": _weighted_quantile(np.asarray(bbox_area_log_abs, dtype=np.float64), 0.95),
            "bbox_area_log_abs_max": float(np.max(bbox_area_log_abs)) if bbox_area_log_abs else None,
        },
    }


def _run_exp1_filtered_metrics(ctx: Dict[str, Any]) -> Dict[str, Any]:
    solve_mesh = ctx["solve_mesh"]
    solve_uv = ctx["solve_uv"]
    face_valid = ctx["face_valid"]
    face_accepted_samples = ctx["face_accepted_samples"]
    face_rel_err = ctx["face_rel_err"]
    face_cosine = ctx["face_cosine"]
    face_log_area_ratio = ctx["face_log_area_ratio"]
    comp_sizes = ctx["component_sizes"]
    accepted = face_accepted_samples > 0
    filters = {
        "all_faces": np.ones((len(solve_mesh.faces),), dtype=np.bool_),
        "valid_faces_only": face_valid,
        "accepted_faces_only": accepted,
        "component_ge_4": comp_sizes >= 4,
        "component_ge_8": comp_sizes >= 8,
        "component_ge_16": comp_sizes >= 16,
        "component_ge_64": comp_sizes >= 64,
        "valid_component_ge_16": face_valid & (comp_sizes >= 16),
        "valid_component_ge_64": face_valid & (comp_sizes >= 64),
    }
    out: Dict[str, Any] = {"status": "ok", "filters": {}}
    for name, mask in filters.items():
        out["filters"][name] = {
            "quality": _masked_quality_summary(solve_mesh, solve_uv, mask),
            "jacobian": _masked_jacobian_summary(face_rel_err, face_cosine, face_log_area_ratio, mask),
        }
    return out


def _run_exp2_curl_audit(ctx: Dict[str, Any]) -> Dict[str, Any]:
    edge_data = _compute_edge_jump_data(ctx["solve_mesh"], ctx["target_jac"], ctx["face_valid"])
    vertex_cycle_residual, face_cycle_residual = _compute_vertex_cycle_residuals(
        ctx["solve_mesh"],
        ctx["target_jac"],
        ctx["face_valid"],
    )
    valid_edge_jump = edge_data["edge_jump_l2"][edge_data["valid_edge_mask"] & np.isfinite(edge_data["edge_jump_l2"])]
    valid_cycle = face_cycle_residual[np.isfinite(face_cycle_residual)]
    jump_thr = float(np.quantile(valid_edge_jump, 0.95)) if valid_edge_jump.size > 0 else None
    cycle_thr = float(np.quantile(valid_cycle, 0.95)) if valid_cycle.size > 0 else None
    comp_ids = ctx["component_ids"]
    comp_sizes = ctx["component_sizes"]
    island_labels = ctx["island_labels"]
    adjacency = edge_data["adjacency"]
    per_component: List[Dict[str, Any]] = []
    if comp_ids.size > 0:
        for comp_id in np.unique(comp_ids[comp_ids >= 0]).tolist():
            face_mask = comp_ids == int(comp_id)
            if not np.any(face_mask):
                continue
            edge_mask = (
                edge_data["valid_edge_mask"]
                & face_mask[adjacency[:, 0]]
                & face_mask[adjacency[:, 1]]
                & np.isfinite(edge_data["edge_jump_l2"])
            )
            comp_jump = edge_data["edge_jump_l2"][edge_mask]
            comp_cycle = face_cycle_residual[face_mask & np.isfinite(face_cycle_residual)]
            comp_faces = np.where(face_mask)[0]
            entry = {
                "island_label": int(island_labels[comp_faces[0]]),
                "component_id": int(comp_id),
                "component_faces": int(comp_sizes[comp_faces[0]]),
                "edge_jump_l2_p95": float(np.quantile(comp_jump, 0.95)) if comp_jump.size > 0 else None,
                "cycle_residual_l2_p95": float(np.quantile(comp_cycle, 0.95)) if comp_cycle.size > 0 else None,
                "high_jump_edge_ratio": float(np.mean(comp_jump > jump_thr)) if comp_jump.size > 0 and jump_thr is not None else 0.0,
                "high_cycle_face_ratio": float(np.mean(comp_cycle > cycle_thr)) if comp_cycle.size > 0 and cycle_thr is not None else 0.0,
            }
            per_component.append(entry)
    per_component.sort(
        key=lambda x: (
            -(float(x["cycle_residual_l2_p95"]) if x["cycle_residual_l2_p95"] is not None else -1.0),
            -(float(x["edge_jump_l2_p95"]) if x["edge_jump_l2_p95"] is not None else -1.0),
            -int(x["component_faces"]),
        )
    )
    return {
        "status": "ok",
        "global": {
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
        },
        "top_components": per_component[:20],
    }


def _solve_variant_summary(
    *,
    ctx: Dict[str, Any],
    variant_name: str,
    face_island_labels: Optional[np.ndarray],
    edge_cut_count: int = 0,
) -> Dict[str, Any]:
    solve_result = _solve_target_field(
        solve_mesh=ctx["solve_mesh"],
        high_mesh=ctx["high_mesh"],
        high_uv=ctx["high_uv"],
        resolved_device=ctx["internal"].resolved_device,
        cfg=ctx["cfg"],
        face_jac=ctx["target_jac"],
        face_weights=ctx["face_weights"],
        face_valid_mask=ctx["face_valid"],
        face_active_mask=ctx["face_active"],
        face_island_labels=face_island_labels,
        anchor_vertex_target_uv=ctx["internal"].anchor_vertex_target_uv,
        anchor_vertex_confidence=ctx["internal"].anchor_vertex_confidence,
        face_smooth_alpha=ctx["internal"].face_smooth_alpha,
    )
    mapped_uv = solve_result["mapped_uv"]
    solved_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], mapped_uv)
    jac_summary, _, _, _ = _jacobian_diagnostics(ctx["target_jac"], solved_jac, ctx["face_valid"])
    sample_summary = _sample_residual_summary_from_arrays(
        mesh=ctx["solve_mesh"],
        vertex_uv=mapped_uv,
        sample_face_ids=ctx["internal"].solve_sample_face_ids,
        sample_bary=ctx["internal"].solve_sample_bary,
        target_uv=ctx["internal"].solve_target_uv,
    )
    island_count = None
    if face_island_labels is not None:
        labels = np.asarray(face_island_labels, dtype=np.int64)
        island_count = int(len(np.unique(labels[labels >= 0])))
    return {
        "status": "ok",
        "variant": variant_name,
        "edge_cut_count": int(edge_cut_count),
        "island_count": island_count,
        "solve_per_island": bool(face_island_labels is not None),
        "solve_mesh_quality": {
            k: v
            for k, v in _quality_with_context(ctx["solve_mesh"], mapped_uv).items()
            if k in {"uv_stretch_p95", "uv_stretch_p99", "uv_bad_tri_ratio_stretch_only", "uv_out_of_bounds_ratio"}
        },
        "jacobian_summary": {
            "frob_rel_error_p95": jac_summary.get("frob_rel_error_p95"),
            "cosine_p05": jac_summary.get("cosine_p05"),
            "log_area_ratio_p95": jac_summary.get("log_area_ratio_p95"),
        },
        "sample_residual_summary": sample_summary,
        "solve_meta": {
            "uv_solver_linear_backend_used": solve_result["solve_meta"].get("uv_solver_linear_backend_used"),
            "uv_m2_system_cond_proxy": solve_result["solve_meta"].get("uv_m2_system_cond_proxy"),
            "uv_solver_residual_u": solve_result["solve_meta"].get("uv_solver_residual_u"),
            "uv_solver_residual_v": solve_result["solve_meta"].get("uv_solver_residual_v"),
        },
    }


def _run_exp3_repartition(ctx: Dict[str, Any], exp2: Dict[str, Any]) -> Dict[str, Any]:
    adjacency = ctx["edge_data"]["adjacency"]
    valid_edge = ctx["edge_data"]["valid_edge_mask"]
    cov_score = np.maximum(ctx["face_target_cov_norm"][adjacency[:, 0]], ctx["face_target_cov_norm"][adjacency[:, 1]])
    jump_score = np.asarray(ctx["edge_data"]["edge_jump_l2"], dtype=np.float64)
    ratios = [0.005, 0.01, 0.02]
    variants: List[Dict[str, Any]] = []
    current_labels = np.asarray(ctx["island_labels"], dtype=np.int64)
    variants.append(_solve_variant_summary(ctx=ctx, variant_name="current_islands", face_island_labels=current_labels))
    connected_labels = _component_labels_from_removed_adjacency(
        n_faces=len(ctx["solve_mesh"].faces),
        adjacency=adjacency,
        face_valid_mask=ctx["face_valid"],
    )
    variants.append(
        _solve_variant_summary(ctx=ctx, variant_name="connected_valid_component", face_island_labels=connected_labels)
    )
    variants.append(_solve_variant_summary(ctx=ctx, variant_name="global_solve", face_island_labels=None))
    for score_name, score in [("cov_split", cov_score), ("jump_split", jump_score)]:
        valid_score_mask = valid_edge & np.isfinite(score)
        valid_idx = np.where(valid_score_mask)[0]
        if valid_idx.size == 0:
            variants.append({"status": "error", "variant": score_name, "error": "no valid adjacency scores"})
            continue
        order = valid_idx[np.argsort(score[valid_idx])[::-1]]
        for ratio in ratios:
            n_cut = max(1, int(math.ceil(float(ratio) * float(order.size))))
            removed = np.zeros((adjacency.shape[0],), dtype=np.bool_)
            removed[order[:n_cut]] = True
            labels = _component_labels_from_removed_adjacency(
                n_faces=len(ctx["solve_mesh"].faces),
                adjacency=adjacency,
                face_valid_mask=ctx["face_valid"],
                removed_edge_mask=removed,
            )
            try:
                variants.append(
                    _solve_variant_summary(
                        ctx=ctx,
                        variant_name=f"{score_name}_{ratio:.3f}",
                        face_island_labels=labels,
                        edge_cut_count=int(n_cut),
                    )
                )
            except Exception as exc:
                variants.append(
                    {
                        "status": "error",
                        "variant": f"{score_name}_{ratio:.3f}",
                        "edge_cut_count": int(n_cut),
                        "error": str(exc),
                    }
                )
    return {"status": "ok", "variants": variants}


def _run_exp4_patch_dirichlet(ctx: Dict[str, Any]) -> Dict[str, Any]:
    neighbors = ctx["neighbors"]
    face_valid = ctx["face_valid"]
    comp_sizes = ctx["component_sizes"]
    accepted = ctx["face_accepted_samples"] > 0
    face_rel = ctx["face_rel_err"]
    face_cov = ctx["face_target_cov_norm"]
    candidate = face_valid & accepted & (comp_sizes >= 64) & np.isfinite(face_rel) & np.isfinite(face_cov)
    if not np.any(candidate):
        return {"status": "error", "error": "no valid candidate faces for patch experiment"}
    seeds: List[Tuple[str, int]] = []
    order_high_jac = np.argsort(np.where(candidate, face_rel, -np.inf))[::-1]
    order_high_cov = np.argsort(np.where(candidate, face_cov, -np.inf))[::-1]
    control_score = np.where(candidate, face_rel + face_cov, np.inf)
    order_good = np.argsort(control_score)
    for name, order in [("high_jac", order_high_jac), ("high_cov", order_high_cov), ("control_good", order_good)]:
        taken = 0
        for fid in order.tolist():
            if fid < 0 or fid >= len(face_valid) or not bool(candidate[fid]):
                continue
            seeds.append((name, int(fid)))
            taken += 1
            if taken >= 3:
                break
    results: List[Dict[str, Any]] = []
    solve_cfg = dict(ctx["cfg"].get("solve", {}))
    lambda_smooth = float(solve_cfg.get("lambda_smooth", 1e-3))
    ridge_eps = max(float(solve_cfg.get("ridge_eps", 1e-8)), 1e-8)
    for category, seed_face in seeds:
        for patch_size in (128, 512):
            patch_faces = _patch_faces_from_seed(
                seed_face=seed_face,
                target_size=patch_size,
                neighbors=neighbors,
                face_valid_mask=face_valid,
                face_island_labels=ctx["island_labels"],
            )
            if patch_faces.size == 0:
                results.append({"status": "error", "category": category, "seed_face": int(seed_face), "patch_size": int(patch_size), "error": "empty_patch"})
                continue
            submesh, global_vid, patch_face_ids = _extract_submesh_for_faces(ctx["solve_mesh"], patch_faces)
            local_face_jac = ctx["target_jac"][patch_face_ids]
            local_face_valid = face_valid[patch_face_ids]
            local_face_weights = np.maximum(ctx["face_weights"][patch_face_ids], 1e-8)
            A, rhs_u, rhs_v, _ = _build_gradient_constraint_system(
                mesh=submesh,
                face_jac=local_face_jac,
                face_weights=local_face_weights,
                face_valid_mask=local_face_valid,
            )
            if A.shape[0] == 0:
                results.append({"status": "error", "category": category, "seed_face": int(seed_face), "patch_size": int(patch_size), "error": "no_gradient_constraints"})
                continue
            boundary_local = _boundary_vertex_ids(submesh)
            boundary_uv = np.asarray(ctx["internal"].anchor_vertex_target_uv[global_vid], dtype=np.float64)
            finite_boundary = np.isfinite(boundary_uv).all(axis=1)
            dirichlet_local = boundary_local[finite_boundary[boundary_local]]
            if dirichlet_local.size < 3:
                results.append({"status": "error", "category": category, "seed_face": int(seed_face), "patch_size": int(patch_size), "error": "insufficient_boundary_constraints"})
                continue
            M = (A.T @ A).tocsr()
            if lambda_smooth > 0.0:
                L = mesh_laplacian(
                    np.asarray(submesh.faces, dtype=np.int64),
                    int(len(submesh.vertices)),
                    vertices=np.asarray(submesh.vertices, dtype=np.float64),
                    mode="cotan",
                )
                M = (M + lambda_smooth * L).tocsr()
            M = (M + ridge_eps * diags(np.ones((int(len(submesh.vertices)),), dtype=np.float64))).tocsr()
            sol_u, meta_u = _solve_spd_with_dirichlet(
                M=M,
                rhs=np.asarray(A.T @ rhs_u, dtype=np.float64),
                dirichlet_ids=dirichlet_local,
                dirichlet_values=boundary_uv[dirichlet_local, 0],
                resolved_device=ctx["internal"].resolved_device,
                solve_cfg=solve_cfg,
                channel_name=f"patch_u_{seed_face}_{patch_size}",
            )
            sol_v, meta_v = _solve_spd_with_dirichlet(
                M=M,
                rhs=np.asarray(A.T @ rhs_v, dtype=np.float64),
                dirichlet_ids=dirichlet_local,
                dirichlet_values=boundary_uv[dirichlet_local, 1],
                resolved_device=ctx["internal"].resolved_device,
                solve_cfg=solve_cfg,
                channel_name=f"patch_v_{seed_face}_{patch_size}",
            )
            patch_uv = np.stack([sol_u, sol_v], axis=1).astype(np.float64)
            patch_pinv, _ = _compute_face_geometry_pinv(submesh)
            solved_jac = _compute_face_jacobians(patch_pinv, submesh, patch_uv)
            jac_summary, _, _, _ = _jacobian_diagnostics(local_face_jac, solved_jac, local_face_valid)
            global_to_local_face = {int(fid): idx for idx, fid in enumerate(patch_face_ids.tolist())}
            sample_mask = np.isin(np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64), patch_face_ids)
            sample_faces_global = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64)[sample_mask]
            sample_face_local = np.asarray([global_to_local_face[int(fid)] for fid in sample_faces_global], dtype=np.int64)
            patch_sample_summary = _sample_residual_summary_from_arrays(
                mesh=submesh,
                vertex_uv=patch_uv,
                sample_face_ids=sample_face_local,
                sample_bary=np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64)[sample_mask],
                target_uv=np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64)[sample_mask],
            )
            results.append(
                {
                    "status": "ok",
                    "category": category,
                    "seed_face": int(seed_face),
                    "patch_size": int(patch_size),
                    "patch_faces": int(patch_face_ids.size),
                    "dirichlet_boundary_vertices": int(dirichlet_local.size),
                    "patch_stretch_p95": _quality_with_context(submesh, patch_uv).get("uv_stretch_p95"),
                    "patch_jac_rel_p95": jac_summary.get("frob_rel_error_p95"),
                    "patch_jac_cos_p05": jac_summary.get("cosine_p05"),
                    "patch_reproj_l1": patch_sample_summary.get("residual_l2_mean"),
                    "solve_backend_u": meta_u.get("backend"),
                    "solve_backend_v": meta_v.get("backend"),
                }
            )
    return {"status": "ok", "patches": results}


def _run_exp5_contributor_mixture(ctx: Dict[str, Any]) -> Dict[str, Any]:
    sample_face_ids = np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64)
    target_face_ids = np.asarray(ctx["internal"].solve_sample_target_face_ids, dtype=np.int64)
    sample_weights = np.asarray(ctx["internal"].solve_sample_weights, dtype=np.float64)
    fallback_mask = np.asarray(ctx["internal"].solve_sample_fallback_mask, dtype=np.bool_)
    target_high_island = np.asarray(ctx["internal"].solve_sample_target_high_island, dtype=np.int64)
    target_uv = np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64)
    high_face_jac, _ = _compute_high_face_jacobians(ctx["high_mesh"], ctx["high_uv"])
    samples_by_face: Dict[int, List[int]] = defaultdict(list)
    for idx, fid in enumerate(sample_face_ids.tolist()):
        samples_by_face[int(fid)].append(int(idx))
    rel = ctx["face_rel_err"]
    cov = ctx["face_target_cov_norm"]
    valid = ctx["face_valid"] & np.isfinite(rel) & np.isfinite(cov) & (ctx["face_accepted_samples"] > 0)
    if not np.any(valid):
        return {"status": "error", "error": "no valid faces for contributor mixture audit"}
    rel_p95 = float(np.quantile(rel[valid], 0.95))
    rel_p50 = float(np.quantile(rel[valid], 0.50))
    cov_p95 = float(np.quantile(cov[valid], 0.95))
    cov_p50 = float(np.quantile(cov[valid], 0.50))
    categories = {
        "high_jac_high_cov": valid & (rel >= rel_p95) & (cov >= cov_p95),
        "high_jac_low_cov": valid & (rel >= rel_p95) & (cov <= cov_p50),
        "low_jac_high_cov": valid & (rel <= rel_p50) & (cov >= cov_p95),
        "control_good": valid & (rel <= rel_p50) & (cov <= cov_p50),
    }
    out_categories: Dict[str, Any] = {}
    for name, mask in categories.items():
        face_ids = np.where(mask)[0]
        if name == "control_good":
            order = face_ids[np.argsort((rel[face_ids] + cov[face_ids]))]
        else:
            order = face_ids[np.argsort((rel[face_ids] + cov[face_ids]))[::-1]]
        entries: List[Dict[str, Any]] = []
        for fid in order[:128].tolist():
            idx = np.asarray(samples_by_face.get(int(fid), []), dtype=np.int64)
            if idx.size == 0:
                continue
            tf = target_face_ids[idx]
            w = np.maximum(sample_weights[idx], 1e-12)
            p = w / max(float(np.sum(w)), 1e-12)
            jac = high_face_jac[np.clip(tf, 0, len(high_face_jac) - 1)]
            flat = jac.reshape(jac.shape[0], -1)
            mean = np.sum(flat * p[:, None], axis=0)
            mean_norm = max(float(np.linalg.norm(mean)), 1e-12)
            dispersion = float(np.sum(np.linalg.norm(flat - mean[None, :], axis=1) * p) / mean_norm)
            sample_norms = np.linalg.norm(flat, axis=1)
            entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-12))))
            uv_span = float(np.linalg.norm(np.max(target_uv[idx], axis=0) - np.min(target_uv[idx], axis=0)))
            entries.append(
                {
                    "face_id": int(fid),
                    "accepted_sample_count": int(idx.size),
                    "unique_high_face_count": int(len(np.unique(tf))),
                    "unique_high_island_count": int(len(np.unique(target_high_island[idx][target_high_island[idx] >= 0]))),
                    "fallback_ratio": float(np.mean(fallback_mask[idx])),
                    "sample_weight_entropy": entropy,
                    "jacobian_sample_dispersion": dispersion,
                    "jacobian_scale_cv": float(np.std(sample_norms) / max(float(np.mean(sample_norms)), 1e-12)),
                    "target_uv_span": uv_span,
                }
            )
        out_categories[name] = {
            "face_count": int(face_ids.size),
            "thresholds": {
                "rel_p95": rel_p95,
                "rel_p50": rel_p50,
                "cov_p95": cov_p95,
                "cov_p50": cov_p50,
            },
            "summary": {
                "unique_high_island_count_p95": _weighted_quantile(
                    np.asarray([e["unique_high_island_count"] for e in entries], dtype=np.float64), 0.95
                ),
                "fallback_ratio_mean": float(np.mean([e["fallback_ratio"] for e in entries])) if entries else None,
                "sample_weight_entropy_p95": _weighted_quantile(
                    np.asarray([e["sample_weight_entropy"] for e in entries], dtype=np.float64), 0.95
                ),
                "jacobian_sample_dispersion_p95": _weighted_quantile(
                    np.asarray([e["jacobian_sample_dispersion"] for e in entries], dtype=np.float64), 0.95
                ),
            },
            "faces": entries,
        }
    return {"status": "ok", "categories": out_categories}


def _run_exp6_pre_post_align(ctx: Dict[str, Any]) -> Dict[str, Any]:
    pre_uv = np.asarray(ctx["internal"].mapped_uv_pre_align, dtype=np.float64)
    post_uv = np.asarray(ctx["solve_uv"], dtype=np.float64)
    solved_pre = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], pre_uv)
    solved_post = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], post_uv)
    jac_pre, _, _, _ = _jacobian_diagnostics(ctx["target_jac"], solved_pre, ctx["face_valid"])
    jac_post, _, _, _ = _jacobian_diagnostics(ctx["target_jac"], solved_post, ctx["face_valid"])
    sample_pre = _sample_residual_summary(ctx["internal"], pre_uv)
    sample_post = _sample_residual_summary(ctx["internal"], post_uv)
    quality_pre = _quality_with_context(ctx["solve_mesh"], pre_uv)
    quality_post = _quality_with_context(ctx["solve_mesh"], post_uv)
    shift_norm = float(np.linalg.norm(np.asarray(ctx["internal"].post_align_shift, dtype=np.float64)))
    max_shift = float(ctx["internal"].post_align_max_shift)
    return {
        "status": "ok",
        "pre_align_metrics": {
            "quality": quality_pre,
            "jacobian": jac_pre,
            "sample_residual": sample_pre,
        },
        "post_align_metrics": {
            "quality": quality_post,
            "jacobian": jac_post,
            "sample_residual": sample_post,
        },
        "delta": {
            "quality": _numeric_deltas(
                quality_pre,
                quality_post,
                ["uv_stretch_p95", "uv_stretch_p99", "uv_bad_tri_ratio_stretch_only", "uv_out_of_bounds_ratio"],
            ),
            "jacobian": _numeric_deltas(
                jac_pre,
                jac_post,
                ["frob_rel_error_p95", "cosine_p05", "log_area_ratio_p95"],
            ),
            "sample_residual": _numeric_deltas(
                sample_pre,
                sample_post,
                ["residual_l2_mean", "residual_l2_p95", "residual_linf"],
            ),
        },
        "post_align_shift_norm": shift_norm,
        "post_align_max_shift": max_shift,
        "post_align_clip_hit": bool(max_shift > 0.0 and shift_norm >= max_shift - 1e-8),
    }


def _run_exact_field_case(
    *,
    ctx: Dict[str, Any],
    case_name: str,
    source_uv: np.ndarray,
    fill_vertex_count: int = 0,
) -> Dict[str, Any]:
    source_uv_np = np.asarray(source_uv, dtype=np.float64)
    target_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], source_uv_np)
    face_valid = np.isfinite(target_jac).all(axis=(1, 2))
    face_weights = np.ones((len(target_jac),), dtype=np.float64)
    face_active = np.ones((len(target_jac),), dtype=np.bool_)
    anchor_conf = np.ones((len(source_uv_np),), dtype=np.float64)
    face_alpha = np.ones((len(target_jac),), dtype=np.float64)
    modes = {
        "per_island": np.asarray(ctx["island_labels"], dtype=np.int64),
        "global_solve": None,
    }
    results: Dict[str, Any] = {}
    for mode_name, labels in modes.items():
        solve_result = _solve_target_field(
            solve_mesh=ctx["solve_mesh"],
            high_mesh=ctx["high_mesh"],
            high_uv=ctx["high_uv"],
            resolved_device=ctx["internal"].resolved_device,
            cfg=ctx["cfg"],
            face_jac=target_jac,
            face_weights=face_weights,
            face_valid_mask=face_valid,
            face_active_mask=face_active,
            face_island_labels=labels,
            anchor_vertex_target_uv=source_uv_np,
            anchor_vertex_confidence=anchor_conf,
            face_smooth_alpha=face_alpha,
        )
        mapped_uv = solve_result["mapped_uv"]
        solved_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], mapped_uv)
        jac_summary, _, _, _ = _jacobian_diagnostics(target_jac, solved_jac, face_valid)
        uv_l2 = np.linalg.norm(mapped_uv - source_uv_np, axis=1)
        results[mode_name] = {
            "solve_success": True,
            "fill_vertex_count": int(fill_vertex_count),
            "uv_l2_mean": float(np.mean(uv_l2)),
            "uv_l2_p95": float(np.quantile(uv_l2, 0.95)),
            "stretch_p95": _quality_with_context(ctx["solve_mesh"], mapped_uv).get("uv_stretch_p95"),
            "jac_rel_p95": jac_summary.get("frob_rel_error_p95"),
            "cond_proxy": solve_result["solve_meta"].get("uv_m2_system_cond_proxy"),
            "residual_u": solve_result["solve_meta"].get("uv_solver_residual_u"),
            "residual_v": solve_result["solve_meta"].get("uv_solver_residual_v"),
            "reproj_l1": None,
        }
    return {"status": "ok", "case_name": case_name, "modes": results}


def _run_exp7_synthetic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    solve_uv = np.asarray(ctx["solve_uv"], dtype=np.float64)
    projected_uv = np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64)
    finite = np.isfinite(projected_uv).all(axis=1)
    fill_count = int(np.count_nonzero(~finite))
    if np.any(~finite):
        projected_uv[~finite] = solve_uv[~finite]
    identity_case: Dict[str, Any]
    high_mesh = ctx["high_mesh"]
    solve_mesh = ctx["solve_mesh"]
    if (
        int(len(high_mesh.vertices)) == int(len(solve_mesh.vertices))
        and int(len(high_mesh.faces)) == int(len(solve_mesh.faces))
        and np.allclose(np.asarray(high_mesh.vertices, dtype=np.float64), np.asarray(solve_mesh.vertices, dtype=np.float64))
        and np.array_equal(np.asarray(high_mesh.faces, dtype=np.int64), np.asarray(solve_mesh.faces, dtype=np.int64))
    ):
        identity_case = _run_exact_field_case(ctx=ctx, case_name="identity_low_equals_high", source_uv=ctx["high_uv"])
    else:
        identity_case = {"status": "skipped", "case_name": "identity_low_equals_high", "reason": "input_meshes_differ"}
    return {
        "status": "ok",
        "cases": [
            _run_exact_field_case(ctx=ctx, case_name="exact_field_reinject_current_low", source_uv=solve_uv),
            _run_exact_field_case(
                ctx=ctx,
                case_name="projected_vertex_uv_reinject",
                source_uv=projected_uv,
                fill_vertex_count=fill_count,
            ),
            identity_case,
        ],
    }


def _run_exp8_target_vs_anchor(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    anchor_face_valid = (
        np.asarray(ctx["face_valid"], dtype=np.bool_)
        & np.isfinite(anchor_jac).all(axis=(1, 2))
    )
    compare_summary, gap_rel, gap_cos, gap_log_area = _compare_jacobian_fields(
        np.asarray(ctx["target_jac"], dtype=np.float64),
        anchor_jac,
        anchor_face_valid,
    )
    mix = ctx["sample_mix_arrays"]
    face_rel = np.asarray(ctx["face_rel_err"], dtype=np.float64)
    cov_norm = np.asarray(ctx["face_target_cov_norm"], dtype=np.float64)
    accepted = np.asarray(ctx["face_accepted_samples"], dtype=np.int64)
    component_sizes = np.asarray(ctx["component_sizes"], dtype=np.int64)
    valid = anchor_face_valid & np.isfinite(gap_rel) & np.isfinite(face_rel) & np.isfinite(cov_norm)
    corr = None
    if np.count_nonzero(valid) >= 8:
        corr = float(np.corrcoef(gap_rel[valid], face_rel[valid])[0, 1])
    p95_err = float(np.quantile(face_rel[valid], 0.95)) if np.any(valid) else None
    p50_err = float(np.quantile(face_rel[valid], 0.50)) if np.any(valid) else None
    p95_cov = float(np.quantile(cov_norm[valid], 0.95)) if np.any(valid) else None
    strata_defs = {
        "high_final_error": valid & (face_rel >= (p95_err if p95_err is not None else np.inf)),
        "control_low_error": valid & (face_rel <= (p50_err if p50_err is not None else -np.inf)),
        "high_cov": valid & (cov_norm >= (p95_cov if p95_cov is not None else np.inf)),
    }
    strata: Dict[str, Any] = {}
    for name, mask in strata_defs.items():
        idx = np.where(mask)[0]
        strata[name] = {
            "face_count": int(idx.size),
            "target_anchor_gap_p50": float(np.quantile(gap_rel[idx], 0.50)) if idx.size > 0 else None,
            "target_anchor_gap_p95": float(np.quantile(gap_rel[idx], 0.95)) if idx.size > 0 else None,
            "gap_cosine_p05": float(np.quantile(gap_cos[idx], 0.05)) if idx.size > 0 else None,
            "gap_log_area_ratio_p95": float(np.quantile(gap_log_area[idx], 0.95)) if idx.size > 0 else None,
            "fallback_ratio_mean": float(np.nanmean(mix["fallback_ratio"][idx])) if idx.size > 0 else None,
            "accepted_samples_p50": float(np.quantile(accepted[idx], 0.50)) if idx.size > 0 else None,
            "component_size_p50": float(np.quantile(component_sizes[idx], 0.50)) if idx.size > 0 else None,
        }
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(gap_rel[valid_idx])[::-1]]
    top_faces: List[Dict[str, Any]] = []
    for fid in order[:20].tolist():
        top_faces.append(
            {
                "face_id": int(fid),
                "target_anchor_gap_rel": float(gap_rel[fid]),
                "target_anchor_cosine": float(gap_cos[fid]) if math.isfinite(float(gap_cos[fid])) else None,
                "target_anchor_log_area_ratio": float(gap_log_area[fid]) if math.isfinite(float(gap_log_area[fid])) else None,
                "current_solve_rel_error": float(face_rel[fid]),
                "cov_norm": float(cov_norm[fid]) if math.isfinite(float(cov_norm[fid])) else None,
                "fallback_ratio": float(mix["fallback_ratio"][fid]) if math.isfinite(float(mix["fallback_ratio"][fid])) else None,
                "accepted_sample_count": int(accepted[fid]),
                "unique_high_face_count": int(mix["unique_high_face_count"][fid]),
                "unique_high_island_count": int(mix["unique_high_island_count"][fid]),
                "component_size": int(component_sizes[fid]),
            }
        )
    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "anchor_uv_quality": {
            k: v
            for k, v in _quality_with_context(ctx["solve_mesh"], anchor_uv).items()
            if k in {"uv_stretch_p95", "uv_stretch_p99", "uv_bad_tri_ratio_stretch_only", "uv_out_of_bounds_ratio"}
        },
        "anchor_sample_residual": _sample_residual_summary_from_arrays(
            mesh=ctx["solve_mesh"],
            vertex_uv=anchor_uv,
            sample_face_ids=np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64),
            sample_bary=np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64),
            target_uv=np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64),
        ),
        "target_vs_anchor_jacobian": compare_summary,
        "gap_correlation_with_current_solve_rel_error": corr,
        "strata": strata,
        "top_faces": top_faces,
    }


def _run_exp9_fallback_ablation(ctx: Dict[str, Any]) -> Dict[str, Any]:
    current_weights = np.asarray(ctx["internal"].solve_sample_weights, dtype=np.float64)
    fallback_mask = np.asarray(ctx["internal"].solve_sample_fallback_mask, dtype=np.bool_)
    current_fallback_weight = float(ctx["cfg"].get("correspondence", {}).get("fallback_weight", 0.7))
    downweight_target = min(0.25, current_fallback_weight)
    if current_fallback_weight > 1e-12:
        downweight_scale = downweight_target / current_fallback_weight
    else:
        downweight_scale = 0.0

    variants = [
        {
            "name": "current",
            "sample_mask": np.ones((current_weights.shape[0],), dtype=np.bool_),
            "sample_weights": current_weights.copy(),
            "note": "baseline weights from method2 runtime",
        },
        {
            "name": "primary_only",
            "sample_mask": ~fallback_mask,
            "sample_weights": current_weights.copy(),
            "note": "drop all fallback samples before face Jacobian aggregation",
        },
        {
            "name": f"fallback_weight_{downweight_target:.2f}",
            "sample_mask": np.ones((current_weights.shape[0],), dtype=np.bool_),
            "sample_weights": np.where(fallback_mask, current_weights * downweight_scale, current_weights),
            "note": "keep fallback samples but downweight them before aggregation",
        },
    ]
    out_variants: List[Dict[str, Any]] = []
    for variant in variants:
        agg = _aggregate_face_field_variant(
            ctx=ctx,
            sample_weights=np.asarray(variant["sample_weights"], dtype=np.float64),
            sample_mask=np.asarray(variant["sample_mask"], dtype=np.bool_),
        )
        compare_summary, _, _, _ = _compare_jacobian_fields(
            np.asarray(ctx["target_jac"], dtype=np.float64),
            np.asarray(agg["face_jac"], dtype=np.float64),
            np.asarray(ctx["face_valid"], dtype=np.bool_) & np.asarray(agg["face_valid"], dtype=np.bool_),
        )
        solve_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=str(variant["name"]),
            face_jac=np.asarray(agg["face_jac"], dtype=np.float64),
            face_weights=np.asarray(agg["face_weights"], dtype=np.float64),
            face_valid=np.asarray(agg["face_valid"], dtype=np.bool_),
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=np.asarray(ctx["target_jac"], dtype=np.float64),
        )
        out_variants.append(
            {
                "status": "ok",
                "variant": str(variant["name"]),
                "note": str(variant["note"]),
                "sample_keep_count": int(agg["sample_keep_count"]),
                "sample_keep_ratio": float(agg["sample_keep_count"] / max(1, current_weights.shape[0])),
                "fallback_sample_ratio_retained": float(np.mean(fallback_mask[np.asarray(variant["sample_mask"], dtype=np.bool_)]))
                if np.any(np.asarray(variant["sample_mask"], dtype=np.bool_))
                else 0.0,
                "field_valid_face_count": int(np.count_nonzero(agg["face_valid"])),
                "field_valid_face_ratio": float(np.mean(np.asarray(agg["face_valid"], dtype=np.bool_))),
                "aggregate_meta": agg["aggregate_meta"],
                "field_vs_current_target": compare_summary,
                "curl_summary": _curl_global_summary(
                    ctx["solve_mesh"],
                    np.asarray(agg["face_jac"], dtype=np.float64),
                    np.asarray(agg["face_valid"], dtype=np.bool_),
                ),
                "solve_summary": solve_summary,
            }
        )
    return {
        "status": "ok",
        "current_fallback_weight": current_fallback_weight,
        "downweight_target": downweight_target,
        "variants": out_variants,
    }


def _run_exp10_anchor_residual(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    base_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    base_face_valid = (
        np.asarray(ctx["face_valid"], dtype=np.bool_)
        & np.isfinite(base_jac).all(axis=(1, 2))
    )
    base_compare, _, _, _ = _compare_jacobian_fields(
        np.asarray(ctx["target_jac"], dtype=np.float64),
        base_jac,
        base_face_valid,
    )
    residual_jac = np.asarray(ctx["target_jac"], dtype=np.float64) - base_jac
    residual_valid = base_face_valid & np.isfinite(residual_jac).all(axis=(1, 2))
    zero_anchor_uv = np.zeros_like(anchor_uv, dtype=np.float64)
    anchor_conf = np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64)
    cfg_override = {"method2": {"anchor_confidence_floor": 0.0}}
    residual_variants = [
        _solve_custom_field_summary(
            ctx=ctx,
            variant_name="anchor_plus_residual_per_island",
            face_jac=residual_jac,
            face_weights=np.asarray(ctx["face_weights"], dtype=np.float64),
            face_valid=residual_valid,
            anchor_vertex_target_uv=zero_anchor_uv,
            anchor_vertex_confidence=anchor_conf,
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=np.asarray(ctx["target_jac"], dtype=np.float64),
            cfg_override=cfg_override,
        ),
        _solve_custom_field_summary(
            ctx=ctx,
            variant_name="anchor_plus_residual_global",
            face_jac=residual_jac,
            face_weights=np.asarray(ctx["face_weights"], dtype=np.float64),
            face_valid=residual_valid,
            anchor_vertex_target_uv=zero_anchor_uv,
            anchor_vertex_confidence=anchor_conf,
            face_island_labels=None,
            compare_target_jac=np.asarray(ctx["target_jac"], dtype=np.float64),
            cfg_override=cfg_override,
        ),
    ]
    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "base_only": {
            "quality": {
                k: v
                for k, v in _quality_with_context(ctx["solve_mesh"], anchor_uv).items()
                if k in {"uv_stretch_p95", "uv_stretch_p99", "uv_bad_tri_ratio_stretch_only", "uv_out_of_bounds_ratio"}
            },
            "sample_residual": _sample_residual_summary_from_arrays(
                mesh=ctx["solve_mesh"],
                vertex_uv=anchor_uv,
                sample_face_ids=np.asarray(ctx["internal"].solve_sample_face_ids, dtype=np.int64),
                sample_bary=np.asarray(ctx["internal"].solve_sample_bary, dtype=np.float64),
                target_uv=np.asarray(ctx["internal"].solve_target_uv, dtype=np.float64),
            ),
            "target_vs_base_jacobian": base_compare,
        },
        "residual_target_field": {
            "valid_face_count": int(np.count_nonzero(residual_valid)),
            "curl_summary": _curl_global_summary(ctx["solve_mesh"], residual_jac, residual_valid),
        },
        "variants": residual_variants,
    }


def _run_exp11_field_builder_variants(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    anchor_valid = np.asarray(ctx["face_valid"], dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))
    samplefit_raw = _fit_face_sample_jacobian_field(ctx=ctx, anchor_uv_prior=None, prior_weight=0.0, min_samples=3)
    samplefit_ridge025 = _fit_face_sample_jacobian_field(
        ctx=ctx,
        anchor_uv_prior=anchor_uv,
        prior_weight=0.25,
        min_samples=3,
    )
    samplefit_ridge1 = _fit_face_sample_jacobian_field(
        ctx=ctx,
        anchor_uv_prior=anchor_uv,
        prior_weight=1.0,
        min_samples=3,
    )
    current_jac = np.asarray(ctx["target_jac"], dtype=np.float64)
    current_valid = np.asarray(ctx["face_valid"], dtype=np.bool_)
    sample_count = np.asarray(samplefit_raw["face_sample_count"], dtype=np.int64)

    variants: List[Tuple[str, np.ndarray, np.ndarray, str]] = []
    variants.append(("current_reference", current_jac.copy(), current_valid.copy(), "baseline current target_jac"))
    variants.append(
        (
            "samplefit_raw_ge3",
            np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "pure low-face sample-fit Jacobian on faces with >=3 samples",
        )
    )
    variants.append(
        (
            "samplefit_anchor_ridge0.25_ge3",
            np.asarray(samplefit_ridge025["face_jac"], dtype=np.float64),
            np.asarray(samplefit_ridge025["face_valid"], dtype=np.bool_),
            "sample-fit with anchor UV ridge prior weight 0.25 on faces with >=3 samples",
        )
    )
    variants.append(
        (
            "samplefit_anchor_ridge1.0_ge3",
            np.asarray(samplefit_ridge1["face_jac"], dtype=np.float64),
            np.asarray(samplefit_ridge1["face_valid"], dtype=np.bool_),
            "sample-fit with anchor UV ridge prior weight 1.0 on faces with >=3 samples",
        )
    )

    hybrid_ge4_raw_jac, hybrid_ge4_raw_valid = _blend_face_fields(
        current_jac,
        current_valid,
        np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
        np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
        sample_count >= 4,
    )
    variants.append(
        (
            "hybrid_ge4_raw_else_current",
            hybrid_ge4_raw_jac,
            hybrid_ge4_raw_valid,
            "replace only faces with >=4 samples by raw sample-fit Jacobian",
        )
    )

    hybrid_ge4_raw_eq3_r025_jac = hybrid_ge4_raw_jac.copy()
    hybrid_ge4_raw_eq3_r025_valid = hybrid_ge4_raw_valid.copy()
    mask_eq3 = sample_count == 3
    replace025 = mask_eq3 & np.asarray(samplefit_ridge025["face_valid"], dtype=np.bool_)
    if np.any(replace025):
        hybrid_ge4_raw_eq3_r025_jac[replace025] = np.asarray(samplefit_ridge025["face_jac"], dtype=np.float64)[replace025]
        hybrid_ge4_raw_eq3_r025_valid[replace025] = True
    variants.append(
        (
            "hybrid_ge4_raw_eq3_ridge0.25_else_current",
            hybrid_ge4_raw_eq3_r025_jac,
            hybrid_ge4_raw_eq3_r025_valid,
            ">=4 faces use raw sample-fit; 3-sample faces use anchor-ridge(0.25); others current",
        )
    )

    hybrid_ge4_raw_eq3_r1_jac = hybrid_ge4_raw_jac.copy()
    hybrid_ge4_raw_eq3_r1_valid = hybrid_ge4_raw_valid.copy()
    replace1 = mask_eq3 & np.asarray(samplefit_ridge1["face_valid"], dtype=np.bool_)
    if np.any(replace1):
        hybrid_ge4_raw_eq3_r1_jac[replace1] = np.asarray(samplefit_ridge1["face_jac"], dtype=np.float64)[replace1]
        hybrid_ge4_raw_eq3_r1_valid[replace1] = True
        hybrid_ge4_raw_eq3_r1_valid[replace1] = True
    variants.append(
        (
            "hybrid_ge4_raw_eq3_ridge1.0_else_current",
            hybrid_ge4_raw_eq3_r1_jac,
            hybrid_ge4_raw_eq3_r1_valid,
            ">=4 faces use raw sample-fit; 3-sample faces use anchor-ridge(1.0); others current",
        )
    )

    out_variants: List[Dict[str, Any]] = []
    for name, field_jac, field_valid, note in variants:
        valid = np.asarray(field_valid, dtype=np.bool_)
        weights = _default_variant_face_weights(ctx, valid)
        compare_current, _, _, _ = _compare_jacobian_fields(
            current_jac,
            np.asarray(field_jac, dtype=np.float64),
            current_valid & valid,
        )
        compare_anchor, _, _, _ = _compare_jacobian_fields(
            anchor_jac,
            np.asarray(field_jac, dtype=np.float64),
            anchor_valid & valid,
        )
        solve_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=name,
            face_jac=np.asarray(field_jac, dtype=np.float64),
            face_weights=weights,
            face_valid=valid,
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=current_jac,
        )
        out_variants.append(
            {
                "status": "ok",
                "variant": name,
                "note": note,
                "field_valid_face_count": int(np.count_nonzero(valid)),
                "field_valid_face_ratio": float(np.mean(valid)),
                "field_sample_explain": _field_sample_explain_summary(ctx, np.asarray(field_jac, dtype=np.float64)),
                "field_curl_summary": _curl_global_summary(ctx["solve_mesh"], np.asarray(field_jac, dtype=np.float64), valid),
                "field_vs_current_target": compare_current,
                "field_vs_anchor": compare_anchor,
                "solve_summary": solve_summary,
            }
        )
    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "samplefit_raw_face_count": int(np.count_nonzero(np.asarray(samplefit_raw["face_valid"], dtype=np.bool_))),
        "samplefit_ridge025_face_count": int(np.count_nonzero(np.asarray(samplefit_ridge025["face_valid"], dtype=np.bool_))),
        "samplefit_ridge1_face_count": int(np.count_nonzero(np.asarray(samplefit_ridge1["face_valid"], dtype=np.bool_))),
        "sample_count_summary": {
            "eq3_count": int(np.count_nonzero(sample_count == 3)),
            "ge4_count": int(np.count_nonzero(sample_count >= 4)),
        },
        "variants": out_variants,
    }


def _run_exp12_field_projection(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    base_valid = np.asarray(ctx["face_active"], dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))
    current_jac = np.asarray(ctx["target_jac"], dtype=np.float64)
    current_valid = np.asarray(ctx["face_valid"], dtype=np.bool_)
    samplefit_raw = _fit_face_sample_jacobian_field(ctx=ctx, anchor_uv_prior=None, prior_weight=0.0, min_samples=3)

    conf_current = _projection_confidence_current(ctx)
    conf_samplefit = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=False)
    conf_samplefit_strict = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=True)

    variants_cfg = [
        {
            "name": "project_current_conf_curl5",
            "note": "project current target field toward anchor base with covariance/fallback-driven trust and lambda_curl=5",
            "data_jac": current_jac,
            "data_valid": current_valid,
            "confidence": conf_current,
            "lambda_base": 1.0,
            "lambda_curl": 5.0,
        },
        {
            "name": "project_samplefit_conf_curl5",
            "note": "project raw sample-fit field toward anchor base with soft trust and lambda_curl=5",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit,
            "lambda_base": 1.0,
            "lambda_curl": 5.0,
        },
        {
            "name": "project_samplefit_strict_curl5",
            "note": "project raw sample-fit field with strict >=4-sample low-fallback gate and lambda_curl=5",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit_strict,
            "lambda_base": 1.0,
            "lambda_curl": 5.0,
        },
        {
            "name": "project_samplefit_strict_curl20",
            "note": "same strict-gated sample-fit projector with stronger field smoothness lambda_curl=20",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit_strict,
            "lambda_base": 1.0,
            "lambda_curl": 20.0,
        },
    ]

    out_variants: List[Dict[str, Any]] = []
    for variant in variants_cfg:
        projected = _project_face_jacobian_field(
            ctx=ctx,
            data_jac=np.asarray(variant["data_jac"], dtype=np.float64),
            data_valid=np.asarray(variant["data_valid"], dtype=np.bool_),
            base_jac=anchor_jac,
            base_valid=base_valid,
            face_confidence=np.asarray(variant["confidence"], dtype=np.float64),
            lambda_base=float(variant["lambda_base"]),
            lambda_curl=float(variant["lambda_curl"]),
        )
        projected_valid = np.asarray(projected["face_valid"], dtype=np.bool_)
        weights = _default_variant_face_weights(ctx, projected_valid)
        data_valid = np.asarray(variant["data_valid"], dtype=np.bool_)
        compare_current, _, _, _ = _compare_jacobian_fields(
            current_jac,
            np.asarray(projected["face_jac"], dtype=np.float64),
            current_valid & projected_valid,
        )
        compare_anchor, _, _, _ = _compare_jacobian_fields(
            anchor_jac,
            np.asarray(projected["face_jac"], dtype=np.float64),
            base_valid & projected_valid,
        )
        compare_data, _, _, _ = _compare_jacobian_fields(
            np.asarray(variant["data_jac"], dtype=np.float64),
            np.asarray(projected["face_jac"], dtype=np.float64),
            data_valid & projected_valid,
        )
        solve_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=str(variant["name"]),
            face_jac=np.asarray(projected["face_jac"], dtype=np.float64),
            face_weights=weights,
            face_valid=projected_valid,
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=current_jac,
            compare_target_valid_mask=current_valid & projected_valid,
        )
        conf_arr = np.asarray(projected["face_confidence"], dtype=np.float64)
        out_variants.append(
            {
                "status": "ok",
                "variant": str(variant["name"]),
                "note": str(variant["note"]),
                "lambda_base": float(variant["lambda_base"]),
                "lambda_curl": float(variant["lambda_curl"]),
                "projected_valid_face_count": int(np.count_nonzero(projected_valid)),
                "projected_valid_face_ratio": float(np.mean(projected_valid)),
                "confidence_mean": float(np.mean(conf_arr[projected_valid])) if np.any(projected_valid) else None,
                "confidence_p50": float(np.quantile(conf_arr[projected_valid], 0.50)) if np.any(projected_valid) else None,
                "confidence_p95": float(np.quantile(conf_arr[projected_valid], 0.95)) if np.any(projected_valid) else None,
                "field_sample_explain": _field_sample_explain_summary(ctx, np.asarray(projected["face_jac"], dtype=np.float64)),
                "field_curl_summary": _curl_global_summary(
                    ctx["solve_mesh"],
                    np.asarray(projected["face_jac"], dtype=np.float64),
                    projected_valid,
                ),
                "field_vs_data": compare_data,
                "field_vs_current_target": compare_current,
                "field_vs_anchor": compare_anchor,
                "projector_meta": {
                    "matrix": projected["matrix_meta"],
                    "solve": projected["solve_meta"],
                },
                "solve_summary": solve_summary,
            }
        )

    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "variants": out_variants,
    }


def _run_exp13_residual_projection(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    base_valid = np.asarray(ctx["face_active"], dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))
    current_jac = np.asarray(ctx["target_jac"], dtype=np.float64)
    current_valid = np.asarray(ctx["face_valid"], dtype=np.bool_)
    current_div = _field_divergence_proxy(
        mesh=ctx["solve_mesh"],
        face_jac=current_jac,
        face_weights=_default_variant_face_weights(ctx, current_valid),
        face_valid=current_valid,
    )
    samplefit_raw = _fit_face_sample_jacobian_field(ctx=ctx, anchor_uv_prior=None, prior_weight=0.0, min_samples=3)

    conf_current_soft = _projection_confidence_current(ctx, strict_gate=False)
    conf_current_strict = _projection_confidence_current(ctx, strict_gate=True)
    conf_samplefit_soft = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=False)
    conf_samplefit_strict = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=True)

    variants_cfg = [
        {
            "name": "residual_current_soft_curl5",
            "note": "residual projector on current target field with soft confidence and lambda_curl=5",
            "data_jac": current_jac,
            "data_valid": current_valid,
            "confidence": conf_current_soft,
            "lambda_decay": 1.0,
            "lambda_curl": 5.0,
        },
        {
            "name": "residual_current_soft_curl20",
            "note": "residual projector on current target field with soft confidence and lambda_curl=20",
            "data_jac": current_jac,
            "data_valid": current_valid,
            "confidence": conf_current_soft,
            "lambda_decay": 1.0,
            "lambda_curl": 20.0,
        },
        {
            "name": "residual_current_strict_curl20",
            "note": "residual projector on current target field with strict >=4-sample low-fallback gate and lambda_curl=20",
            "data_jac": current_jac,
            "data_valid": current_valid,
            "confidence": conf_current_strict,
            "lambda_decay": 1.0,
            "lambda_curl": 20.0,
        },
        {
            "name": "residual_samplefit_soft_curl5",
            "note": "residual projector on raw sample-fit field with soft confidence and lambda_curl=5",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit_soft,
            "lambda_decay": 1.0,
            "lambda_curl": 5.0,
        },
        {
            "name": "residual_samplefit_soft_curl20",
            "note": "residual projector on raw sample-fit field with soft confidence and lambda_curl=20",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit_soft,
            "lambda_decay": 1.0,
            "lambda_curl": 20.0,
        },
        {
            "name": "residual_samplefit_strict_curl20",
            "note": "residual projector on raw sample-fit field with strict >=4-sample low-fallback gate and lambda_curl=20",
            "data_jac": np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
            "data_valid": np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
            "confidence": conf_samplefit_strict,
            "lambda_decay": 1.0,
            "lambda_curl": 20.0,
        },
    ]

    out_variants: List[Dict[str, Any]] = []
    for variant in variants_cfg:
        projected = _project_face_jacobian_residual_field(
            ctx=ctx,
            data_jac=np.asarray(variant["data_jac"], dtype=np.float64),
            data_valid=np.asarray(variant["data_valid"], dtype=np.bool_),
            base_jac=anchor_jac,
            base_valid=base_valid,
            face_confidence=np.asarray(variant["confidence"], dtype=np.float64),
            lambda_decay=float(variant["lambda_decay"]),
            lambda_curl=float(variant["lambda_curl"]),
        )
        projected_valid = np.asarray(projected["face_valid"], dtype=np.bool_)
        weights = _default_variant_face_weights(ctx, projected_valid)
        data_valid = np.asarray(variant["data_valid"], dtype=np.bool_)

        compare_current, _, _, _ = _compare_jacobian_fields(
            current_jac,
            np.asarray(projected["face_jac"], dtype=np.float64),
            current_valid & projected_valid,
        )
        compare_anchor, _, _, _ = _compare_jacobian_fields(
            anchor_jac,
            np.asarray(projected["face_jac"], dtype=np.float64),
            base_valid & projected_valid,
        )
        compare_data, _, _, _ = _compare_jacobian_fields(
            np.asarray(variant["data_jac"], dtype=np.float64),
            np.asarray(projected["face_jac"], dtype=np.float64),
            data_valid & projected_valid,
        )
        compare_residual, _, _, _ = _compare_jacobian_fields(
            np.asarray(projected["residual_data_jac"], dtype=np.float64),
            np.asarray(projected["residual_jac"], dtype=np.float64),
            np.asarray(projected["residual_data_valid"], dtype=np.bool_) & np.asarray(projected["residual_valid"], dtype=np.bool_),
        )
        residual_curl = _curl_global_summary(
            ctx["solve_mesh"],
            np.asarray(projected["residual_jac"], dtype=np.float64),
            np.asarray(projected["residual_valid"], dtype=np.bool_),
        )
        final_curl = _curl_global_summary(
            ctx["solve_mesh"],
            np.asarray(projected["face_jac"], dtype=np.float64),
            projected_valid,
        )
        final_div = _field_divergence_proxy(
            mesh=ctx["solve_mesh"],
            face_jac=np.asarray(projected["face_jac"], dtype=np.float64),
            face_weights=weights,
            face_valid=projected_valid,
        )
        solve_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=str(variant["name"]),
            face_jac=np.asarray(projected["face_jac"], dtype=np.float64),
            face_weights=weights,
            face_valid=projected_valid,
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=current_jac,
            compare_target_valid_mask=current_valid & projected_valid,
            include_mapped_uv=True,
        )
        solve_uv = np.asarray(solve_summary.pop("mapped_uv"), dtype=np.float64)
        solve_oob_overlap = _high_oob_high_divergence_overlap(
            mesh=ctx["solve_mesh"],
            vertex_uv=solve_uv,
            face_rhs_norm=np.asarray(final_div["face_rhs_norm"], dtype=np.float64),
            face_mask=projected_valid,
        )
        conf_arr = np.asarray(projected["face_confidence"], dtype=np.float64)
        out_variants.append(
            {
                "status": "ok",
                "variant": str(variant["name"]),
                "note": str(variant["note"]),
                "lambda_decay": float(variant["lambda_decay"]),
                "lambda_curl": float(variant["lambda_curl"]),
                "projected_valid_face_count": int(np.count_nonzero(projected_valid)),
                "projected_valid_face_ratio": float(np.mean(projected_valid)),
                "confidence_mean": float(np.mean(conf_arr[projected_valid])) if np.any(projected_valid) else None,
                "confidence_p50": float(np.quantile(conf_arr[projected_valid], 0.50)) if np.any(projected_valid) else None,
                "confidence_p95": float(np.quantile(conf_arr[projected_valid], 0.95)) if np.any(projected_valid) else None,
                "field_sample_explain": _field_sample_explain_summary(ctx, np.asarray(projected["face_jac"], dtype=np.float64)),
                "field_curl_summary": final_curl,
                "residual_field_curl_summary": residual_curl,
                "field_divergence_summary": final_div["summary"],
                "field_vs_data": compare_data,
                "field_vs_current_target": compare_current,
                "field_vs_anchor": compare_anchor,
                "residual_vs_data": compare_residual,
                "divergence_shift_vs_current": _divergence_shift_summary(
                    np.asarray(current_div["face_rhs_norm"], dtype=np.float64),
                    np.asarray(final_div["face_rhs_norm"], dtype=np.float64),
                    projected_valid & current_valid,
                ),
                "current_divergence_summary": current_div["summary"],
                "solve_divergence_overlap_using_baseline_uv": solve_oob_overlap,
                "projector_meta": {
                    "matrix": projected["matrix_meta"],
                    "solve": projected["solve_meta"],
                },
                "solve_summary": solve_summary,
            }
        )

    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "current_divergence_summary": current_div["summary"],
        "variants": out_variants,
    }


def _run_exp14_solve_space_constraints(ctx: Dict[str, Any]) -> Dict[str, Any]:
    anchor_uv, fill_meta = _complete_anchor_vertex_uv(ctx)
    anchor_jac = _compute_face_jacobians(ctx["internal"].face_geom_pinv, ctx["solve_mesh"], anchor_uv)
    base_valid = np.asarray(ctx["face_active"], dtype=np.bool_) & np.isfinite(anchor_jac).all(axis=(1, 2))
    current_jac = np.asarray(ctx["target_jac"], dtype=np.float64)
    current_valid = np.asarray(ctx["face_valid"], dtype=np.bool_)
    samplefit_raw = _fit_face_sample_jacobian_field(ctx=ctx, anchor_uv_prior=None, prior_weight=0.0, min_samples=3)
    conf_samplefit = _projection_confidence_samplefit(ctx, samplefit_raw, strict_gate=False)

    exp12_best = _project_face_jacobian_field(
        ctx=ctx,
        data_jac=np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
        data_valid=np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
        base_jac=anchor_jac,
        base_valid=base_valid,
        face_confidence=conf_samplefit,
        lambda_base=1.0,
        lambda_curl=5.0,
    )
    exp13_best = _project_face_jacobian_residual_field(
        ctx=ctx,
        data_jac=np.asarray(samplefit_raw["face_jac"], dtype=np.float64),
        data_valid=np.asarray(samplefit_raw["face_valid"], dtype=np.bool_),
        base_jac=anchor_jac,
        base_valid=base_valid,
        face_confidence=conf_samplefit,
        lambda_decay=1.0,
        lambda_curl=20.0,
    )

    field_sources = [
        {
            "field_source": "current_target",
            "note": "current Method2 face Jacobian target",
            "face_jac": current_jac,
            "face_valid": current_valid,
        },
        {
            "field_source": "exp12_project_samplefit_conf_curl5",
            "note": "Exp12 best field projector output",
            "face_jac": np.asarray(exp12_best["face_jac"], dtype=np.float64),
            "face_valid": np.asarray(exp12_best["face_valid"], dtype=np.bool_),
        },
        {
            "field_source": "exp13_residual_samplefit_soft_curl20",
            "note": "Exp13 best residual-space projector output",
            "face_jac": np.asarray(exp13_best["face_jac"], dtype=np.float64),
            "face_valid": np.asarray(exp13_best["face_valid"], dtype=np.bool_),
        },
    ]
    box_levels = {
        "box_low": 10.0,
        "box_medium": 100.0,
        "box_high": 1000.0,
    }

    out_variants: List[Dict[str, Any]] = []
    zero_summary = _norm_summary(np.zeros((0,), dtype=np.float64))
    for source in field_sources:
        field_jac = np.asarray(source["face_jac"], dtype=np.float64)
        field_valid = np.asarray(source["face_valid"], dtype=np.bool_)
        face_weights = _default_variant_face_weights(ctx, field_valid)
        reference_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=f"{source['field_source']}_unconstrained",
            face_jac=field_jac,
            face_weights=face_weights,
            face_valid=field_valid,
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=current_jac,
            compare_target_valid_mask=current_valid & field_valid,
            include_mapped_uv=True,
        )
        reference_uv = np.asarray(reference_summary.pop("mapped_uv"), dtype=np.float64)
        reference_feas = summarize_uv_box_feasibility(ctx["solve_mesh"], reference_uv, margin=0.0)
        out_variants.append(
            {
                "status": "ok",
                "field_source": str(source["field_source"]),
                "constraint_variant": "unconstrained",
                "note": str(source["note"]),
                "box_weight": 0.0,
                "box_margin": 0.0,
                "field_valid_face_count": int(np.count_nonzero(field_valid)),
                "field_valid_face_ratio": float(np.mean(field_valid)),
                "field_curl_summary": _curl_global_summary(ctx["solve_mesh"], field_jac, field_valid),
                "solve_summary": reference_summary,
                "solve_feasibility_vs_reference": {
                    "reference": reference_feas,
                    "candidate": reference_feas,
                    "delta": {
                        "vertex_oob_ratio": 0.0,
                        "face_oob_ratio": 0.0,
                        "max_oob_overshoot": 0.0,
                        "barrier_active_vertex_ratio": 0.0,
                        "barrier_active_face_ratio": 0.0,
                        "max_barrier_overshoot": 0.0,
                    },
                    "all_vertex_displacement": _norm_summary(np.zeros((len(reference_uv),), dtype=np.float64)),
                    "boundary_vertex_displacement": zero_summary,
                    "barrier_active_vertex_displacement": zero_summary,
                    "island_drift": {
                        "island_count": 0,
                        "centroid_shift": zero_summary,
                        "bbox_area_ratio_p50": None,
                        "bbox_area_ratio_p95": None,
                        "bbox_area_log_abs_p50": None,
                        "bbox_area_log_abs_p95": None,
                        "bbox_area_log_abs_max": None,
                    },
                },
            }
        )

        for level_name, box_weight in box_levels.items():
            solve_summary = _solve_custom_field_summary(
                ctx=ctx,
                variant_name=f"{source['field_source']}_{level_name}",
                face_jac=field_jac,
                face_weights=face_weights,
                face_valid=field_valid,
                anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
                anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
                face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
                compare_target_jac=current_jac,
                compare_target_valid_mask=current_valid & field_valid,
                cfg_override={
                    "solve": {
                        "constraint_mode": "box_barrier",
                        "constraint_device": "cpu",
                        "constraint_box_weight": float(box_weight),
                        "constraint_box_margin": 0.0,
                        "constraint_refine_iters": 60,
                        "constraint_refine_lr": 0.05,
                        "constraint_grad_clip": 5.0,
                        "constraint_early_stop_rel_tol": 1e-5,
                        "constraint_early_stop_patience": 8,
                    }
                },
                include_mapped_uv=True,
            )
            solve_uv = np.asarray(solve_summary.pop("mapped_uv"), dtype=np.float64)
            out_variants.append(
                {
                    "status": "ok",
                    "field_source": str(source["field_source"]),
                    "constraint_variant": level_name,
                    "note": str(source["note"]),
                    "box_weight": float(box_weight),
                    "box_margin": 0.0,
                    "field_valid_face_count": int(np.count_nonzero(field_valid)),
                    "field_valid_face_ratio": float(np.mean(field_valid)),
                    "field_curl_summary": _curl_global_summary(ctx["solve_mesh"], field_jac, field_valid),
                    "solve_summary": solve_summary,
                    "solve_feasibility_vs_reference": _route_b_feasibility_summary(
                        mesh=ctx["solve_mesh"],
                        reference_uv=reference_uv,
                        candidate_uv=solve_uv,
                        face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
                        box_margin=0.0,
                    ),
                }
            )

    return {
        "status": "ok",
        "anchor_fill": fill_meta,
        "box_weight_levels": box_levels,
        "variants": out_variants,
    }


def _run_exp15_topology_release_cuts(ctx: Dict[str, Any]) -> Dict[str, Any]:
    field_sources = _route_c_build_field_sources(ctx)
    solve_mesh = ctx["solve_mesh"]
    solve_faces = np.asarray(solve_mesh.faces, dtype=np.int64)
    original_vertex_count = int(len(solve_mesh.vertices))
    reference_component_labels = _connected_face_labels(solve_mesh)
    reference_component_count = int(len(np.unique(reference_component_labels[reference_component_labels >= 0])))
    budget_levels = [
        ("budget_none", 0.0),
        ("budget_top_0p1pct", 0.001),
        ("budget_top_0p5pct", 0.005),
        ("budget_top_1p0pct", 0.010),
    ]

    quality_delta_keys = [
        "uv_stretch_p95",
        "uv_stretch_p99",
        "uv_bad_tri_ratio_stretch_only",
        "uv_out_of_bounds_ratio",
    ]
    jac_delta_keys = [
        "frob_rel_error_p50",
        "frob_rel_error_p95",
        "frob_rel_error_p99",
        "cosine_p05",
        "log_area_ratio_p95",
    ]
    sample_delta_keys = [
        "residual_l2_mean",
        "residual_l2_p95",
        "residual_linf",
    ]

    out_variants: List[Dict[str, Any]] = []
    for source in field_sources:
        field_jac = np.asarray(source["face_jac"], dtype=np.float64)
        field_valid = np.asarray(source["face_valid"], dtype=np.bool_)
        face_weights = _default_variant_face_weights(ctx, field_valid)

        reference_summary = _solve_custom_field_summary(
            ctx=ctx,
            variant_name=f"{source['field_source']}_uncut",
            face_jac=field_jac,
            face_weights=face_weights,
            face_valid=field_valid,
            anchor_vertex_target_uv=np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
            anchor_vertex_confidence=np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
            face_island_labels=np.asarray(ctx["island_labels"], dtype=np.int64),
            compare_target_jac=np.asarray(ctx["target_jac"], dtype=np.float64),
            compare_target_valid_mask=np.asarray(ctx["face_valid"], dtype=np.bool_) & field_valid,
            include_mapped_uv=True,
        )
        reference_uv = np.asarray(reference_summary.pop("mapped_uv"), dtype=np.float64)
        reference_field_curl = _curl_global_summary(solve_mesh, field_jac, field_valid)
        edge_jump_data = _compute_edge_jump_data(solve_mesh, field_jac, field_valid)
        _, face_cycle_residual = _compute_vertex_cycle_residuals(solve_mesh, field_jac, field_valid)
        face_stretch = np.asarray(face_stretch_anisotropy(solve_mesh, reference_uv), dtype=np.float64)
        field_div = _field_divergence_proxy(
            mesh=solve_mesh,
            face_jac=field_jac,
            face_weights=face_weights,
            face_valid=field_valid,
        )
        edge_table = build_interior_edge_table(mesh=solve_mesh, face_valid_mask=field_valid, existing_seam_edges=None)
        scored_edges = score_route_c_cut_edges(
            edge_table=edge_table,
            edge_jump_l2=np.asarray(edge_jump_data["edge_jump_l2"], dtype=np.float64),
            face_cycle_residual=np.asarray(face_cycle_residual, dtype=np.float64),
            face_stretch=face_stretch,
            face_divergence=np.asarray(field_div["face_rhs_norm"], dtype=np.float64),
        )
        common_cut_meta = {
            "eligible_edges": dict(edge_table["summary"]),
            "score_scales": dict(scored_edges["scales"]),
            "score_summary_all": dict(scored_edges["summary"]["score"]),
        }

        for budget_name, fraction in budget_levels:
            selection = select_budgeted_cut_edges(scored_edges=scored_edges, fraction=float(fraction))
            cut_meta = {
                **common_cut_meta,
                "budget_fraction": float(fraction),
                "selection": {
                    "selected_edge_count": int(selection["selected_count"]),
                    "selected_edge_ratio": float(selection["selected_ratio"]),
                    "selected_edge_length": float(selection["selected_edge_length"]),
                    "selected_edge_length_ratio": float(selection["selected_edge_length_ratio"]),
                    "selected_score_summary": dict(selection["selected_score_summary"]),
                    "all_score_summary": dict(selection["all_score_summary"]),
                },
            }

            if float(fraction) <= 0.0 or int(selection["selected_count"]) == 0:
                identity_map = np.arange(original_vertex_count, dtype=np.int64)
                out_variants.append(
                    {
                        "status": "ok",
                        "field_source": str(source["field_source"]),
                        "budget_variant": budget_name,
                        "note": str(source["note"]),
                        "field_valid_face_count": int(np.count_nonzero(field_valid)),
                        "field_valid_face_ratio": float(np.mean(field_valid)),
                        "cut_proposal": cut_meta,
                        "topology_mutation": {
                            "selected_cut_edges": 0,
                            "split_vertices_added": 0,
                            "new_solve_vertex_count": int(len(solve_mesh.vertices)),
                            "new_solve_face_count": int(len(solve_mesh.faces)),
                            "reference_island_count": int(reference_component_count),
                            "candidate_island_count": int(reference_component_count),
                            "delta_island_count": 0,
                        },
                        "field_curl_summary": reference_field_curl,
                        "cut_domain_field_curl_summary": reference_field_curl,
                        "solve_summary": reference_summary,
                        "delta_vs_same_field_uncut_reference": {
                            "solve_mesh_quality": _numeric_deltas(
                                reference_summary["solve_mesh_quality"],
                                reference_summary["solve_mesh_quality"],
                                quality_delta_keys,
                            ),
                            "solve_vs_variant_target_jacobian": _numeric_deltas(
                                reference_summary["solve_vs_variant_target_jacobian"],
                                reference_summary["solve_vs_variant_target_jacobian"],
                                jac_delta_keys,
                            ),
                            "sample_residual_summary": _numeric_deltas(
                                reference_summary["sample_residual_summary"],
                                reference_summary["sample_residual_summary"],
                                sample_delta_keys,
                            ),
                            "topology_delta": _route_c_topology_delta_summary(
                                reference_mesh=solve_mesh,
                                reference_uv=reference_uv,
                                candidate_mesh=solve_mesh,
                                candidate_uv=reference_uv,
                                candidate_face_labels=reference_component_labels,
                                candidate_new_to_old=identity_map,
                            ),
                        },
                    }
                )
                continue

            split_vertices, split_faces, split_meta = split_vertices_along_cut_edges(
                vertices=np.asarray(solve_mesh.vertices, dtype=np.float32),
                faces=solve_faces,
                cut_edges=np.asarray(selection["selected_edges"], dtype=np.int64),
            )
            split_mesh = trimesh.Trimesh(vertices=split_vertices, faces=split_faces, process=False)
            split_parent = _infer_split_vertex_parent_map(
                original_faces=solve_faces,
                split_faces=np.asarray(split_faces, dtype=np.int64),
                original_vertex_count=original_vertex_count,
                split_vertex_count=int(np.asarray(split_vertices).shape[0]),
            )
            split_anchor_uv = _remap_vertex_array_to_split(
                np.asarray(ctx["internal"].anchor_vertex_target_uv, dtype=np.float64),
                split_parent,
            )
            split_anchor_conf = _remap_vertex_array_to_split(
                np.asarray(ctx["internal"].anchor_vertex_confidence, dtype=np.float64),
                split_parent,
            )
            split_labels = _connected_face_labels(split_mesh)
            split_face_geom_pinv, _ = _compute_face_geometry_pinv(split_mesh)
            solve_summary = _solve_custom_field_summary(
                ctx=ctx,
                variant_name=f"{source['field_source']}_{budget_name}",
                face_jac=field_jac,
                face_weights=face_weights,
                face_valid=field_valid,
                anchor_vertex_target_uv=split_anchor_uv,
                anchor_vertex_confidence=split_anchor_conf,
                face_island_labels=split_labels,
                compare_target_jac=np.asarray(ctx["target_jac"], dtype=np.float64),
                compare_target_valid_mask=np.asarray(ctx["face_valid"], dtype=np.bool_) & field_valid,
                include_mapped_uv=True,
                solve_mesh_override=split_mesh,
                face_geom_pinv_override=np.asarray(split_face_geom_pinv, dtype=np.float64),
            )
            split_uv = np.asarray(solve_summary.pop("mapped_uv"), dtype=np.float64)
            split_island_count = int(len(np.unique(split_labels[split_labels >= 0])))
            out_variants.append(
                {
                    "status": "ok",
                    "field_source": str(source["field_source"]),
                    "budget_variant": budget_name,
                    "note": str(source["note"]),
                    "field_valid_face_count": int(np.count_nonzero(field_valid)),
                    "field_valid_face_ratio": float(np.mean(field_valid)),
                    "cut_proposal": cut_meta,
                    "topology_mutation": {
                        "selected_cut_edges": int(selection["selected_count"]),
                        "split_vertices_added": int(split_meta.get("split_vertices_added", 0)),
                        "new_solve_vertex_count": int(split_mesh.vertices.shape[0]),
                        "new_solve_face_count": int(split_mesh.faces.shape[0]),
                        "reference_island_count": int(reference_component_count),
                        "candidate_island_count": int(split_island_count),
                        "delta_island_count": int(split_island_count - reference_component_count),
                    },
                    "field_curl_summary": reference_field_curl,
                    "cut_domain_field_curl_summary": _curl_global_summary(split_mesh, field_jac, field_valid),
                    "solve_summary": solve_summary,
                    "delta_vs_same_field_uncut_reference": {
                        "solve_mesh_quality": _numeric_deltas(
                            reference_summary["solve_mesh_quality"],
                            solve_summary["solve_mesh_quality"],
                            quality_delta_keys,
                        ),
                        "solve_vs_variant_target_jacobian": _numeric_deltas(
                            reference_summary["solve_vs_variant_target_jacobian"],
                            solve_summary["solve_vs_variant_target_jacobian"],
                            jac_delta_keys,
                        ),
                        "sample_residual_summary": _numeric_deltas(
                            reference_summary["sample_residual_summary"],
                            solve_summary["sample_residual_summary"],
                            sample_delta_keys,
                        ),
                        "topology_delta": _route_c_topology_delta_summary(
                            reference_mesh=solve_mesh,
                            reference_uv=reference_uv,
                            candidate_mesh=split_mesh,
                            candidate_uv=split_uv,
                            candidate_face_labels=split_labels,
                            candidate_new_to_old=split_parent,
                        ),
                    },
                }
            )

    return {
        "status": "ok",
        "budget_levels": {name: frac for name, frac in budget_levels},
        "variants": out_variants,
    }


def _run_all_experiments(ctx: Dict[str, Any]) -> Dict[str, Any]:
    exp1 = _run_exp1_filtered_metrics(ctx)
    exp2 = _run_exp2_curl_audit(ctx)
    exp3 = _run_exp3_repartition(ctx, exp2)
    exp4 = _run_exp4_patch_dirichlet(ctx)
    exp5 = _run_exp5_contributor_mixture(ctx)
    exp6 = _run_exp6_pre_post_align(ctx)
    exp7 = _run_exp7_synthetic(ctx)
    exp8 = _run_exp8_target_vs_anchor(ctx)
    exp9 = _run_exp9_fallback_ablation(ctx)
    exp10 = _run_exp10_anchor_residual(ctx)
    exp11 = _run_exp11_field_builder_variants(ctx)
    exp12 = _run_exp12_field_projection(ctx)
    exp13 = _run_exp13_residual_projection(ctx)
    exp14 = _run_exp14_solve_space_constraints(ctx)
    exp15 = _run_exp15_topology_release_cuts(ctx)
    return {
        "exp1_filtered_metrics": exp1,
        "exp2_curl_audit": exp2,
        "exp3_fixed_field_repartition": exp3,
        "exp4_local_patch_dirichlet": exp4,
        "exp5_contributor_mixture": exp5,
        "exp6_pre_post_align": exp6,
        "exp7_synthetic_exact_recovery": exp7,
        "exp8_target_vs_anchor_field": exp8,
        "exp9_fallback_ablation": exp9,
        "exp10_anchor_residual_field": exp10,
        "exp11_field_builder_variants": exp11,
        "exp12_field_projection_variants": exp12,
        "exp13_residual_projection_variants": exp13,
        "exp14_solve_space_constraints": exp14,
        "exp15_topology_release_cuts": exp15,
    }
