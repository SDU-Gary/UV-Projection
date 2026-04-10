from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, lsmr, spsolve
from scipy.spatial import cKDTree

def nearest_vertex_uv(low_mesh: trimesh.Trimesh, high_mesh: trimesh.Trimesh, high_uv: np.ndarray) -> np.ndarray:
    tree = cKDTree(np.asarray(high_mesh.vertices))
    _, idx = tree.query(np.asarray(low_mesh.vertices), k=1)
    idx = np.asarray(idx, dtype=np.int64)
    return high_uv[np.clip(idx, 0, len(high_uv) - 1)]


def mesh_laplacian(
    faces: np.ndarray,
    n_vertices: int,
    face_mask: Optional[np.ndarray] = None,
    *,
    vertices: Optional[np.ndarray] = None,
    mode: str = "uniform",
) -> csr_matrix:
    if face_mask is not None:
        mask = np.asarray(face_mask, dtype=np.bool_)
        if mask.shape[0] != faces.shape[0]:
            raise ValueError("face_mask length mismatch for Laplacian construction")
        faces = faces[mask]
        if faces.size == 0:
            return csr_matrix((n_vertices, n_vertices), dtype=np.float64)

    mode_norm = str(mode or "uniform").strip().lower()
    if mode_norm not in {"uniform", "cotan"}:
        raise ValueError(f"Unsupported Laplacian mode: {mode}")

    if mode_norm == "uniform":
        edges = np.vstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(len(row), dtype=np.float64)
        W = coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices), dtype=np.float64).tocsr()
    else:
        if vertices is None:
            raise ValueError("vertices are required for cotan Laplacian")
        verts = np.asarray(vertices, dtype=np.float64)
        tri = verts[np.asarray(faces, dtype=np.int64)]
        if tri.size == 0:
            return csr_matrix((n_vertices, n_vertices), dtype=np.float64)

        p0 = tri[:, 0]
        p1 = tri[:, 1]
        p2 = tri[:, 2]

        def cotangent(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            cross = np.linalg.norm(np.cross(a, b), axis=1)
            dot = np.sum(a * b, axis=1)
            out = np.zeros_like(dot)
            ok = cross > 1e-20
            out[ok] = dot[ok] / cross[ok]
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        cot0 = cotangent(p1 - p0, p2 - p0)
        cot1 = cotangent(p2 - p1, p0 - p1)
        cot2 = cotangent(p0 - p2, p1 - p2)

        i0 = faces[:, 0]
        i1 = faces[:, 1]
        i2 = faces[:, 2]

        # Edge (1,2) opposite vertex 0, etc.
        e_i = np.concatenate([i1, i2, i2, i0, i0, i1])
        e_j = np.concatenate([i2, i1, i0, i2, i1, i0])
        e_w = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]).astype(np.float64)
        e_w = np.maximum(e_w, 0.0)
        valid = np.isfinite(e_w) & (e_w > 0.0)
        W = coo_matrix(
            (e_w[valid], (e_i[valid], e_j[valid])),
            shape=(n_vertices, n_vertices),
            dtype=np.float64,
        ).tocsr()

    degree = np.array(W.sum(axis=1)).reshape(-1)
    D = diags(degree)
    return (D - W).tocsr()


def connected_components_labels(faces: np.ndarray, n_vertices: int) -> Tuple[np.ndarray, int]:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    graph = coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    return labels, int(n_comp)


def interpolate_sample_uv(
    faces: np.ndarray,
    sample_face_ids: np.ndarray,
    sample_bary: np.ndarray,
    vertex_uv: np.ndarray,
) -> np.ndarray:
    tri = vertex_uv[faces[sample_face_ids]]
    return (
        tri[:, 0] * sample_bary[:, [0]]
        + tri[:, 1] * sample_bary[:, [1]]
        + tri[:, 2] * sample_bary[:, [2]]
    )


def build_cuda_sparse_system(*, M: csr_matrix, device: str):
    import torch

    cuda_device = torch.device(device if str(device).startswith("cuda") else "cuda")
    if cuda_device.type != "cuda":
        raise RuntimeError(f"Expected CUDA device for sparse PCG, got '{cuda_device}'")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")

    M_coo = M.tocoo()
    indices = np.vstack([M_coo.row, M_coo.col]).astype(np.int64, copy=False)
    values = M_coo.data.astype(np.float32, copy=False)

    idx_t = torch.from_numpy(indices).to(device=cuda_device, dtype=torch.long, non_blocking=True)
    val_t = torch.from_numpy(values).to(device=cuda_device, dtype=torch.float32, non_blocking=True)
    M_cuda = torch.sparse_coo_tensor(
        idx_t,
        val_t,
        size=M_coo.shape,
        dtype=torch.float32,
        device=cuda_device,
    ).coalesce()
    diag = M.diagonal().astype(np.float32, copy=False)
    diag_t = torch.from_numpy(diag).to(device=cuda_device, dtype=torch.float32, non_blocking=True)
    return M_cuda, diag_t


def solve_linear_cuda_pcg(
    *,
    M_cuda,
    M_diag_cuda,
    rhs: np.ndarray,
    pcg_max_iter: int,
    pcg_tol: float,
    pcg_check_every: int,
    pcg_preconditioner: str,
    channel_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch

    rhs_t = torch.from_numpy(np.asarray(rhs, dtype=np.float32)).to(device=M_cuda.device, non_blocking=True)
    n = int(rhs_t.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64), {
            "backend": "cuda_pcg",
            "cg_info": 0,
            "cg2_info": 0,
            "iters": 0,
            "residual": 0.0,
            "converged": True,
        }

    check_every = max(1, int(pcg_check_every))
    max_iter = max(1, int(pcg_max_iter))
    tol = max(float(pcg_tol), 1e-12)
    preconditioner = str(pcg_preconditioner or "jacobi").strip().lower()
    if preconditioner not in {"jacobi", "none"}:
        preconditioner = "jacobi"

    diag = M_diag_cuda
    diag_safe = torch.where(torch.abs(diag) > 1e-8, diag, torch.ones_like(diag))
    x = rhs_t / diag_safe
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def spmv(vec):
        return torch.sparse.mm(M_cuda, vec.unsqueeze(1)).squeeze(1)

    r = rhs_t - spmv(x)
    if preconditioner == "jacobi":
        inv_diag = torch.where(torch.abs(diag) > 1e-8, 1.0 / diag, torch.ones_like(diag))
        z = r * inv_diag
    else:
        z = r.clone()
    p = z.clone()

    rz_old = torch.dot(r, z)
    rhs_norm = torch.linalg.norm(rhs_t)
    rhs_norm_val = float(rhs_norm.detach().cpu().item())
    if rhs_norm_val <= 1e-20:
        return np.zeros((n,), dtype=np.float64), {
            "backend": "cuda_pcg",
            "cg_info": 0,
            "cg2_info": 0,
            "iters": 0,
            "residual": 0.0,
            "converged": True,
        }
    tol_abs = max(1e-10, tol * rhs_norm_val)
    residual = float(torch.linalg.norm(r).detach().cpu().item())
    if not np.isfinite(residual):
        raise RuntimeError(f"Initial residual is non-finite for channel '{channel_name}'")
    if residual <= tol_abs:
        return x.detach().cpu().numpy().astype(np.float64), {
            "backend": "cuda_pcg",
            "cg_info": 0,
            "cg2_info": 0,
            "iters": 0,
            "residual": residual,
            "converged": True,
        }

    converged = False
    iters = 0
    for it in range(1, max_iter + 1):
        Ap = spmv(p)
        denom = torch.dot(p, Ap)
        denom_val = float(denom.detach().cpu().item())
        if not np.isfinite(denom_val) or abs(denom_val) <= 1e-20:
            raise RuntimeError(
                f"PCG breakdown for channel '{channel_name}' at iter={it}: denom={denom_val}"
            )

        alpha = rz_old / denom
        x = x + alpha * p

        if it % check_every == 0:
            r = rhs_t - spmv(x)
        else:
            r = r - alpha * Ap

        residual = float(torch.linalg.norm(r).detach().cpu().item())
        iters = it
        if not np.isfinite(residual):
            raise RuntimeError(f"Residual became non-finite for channel '{channel_name}' at iter={it}")
        if residual <= tol_abs:
            converged = True
            break

        if preconditioner == "jacobi":
            z = r * inv_diag
        else:
            z = r
        rz_new = torch.dot(r, z)
        rz_old_val = float(rz_old.detach().cpu().item())
        if not np.isfinite(rz_old_val) or abs(rz_old_val) <= 1e-20:
            raise RuntimeError(
                f"PCG breakdown for channel '{channel_name}' at iter={it}: rz_old={rz_old_val}"
            )
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    if not converged:
        raise RuntimeError(
            f"CUDA PCG did not converge for channel '{channel_name}' "
            f"(iters={iters}, residual={residual:.6e}, tol_abs={tol_abs:.6e})"
        )

    x_np = x.detach().cpu().numpy().astype(np.float64)
    return x_np, {
        "backend": "cuda_pcg",
        "cg_info": 0,
        "cg2_info": 0,
        "iters": iters,
        "residual": residual,
        "converged": True,
    }


def solve_linear_robust(
    *,
    M: csr_matrix,
    rhs: np.ndarray,
    cg_max_iter: int,
    cg_tol: float,
    channel_name: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    def residual_norm(x: np.ndarray) -> float:
        return float(np.linalg.norm(M.dot(x) - rhs))

    diag = M.diagonal().astype(np.float64)
    diag_safe = np.maximum(np.abs(diag), 1e-8)
    x0 = (rhs / diag_safe).astype(np.float64)

    cg_iter = max(100, int(cg_max_iter))
    x1, info1 = cg(M, rhs, x0=x0, maxiter=cg_iter, rtol=float(cg_tol), atol=0.0)
    if info1 == 0 and np.all(np.isfinite(x1)):
        return x1, {
            "backend": "cg",
            "cg_info": info1,
            "cg2_info": 0,
            "iters": -1,
            "residual": residual_norm(x1),
            "converged": True,
        }

    precond = diags(1.0 / diag_safe)
    cg2_iter = max(cg_iter * 4, cg_iter + 400)
    cg2_tol = max(float(cg_tol) * 10.0, 1e-5)
    x2, info2 = cg(M, rhs, x0=x0, M=precond, maxiter=cg2_iter, rtol=cg2_tol, atol=0.0)
    if info2 == 0 and np.all(np.isfinite(x2)):
        return x2, {
            "backend": "cg_jacobi",
            "cg_info": info1,
            "cg2_info": info2,
            "iters": -1,
            "residual": residual_norm(x2),
            "converged": True,
        }

    try:
        from sksparse.cholmod import cholesky  # type: ignore

        factor = cholesky(M.tocsc())
        x_ch = factor(rhs)
        if np.all(np.isfinite(x_ch)):
            x_ch = np.asarray(x_ch, dtype=np.float64).reshape(-1)
            return x_ch, {
                "backend": "cholmod",
                "cg_info": info1,
                "cg2_info": info2,
                "iters": 1,
                "residual": residual_norm(x_ch),
                "converged": True,
            }
    except Exception:
        pass

    try:
        import pyamg  # type: ignore

        ml = pyamg.smoothed_aggregation_solver(M)
        M_prec = ml.aspreconditioner()
        x3a, info3a = cg(
            M,
            rhs,
            x0=x0,
            M=M_prec,
            maxiter=max(cg2_iter, 3000),
            rtol=max(float(cg_tol), 1e-8),
            atol=0.0,
        )
        if info3a == 0 and np.all(np.isfinite(x3a)):
            x3a = x3a.astype(np.float64)
            return x3a, {
                "backend": "cg_pyamg",
                "cg_info": info1,
                "cg2_info": info2,
                "iters": -1,
                "residual": residual_norm(x3a),
                "converged": True,
            }
    except Exception:
        pass

    try:
        x3 = spsolve(M.tocsc(), rhs)
        if np.all(np.isfinite(x3)):
            x3 = x3.astype(np.float64)
            return x3, {
                "backend": "spsolve",
                "cg_info": info1,
                "cg2_info": info2,
                "iters": 1,
                "residual": residual_norm(x3),
                "converged": True,
            }
    except Exception:
        pass

    lsmr_res = lsmr(
        M,
        rhs,
        atol=max(float(cg_tol), 1e-8),
        btol=max(float(cg_tol), 1e-8),
        maxiter=max(cg2_iter, 2000),
    )
    x4 = lsmr_res[0]
    if np.all(np.isfinite(x4)):
        x4 = x4.astype(np.float64)
        return x4, {
            "backend": "lsmr",
            "cg_info": info1,
            "cg2_info": info2,
            "iters": int(lsmr_res[2]),
            "residual": residual_norm(x4),
            "converged": True,
        }

    raise RuntimeError(
        f"UV linear solve did not converge for {channel_name} "
        f"(cg1_info={info1}, cg2_info={info2})"
    )


def solve_global_uv(
    *,
    low_mesh: trimesh.Trimesh,
    sample_face_ids: np.ndarray,
    sample_bary: np.ndarray,
    target_uv: np.ndarray,
    sample_weights: np.ndarray,
    backend: str,
    lambda_smooth: float,
    pcg_max_iter: int,
    pcg_tol: float,
    pcg_check_every: int,
    pcg_preconditioner: str,
    anchor_weight: float,
    ridge_eps: float,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    smooth_face_mask: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    low_faces = np.asarray(low_mesh.faces, dtype=np.int64)
    n_vertices = int(len(low_mesh.vertices))
    n_samples = int(len(sample_face_ids))

    tri_vidx = low_faces[sample_face_ids]

    rows = np.repeat(np.arange(n_samples, dtype=np.int64), 3)
    cols = tri_vidx.reshape(-1)
    sqrt_w = np.sqrt(np.maximum(sample_weights, 1e-12))
    data = (sample_bary.reshape(-1) * np.repeat(sqrt_w, 3)).astype(np.float64)

    A = coo_matrix((data, (rows, cols)), shape=(n_samples, n_vertices), dtype=np.float64).tocsr()
    b_u = (target_uv[:, 0] * sqrt_w).astype(np.float64)
    b_v = (target_uv[:, 1] * sqrt_w).astype(np.float64)

    L = mesh_laplacian(low_faces, n_vertices, face_mask=smooth_face_mask)

    M = (A.T @ A).tocsr()
    if lambda_smooth > 0.0:
        M = (M + lambda_smooth * L).tocsr()
    diag_eps = max(float(ridge_eps), 1e-6)
    M = (M + diag_eps * diags(np.ones(n_vertices, dtype=np.float64))).tocsr()

    rhs_u = A.T @ b_u
    rhs_v = A.T @ b_v

    if anchor_weight > 0.0:
        M = M.tolil(copy=False)
        anchor_uv = nearest_vertex_uv(low_mesh, high_mesh, high_uv)
        labels, n_comp = connected_components_labels(low_faces, n_vertices)
        for comp in range(n_comp):
            idx = int(np.where(labels == comp)[0][0])
            M[idx, idx] = M[idx, idx] + anchor_weight
            rhs_u[idx] += anchor_weight * anchor_uv[idx, 0]
            rhs_v[idx] += anchor_weight * anchor_uv[idx, 1]
        M = M.tocsr()

    backend_requested = str(backend or "auto").strip().lower()
    if backend_requested not in {
        "auto",
        "cuda_pcg",
        "cpu_scipy",
    }:
        raise ValueError(
            f"Unsupported UV solve backend '{backend_requested}'. "
            "Expected one of: auto, cuda_pcg, cpu_scipy"
        )

    backend_used = "cpu_scipy"
    fallback_reason = None
    solve_u: Dict[str, Any]
    solve_v: Dict[str, Any]
    linear_backend_requested = backend_requested

    want_cuda = str(device).startswith("cuda")
    use_cuda_first = linear_backend_requested == "cuda_pcg" or (linear_backend_requested == "auto" and want_cuda)
    if use_cuda_first:
        try:
            M_cuda, M_diag_cuda = build_cuda_sparse_system(M=M, device=device)
            uv_u, solve_u = solve_linear_cuda_pcg(
                M_cuda=M_cuda,
                M_diag_cuda=M_diag_cuda,
                rhs=rhs_u,
                pcg_max_iter=pcg_max_iter,
                pcg_tol=pcg_tol,
                pcg_check_every=pcg_check_every,
                pcg_preconditioner=pcg_preconditioner,
                channel_name="u",
            )
            uv_v, solve_v = solve_linear_cuda_pcg(
                M_cuda=M_cuda,
                M_diag_cuda=M_diag_cuda,
                rhs=rhs_v,
                pcg_max_iter=pcg_max_iter,
                pcg_tol=pcg_tol,
                pcg_check_every=pcg_check_every,
                pcg_preconditioner=pcg_preconditioner,
                channel_name="v",
            )
            backend_used = "cuda_pcg"
        except Exception as exc:
            fallback_reason = f"cuda_pcg_failed: {exc}"
            uv_u, solve_u = solve_linear_robust(
                M=M,
                rhs=rhs_u,
                cg_max_iter=pcg_max_iter,
                cg_tol=pcg_tol,
                channel_name="u",
            )
            uv_v, solve_v = solve_linear_robust(
                M=M,
                rhs=rhs_v,
                cg_max_iter=pcg_max_iter,
                cg_tol=pcg_tol,
                channel_name="v",
            )
            backend_used = "cpu_scipy"
    else:
        uv_u, solve_u = solve_linear_robust(
            M=M,
            rhs=rhs_u,
            cg_max_iter=pcg_max_iter,
            cg_tol=pcg_tol,
            channel_name="u",
        )
        uv_v, solve_v = solve_linear_robust(
            M=M,
            rhs=rhs_v,
            cg_max_iter=pcg_max_iter,
            cg_tol=pcg_tol,
            channel_name="v",
        )
        backend_used = "cpu_scipy"

    uv_out = np.stack([uv_u, uv_v], axis=1).astype(np.float32)

    solve_meta = {
        "uv_solver_backend_requested": backend_requested,
        "uv_solver_backend_used": backend_used,
        "uv_solver_linear_backend_requested": linear_backend_requested,
        "uv_solver_linear_backend_used": solve_u["backend"],
        "uv_solver_backend_u": solve_u["backend"],
        "uv_solver_backend_v": solve_v["backend"],
        "uv_solver_iters_u": int(solve_u.get("iters", -1)),
        "uv_solver_iters_v": int(solve_v.get("iters", -1)),
        "uv_solver_residual_u": float(solve_u.get("residual", float("nan"))),
        "uv_solver_residual_v": float(solve_v.get("residual", float("nan"))),
        "uv_solver_converged_u": bool(solve_u.get("converged", False)),
        "uv_solver_converged_v": bool(solve_v.get("converged", False)),
        "uv_solver_u": solve_u["backend"],
        "uv_solver_v": solve_v["backend"],
        "uv_solver_cg_info_u": int(solve_u["cg_info"]),
        "uv_solver_cg_info_v": int(solve_v["cg_info"]),
        "uv_solver_refine_cg_info_u": int(solve_u["cg2_info"]),
        "uv_solver_refine_cg_info_v": int(solve_v["cg2_info"]),
    }
    if fallback_reason:
        solve_meta["uv_solver_fallback_reason"] = fallback_reason
    return uv_out.astype(np.float32), solve_meta


__all__ = [
    "build_cuda_sparse_system",
    "connected_components_labels",
    "interpolate_sample_uv",
    "mesh_laplacian",
    "nearest_vertex_uv",
    "solve_global_uv",
    "solve_linear_cuda_pcg",
    "solve_linear_robust",
]
