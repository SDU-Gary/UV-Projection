from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh
from scipy.sparse import csr_matrix


def resolve_constraint_device(requested: str, resolved_device: str) -> str:
    req = str(requested or "auto").strip().lower()
    if req == "auto":
        req = resolved_device if str(resolved_device).startswith("cuda") else "cpu"
    if req.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return req
        except Exception:
            pass
    return "cpu"


def compute_uv_box_feasibility_arrays(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    *,
    margin: float = 0.0,
) -> Dict[str, np.ndarray]:
    uv_np = np.asarray(uv, dtype=np.float64)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        raise RuntimeError("uv box feasibility expects [N,2] UV array")
    faces = np.asarray(mesh.faces, dtype=np.int64)

    actual_low = np.maximum(-uv_np, 0.0)
    actual_high = np.maximum(uv_np - 1.0, 0.0)
    actual_violation = actual_low + actual_high
    vertex_oob_overshoot = np.max(actual_violation, axis=1) if uv_np.shape[0] > 0 else np.zeros((0,), dtype=np.float64)
    vertex_oob_mask = vertex_oob_overshoot > 0.0

    box_margin = float(np.clip(float(margin), 0.0, 0.49))
    lower = box_margin
    upper = 1.0 - box_margin
    barrier_low = np.maximum(lower - uv_np, 0.0)
    barrier_high = np.maximum(uv_np - upper, 0.0)
    barrier_violation = barrier_low + barrier_high
    vertex_barrier_overshoot = (
        np.max(barrier_violation, axis=1) if uv_np.shape[0] > 0 else np.zeros((0,), dtype=np.float64)
    )
    vertex_barrier_mask = vertex_barrier_overshoot > 0.0

    if faces.size == 0 or uv_np.shape[0] == 0:
        face_oob_mask = np.zeros((0,), dtype=np.bool_)
        face_barrier_mask = np.zeros((0,), dtype=np.bool_)
        face_oob_overshoot = np.zeros((0,), dtype=np.float64)
        face_barrier_overshoot = np.zeros((0,), dtype=np.float64)
    else:
        face_oob_mask = np.any(vertex_oob_mask[faces], axis=1)
        face_barrier_mask = np.any(vertex_barrier_mask[faces], axis=1)
        face_oob_overshoot = np.max(vertex_oob_overshoot[faces], axis=1)
        face_barrier_overshoot = np.max(vertex_barrier_overshoot[faces], axis=1)

    return {
        "vertex_oob_mask": vertex_oob_mask,
        "vertex_oob_overshoot": vertex_oob_overshoot,
        "face_oob_mask": face_oob_mask,
        "face_oob_overshoot": face_oob_overshoot,
        "vertex_barrier_mask": vertex_barrier_mask,
        "vertex_barrier_overshoot": vertex_barrier_overshoot,
        "face_barrier_mask": face_barrier_mask,
        "face_barrier_overshoot": face_barrier_overshoot,
    }


def summarize_uv_box_feasibility(
    mesh: trimesh.Trimesh,
    uv: np.ndarray,
    *,
    margin: float = 0.0,
) -> Dict[str, Any]:
    arrays = compute_uv_box_feasibility_arrays(mesh, uv, margin=margin)
    v_oob = np.asarray(arrays["vertex_oob_overshoot"], dtype=np.float64)
    f_oob = np.asarray(arrays["face_oob_overshoot"], dtype=np.float64)
    v_bar = np.asarray(arrays["vertex_barrier_overshoot"], dtype=np.float64)
    f_bar = np.asarray(arrays["face_barrier_overshoot"], dtype=np.float64)
    v_oob_mask = np.asarray(arrays["vertex_oob_mask"], dtype=np.bool_)
    f_oob_mask = np.asarray(arrays["face_oob_mask"], dtype=np.bool_)
    v_bar_mask = np.asarray(arrays["vertex_barrier_mask"], dtype=np.bool_)
    f_bar_mask = np.asarray(arrays["face_barrier_mask"], dtype=np.bool_)

    def _sel_stats(values: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        vals = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=np.bool_)]
        if vals.size == 0:
            return None, None
        return float(np.mean(vals)), float(np.quantile(vals, 0.95))

    oob_mean, oob_p95 = _sel_stats(v_oob, v_oob_mask)
    bar_mean, bar_p95 = _sel_stats(v_bar, v_bar_mask)
    return {
        "box_margin": float(np.clip(float(margin), 0.0, 0.49)),
        "vertex_count": int(v_oob.shape[0]),
        "face_count": int(f_oob.shape[0]),
        "vertex_oob_ratio": float(np.mean(v_oob_mask)) if v_oob_mask.size > 0 else 0.0,
        "face_oob_ratio": float(np.mean(f_oob_mask)) if f_oob_mask.size > 0 else 0.0,
        "max_oob_overshoot": float(np.max(v_oob)) if v_oob.size > 0 else 0.0,
        "oob_overshoot_mean_active": oob_mean,
        "oob_overshoot_p95_active": oob_p95,
        "barrier_active_vertex_ratio": float(np.mean(v_bar_mask)) if v_bar_mask.size > 0 else 0.0,
        "barrier_active_face_ratio": float(np.mean(f_bar_mask)) if f_bar_mask.size > 0 else 0.0,
        "max_barrier_overshoot": float(np.max(v_bar)) if v_bar.size > 0 else 0.0,
        "barrier_overshoot_mean_active": bar_mean,
        "barrier_overshoot_p95_active": bar_p95,
        "face_barrier_overshoot_p95_active": float(np.quantile(f_bar[f_bar_mask], 0.95))
        if np.any(f_bar_mask)
        else None,
    }


def _csr_to_torch_sparse(M: Optional[csr_matrix], *, device: str):
    import torch

    if M is None:
        return None
    mat = M.tocoo()
    if mat.shape[0] == 0 or mat.shape[1] == 0 or mat.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            size=mat.shape,
            dtype=torch.float32,
            device=device,
        ).coalesce()
    indices = np.vstack([mat.row, mat.col]).astype(np.int64, copy=False)
    values = mat.data.astype(np.float32, copy=False)
    return torch.sparse_coo_tensor(
        torch.from_numpy(indices).to(device=device, dtype=torch.long),
        torch.from_numpy(values).to(device=device, dtype=torch.float32),
        size=mat.shape,
        dtype=torch.float32,
        device=device,
    ).coalesce()


def _zero_scalar(device: str):
    import torch

    return torch.zeros((), dtype=torch.float32, device=device)


def refine_uv_with_soft_box_constraint(
    *,
    mesh: trimesh.Trimesh,
    uv_init: np.ndarray,
    A: csr_matrix,
    rhs_u: np.ndarray,
    rhs_v: np.ndarray,
    smooth_matrix: Optional[csr_matrix],
    anchor_ids: np.ndarray,
    anchor_target_uv: np.ndarray,
    anchor_weights: np.ndarray,
    solve_cfg: Dict[str, Any],
    resolved_device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode = str(solve_cfg.get("constraint_mode", "none")).strip().lower()
    if mode not in {"none", "box_barrier"}:
        mode = "none"

    box_weight = max(float(solve_cfg.get("constraint_box_weight", 0.0)), 0.0)
    box_margin = float(np.clip(float(solve_cfg.get("constraint_box_margin", 0.0)), 0.0, 0.49))
    refine_iters = max(0, int(solve_cfg.get("constraint_refine_iters", 80)))
    lr = max(float(solve_cfg.get("constraint_refine_lr", 0.05)), 1e-6)
    grad_clip = max(float(solve_cfg.get("constraint_grad_clip", 5.0)), 0.0)
    early_rel_tol = max(float(solve_cfg.get("constraint_early_stop_rel_tol", 1e-5)), 0.0)
    early_patience = max(1, int(solve_cfg.get("constraint_early_stop_patience", 10)))
    device_name = resolve_constraint_device(str(solve_cfg.get("constraint_device", "cpu")), resolved_device)

    uv0 = np.asarray(uv_init, dtype=np.float64)
    init_feas = summarize_uv_box_feasibility(mesh, uv0, margin=box_margin)
    meta: Dict[str, Any] = {
        "uv_solver_constraint_mode": mode,
        "uv_solver_constraint_applied": False,
        "uv_solver_constraint_status": "disabled_by_config" if mode == "none" else "not_run",
        "uv_solver_constraint_device": device_name,
        "uv_solver_constraint_optimizer": "adam",
        "uv_solver_constraint_iters": 0,
        "uv_solver_constraint_stop_reason": None,
        "uv_solver_constraint_box_weight": float(box_weight),
        "uv_solver_constraint_box_margin": float(box_margin),
        "uv_solver_constraint_refine_lr": float(lr),
        "uv_solver_constraint_grad_clip": float(grad_clip),
        "uv_solver_constraint_energy_init": None,
        "uv_solver_constraint_energy_final": None,
        "uv_solver_constraint_energy_data_init": None,
        "uv_solver_constraint_energy_data_final": None,
        "uv_solver_constraint_energy_smooth_init": None,
        "uv_solver_constraint_energy_smooth_final": None,
        "uv_solver_constraint_energy_anchor_init": None,
        "uv_solver_constraint_energy_anchor_final": None,
        "uv_solver_constraint_energy_box_init": None,
        "uv_solver_constraint_energy_box_final": None,
        "uv_solver_constraint_vertex_oob_ratio_init": init_feas["vertex_oob_ratio"],
        "uv_solver_constraint_vertex_oob_ratio_final": init_feas["vertex_oob_ratio"],
        "uv_solver_constraint_face_oob_ratio_init": init_feas["face_oob_ratio"],
        "uv_solver_constraint_face_oob_ratio_final": init_feas["face_oob_ratio"],
        "uv_solver_constraint_barrier_active_vertex_ratio_init": init_feas["barrier_active_vertex_ratio"],
        "uv_solver_constraint_barrier_active_vertex_ratio_final": init_feas["barrier_active_vertex_ratio"],
        "uv_solver_constraint_barrier_active_face_ratio_init": init_feas["barrier_active_face_ratio"],
        "uv_solver_constraint_barrier_active_face_ratio_final": init_feas["barrier_active_face_ratio"],
        "uv_solver_constraint_max_oob_overshoot_init": init_feas["max_oob_overshoot"],
        "uv_solver_constraint_max_oob_overshoot_final": init_feas["max_oob_overshoot"],
        "uv_solver_constraint_max_barrier_overshoot_init": init_feas["max_barrier_overshoot"],
        "uv_solver_constraint_max_barrier_overshoot_final": init_feas["max_barrier_overshoot"],
    }
    if mode == "none":
        return uv0.astype(np.float32, copy=False), meta
    if uv0.size == 0 or int(len(mesh.vertices)) == 0:
        meta["uv_solver_constraint_status"] = "skipped_empty_mesh"
        return uv0.astype(np.float32, copy=False), meta
    if box_weight <= 0.0:
        meta["uv_solver_constraint_status"] = "skipped_zero_box_weight"
        return uv0.astype(np.float32, copy=False), meta
    if refine_iters <= 0:
        meta["uv_solver_constraint_status"] = "skipped_zero_iters"
        return uv0.astype(np.float32, copy=False), meta
    if float(init_feas["barrier_active_vertex_ratio"]) <= 0.0:
        meta["uv_solver_constraint_status"] = "skipped_already_feasible"
        return uv0.astype(np.float32, copy=False), meta

    try:
        import torch
    except Exception as exc:
        meta["uv_solver_constraint_status"] = "skipped_torch_unavailable"
        meta["uv_solver_constraint_stop_reason"] = str(exc)
        return uv0.astype(np.float32, copy=False), meta

    dev = torch.device(device_name)
    try:
        A_t = _csr_to_torch_sparse(A, device=str(dev))
        L_t = _csr_to_torch_sparse(smooth_matrix, device=str(dev)) if smooth_matrix is not None else None
        uv_var = torch.tensor(uv0.astype(np.float32), dtype=torch.float32, device=dev, requires_grad=True)
        rhs = np.stack([np.asarray(rhs_u, dtype=np.float32), np.asarray(rhs_v, dtype=np.float32)], axis=1)
        rhs_t = torch.tensor(rhs, dtype=torch.float32, device=dev)
        anchor_ids_np = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
        anchor_uv_np = np.asarray(anchor_target_uv, dtype=np.float32)
        anchor_w_np = np.asarray(anchor_weights, dtype=np.float32).reshape(-1)
        if anchor_ids_np.size > 0 and anchor_uv_np.shape[0] == anchor_ids_np.size and anchor_w_np.shape[0] == anchor_ids_np.size:
            anchor_ids_t = torch.tensor(anchor_ids_np, dtype=torch.long, device=dev)
            anchor_uv_t = torch.tensor(anchor_uv_np, dtype=torch.float32, device=dev)
            anchor_w_t = torch.tensor(anchor_w_np, dtype=torch.float32, device=dev)
        else:
            anchor_ids_t = None
            anchor_uv_t = None
            anchor_w_t = None
    except Exception as exc:
        meta["uv_solver_constraint_status"] = "setup_failed"
        meta["uv_solver_constraint_stop_reason"] = str(exc)
        return uv0.astype(np.float32, copy=False), meta

    optimizer = torch.optim.Adam([uv_var], lr=lr)
    n_rows = max(int(A.shape[0]), 1)
    n_vertices = max(int(len(mesh.vertices)), 1)
    lower = float(box_margin)
    upper = float(1.0 - box_margin)

    def _terms(x):
        data_term = _zero_scalar(str(dev))
        if A_t is not None and A.shape[0] > 0:
            data_res = torch.sparse.mm(A_t, x) - rhs_t
            data_term = torch.sum(data_res * data_res) / float(max(2 * n_rows, 1))

        smooth_term = _zero_scalar(str(dev))
        if L_t is not None and smooth_matrix is not None and smooth_matrix.nnz > 0:
            lx = torch.sparse.mm(L_t, x)
            smooth_term = torch.sum(x * lx) / float(max(2 * n_vertices, 1))

        anchor_term = _zero_scalar(str(dev))
        if anchor_ids_t is not None and anchor_uv_t is not None and anchor_w_t is not None and anchor_ids_t.numel() > 0:
            diff = x[anchor_ids_t] - anchor_uv_t
            anchor_term = torch.sum(anchor_w_t[:, None] * (diff * diff)) / float(max(2 * anchor_ids_t.numel(), 1))

        low_pen = torch.relu(lower - x)
        high_pen = torch.relu(x - upper)
        box_term = torch.mean(low_pen * low_pen + high_pen * high_pen)
        total = data_term + smooth_term + anchor_term + float(box_weight) * box_term
        return total, data_term, smooth_term, anchor_term, box_term

    def _eval_no_grad(x):
        with torch.no_grad():
            vals = _terms(x)
            return tuple(float(v.detach().cpu().item()) for v in vals)

    init_total, init_data, init_smooth, init_anchor, init_box = _eval_no_grad(uv_var)
    meta["uv_solver_constraint_energy_init"] = init_total
    meta["uv_solver_constraint_energy_data_init"] = init_data
    meta["uv_solver_constraint_energy_smooth_init"] = init_smooth
    meta["uv_solver_constraint_energy_anchor_init"] = init_anchor
    meta["uv_solver_constraint_energy_box_init"] = init_box

    best_uv = uv_var.detach().clone()
    best_total = float(init_total)
    best_box = float(init_box)
    prev_total: Optional[float] = None
    patience_count = 0
    stop_reason = "max_iters_reached"
    iters_done = 0

    for step in range(1, refine_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        total_t, _, _, _, _ = _terms(uv_var)
        if not torch.isfinite(total_t):
            stop_reason = "nonfinite_energy"
            break
        total_t.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([uv_var], grad_clip)
        optimizer.step()
        iters_done = step

        cur_total, _, _, _, cur_box = _eval_no_grad(uv_var)
        better_box = cur_box < (best_box - 1e-12)
        better_total = cur_total < (best_total - 1e-12)
        if better_box or (abs(cur_box - best_box) <= 1e-12 and better_total):
            best_uv = uv_var.detach().clone()
            best_total = float(cur_total)
            best_box = float(cur_box)

        if prev_total is not None and np.isfinite(prev_total):
            rel = abs(prev_total - cur_total) / max(abs(prev_total), 1e-12)
            if rel <= early_rel_tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= early_patience:
                stop_reason = "early_stop_rel_tol"
                break
        prev_total = float(cur_total)

    uv_out = best_uv.detach().cpu().numpy().astype(np.float32)
    final_total, final_data, final_smooth, final_anchor, final_box = _eval_no_grad(best_uv)
    final_feas = summarize_uv_box_feasibility(mesh, uv_out, margin=box_margin)
    meta.update(
        {
            "uv_solver_constraint_applied": True,
            "uv_solver_constraint_status": "ok",
            "uv_solver_constraint_iters": int(iters_done),
            "uv_solver_constraint_stop_reason": stop_reason,
            "uv_solver_constraint_energy_final": final_total,
            "uv_solver_constraint_energy_data_final": final_data,
            "uv_solver_constraint_energy_smooth_final": final_smooth,
            "uv_solver_constraint_energy_anchor_final": final_anchor,
            "uv_solver_constraint_energy_box_final": final_box,
            "uv_solver_constraint_vertex_oob_ratio_final": final_feas["vertex_oob_ratio"],
            "uv_solver_constraint_face_oob_ratio_final": final_feas["face_oob_ratio"],
            "uv_solver_constraint_barrier_active_vertex_ratio_final": final_feas["barrier_active_vertex_ratio"],
            "uv_solver_constraint_barrier_active_face_ratio_final": final_feas["barrier_active_face_ratio"],
            "uv_solver_constraint_max_oob_overshoot_final": final_feas["max_oob_overshoot"],
            "uv_solver_constraint_max_barrier_overshoot_final": final_feas["max_barrier_overshoot"],
        }
    )
    return uv_out, meta


__all__ = [
    "compute_uv_box_feasibility_arrays",
    "refine_uv_with_soft_box_constraint",
    "resolve_constraint_device",
    "summarize_uv_box_feasibility",
]
