from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import trimesh

from .linear_solver import interpolate_sample_uv
from .method2_pipeline import Method2InternalState, run_method2_gradient_poisson
from .quality import texture_reprojection_error


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return edges.astype(np.int64)


def _triangle_det_uv(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = uv[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _pre_repair_inverted_uv(
    *,
    uv_init: np.ndarray,
    faces: np.ndarray,
    det_eps: float,
    max_iters: int,
    step: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    uv = np.asarray(uv_init, dtype=np.float32).copy()
    if uv.size == 0 or faces.size == 0 or max_iters <= 0 or step <= 0.0:
        det0 = _triangle_det_uv(uv, faces) if faces.size > 0 else np.zeros((0,), dtype=np.float32)
        viol0 = int(np.count_nonzero(det0 <= det_eps))
        return uv, {
            "uv_m4_pre_repair_enabled": bool(max_iters > 0 and step > 0.0),
            "uv_m4_pre_repair_iters_used": 0,
            "uv_m4_pre_repair_initial_violations": int(viol0),
            "uv_m4_pre_repair_final_violations": int(viol0),
        }

    det = _triangle_det_uv(uv, faces)
    init_viol = int(np.count_nonzero(det <= det_eps))
    if init_viol == 0:
        return uv, {
            "uv_m4_pre_repair_enabled": True,
            "uv_m4_pre_repair_iters_used": 0,
            "uv_m4_pre_repair_initial_violations": 0,
            "uv_m4_pre_repair_final_violations": 0,
        }

    iters_used = 0
    for it in range(max_iters):
        det = _triangle_det_uv(uv, faces)
        bad = det <= det_eps
        if not np.any(bad):
            break
        iters_used = it + 1
        fb = faces[bad]
        tri = uv[fb]
        det_bad = det[bad]

        # Analytic gradients of oriented area wrt each UV vertex.
        g0 = np.stack([tri[:, 2, 1] - tri[:, 1, 1], tri[:, 1, 0] - tri[:, 2, 0]], axis=1)
        g1 = np.stack([tri[:, 0, 1] - tri[:, 2, 1], tri[:, 2, 0] - tri[:, 0, 0]], axis=1)
        g2 = np.stack([tri[:, 1, 1] - tri[:, 0, 1], tri[:, 0, 0] - tri[:, 1, 0]], axis=1)

        gn2 = np.sum(g0 * g0, axis=1) + np.sum(g1 * g1, axis=1) + np.sum(g2 * g2, axis=1)
        scale = (step * (det_eps - det_bad) / np.maximum(gn2, 1e-12)).astype(np.float32, copy=False)
        d0 = g0 * scale[:, None]
        d1 = g1 * scale[:, None]
        d2 = g2 * scale[:, None]

        corr = np.zeros_like(uv, dtype=np.float32)
        cnt = np.zeros((uv.shape[0],), dtype=np.float32)
        np.add.at(corr, fb[:, 0], d0)
        np.add.at(corr, fb[:, 1], d1)
        np.add.at(corr, fb[:, 2], d2)
        np.add.at(cnt, fb[:, 0], 1.0)
        np.add.at(cnt, fb[:, 1], 1.0)
        np.add.at(cnt, fb[:, 2], 1.0)

        valid = cnt > 0.0
        if not np.any(valid):
            break
        uv[valid] += corr[valid] / cnt[valid, None]

    det_final = _triangle_det_uv(uv, faces)
    final_viol = int(np.count_nonzero(det_final <= det_eps))
    return uv, {
        "uv_m4_pre_repair_enabled": True,
        "uv_m4_pre_repair_iters_used": int(iters_used),
        "uv_m4_pre_repair_initial_violations": int(init_viol),
        "uv_m4_pre_repair_final_violations": int(final_viol),
    }


def _resolve_opt_device(requested: str, default_device: str) -> str:
    req = str(requested or "auto").strip().lower()
    if req == "auto":
        req = default_device if default_device.startswith("cuda") else "cpu"
    if req.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return req
        except Exception:
            pass
    return "cpu"


def _validate_method4_state(state: Method2InternalState) -> None:
    solve_mesh = state.solve_mesh
    faces_np = np.asarray(solve_mesh.faces, dtype=np.int64)
    n_faces = int(len(faces_np))
    n_vertices = int(len(solve_mesh.vertices))

    mapped_uv_init = np.asarray(state.mapped_uv_init)
    if mapped_uv_init.shape != (n_vertices, 2):
        raise RuntimeError(
            f"method4 invalid mapped_uv_init shape: expected {(n_vertices, 2)}, got {mapped_uv_init.shape}"
        )

    face_pinv = np.asarray(state.face_geom_pinv)
    if face_pinv.shape != (n_faces, 2, 3):
        raise RuntimeError(
            f"method4 invalid face_geom_pinv shape: expected {(n_faces, 2, 3)}, got {face_pinv.shape}"
        )

    face_target_jacobian = np.asarray(state.face_target_jacobian)
    if face_target_jacobian.shape != (n_faces, 2, 3):
        raise RuntimeError(
            f"method4 invalid face_target_jacobian shape: expected {(n_faces, 2, 3)}, got {face_target_jacobian.shape}"
        )

    face_target_valid_mask = np.asarray(state.face_target_valid_mask, dtype=np.bool_).reshape(-1)
    if face_target_valid_mask.shape[0] != n_faces:
        raise RuntimeError(
            f"method4 invalid face_target_valid_mask length: expected {n_faces}, got {face_target_valid_mask.shape[0]}"
        )

    face_target_weights = np.asarray(state.face_target_weights).reshape(-1)
    if face_target_weights.shape[0] != n_faces:
        raise RuntimeError(
            f"method4 invalid face_target_weights length: expected {n_faces}, got {face_target_weights.shape[0]}"
        )
    if np.any(face_target_valid_mask & ~np.isfinite(face_target_weights)):
        raise RuntimeError("method4 got non-finite face_target_weights on valid faces")
    if np.any(face_target_valid_mask & (face_target_weights < 0.0)):
        raise RuntimeError("method4 got negative face_target_weights on valid faces")

    if np.any(face_target_valid_mask & ~np.isfinite(face_target_jacobian).all(axis=(1, 2))):
        raise RuntimeError("method4 got non-finite face_target_jacobian on valid faces")

    anchor_ids_np = np.asarray(state.anchor_vertex_ids, dtype=np.int64).reshape(-1)
    anchor_uv_np = np.asarray(state.anchor_uv)
    if anchor_ids_np.size > 0:
        if anchor_uv_np.shape != (anchor_ids_np.size, 2):
            raise RuntimeError(
                f"method4 invalid anchor_uv shape: expected {(anchor_ids_np.size, 2)}, got {anchor_uv_np.shape}"
            )
        if np.any((anchor_ids_np < 0) | (anchor_ids_np >= n_vertices)):
            raise RuntimeError("method4 got out-of-range anchor vertex ids")
        if not np.isfinite(anchor_uv_np).all():
            raise RuntimeError("method4 got non-finite anchor_uv values")


def _run_nonlinear_refine(
    *,
    state: Method2InternalState,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch

    m4_cfg = cfg.get("method4", {})
    max_iters = max(1, int(m4_cfg.get("max_iters", 120)))
    lr = float(m4_cfg.get("lr", 0.25))
    jacobian_w = float(m4_cfg.get("jacobian_weight", 1.0))
    smooth_w = float(m4_cfg.get("smooth_weight", 1e-6))
    symd_w = float(m4_cfg.get("sym_dirichlet_weight", 2e-2))
    logdet_w = float(m4_cfg.get("logdet_barrier_weight", 1e-2))
    flip_barrier_w = float(m4_cfg.get("flip_barrier_weight", 5e-2))
    legacy_barrier_w = float(m4_cfg.get("barrier_weight", 0.0))
    anchor_w = float(m4_cfg.get("anchor_weight", 5e-4))
    det_eps = float(m4_cfg.get("det_eps", 1e-7))
    det_softplus_beta = float(m4_cfg.get("det_softplus_beta", 40.0))
    area_eps = float(m4_cfg.get("area_eps", 1e-10))
    grad_clip = float(m4_cfg.get("grad_clip", 5.0))
    early_rel_tol = float(m4_cfg.get("early_stop_rel_tol", 1e-5))
    early_patience = max(1, int(m4_cfg.get("early_stop_patience", 10)))
    max_line_search_fail = max(1, int(m4_cfg.get("max_line_search_fail", 16)))
    line_alpha = float(m4_cfg.get("line_search_alpha", 0.5))
    line_c1 = float(m4_cfg.get("line_search_c1", 1e-4))
    recovery_mode_enabled = bool(m4_cfg.get("recovery_mode_enabled", False))
    recovery_det_improve_eps = max(0.0, float(m4_cfg.get("recovery_det_improve_eps", 1e-8)))
    patch_rounds = max(0, int(m4_cfg.get("patch_refine_rounds", 3)))
    patch_steps = max(1, int(m4_cfg.get("patch_refine_steps", 80)))
    patch_lr = float(m4_cfg.get("patch_refine_lr", 0.05))
    homotopy_enabled = bool(m4_cfg.get("barrier_homotopy_enabled", True))
    homotopy_warmup = max(1, int(m4_cfg.get("barrier_homotopy_warmup_iters", 40)))
    pre_repair_enabled = bool(m4_cfg.get("pre_repair_enabled", True))
    pre_repair_iters = max(0, int(m4_cfg.get("pre_repair_iters", 8)))
    pre_repair_step = float(m4_cfg.get("pre_repair_step", 0.25))
    optimizer_name = str(m4_cfg.get("optimizer", "lbfgs")).strip().lower()
    if optimizer_name not in {"lbfgs", "adam"}:
        optimizer_name = "lbfgs"

    solve_mesh = state.solve_mesh
    faces_np = np.asarray(solve_mesh.faces, dtype=np.int64)
    if faces_np.size == 0:
        return state.mapped_uv_init.copy(), {
            "uv_m4_enabled": True,
            "uv_m4_refine_status": "skipped_empty_mesh",
            "uv_m4_nonlinear_iters": 0,
            "uv_m4_energy_init": None,
            "uv_m4_energy_final": None,
            "uv_m4_barrier_violations": 0,
            "uv_m4_line_search_fail_count": 0,
            "uv_m4_patch_refine_rounds": 0,
            "uv_m4_det_min": None,
            "uv_m4_det_p01": None,
        }

    valid_face = np.asarray(state.face_target_valid_mask, dtype=np.bool_)
    if not np.any(valid_face):
        return state.mapped_uv_init.copy(), {
            "uv_m4_enabled": True,
            "uv_m4_refine_status": "skipped_no_valid_jacobians",
            "uv_m4_nonlinear_iters": 0,
            "uv_m4_energy_init": None,
            "uv_m4_energy_final": None,
            "uv_m4_barrier_violations": 0,
            "uv_m4_line_search_fail_count": 0,
            "uv_m4_patch_refine_rounds": 0,
            "uv_m4_det_min": None,
            "uv_m4_det_p01": None,
        }

    opt_device = _resolve_opt_device(str(m4_cfg.get("device", "auto")), state.resolved_device)
    dev = torch.device(opt_device)

    faces_t = torch.tensor(faces_np, dtype=torch.long, device=dev)
    edges_np = _mesh_edges(faces_np)
    edges_t = (
        torch.tensor(edges_np, dtype=torch.long, device=dev)
        if edges_np.size > 0
        else torch.zeros((0, 2), dtype=torch.long, device=dev)
    )
    pinv_t = torch.tensor(np.asarray(state.face_geom_pinv, dtype=np.float32), dtype=torch.float32, device=dev)
    # Invalid faces may carry NaN placeholders from upstream projectors; zero them here so
    # Method4 can rely on face_valid_t/face_w_t masking without 0 * NaN poisoning the data term.
    jac_t_np = np.nan_to_num(
        np.asarray(state.face_target_jacobian, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    jac_t = torch.tensor(jac_t_np, dtype=torch.float32, device=dev)
    face_w_np = np.nan_to_num(
        np.asarray(state.face_target_weights, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    face_w_np = np.maximum(face_w_np, 0.0)
    face_w_np[~valid_face] = 0.0
    if np.any(valid_face):
        mean_w = float(np.mean(face_w_np[valid_face]))
        if mean_w > 0:
            face_w_np = face_w_np / mean_w
    face_w_t = torch.tensor(face_w_np, dtype=torch.float32, device=dev)
    face_valid_t = torch.tensor(valid_face.astype(np.float32), dtype=torch.float32, device=dev)

    anchor_ids_np = np.asarray(state.anchor_vertex_ids, dtype=np.int64)
    if anchor_ids_np.size > 0:
        anchor_ids_t = torch.tensor(anchor_ids_np, dtype=torch.long, device=dev)
        anchor_uv_t = torch.tensor(np.asarray(state.anchor_uv, dtype=np.float32), dtype=torch.float32, device=dev)
    else:
        anchor_ids_t = None
        anchor_uv_t = None

    uv_init_np = np.asarray(state.mapped_uv_init, dtype=np.float32)
    pre_repair_meta: Dict[str, Any] = {
        "uv_m4_pre_repair_enabled": bool(pre_repair_enabled),
        "uv_m4_pre_repair_iters_used": 0,
        "uv_m4_pre_repair_initial_violations": 0,
        "uv_m4_pre_repair_final_violations": 0,
    }
    if pre_repair_enabled:
        uv_init_np, pre_repair_meta = _pre_repair_inverted_uv(
            uv_init=uv_init_np,
            faces=faces_np,
            det_eps=det_eps,
            max_iters=pre_repair_iters,
            step=pre_repair_step,
        )
    uv_var = torch.tensor(np.asarray(uv_init_np, dtype=np.float32), dtype=torch.float32, device=dev)
    uv_var.requires_grad_(True)
    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([uv_var], lr=lr, max_iter=1, history_size=20, line_search_fn=None)
    else:
        optimizer = torch.optim.Adam([uv_var], lr=lr)

    def compute_terms(x, *, logdet_scale: float = 1.0):
        tri_uv = x[faces_t]
        du1 = tri_uv[:, 1] - tri_uv[:, 0]
        du2 = tri_uv[:, 2] - tri_uv[:, 0]
        fmat = torch.stack([du1, du2], dim=2)  # [F, 2, 2]
        jac_now = torch.bmm(fmat, pinv_t)  # [F, 2, 3]

        valid_w = face_w_t * face_valid_t
        wsum = torch.clamp(torch.sum(valid_w), min=1e-8)

        jac_res = jac_now - jac_t
        data_term = torch.sum(torch.sum(jac_res * jac_res, dim=(1, 2)) * valid_w) / wsum

        if edges_t.numel() > 0:
            edge_delta = x[edges_t[:, 0]] - x[edges_t[:, 1]]
            smooth_term = torch.mean(torch.sum(edge_delta * edge_delta, dim=1))
        else:
            smooth_term = torch.zeros((), device=x.device, dtype=torch.float32)

        det = fmat[:, 0, 0] * fmat[:, 1, 1] - fmat[:, 0, 1] * fmat[:, 1, 0]
        beta = max(float(det_softplus_beta), 1e-3)
        det_pos = torch.nn.functional.softplus((det - det_eps) * beta) / beta + det_eps
        adj00 = fmat[:, 1, 1]
        adj01 = -fmat[:, 0, 1]
        adj10 = -fmat[:, 1, 0]
        adj11 = fmat[:, 0, 0]
        inv = torch.stack(
            [
                torch.stack([adj00 / det_pos, adj01 / det_pos], dim=1),
                torch.stack([adj10 / det_pos, adj11 / det_pos], dim=1),
            ],
            dim=1,
        )
        norm_f = torch.sum(fmat * fmat, dim=(1, 2))
        norm_inv = torch.sum(inv * inv, dim=(1, 2))
        symd_term = torch.sum((norm_f + norm_inv) * valid_w) / wsum

        logdet_term = torch.mean(-torch.log(torch.clamp(det_pos, min=1e-12)))
        flip_term = torch.mean(torch.relu(det_eps - det) ** 2)
        if legacy_barrier_w > 0.0:
            legacy_barrier_term = torch.mean(torch.relu(area_eps - det) ** 2)
        else:
            legacy_barrier_term = torch.zeros((), device=x.device, dtype=torch.float32)

        if anchor_ids_t is not None and anchor_uv_t is not None and anchor_ids_t.numel() > 0:
            anchor_term = torch.mean((x[anchor_ids_t] - anchor_uv_t) ** 2)
        else:
            anchor_term = torch.zeros((), device=x.device, dtype=torch.float32)

        total = (
            jacobian_w * data_term
            + smooth_w * smooth_term
            + symd_w * symd_term
            + (logdet_w * float(logdet_scale)) * logdet_term
            + (flip_barrier_w * float(logdet_scale)) * flip_term
            + legacy_barrier_w * legacy_barrier_term
            + anchor_w * anchor_term
        )
        return total, data_term, smooth_term, symd_term, logdet_term, flip_term, legacy_barrier_term, anchor_term, det

    def eval_no_grad(x, *, logdet_scale: float = 1.0):
        with torch.no_grad():
            t = compute_terms(x, logdet_scale=logdet_scale)
            energy = float(t[0].detach().cpu().item())
            det_np = t[8].detach().cpu().numpy().astype(np.float64)
        return t, energy, det_np

    init_terms, init_energy, _ = eval_no_grad(uv_var, logdet_scale=0.0 if homotopy_enabled else 1.0)
    best_energy = init_energy
    best_uv = uv_var.detach().clone()
    stop_reason = "max_iters"
    line_search_fail_count = 0
    line_search_backtrack_count = 0
    accepted_step_count = 0
    recovery_accepted_step_count = 0
    iters_done = 0
    patience_count = 0
    prev_energy: Optional[float] = None

    def accept_candidate(
        *,
        prev_energy_cur: float,
        prev_det: np.ndarray,
        cand_energy: float,
        cand_det: np.ndarray,
    ) -> Tuple[bool, str]:
        if not np.isfinite(cand_energy):
            return False, "nonfinite_energy"
        energy_ok = cand_energy <= prev_energy_cur * (1.0 + line_c1)
        cand_det_min = float(np.min(cand_det)) if cand_det.size > 0 else float("inf")
        if energy_ok and cand_det_min > det_eps * 0.1:
            return True, "strict"

        prev_viol = int(np.count_nonzero(prev_det <= det_eps))
        if not recovery_mode_enabled or prev_viol <= 0 or not energy_ok:
            return False, "strict_reject"

        cand_viol = int(np.count_nonzero(cand_det <= det_eps))
        prev_det_min = float(np.min(prev_det)) if prev_det.size > 0 else float("inf")
        if cand_viol < prev_viol:
            return True, "recovery_lower_violation_count"
        if cand_viol == prev_viol and cand_det_min > prev_det_min + recovery_det_improve_eps:
            return True, "recovery_same_violation_better_det"
        return False, "recovery_reject"

    for step in range(1, max_iters + 1):
        if homotopy_enabled:
            logdet_scale = min(float(step) / float(homotopy_warmup), 1.0)
        else:
            logdet_scale = 1.0
        prev_uv = uv_var.detach().clone()
        _, prev_energy_cur, prev_det = eval_no_grad(prev_uv, logdet_scale=logdet_scale)

        if optimizer_name == "lbfgs":
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = compute_terms(uv_var, logdet_scale=logdet_scale)[0]
                if not torch.isfinite(loss):
                    loss = torch.nan_to_num(loss, nan=1e12, posinf=1e12, neginf=1e12)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_([uv_var], grad_clip)
                return loss

            optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_terms(uv_var, logdet_scale=logdet_scale)[0]
            if not torch.isfinite(loss):
                stop_reason = "nonfinite_energy"
                break
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([uv_var], grad_clip)
            optimizer.step()

        iters_done = step
        _, cur_energy, cur_det = eval_no_grad(uv_var, logdet_scale=logdet_scale)
        accepted_step, accept_mode = accept_candidate(
            prev_energy_cur=prev_energy_cur,
            prev_det=prev_det,
            cand_energy=cur_energy,
            cand_det=cur_det,
        )
        needs_backtrack = not accepted_step
        if needs_backtrack:
            line_search_backtrack_count += 1
            accepted = False
            accepted_mode = "none"
            direction = uv_var.detach() - prev_uv
            t = line_alpha
            while t > 1e-4:
                candidate = prev_uv + t * direction
                _, cand_energy, cand_det = eval_no_grad(candidate, logdet_scale=logdet_scale)
                cand_accept, cand_mode = accept_candidate(
                    prev_energy_cur=prev_energy_cur,
                    prev_det=prev_det,
                    cand_energy=cand_energy,
                    cand_det=cand_det,
                )
                if cand_accept:
                    with torch.no_grad():
                        uv_var.copy_(candidate)
                    cur_energy = cand_energy
                    cur_det = cand_det
                    accepted = True
                    accepted_step = True
                    accept_mode = cand_mode
                    accepted_mode = cand_mode
                    break
                t *= line_alpha
            if not accepted:
                line_search_fail_count += 1
                accepted_step = False
                with torch.no_grad():
                    uv_var.copy_(prev_uv)
                cur_energy = prev_energy_cur
                cur_det = prev_det
            if line_search_fail_count >= max_line_search_fail:
                stop_reason = "line_search_fail_limit"
                break
        else:
            accepted_mode = accept_mode

        if accepted_step:
            accepted_step_count += 1
            if str(accepted_mode).startswith("recovery_"):
                recovery_accepted_step_count += 1

        if cur_energy < best_energy:
            best_energy = cur_energy
            best_uv = uv_var.detach().clone()

        if prev_energy is not None and np.isfinite(prev_energy):
            rel = abs(prev_energy - cur_energy) / max(abs(prev_energy), 1e-12)
            if rel <= early_rel_tol:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= early_patience:
                stop_reason = "early_stop_rel_tol"
                break
        prev_energy = cur_energy

    patch_rounds_used = 0
    best_terms, best_energy_eval, best_det = eval_no_grad(best_uv, logdet_scale=1.0)
    best_det_min = float(np.min(best_det)) if best_det.size > 0 else float("inf")
    best_det_p01 = float(np.percentile(best_det, 1)) if best_det.size > 0 else float("inf")
    barrier_violations = int(np.count_nonzero(best_det <= det_eps))

    if barrier_violations > 0 and patch_rounds > 0:
        uv_patch = best_uv.detach().clone().requires_grad_(True)
        patch_optimizer = torch.optim.Adam([uv_patch], lr=patch_lr)
        for r in range(1, patch_rounds + 1):
            _, _, det_now = eval_no_grad(uv_patch, logdet_scale=1.0)
            bad_faces = np.where(det_now <= det_eps)[0]
            if bad_faces.size == 0:
                break
            patch_rounds_used = r
            bad_vid = np.unique(faces_np[bad_faces].reshape(-1))
            update_mask = torch.zeros((uv_patch.shape[0],), dtype=torch.bool, device=uv_patch.device)
            update_mask[torch.tensor(bad_vid, dtype=torch.long, device=uv_patch.device)] = True

            for _ in range(patch_steps):
                patch_optimizer.zero_grad(set_to_none=True)
                loss = compute_terms(uv_patch, logdet_scale=1.0)[0]
                if not torch.isfinite(loss):
                    break
                loss.backward()
                if uv_patch.grad is not None:
                    uv_patch.grad[~update_mask] = 0.0
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_([uv_patch], grad_clip)
                patch_optimizer.step()

        patch_terms, patch_energy, patch_det = eval_no_grad(uv_patch, logdet_scale=1.0)
        patch_viol = int(np.count_nonzero(patch_det <= det_eps))
        if patch_viol < barrier_violations or (patch_viol == barrier_violations and patch_energy < best_energy_eval):
            best_uv = uv_patch.detach().clone()
            best_terms = patch_terms
            best_energy_eval = patch_energy
            best_det = patch_det
            barrier_violations = patch_viol
            best_det_min = float(np.min(best_det)) if best_det.size > 0 else float("inf")
            best_det_p01 = float(np.percentile(best_det, 1)) if best_det.size > 0 else float("inf")

    uv_out = best_uv.detach().cpu().numpy().astype(np.float32)
    final_total = float(best_energy_eval)
    final_data = float(best_terms[1].detach().cpu().item())
    final_smooth = float(best_terms[2].detach().cpu().item())
    final_symd = float(best_terms[3].detach().cpu().item())
    final_logdet = float(best_terms[4].detach().cpu().item())
    final_flip = float(best_terms[5].detach().cpu().item())
    final_legacy = float(best_terms[6].detach().cpu().item())
    final_anchor = float(best_terms[7].detach().cpu().item())

    refine_status = "ok" if accepted_step_count > 0 else "stalled_no_accepted_step"
    meta = {
        "uv_m4_enabled": True,
        "uv_m4_refine_status": refine_status,
        "uv_m4_device": str(dev),
        "uv_m4_optimizer": optimizer_name,
        "uv_m4_nonlinear_iters": int(iters_done),
        "uv_m4_accepted_step_count": int(accepted_step_count),
        "uv_m4_energy_init": float(init_energy),
        "uv_m4_energy_final": float(final_total),
        "uv_m4_energy_data_init": float(init_terms[1].detach().cpu().item()),
        "uv_m4_energy_data_final": float(final_data),
        "uv_m4_energy_smooth_final": float(final_smooth),
        "uv_m4_energy_symd_final": float(final_symd),
        "uv_m4_energy_logdet_final": float(final_logdet),
        "uv_m4_energy_flip_final": float(final_flip),
        "uv_m4_energy_barrier_final": float(final_legacy),
        "uv_m4_energy_anchor_final": float(final_anchor),
        "uv_m4_barrier_violations": int(barrier_violations),
        "uv_m4_barrier_homotopy_enabled": bool(homotopy_enabled),
        "uv_m4_barrier_homotopy_warmup_iters": int(homotopy_warmup),
        "uv_m4_line_search_fail_count": int(line_search_fail_count),
        "uv_m4_line_search_backtrack_count": int(line_search_backtrack_count),
        "uv_m4_recovery_mode_enabled": bool(recovery_mode_enabled),
        "uv_m4_recovery_det_improve_eps": float(recovery_det_improve_eps),
        "uv_m4_recovery_accepted_step_count": int(recovery_accepted_step_count),
        "uv_m4_patch_refine_rounds": int(patch_rounds_used),
        "uv_m4_det_min": float(best_det_min),
        "uv_m4_det_p01": float(best_det_p01),
        "uv_m4_stop_reason": stop_reason,
        **pre_repair_meta,
    }
    return uv_out, meta


def run_method4_from_internal_state(
    *,
    internal: Method2InternalState,
    base_stats: Dict[str, Any],
    export_payload: Dict[str, Any],
    image,
    cfg: Dict[str, Any],
    success_solver_stage: str = "m4",
    fallback_solver_stage: str = "m2_fallback_after_m4",
    disabled_solver_stage: str = "m2",
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    m4_cfg = cfg.get("method4", {})
    if not bool(m4_cfg.get("enabled", True)):
        out_stats = dict(base_stats)
        out_stats.update(
            {
                "uv_solver_stage": disabled_solver_stage,
                "uv_m4_enabled": False,
                "uv_m4_refine_status": "disabled_by_config",
                "uv_m4_nonlinear_iters": 0,
                "uv_m4_energy_init": None,
                "uv_m4_energy_final": None,
                "uv_m4_barrier_violations": None,
                "uv_m4_line_search_fail_count": 0,
                "uv_m4_line_search_backtrack_count": 0,
                "uv_m4_patch_refine_rounds": 0,
                "uv_m4_det_min": None,
                "uv_m4_det_p01": None,
            }
        )
        return internal.mapped_uv_init.copy(), out_stats, export_payload

    _validate_method4_state(internal)

    mapped_uv_m4, m4_meta = _run_nonlinear_refine(state=internal, cfg=cfg)
    fallback_on_violation = bool(m4_cfg.get("fallback_to_method2_on_violation", True))
    barrier_viol = int(m4_meta.get("uv_m4_barrier_violations", 0))
    valid_faces = int(np.count_nonzero(np.asarray(internal.face_target_valid_mask, dtype=np.bool_)))
    violation_ratio = float(barrier_viol / max(1, valid_faces))
    viol_ratio_tol = max(0.0, float(m4_cfg.get("fallback_violation_ratio_tol", 0.02)))
    viol_count_tol = max(0, int(m4_cfg.get("fallback_violation_count_tol", 8)))
    m4_meta["uv_m4_barrier_violation_ratio"] = violation_ratio
    m4_meta["uv_m4_barrier_violation_ratio_tol"] = float(viol_ratio_tol)
    m4_meta["uv_m4_barrier_violation_count_tol"] = int(viol_count_tol)

    fallback_triggered = barrier_viol > viol_count_tol and violation_ratio > viol_ratio_tol
    if fallback_triggered and fallback_on_violation:
        out_stats = dict(base_stats)
        out_stats.update(m4_meta)
        out_stats["uv_solver_stage"] = fallback_solver_stage
        out_stats["uv_m4_refine_status"] = "fallback_to_m2_injective_failed"
        return internal.mapped_uv_init.copy(), out_stats, export_payload

    solve_mesh = internal.solve_mesh
    if internal.solve_sample_face_ids.size > 0:
        pred_uv = interpolate_sample_uv(
            np.asarray(solve_mesh.faces, dtype=np.int64),
            internal.solve_sample_face_ids,
            internal.solve_sample_bary,
            mapped_uv_m4,
        )
        color_l1, color_l2 = texture_reprojection_error(image, internal.solve_target_uv, pred_uv)
    else:
        color_l1, color_l2 = None, None

    out_stats = dict(base_stats)
    out_stats["uv_solver_stage"] = success_solver_stage
    out_stats["uv_color_reproj_l1"] = color_l1
    out_stats["uv_color_reproj_l2"] = color_l2
    out_stats.update(m4_meta)
    return mapped_uv_m4, out_stats, export_payload


def run_method4_jacobian_injective(
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
    m2_out = run_method2_gradient_poisson(
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
    mapped_uv_m2, method2_stats, export_payload, internal = m2_out

    if internal is None:
        out_stats = dict(method2_stats)
        out_stats.update(
            {
                "uv_solver_stage": "m2",
                "uv_m4_enabled": True,
                "uv_m4_refine_status": "skipped_no_internal_state",
                "uv_m4_nonlinear_iters": 0,
                "uv_m4_energy_init": None,
                "uv_m4_energy_final": None,
                "uv_m4_barrier_violations": None,
                "uv_m4_line_search_fail_count": 0,
                "uv_m4_line_search_backtrack_count": 0,
                "uv_m4_patch_refine_rounds": 0,
                "uv_m4_det_min": None,
                "uv_m4_det_p01": None,
            }
        )
        return mapped_uv_m2, out_stats, export_payload

    return run_method4_from_internal_state(
        internal=internal,
        base_stats=method2_stats,
        export_payload=export_payload,
        image=image,
        cfg=cfg,
        success_solver_stage="m4",
        fallback_solver_stage="m2_fallback_after_m4",
        disabled_solver_stage="m2",
    )


__all__ = ["run_method4_from_internal_state", "run_method4_jacobian_injective"]
