from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh


def _face_adjacency_lists(mesh: trimesh.Trimesh) -> List[List[int]]:
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


def _majority_vote_face_labels(
    labels: np.ndarray,
    neighbors: List[List[int]],
    *,
    iters: int,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(labels, dtype=np.int64).copy()
    total_changes = 0
    for _ in range(max(0, int(iters))):
        changed = 0
        nxt = out.copy()
        for f, nbs in enumerate(neighbors):
            lf = int(out[f])
            if lf < 0 or len(nbs) == 0:
                continue
            vals = [int(out[j]) for j in nbs if int(out[j]) >= 0]
            if len(vals) < 2:
                continue
            if lf in vals:
                continue
            uniq, cnt = np.unique(np.asarray(vals, dtype=np.int64), return_counts=True)
            pick = int(uniq[int(np.argmax(cnt))])
            if pick != lf:
                nxt[f] = pick
                changed += 1
        out = nxt
        total_changes += changed
        if changed == 0:
            break
    return out, int(total_changes)


def _majority_nonneg(values: List[int]) -> int:
    counts: Dict[int, int] = {}
    best_label = -1
    best_count = 0
    for v in values:
        iv = int(v)
        if iv < 0:
            continue
        c = counts.get(iv, 0) + 1
        counts[iv] = c
        if c > best_count or (c == best_count and (best_label < 0 or iv < best_label)):
            best_label = iv
            best_count = c
    return int(best_label)


def _fill_unknown_face_labels(
    labels: np.ndarray,
    neighbors: List[List[int]],
    *,
    iters: int,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(labels, dtype=np.int64).copy()
    total_changes = 0
    for _ in range(max(0, int(iters))):
        changed = 0
        nxt = out.copy()
        for f, nbs in enumerate(neighbors):
            if int(out[f]) >= 0 or len(nbs) == 0:
                continue
            vals = [int(out[j]) for j in nbs if int(out[j]) >= 0]
            if len(vals) == 0:
                continue
            pick = _majority_nonneg(vals)
            if pick >= 0:
                nxt[f] = pick
                changed += 1
        out = nxt
        total_changes += changed
        if changed == 0:
            break
    return out, int(total_changes)


def _morphological_close_face_labels(
    labels: np.ndarray,
    neighbors: List[List[int]],
    *,
    iters: int,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(labels, dtype=np.int64).copy()
    total_changes = 0
    for _ in range(max(0, int(iters))):
        # Dilation on face graph: pull to local majority in one-ring.
        dil = out.copy()
        for f, nbs in enumerate(neighbors):
            if len(nbs) == 0:
                continue
            vals = [int(out[f])]
            vals.extend(int(out[j]) for j in nbs)
            pick = _majority_nonneg(vals)
            if pick >= 0:
                dil[f] = pick

        # Erosion: suppress isolated spikes that disagree with one-ring majority.
        ero = dil.copy()
        for f, nbs in enumerate(neighbors):
            if len(nbs) == 0:
                continue
            vals = [int(dil[f])]
            vals.extend(int(dil[j]) for j in nbs)
            pick = _majority_nonneg(vals)
            if pick < 0:
                continue
            if int(ero[f]) != pick:
                ero[f] = pick
                total_changes += 1
        out = ero
    return out, int(total_changes)


def _face_label_confidence(labels: np.ndarray, neighbors: List[List[int]]) -> np.ndarray:
    out = np.zeros((len(labels),), dtype=np.float32)
    for f, nbs in enumerate(neighbors):
        lf = int(labels[f])
        if lf < 0 or len(nbs) == 0:
            continue
        vals = [int(labels[j]) for j in nbs if int(labels[j]) >= 0]
        if len(vals) == 0:
            continue
        same = sum(1 for v in vals if v == lf)
        out[f] = float(same / max(1, len(vals)))
    return out


def transfer_face_semantics_by_projection(
    *,
    high_ctx: Dict[str, Any],
    high_face_island: np.ndarray,
    low_mesh: trimesh.Trimesh,
    seam_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    import torch

    tri = np.asarray(low_mesh.vertices, dtype=np.float32)[np.asarray(low_mesh.faces, dtype=np.int64)]
    if tri.size == 0:
        n_faces = int(len(low_mesh.faces))
        labels = np.full((n_faces,), -1, dtype=np.int64)
        conf = np.zeros((n_faces,), dtype=np.float32)
        return {
            "low_face_island": labels,
            "low_face_conflict": np.ones((n_faces,), dtype=np.bool_),
            "low_face_confidence": conf,
            "meta": {
                "uv_semantic_transfer_points": 0,
                "uv_semantic_transfer_hits": 0,
                "uv_semantic_transfer_hit_ratio": 0.0,
                "uv_semantic_transfer_failures": 0,
                "uv_semantic_transfer_failure_ratio": 0.0,
                "uv_semantic_transfer_err_miss": 0,
                "uv_semantic_transfer_err_miss_ratio": 0.0,
                "uv_semantic_transfer_err_miss_ratio_in_failed": 0.0,
                "uv_semantic_transfer_err_angle_reject": 0,
                "uv_semantic_transfer_err_angle_reject_ratio": 0.0,
                "uv_semantic_transfer_err_angle_reject_ratio_in_failed": 0.0,
                "uv_semantic_transfer_err_distance": 0,
                "uv_semantic_transfer_err_distance_ratio": 0.0,
                "uv_semantic_transfer_err_distance_ratio_in_failed": 0.0,
                "uv_semantic_transfer_err_unclassified": 0,
                "uv_semantic_transfer_err_unclassified_ratio": 0.0,
                "uv_semantic_transfer_unknown_faces": n_faces,
            },
        }

    centroids = np.mean(tri, axis=1).astype(np.float32)
    normals = np.asarray(low_mesh.face_normals, dtype=np.float32)
    nrm_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nrm_norm, 1e-8)

    v_t = high_ctx["v_t"]
    bvh = high_ctx["bvh"]
    face_normals_t = high_ctx["face_normals_t"]
    bbox_diag = max(1e-6, float(high_ctx["bbox_diag"]))
    high_island = np.asarray(high_face_island, dtype=np.int64)

    # Acceptance policy:
    # 1) Near-hit bypass: if hit distance is sufficiently close, ignore angle mismatch.
    # 2) Otherwise use a relaxed angle threshold (default 60 deg).
    angle_deg = float(seam_cfg.get("transfer_relaxed_normal_angle_deg", 60.0))
    cos_min = float(math.cos(math.radians(max(0.0, min(89.9, angle_deg)))))
    max_dist = float(corr_cfg.get("ray_max_dist_ratio", 0.08)) * bbox_diag
    safe_dist = float(seam_cfg.get("transfer_near_accept_dist_ratio", 0.005)) * bbox_diag
    chunk = max(1, int(corr_cfg.get("bvh_chunk_size", 200000)))
    vote_iters = int(seam_cfg.get("transfer_majority_vote_iters", 1))
    fill_unknown_iters = int(seam_cfg.get("transfer_fill_unknown_iters", 2))
    morph_close_iters = int(seam_cfg.get("transfer_morph_close_iters", 2))
    conf_min = float(seam_cfg.get("uv_island_guard_confidence_min", 0.55))

    n_faces = int(centroids.shape[0])
    selected_face = np.full((n_faces,), -1, dtype=np.int64)
    valid = np.zeros((n_faces,), dtype=np.bool_)
    err_miss = 0
    err_angle_reject = 0
    err_distance = 0
    err_unclassified = 0

    for st in range(0, n_faces, chunk):
        ed = min(st + chunk, n_faces)
        pts_t = torch.tensor(centroids[st:ed], dtype=torch.float32, device=v_t.device)
        nrm_t = torch.tensor(normals[st:ed], dtype=torch.float32, device=v_t.device)
        nrm_t = nrm_t / torch.clamp(torch.linalg.norm(nrm_t, dim=1, keepdim=True), min=1e-8)

        r_pos = bvh.intersect_ray(pts_t, nrm_t, max_t=max_dist)
        r_neg = bvh.intersect_ray(pts_t, -nrm_t, max_t=max_dist)

        face_pos = torch.clamp(r_pos.face_ids.long(), 0, face_normals_t.shape[0] - 1)
        face_neg = torch.clamp(r_neg.face_ids.long(), 0, face_normals_t.shape[0] - 1)
        hit_pos = r_pos.hit.bool() & (r_pos.face_ids.long() >= 0) & torch.isfinite(r_pos.t.float())
        hit_neg = r_neg.hit.bool() & (r_neg.face_ids.long() >= 0) & torch.isfinite(r_neg.t.float())

        dot_pos = torch.sum(face_normals_t[face_pos] * nrm_t, dim=1)
        dot_neg = torch.sum(face_normals_t[face_neg] * (-nrm_t), dim=1)
        t_pos = r_pos.t.float()
        t_neg = r_neg.t.float()
        near_pos = hit_pos & (t_pos <= float(safe_dist))
        near_neg = hit_neg & (t_neg <= float(safe_dist))
        val_pos = hit_pos & (near_pos | (dot_pos >= cos_min))
        val_neg = hit_neg & (near_neg | (dot_neg >= cos_min))
        choose_pos = val_pos & (~val_neg | (t_pos <= t_neg))
        choose_neg = val_neg & (~val_pos | (t_neg < t_pos))

        face_sel = torch.full((ed - st,), -1, dtype=torch.long, device=v_t.device)
        face_sel[choose_pos] = face_pos[choose_pos]
        face_sel[choose_neg] = face_neg[choose_neg]
        ok = choose_pos | choose_neg

        # Failure reason instrumentation for unmatched faces.
        fail = ~ok
        near_hit_any = hit_pos | hit_neg
        angle_reject_mask = fail & near_hit_any
        dist_reject_mask = torch.zeros_like(fail)
        miss_mask = torch.zeros_like(fail)

        # Distance rejection requires probing without max_t to detect far hits
        # (hits exist but are filtered out by current max_dist).
        need_far_probe = fail & (~near_hit_any)
        if bool(torch.any(need_far_probe)):
            probe_idx = torch.nonzero(need_far_probe, as_tuple=False).reshape(-1)
            pts_probe = pts_t[probe_idx]
            nrm_probe = nrm_t[probe_idx]

            r_pos_far = bvh.intersect_ray(pts_probe, nrm_probe)
            r_neg_far = bvh.intersect_ray(pts_probe, -nrm_probe)

            hit_pos_far = r_pos_far.hit.bool() & (r_pos_far.face_ids.long() >= 0) & torch.isfinite(r_pos_far.t.float())
            hit_neg_far = r_neg_far.hit.bool() & (r_neg_far.face_ids.long() >= 0) & torch.isfinite(r_neg_far.t.float())
            far_hit_any = hit_pos_far | hit_neg_far

            far_over_dist = (hit_pos_far & (r_pos_far.t.float() > float(max_dist))) | (
                hit_neg_far & (r_neg_far.t.float() > float(max_dist))
            )
            probe_dist_reject = far_hit_any & far_over_dist
            probe_miss = ~far_hit_any
            probe_unclassified = far_hit_any & (~far_over_dist)

            dist_reject_mask[probe_idx] = probe_dist_reject
            miss_mask[probe_idx] = probe_miss
            # Rare numerical edge case: far probe sees a hit but not over distance,
            # while near probe had no hit.
            if bool(torch.any(probe_unclassified)):
                uc = torch.zeros_like(fail)
                uc[probe_idx] = probe_unclassified
                err_unclassified += int(torch.count_nonzero(uc).item())
        else:
            miss_mask = fail & (~near_hit_any)

        err_angle_reject += int(torch.count_nonzero(angle_reject_mask).item())
        err_distance += int(torch.count_nonzero(dist_reject_mask).item())
        err_miss += int(torch.count_nonzero(miss_mask).item())

        selected_face[st:ed] = face_sel.detach().cpu().numpy().astype(np.int64)
        valid[st:ed] = ok.detach().cpu().numpy().astype(np.bool_)

    labels = np.full((n_faces,), -1, dtype=np.int64)
    ok_ids = np.where(valid & (selected_face >= 0) & (selected_face < high_island.shape[0]))[0]
    if ok_ids.size > 0:
        labels[ok_ids] = high_island[selected_face[ok_ids]]

    neighbors = _face_adjacency_lists(low_mesh)
    labels_filled, fill_changes = _fill_unknown_face_labels(labels, neighbors, iters=fill_unknown_iters)
    labels_smoothed, label_changes = _majority_vote_face_labels(labels_filled, neighbors, iters=vote_iters)
    labels_closed, morph_changes = _morphological_close_face_labels(labels_smoothed, neighbors, iters=morph_close_iters)
    confidence = _face_label_confidence(labels_closed, neighbors)
    conflict = (labels_closed < 0) | (confidence < conf_min)

    hit_count = int(np.count_nonzero(labels >= 0))
    fail_count = int(max(0, n_faces - hit_count))
    mapped_count = int(np.count_nonzero(labels_closed >= 0))
    meta = {
        "uv_semantic_transfer_points": int(n_faces),
        "uv_semantic_transfer_hits": hit_count,
        "uv_semantic_transfer_hit_ratio": float(hit_count / max(1, n_faces)),
        "uv_semantic_transfer_failures": int(fail_count),
        "uv_semantic_transfer_failure_ratio": float(fail_count / max(1, n_faces)),
        "uv_semantic_transfer_err_miss": int(err_miss),
        "uv_semantic_transfer_err_miss_ratio": float(err_miss / max(1, n_faces)),
        "uv_semantic_transfer_err_miss_ratio_in_failed": float(err_miss / max(1, fail_count)),
        "uv_semantic_transfer_err_angle_reject": int(err_angle_reject),
        "uv_semantic_transfer_err_angle_reject_ratio": float(err_angle_reject / max(1, n_faces)),
        "uv_semantic_transfer_err_angle_reject_ratio_in_failed": float(err_angle_reject / max(1, fail_count)),
        "uv_semantic_transfer_err_distance": int(err_distance),
        "uv_semantic_transfer_err_distance_ratio": float(err_distance / max(1, n_faces)),
        "uv_semantic_transfer_err_distance_ratio_in_failed": float(err_distance / max(1, fail_count)),
        "uv_semantic_transfer_err_unclassified": int(err_unclassified),
        "uv_semantic_transfer_err_unclassified_ratio": float(err_unclassified / max(1, n_faces)),
        "uv_semantic_transfer_angle_deg": float(angle_deg),
        "uv_semantic_transfer_max_dist": float(max_dist),
        "uv_semantic_transfer_safe_dist": float(safe_dist),
        "uv_semantic_transfer_fill_unknown_iters": int(fill_unknown_iters),
        "uv_semantic_transfer_fill_unknown_changes": int(fill_changes),
        "uv_semantic_transfer_majority_vote_iters": int(vote_iters),
        "uv_semantic_transfer_label_changes": int(label_changes),
        "uv_semantic_transfer_morph_close_iters": int(morph_close_iters),
        "uv_semantic_transfer_morph_close_changes": int(morph_changes),
        "uv_semantic_transfer_unknown_faces": int(np.count_nonzero(labels_closed < 0)),
        "uv_semantic_transfer_mapped_faces": int(mapped_count),
        "uv_semantic_transfer_conflict_faces": int(np.count_nonzero(conflict)),
        "uv_semantic_transfer_confidence_mean": float(np.mean(confidence[labels_closed >= 0]))
        if np.any(labels_closed >= 0)
        else 0.0,
    }

    return {
        "low_face_island": labels_closed.astype(np.int64, copy=False),
        "low_face_conflict": conflict.astype(np.bool_, copy=False),
        "low_face_confidence": confidence.astype(np.float32, copy=False),
        "meta": meta,
    }


__all__ = [
    "transfer_face_semantics_by_projection",
]
