from __future__ import annotations

import math
import heapq
from collections import deque
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


def _transfer_face_semantics_single_point(
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


def _build_weighted_face_adjacency(mesh: trimesh.Trimesh) -> Tuple[List[List[int]], List[List[float]]]:
    n_faces = int(len(mesh.faces))
    neigh: List[List[int]] = [[] for _ in range(n_faces)]
    weights: List[List[float]] = [[] for _ in range(n_faces)]
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 2:
        return neigh, weights

    edge_len = np.ones((adj.shape[0],), dtype=np.float64)
    try:
        ae = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        if ae.ndim == 2 and ae.shape[1] == 2 and ae.shape[0] == adj.shape[0]:
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            v0 = verts[ae[:, 0]]
            v1 = verts[ae[:, 1]]
            edge_len = np.linalg.norm(v1 - v0, axis=1)
            edge_len = np.maximum(edge_len, 1e-8)
    except Exception:
        pass

    for i, (a, b) in enumerate(adj.tolist()):
        ia = int(a)
        ib = int(b)
        if ia < 0 or ib < 0 or ia >= n_faces or ib >= n_faces or ia == ib:
            continue
        w = float(edge_len[i]) if np.isfinite(edge_len[i]) and edge_len[i] > 0.0 else 1.0
        neigh[ia].append(ib)
        weights[ia].append(w)
        neigh[ib].append(ia)
        weights[ib].append(w)
    return neigh, weights


def _component_count_per_label(labels: np.ndarray, neighbors: List[List[int]]) -> Dict[int, int]:
    lbl = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_faces = int(lbl.shape[0])
    out: Dict[int, int] = {}
    valid_ids = np.unique(lbl[lbl >= 0]).astype(np.int64)
    for lid in valid_ids.tolist():
        mask = lbl == int(lid)
        if not np.any(mask):
            continue
        seen = np.zeros((n_faces,), dtype=np.bool_)
        comp = 0
        for fid in np.where(mask)[0].tolist():
            if seen[fid]:
                continue
            comp += 1
            q: deque[int] = deque([int(fid)])
            seen[fid] = True
            while q:
                cur = q.popleft()
                for nb in neighbors[cur]:
                    if mask[nb] and (not seen[nb]):
                        seen[nb] = True
                        q.append(int(nb))
        out[int(lid)] = int(comp)
    return out


def _soft_unary_cost(
    *,
    face_id: int,
    label: int,
    candidate_prob_maps: List[Dict[int, float]],
    unary_eps: float,
    other_penalty: float,
    unknown_penalty: float,
    main_shell_labels: set[int],
    prefer_main_shells: bool,
    micro_shell_penalty: float,
) -> float:
    fmap = candidate_prob_maps[face_id]
    if int(label) in fmap:
        cost = float(-math.log(max(float(fmap[int(label)]), unary_eps)))
    elif len(fmap) == 0:
        cost = float(unknown_penalty)
    else:
        cost = float(other_penalty)
    if prefer_main_shells and len(main_shell_labels) > 0 and int(label) not in main_shell_labels:
        cost += float(max(0.0, micro_shell_penalty))
    return float(cost)


def _run_soft_priority_flood_icm(
    *,
    top1_label: np.ndarray,
    top1_prob: np.ndarray,
    top2_prob: np.ndarray,
    candidate_prob_maps: List[Dict[int, float]],
    neighbors: List[List[int]],
    edge_weights: List[List[float]],
    main_shell_labels: set[int],
    seam_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_faces = int(top1_label.shape[0])
    labels = np.full((n_faces,), -1, dtype=np.int64)
    if n_faces == 0:
        return labels, np.zeros((0,), dtype=np.bool_), {
            "seed_faces": 0,
            "priority_assigned_faces": 0,
            "fallback_assigned_faces": 0,
            "icm_relabel_faces": 0,
            "seed_prob_min": 0.0,
            "seed_margin_min": 0.0,
            "smoothness_weight": 0.0,
        }

    seed_prob_min = float(seam_cfg.get("transfer_soft_seed_prob_min", 0.95))
    seed_margin_min = float(seam_cfg.get("transfer_soft_seed_margin_min", 0.25))
    unary_eps = float(seam_cfg.get("transfer_soft_unary_eps", 1e-4))
    other_penalty = float(seam_cfg.get("transfer_soft_other_label_penalty", 2.5))
    unknown_penalty = float(seam_cfg.get("transfer_soft_unknown_label_penalty", 1.25))
    smoothness_weight = float(seam_cfg.get("transfer_soft_smoothness_weight", 0.35))
    icm_iters = int(max(0, seam_cfg.get("transfer_soft_icm_iters", 2)))
    prefer_main_shells = bool(seam_cfg.get("transfer_soft_prefer_main_shells", True))
    micro_shell_penalty = float(seam_cfg.get("transfer_soft_micro_shell_penalty", 0.75))

    known_mask = np.asarray(top1_label >= 0, dtype=np.bool_, copy=False)
    top_gap = np.asarray(top1_prob - top2_prob, dtype=np.float32, copy=False)
    if len(main_shell_labels) > 0:
        main_mask = np.isin(top1_label, np.asarray(sorted(main_shell_labels), dtype=np.int64))
    else:
        main_mask = np.ones((n_faces,), dtype=np.bool_)

    seed_mask = known_mask & (top1_prob >= float(seed_prob_min)) & (top_gap >= float(seed_margin_min))
    if prefer_main_shells and len(main_shell_labels) > 0:
        seed_mask &= main_mask
    if not np.any(seed_mask):
        fallback_seed_mask = known_mask & main_mask if np.any(main_mask & known_mask) else known_mask
        if np.any(fallback_seed_mask):
            relaxed_prob = max(0.75, float(seed_prob_min) - 0.20)
            relaxed_margin = max(0.05, float(seed_margin_min) * 0.5)
            seed_mask = fallback_seed_mask & (top1_prob >= relaxed_prob) & (top_gap >= relaxed_margin)
        if not np.any(seed_mask):
            seed_mask = fallback_seed_mask.astype(np.bool_, copy=True)

    flat_edge_weights = np.asarray([w for row in edge_weights for w in row if w > 0.0], dtype=np.float64)
    mean_edge = float(np.mean(flat_edge_weights)) if flat_edge_weights.size > 0 else 1.0

    best_cost = np.full((n_faces,), np.inf, dtype=np.float64)
    heap: List[Tuple[float, int, int]] = []
    for fid in np.where(seed_mask)[0].tolist():
        label = int(top1_label[fid])
        if label < 0:
            continue
        cost0 = _soft_unary_cost(
            face_id=int(fid),
            label=label,
            candidate_prob_maps=candidate_prob_maps,
            unary_eps=unary_eps,
            other_penalty=other_penalty,
            unknown_penalty=unknown_penalty,
            main_shell_labels=main_shell_labels,
            prefer_main_shells=prefer_main_shells,
            micro_shell_penalty=micro_shell_penalty,
        )
        if cost0 < best_cost[fid]:
            best_cost[fid] = cost0
            labels[fid] = label
            heapq.heappush(heap, (cost0, int(fid), label))

    while heap:
        cost, fid, label = heapq.heappop(heap)
        if cost > best_cost[fid] + 1e-12 or int(label) != int(labels[fid]):
            continue
        for nb, edge_w in zip(neighbors[fid], edge_weights[fid]):
            step_cost = float(smoothness_weight) * float(mean_edge / max(float(edge_w), 1e-8))
            unary_cost = _soft_unary_cost(
                face_id=int(nb),
                label=int(label),
                candidate_prob_maps=candidate_prob_maps,
                unary_eps=unary_eps,
                other_penalty=other_penalty,
                unknown_penalty=unknown_penalty,
                main_shell_labels=main_shell_labels,
                prefer_main_shells=prefer_main_shells,
                micro_shell_penalty=micro_shell_penalty,
            )
            cand_cost = float(cost + step_cost + unary_cost)
            if cand_cost + 1e-12 < best_cost[nb] or (
                abs(cand_cost - best_cost[nb]) <= 1e-12 and (labels[nb] < 0 or int(label) < int(labels[nb]))
            ):
                best_cost[nb] = cand_cost
                labels[nb] = int(label)
                heapq.heappush(heap, (cand_cost, int(nb), int(label)))

    priority_assigned = int(np.count_nonzero((labels >= 0) & (~seed_mask)))
    fallback_assigned = 0

    unresolved_known = np.where((labels < 0) & known_mask)[0]
    if unresolved_known.size > 0:
        labels[unresolved_known] = top1_label[unresolved_known]
        fallback_assigned += int(unresolved_known.size)

    unresolved = np.where(labels < 0)[0]
    for _ in range(8):
        if unresolved.size == 0:
            break
        changed = 0
        for fid in unresolved.tolist():
            votes: Dict[int, float] = {}
            for nb, w in zip(neighbors[fid], edge_weights[fid]):
                lb = int(labels[nb])
                if lb < 0:
                    continue
                votes[lb] = votes.get(lb, 0.0) + float(max(w, 1e-8))
            if not votes:
                continue
            best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            labels[fid] = int(best)
            changed += 1
            fallback_assigned += 1
        if changed == 0:
            break
        unresolved = np.where(labels < 0)[0]

    unresolved = np.where(labels < 0)[0]
    if unresolved.size > 0:
        known_labels = labels[labels >= 0]
        global_major = int(np.bincount(known_labels).argmax()) if known_labels.size > 0 else -1
        if global_major >= 0:
            labels[unresolved] = int(global_major)
            fallback_assigned += int(unresolved.size)

    icm_relabels = 0
    for _ in range(icm_iters):
        changed = 0
        nxt = labels.copy()
        for fid in range(n_faces):
            candidate_labels = set(candidate_prob_maps[fid].keys())
            if labels[fid] >= 0:
                candidate_labels.add(int(labels[fid]))
            for nb in neighbors[fid]:
                if labels[nb] >= 0:
                    candidate_labels.add(int(labels[nb]))
            if not candidate_labels:
                continue
            best_label = int(labels[fid]) if labels[fid] >= 0 else -1
            best_energy = np.inf
            for cand in sorted(candidate_labels):
                unary_cost = _soft_unary_cost(
                    face_id=int(fid),
                    label=int(cand),
                    candidate_prob_maps=candidate_prob_maps,
                    unary_eps=unary_eps,
                    other_penalty=other_penalty,
                    unknown_penalty=unknown_penalty,
                    main_shell_labels=main_shell_labels,
                    prefer_main_shells=prefer_main_shells,
                    micro_shell_penalty=micro_shell_penalty,
                )
                pair_cost = 0.0
                for nb, edge_w in zip(neighbors[fid], edge_weights[fid]):
                    nb_label = int(labels[nb])
                    if nb_label < 0 or nb_label == int(cand):
                        continue
                    pair_cost += float(smoothness_weight) * float(mean_edge / max(float(edge_w), 1e-8))
                energy = float(unary_cost + pair_cost)
                if energy + 1e-12 < best_energy or (
                    abs(energy - best_energy) <= 1e-12 and (best_label < 0 or int(cand) < int(best_label))
                ):
                    best_energy = energy
                    best_label = int(cand)
            if best_label >= 0 and int(best_label) != int(labels[fid]):
                nxt[fid] = int(best_label)
                changed += 1
        labels = nxt
        icm_relabels += int(changed)
        if changed == 0:
            break

    return labels.astype(np.int64, copy=False), seed_mask.astype(np.bool_, copy=False), {
        "seed_faces": int(np.count_nonzero(seed_mask)),
        "priority_assigned_faces": int(priority_assigned),
        "fallback_assigned_faces": int(fallback_assigned),
        "icm_relabel_faces": int(icm_relabels),
        "seed_prob_min": float(seed_prob_min),
        "seed_margin_min": float(seed_margin_min),
        "smoothness_weight": float(smoothness_weight),
    }


def _compute_soft_face_evidence(
    *,
    sample_ids: np.ndarray,
    sample_valid: np.ndarray,
) -> Dict[str, Any]:
    n_faces = int(sample_ids.shape[0])
    top1_label = np.full((n_faces,), -1, dtype=np.int64)
    top1_prob = np.zeros((n_faces,), dtype=np.float32)
    top2_label = np.full((n_faces,), -1, dtype=np.int64)
    top2_prob = np.zeros((n_faces,), dtype=np.float32)
    entropy = np.zeros((n_faces,), dtype=np.float32)
    candidate_count = np.zeros((n_faces,), dtype=np.int32)
    candidate_labels: List[np.ndarray] = []
    candidate_probs: List[np.ndarray] = []
    candidate_prob_maps: List[Dict[int, float]] = []

    for fid in range(n_faces):
        valid = np.asarray(sample_valid[fid], dtype=np.bool_)
        if not np.any(valid):
            candidate_labels.append(np.zeros((0,), dtype=np.int64))
            candidate_probs.append(np.zeros((0,), dtype=np.float32))
            candidate_prob_maps.append({})
            continue
        vals = np.asarray(sample_ids[fid, valid], dtype=np.int64)
        vals = vals[vals >= 0]
        if vals.size == 0:
            candidate_labels.append(np.zeros((0,), dtype=np.int64))
            candidate_probs.append(np.zeros((0,), dtype=np.float32))
            candidate_prob_maps.append({})
            continue
        uniq, cnt = np.unique(vals, return_counts=True)
        order = np.lexsort((uniq, -cnt))
        uniq = uniq[order]
        cnt = cnt[order]
        prob = cnt.astype(np.float64) / max(1, int(vals.size))
        prob32 = prob.astype(np.float32, copy=False)
        candidate_count[fid] = int(uniq.size)
        top1_label[fid] = int(uniq[0])
        top1_prob[fid] = float(prob[0])
        candidate_labels.append(uniq.astype(np.int64, copy=False))
        candidate_probs.append(prob32)
        candidate_prob_maps.append(
            {int(label): float(p) for label, p in zip(uniq.tolist(), prob.tolist())}
        )
        if uniq.size > 1:
            top2_label[fid] = int(uniq[1])
            top2_prob[fid] = float(prob[1])
            ent = float(-np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0))))
            entropy[fid] = float(ent / max(math.log(float(uniq.size)), 1e-12))
        else:
            entropy[fid] = 0.0

    return {
        "soft_top1_label": top1_label,
        "soft_top1_prob": top1_prob,
        "soft_top2_label": top2_label,
        "soft_top2_prob": top2_prob,
        "soft_entropy": entropy,
        "soft_candidate_count": candidate_count,
        "candidate_labels": candidate_labels,
        "candidate_probs": candidate_probs,
        "candidate_prob_maps": candidate_prob_maps,
    }


def _transfer_face_semantics_four_point_bfs(
    *,
    high_ctx: Dict[str, Any],
    high_face_island: np.ndarray,
    low_mesh: trimesh.Trimesh,
    seam_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    import torch

    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    verts = np.asarray(low_mesh.vertices, dtype=np.float32)
    n_faces = int(faces.shape[0])
    if n_faces == 0 or verts.shape[0] == 0:
        labels = np.full((n_faces,), -1, dtype=np.int64)
        conf = np.zeros((n_faces,), dtype=np.float32)
        return {
            "low_face_island": labels,
            "low_face_conflict": np.ones((n_faces,), dtype=np.bool_),
            "low_face_confidence": conf,
            "meta": {
                "uv_semantic_transfer_points": int(n_faces),
                "uv_semantic_transfer_hits": 0,
                "uv_semantic_transfer_hit_ratio": 0.0,
                "uv_semantic_transfer_failures": int(n_faces),
                "uv_semantic_transfer_failure_ratio": 1.0 if n_faces > 0 else 0.0,
                "uv_semantic_transfer_unknown_faces": int(n_faces),
                "uv_semantic_transfer_mode": "four_point_soft_flood",
            },
        }

    tri = verts[faces]
    vnorm = np.asarray(low_mesh.vertex_normals, dtype=np.float32)
    if vnorm.ndim != 2 or vnorm.shape[0] != verts.shape[0]:
        fn = np.asarray(low_mesh.face_normals, dtype=np.float32)
        vnorm = np.zeros_like(verts, dtype=np.float32)
        np.add.at(vnorm, faces[:, 0], fn)
        np.add.at(vnorm, faces[:, 1], fn)
        np.add.at(vnorm, faces[:, 2], fn)
    vnorm = vnorm / np.maximum(np.linalg.norm(vnorm, axis=1, keepdims=True), 1e-8)
    tri_n = vnorm[faces]

    default_bary = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
        dtype=np.float32,
    )
    bary_w = np.asarray(seam_cfg.get("transfer_four_point_barycentric", default_bary), dtype=np.float32)
    if bary_w.shape != (4, 3):
        bary_w = default_bary

    sample_pos = np.einsum("sk,fkc->fsc", bary_w, tri, optimize=True).astype(np.float32, copy=False)
    sample_nrm = np.einsum("sk,fkc->fsc", bary_w, tri_n, optimize=True).astype(np.float32, copy=False)
    sample_nrm = sample_nrm / np.maximum(np.linalg.norm(sample_nrm, axis=2, keepdims=True), 1e-8)

    v_t = high_ctx["v_t"]
    bvh = high_ctx["bvh"]
    face_normals_t = high_ctx.get("face_normals_t", None)
    bbox_diag = max(1e-6, float(high_ctx["bbox_diag"]))
    high_island = np.asarray(high_face_island, dtype=np.int64)

    max_dist = float(seam_cfg.get("transfer_max_dist_ratio", 0.005)) * bbox_diag
    angle_deg = float(seam_cfg.get("transfer_relaxed_normal_angle_deg", 60.0))
    cos_min = float(math.cos(math.radians(max(0.0, min(89.9, angle_deg)))))
    safe_dist = float(seam_cfg.get("transfer_near_accept_dist_ratio", 0.005)) * bbox_diag
    ray_origin_eps = float(seam_cfg.get("transfer_ray_origin_offset_ratio", 1e-6)) * bbox_diag
    conf_min = float(seam_cfg.get("uv_island_guard_confidence_min", 0.55))
    base_chunk = max(1, int(corr_cfg.get("bvh_chunk_size", 200000)))
    chunk_faces = max(1, base_chunk // 4)

    sample_ids = np.full((n_faces, 4), -1, dtype=np.int64)
    sample_valid = np.zeros((n_faces, 4), dtype=np.bool_)
    # 0=success/unset, 1=miss, 2=distance reject, 3=angle reject(reserved), 4=other
    sample_fail_kind = np.zeros((n_faces, 4), dtype=np.uint8)

    for st in range(0, n_faces, chunk_faces):
        ed = min(st + chunk_faces, n_faces)
        loc_pos = sample_pos[st:ed].reshape(-1, 3)
        loc_nrm = sample_nrm[st:ed].reshape(-1, 3)

        pts_t = torch.tensor(loc_pos, dtype=torch.float32, device=v_t.device)
        nrm_t = torch.tensor(loc_nrm, dtype=torch.float32, device=v_t.device)
        nrm_t = nrm_t / torch.clamp(torch.linalg.norm(nrm_t, dim=1, keepdim=True), min=1e-8)

        # Opposite-direction origin offsets reduce t~=0 self-hit miss for aligned surfaces.
        r_pos = bvh.intersect_ray(pts_t - nrm_t * float(ray_origin_eps), nrm_t, max_t=max_dist)
        r_neg = bvh.intersect_ray(pts_t + nrm_t * float(ray_origin_eps), -nrm_t, max_t=max_dist)

        fid_pos = torch.clamp(r_pos.face_ids.long(), 0, high_island.shape[0] - 1)
        fid_neg = torch.clamp(r_neg.face_ids.long(), 0, high_island.shape[0] - 1)
        hit_pos = r_pos.hit.bool() & (r_pos.face_ids.long() >= 0) & torch.isfinite(r_pos.t.float())
        hit_neg = r_neg.hit.bool() & (r_neg.face_ids.long() >= 0) & torch.isfinite(r_neg.t.float())
        t_pos = r_pos.t.float()
        t_neg = r_neg.t.float()
        near_pos = hit_pos & (t_pos <= float(safe_dist))
        near_neg = hit_neg & (t_neg <= float(safe_dist))

        if face_normals_t is not None:
            dot_pos = torch.sum(face_normals_t[fid_pos] * nrm_t, dim=1)
            dot_neg = torch.sum(face_normals_t[fid_neg] * (-nrm_t), dim=1)
            val_pos = hit_pos & (near_pos | (dot_pos >= cos_min))
            val_neg = hit_neg & (near_neg | (dot_neg >= cos_min))
        else:
            val_pos = hit_pos
            val_neg = hit_neg

        choose_pos = val_pos & (~val_neg | (t_pos <= t_neg))
        choose_neg = val_neg & (~val_pos | (t_neg < t_pos))
        ok = choose_pos | choose_neg

        sel = torch.full((pts_t.shape[0],), -1, dtype=torch.long, device=v_t.device)
        sel[choose_pos] = fid_pos[choose_pos]
        sel[choose_neg] = fid_neg[choose_neg]

        sel_np = sel.detach().cpu().numpy().astype(np.int64)
        ok_np = ok.detach().cpu().numpy().astype(np.bool_)
        loc_faces = (np.arange(sel_np.shape[0], dtype=np.int64) // 4) + int(st)
        loc_sidx = np.arange(sel_np.shape[0], dtype=np.int64) % 4
        # IMPORTANT: store semantic-island IDs, not high-face indices.
        # Using raw face indices explodes label cardinality and destroys BFS growth.
        sem_np = np.full_like(sel_np, -1, dtype=np.int64)
        if np.any(ok_np):
            sem_np[ok_np] = high_island[np.clip(sel_np[ok_np], 0, high_island.shape[0] - 1)]
        sample_ids[loc_faces, loc_sidx] = sem_np
        sample_valid[loc_faces, loc_sidx] = ok_np

        fail = ~ok
        near_hit_any = hit_pos | hit_neg
        kind_all = torch.zeros_like(fail, dtype=torch.uint8)
        # 3 = angle reject: intersected nearby surfaces but rejected by angle filter.
        angle_reject = fail & near_hit_any
        kind_all[angle_reject] = 3

        need_far_probe = fail & (~near_hit_any)
        fail_idx = torch.nonzero(fail, as_tuple=False).reshape(-1)
        if fail_idx.numel() == 0:
            continue
        if bool(torch.any(need_far_probe)):
            probe_idx = torch.nonzero(need_far_probe, as_tuple=False).reshape(-1)
            pts_probe = pts_t[probe_idx]
            nrm_probe = nrm_t[probe_idx]
            r_pos_far = bvh.intersect_ray(pts_probe - nrm_probe * float(ray_origin_eps), nrm_probe)
            r_neg_far = bvh.intersect_ray(pts_probe + nrm_probe * float(ray_origin_eps), -nrm_probe)
            hit_pos_far = (
                r_pos_far.hit.bool() & (r_pos_far.face_ids.long() >= 0) & torch.isfinite(r_pos_far.t.float())
            )
            hit_neg_far = (
                r_neg_far.hit.bool() & (r_neg_far.face_ids.long() >= 0) & torch.isfinite(r_neg_far.t.float())
            )
            far_hit = hit_pos_far | hit_neg_far
            kind_all[probe_idx[far_hit]] = 2  # distance reject
            kind_all[probe_idx[~far_hit]] = 1  # miss

        fail_face = (fail_idx.detach().cpu().numpy().astype(np.int64) // 4) + int(st)
        fail_sidx = fail_idx.detach().cpu().numpy().astype(np.int64) % 4
        kind = kind_all[fail_idx].detach().cpu().numpy().astype(np.uint8, copy=False)
        sample_fail_kind[fail_face, fail_sidx] = kind

    face_hit_count = np.count_nonzero(sample_valid, axis=1).astype(np.int32, copy=False)
    soft_evidence = _compute_soft_face_evidence(sample_ids=sample_ids, sample_valid=sample_valid)
    soft_top1_label = np.asarray(soft_evidence["soft_top1_label"], dtype=np.int64, copy=False)
    soft_top1_prob = np.asarray(soft_evidence["soft_top1_prob"], dtype=np.float32, copy=False)
    soft_top2_prob = np.asarray(soft_evidence["soft_top2_prob"], dtype=np.float32, copy=False)
    soft_entropy = np.asarray(soft_evidence["soft_entropy"], dtype=np.float32, copy=False)
    soft_candidate_count = np.asarray(soft_evidence["soft_candidate_count"], dtype=np.int32, copy=False)
    soft_known = soft_top1_label >= 0
    main_shell_labels = {
        int(x)
        for x in np.asarray(seam_cfg.get("transfer_main_shell_labels", []), dtype=np.int64).reshape(-1).tolist()
    }

    neighbors, edge_weights = _build_weighted_face_adjacency(low_mesh)
    labels, seed_mask, soft_assign_meta = _run_soft_priority_flood_icm(
        top1_label=soft_top1_label,
        top1_prob=soft_top1_prob,
        top2_prob=soft_top2_prob,
        candidate_prob_maps=list(soft_evidence["candidate_prob_maps"]),
        neighbors=neighbors,
        edge_weights=edge_weights,
        main_shell_labels=main_shell_labels,
        seam_cfg=seam_cfg,
    )

    unknown_mask = ~soft_known
    boundary_mask = soft_known & (~seed_mask)
    pre_bfs_label = soft_top1_label.astype(np.int64, copy=True)
    pre_bfs_conf = soft_top1_prob.astype(np.float32, copy=True)
    pre_bfs_state = np.zeros((n_faces,), dtype=np.uint8)
    pre_bfs_state[boundary_mask] = np.uint8(1)
    pre_bfs_state[seed_mask] = np.uint8(2)

    confidence = _face_label_confidence(labels, neighbors)
    conflict = (labels < 0) | (confidence < conf_min)

    hit_count = int(np.count_nonzero(face_hit_count > 0))
    fail_count = int(max(0, n_faces - hit_count))
    err_angle = 0
    err_distance = 0
    err_miss = 0
    err_other = 0
    for fid in np.where(face_hit_count == 0)[0].tolist():
        kinds = sample_fail_kind[fid]
        if np.any(kinds == 3):
            err_angle += 1
        elif np.any(kinds == 2):
            err_distance += 1
        elif np.any(kinds == 1):
            err_miss += 1
        else:
            err_other += 1

    comp_per_id = _component_count_per_label(labels, neighbors)
    mapped_count = int(np.count_nonzero(labels >= 0))
    unique_labels = np.unique(labels[labels >= 0]).astype(np.int64, copy=False)
    final_relabel_count = int(np.count_nonzero(soft_known & (labels != soft_top1_label)))
    meta = {
        "uv_semantic_transfer_mode": "four_point_soft_flood",
        "uv_semantic_transfer_points": int(n_faces),
        "uv_semantic_transfer_sample_points": int(n_faces * 4),
        "uv_semantic_transfer_hits": int(hit_count),
        "uv_semantic_transfer_hit_ratio": float(hit_count / max(1, n_faces)),
        "uv_semantic_transfer_failures": int(fail_count),
        "uv_semantic_transfer_failure_ratio": float(fail_count / max(1, n_faces)),
        "uv_semantic_transfer_err_miss": int(err_miss),
        "uv_semantic_transfer_err_miss_ratio": float(err_miss / max(1, n_faces)),
        "uv_semantic_transfer_err_miss_ratio_in_failed": float(err_miss / max(1, fail_count)),
        "uv_semantic_transfer_err_angle_reject": int(err_angle),
        "uv_semantic_transfer_err_angle_reject_ratio": float(err_angle / max(1, n_faces)),
        "uv_semantic_transfer_err_angle_reject_ratio_in_failed": float(err_angle / max(1, fail_count)),
        "uv_semantic_transfer_err_distance": int(err_distance),
        "uv_semantic_transfer_err_distance_ratio": float(err_distance / max(1, n_faces)),
        "uv_semantic_transfer_err_distance_ratio_in_failed": float(err_distance / max(1, fail_count)),
        "uv_semantic_transfer_err_unclassified": int(err_other),
        "uv_semantic_transfer_err_unclassified_ratio": float(err_other / max(1, n_faces)),
        "uv_semantic_transfer_strong_seed_faces": int(np.count_nonzero(seed_mask)),
        "uv_semantic_transfer_boundary_faces": int(np.count_nonzero(boundary_mask)),
        "uv_semantic_transfer_unknown_faces_initial": int(np.count_nonzero(unknown_mask)),
        "uv_semantic_transfer_pre_bfs_confidence_mean": float(np.mean(pre_bfs_conf)),
        "uv_semantic_transfer_pre_bfs_confidence_p10": float(np.percentile(pre_bfs_conf, 10.0)),
        "uv_semantic_transfer_pre_bfs_confidence_p50": float(np.percentile(pre_bfs_conf, 50.0)),
        "uv_semantic_transfer_pre_bfs_confidence_p90": float(np.percentile(pre_bfs_conf, 90.0)),
        "uv_semantic_transfer_bfs_assigned_faces": int(soft_assign_meta["priority_assigned_faces"]),
        "uv_semantic_transfer_vote_assigned_faces": int(soft_assign_meta["fallback_assigned_faces"]),
        "uv_semantic_transfer_priority_assigned_faces": int(soft_assign_meta["priority_assigned_faces"]),
        "uv_semantic_transfer_priority_seed_faces": int(soft_assign_meta["seed_faces"]),
        "uv_semantic_transfer_priority_seed_prob_min": float(soft_assign_meta["seed_prob_min"]),
        "uv_semantic_transfer_priority_seed_margin_min": float(soft_assign_meta["seed_margin_min"]),
        "uv_semantic_transfer_priority_smoothness_weight": float(soft_assign_meta["smoothness_weight"]),
        "uv_semantic_transfer_icm_relabel_faces": int(soft_assign_meta["icm_relabel_faces"]),
        "uv_semantic_transfer_soft_relabel_faces": int(final_relabel_count),
        "uv_semantic_transfer_unknown_faces": int(np.count_nonzero(labels < 0)),
        "uv_semantic_transfer_mapped_faces": int(mapped_count),
        "uv_semantic_transfer_final_label_count": int(unique_labels.size),
        "uv_semantic_transfer_conflict_faces": int(np.count_nonzero(conflict)),
        "uv_semantic_transfer_confidence_mean": float(np.mean(confidence[labels >= 0])) if np.any(labels >= 0) else 0.0,
        "uv_semantic_transfer_component_count_per_id": {str(k): int(v) for k, v in sorted(comp_per_id.items())},
        "uv_semantic_transfer_component_count_max": int(max(comp_per_id.values())) if len(comp_per_id) > 0 else 0,
        "uv_semantic_transfer_soft_top1_prob_mean": float(np.mean(soft_top1_prob[soft_known]))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_soft_top1_prob_p10": float(np.percentile(soft_top1_prob[soft_known], 10.0))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_soft_top1_prob_p50": float(np.percentile(soft_top1_prob[soft_known], 50.0))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_soft_top1_prob_p90": float(np.percentile(soft_top1_prob[soft_known], 90.0))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_soft_entropy_mean": float(np.mean(soft_entropy[soft_known]))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_soft_candidate_count_mean": float(np.mean(soft_candidate_count[soft_known]))
        if np.any(soft_known)
        else 0.0,
        "uv_semantic_transfer_max_dist_ratio": float(seam_cfg.get("transfer_max_dist_ratio", 0.005)),
        "uv_semantic_transfer_max_dist": float(max_dist),
        "uv_semantic_transfer_angle_deg": float(angle_deg),
        "uv_semantic_transfer_safe_dist": float(safe_dist),
        "uv_semantic_transfer_ray_origin_eps": float(ray_origin_eps),
    }
    return {
        "low_face_island": labels.astype(np.int64, copy=False),
        "low_face_conflict": conflict.astype(np.bool_, copy=False),
        "low_face_confidence": confidence.astype(np.float32, copy=False),
        "low_face_pre_bfs_label": pre_bfs_label.astype(np.int64, copy=False),
        "low_face_pre_bfs_confidence": pre_bfs_conf.astype(np.float32, copy=False),
        "low_face_pre_bfs_state": pre_bfs_state.astype(np.uint8, copy=False),
        "low_face_soft_top1_label": np.asarray(soft_evidence["soft_top1_label"], dtype=np.int64, copy=False),
        "low_face_soft_top1_prob": np.asarray(soft_evidence["soft_top1_prob"], dtype=np.float32, copy=False),
        "low_face_soft_top2_label": np.asarray(soft_evidence["soft_top2_label"], dtype=np.int64, copy=False),
        "low_face_soft_top2_prob": np.asarray(soft_evidence["soft_top2_prob"], dtype=np.float32, copy=False),
        "low_face_soft_entropy": np.asarray(soft_evidence["soft_entropy"], dtype=np.float32, copy=False),
        "low_face_soft_candidate_count": np.asarray(
            soft_evidence["soft_candidate_count"], dtype=np.int32, copy=False
        ),
        "meta": meta,
    }


def transfer_face_semantics_by_projection(
    *,
    high_ctx: Dict[str, Any],
    high_face_island: np.ndarray,
    low_mesh: trimesh.Trimesh,
    seam_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    mode = str(seam_cfg.get("transfer_sampling_mode", "single_point_projection")).strip().lower()
    if mode in {
        "four_point_soft_flood",
        "four_point_soft",
        "four_point_priority_flood",
        "four_point_bfs",
        "4point_bfs",
        "four_point",
    }:
        return _transfer_face_semantics_four_point_bfs(
            high_ctx=high_ctx,
            high_face_island=high_face_island,
            low_mesh=low_mesh,
            seam_cfg=seam_cfg,
            corr_cfg=corr_cfg,
        )
    out = _transfer_face_semantics_single_point(
        high_ctx=high_ctx,
        high_face_island=high_face_island,
        low_mesh=low_mesh,
        seam_cfg=seam_cfg,
        corr_cfg=corr_cfg,
    )
    meta = dict(out.get("meta", {}))
    meta.setdefault("uv_semantic_transfer_mode", "single_point_projection")
    out["meta"] = meta
    return out


__all__ = [
    "transfer_face_semantics_by_projection",
]
