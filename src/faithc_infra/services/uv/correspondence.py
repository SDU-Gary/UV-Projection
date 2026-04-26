from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh

from ..atom3d_runtime import ensure_atom3d_cuda_runtime


def build_high_cuda_context(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    device: str,
) -> Dict[str, Any]:
    runtime_diag: Dict[str, Any] = {}
    if str(device).strip().lower() == "cuda":
        runtime_diag = ensure_atom3d_cuda_runtime(device, strict=True, require_cuda=True)

    import torch
    from atom3d import MeshBVH

    high_v = np.asarray(high_mesh.vertices, dtype=np.float32)
    high_f = np.asarray(high_mesh.faces, dtype=np.int64)
    high_fn = np.asarray(high_mesh.face_normals, dtype=np.float32)
    fn_norm = np.linalg.norm(high_fn, axis=1, keepdims=True)
    high_fn = high_fn / np.maximum(fn_norm, 1e-8)

    bbox_min = np.min(high_v, axis=0)
    bbox_max = np.max(high_v, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    v_t = torch.tensor(high_v, dtype=torch.float32, device=device)
    f_t = torch.tensor(high_f, dtype=torch.long, device=device)
    uv_t = torch.tensor(np.asarray(high_uv, dtype=np.float32), dtype=torch.float32, device=device)
    face_normals_t = torch.tensor(high_fn, dtype=torch.float32, device=device)
    bvh = MeshBVH(v_t, f_t, device=device)

    return {
        "v_t": v_t,
        "f_t": f_t,
        "uv_t": uv_t,
        "face_normals_t": face_normals_t,
        "bvh": bvh,
        "bbox_diag": bbox_diag,
        "runtime_diag": runtime_diag,
    }


def barycentric_from_points_torch(points_t, tri_v_t):
    import torch

    a = tri_v_t[:, 0]
    b = tri_v_t[:, 1]
    c = tri_v_t[:, 2]

    v0 = b - a
    v1 = c - a
    v2 = points_t - a

    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    denom = d00 * d11 - d01 * d01
    denom = torch.clamp(denom, min=1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    bary = torch.stack([u, v, w], dim=1)
    bary = torch.clamp(bary, min=0.0)
    bary = bary / torch.clamp(torch.sum(bary, dim=1, keepdim=True), min=1e-8)
    return bary


def ray_result_to_uv(
    *,
    ray_result,
    point_normals_t,
    f_t,
    v_t,
    uv_t,
    face_normals_t,
    normal_dot_min: float,
):
    import torch

    face_ids_raw = ray_result.face_ids.long()
    hit_mask = ray_result.hit.bool()
    t = ray_result.t.float()

    valid_face = face_ids_raw >= 0
    face_ids = torch.clamp(face_ids_raw, 0, f_t.shape[0] - 1)
    face_normals = face_normals_t[face_ids]
    dot = torch.sum(face_normals * point_normals_t, dim=1)

    valid = hit_mask & valid_face & torch.isfinite(t) & (dot >= normal_dot_min)

    tri_vidx = f_t[face_ids]
    tri_v = v_t[tri_vidx]
    bary = barycentric_from_points_torch(ray_result.hit_points.float(), tri_v)
    tri_uv = uv_t[tri_vidx]
    uv = (tri_uv * bary.unsqueeze(-1)).sum(dim=1)
    return uv, face_ids.long(), t, valid


def island_compatible_mask_torch(
    *,
    face_ids_t,
    expected_island_t,
    high_face_island_t,
    allow_unknown: bool,
):
    import torch

    expected_known = expected_island_t >= 0
    if not torch.any(expected_known):
        return torch.ones_like(face_ids_t, dtype=torch.bool)

    clamped = torch.clamp(face_ids_t.long(), 0, high_face_island_t.shape[0] - 1)
    hit_island = high_face_island_t[clamped]

    compat = ~expected_known
    compat = compat | (hit_island == expected_island_t)
    if allow_unknown:
        compat = compat | (hit_island < 0)
    return compat


def correspond_points_hybrid(
    *,
    points: np.ndarray,
    point_normals: np.ndarray,
    corr_cfg: Dict[str, Any],
    high_ctx: Dict[str, Any],
    island_guard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch

    pts = np.asarray(points, dtype=np.float32)
    nrm = np.asarray(point_normals, dtype=np.float32)
    n = int(len(pts))
    if n == 0:
        return {
            "target_uv": np.zeros((0, 2), dtype=np.float32),
            "target_face_ids": np.zeros((0,), dtype=np.int64),
            "valid_mask": np.zeros((0,), dtype=np.bool_),
            "primary_mask": np.zeros((0,), dtype=np.bool_),
            "fallback_used_mask": np.zeros((0,), dtype=np.bool_),
            "island_guard_stats": {
                "enabled": False,
                "constrained_points": 0,
                "reject_count": 0,
                "fallback_success_count": 0,
                "invalid_after_guard_count": 0,
            },
        }

    nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(nrm_len, 1e-8)

    target_uv = np.zeros((n, 2), dtype=np.float32)
    target_face_ids = np.full((n,), -1, dtype=np.int64)
    valid_mask = np.zeros((n,), dtype=np.bool_)
    primary_mask = np.zeros((n,), dtype=np.bool_)
    fallback_used_mask = np.zeros((n,), dtype=np.bool_)

    v_t = high_ctx["v_t"]
    f_t = high_ctx["f_t"]
    uv_t = high_ctx["uv_t"]
    face_normals_t = high_ctx["face_normals_t"]
    bvh = high_ctx["bvh"]

    normal_dot_min = float(corr_cfg.get("normal_dot_min", 0.7))
    max_dist = float(corr_cfg.get("ray_max_dist_ratio", 0.08)) * max(1e-6, float(high_ctx["bbox_diag"]))
    chunk_size = max(1, int(corr_cfg.get("bvh_chunk_size", 200000)))

    guard_enabled = False
    guard_mode = "soft"
    guard_use_fallback = True
    guard_allow_unknown = False
    guard_expected_island = np.full((n,), -1, dtype=np.int64)
    guard_constrained_mask = np.zeros((n,), dtype=np.bool_)
    guard_reject_mask = np.zeros((n,), dtype=np.bool_)
    guard_fallback_success_mask = np.zeros((n,), dtype=np.bool_)
    guard_invalid_after_mask = np.zeros((n,), dtype=np.bool_)
    high_face_island_t = None
    if island_guard is not None and bool(island_guard.get("enabled", False)):
        high_face_island_np = np.asarray(island_guard.get("high_face_island"), dtype=np.int64)
        expected_island_np = np.asarray(island_guard.get("expected_island"), dtype=np.int64)
        if high_face_island_np.ndim == 1 and expected_island_np.shape[0] == n:
            guard_enabled = True
            guard_mode = str(island_guard.get("mode", "soft")).strip().lower()
            if guard_mode not in {"soft", "strict"}:
                guard_mode = "soft"
            guard_use_fallback = guard_mode != "strict"
            guard_allow_unknown = bool(island_guard.get("allow_unknown", False))
            guard_expected_island = expected_island_np
            guard_constrained_mask = guard_expected_island >= 0
            high_face_island_t = torch.tensor(high_face_island_np, dtype=torch.long, device=v_t.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        pts_t = torch.tensor(pts[start:end], dtype=torch.float32, device=v_t.device)
        nrm_t = torch.tensor(nrm[start:end], dtype=torch.float32, device=v_t.device)
        nrm_t = nrm_t / torch.clamp(torch.linalg.norm(nrm_t, dim=1, keepdim=True), min=1e-8)
        expected_t = None
        if guard_enabled:
            expected_np = guard_expected_island[start:end]
            expected_t = torch.tensor(expected_np, dtype=torch.long, device=v_t.device)

        ray_pos = bvh.intersect_ray(pts_t, nrm_t, max_t=max_dist)
        ray_neg = bvh.intersect_ray(pts_t, -nrm_t, max_t=max_dist)

        uv_pos, face_pos, t_pos, valid_pos = ray_result_to_uv(
            ray_result=ray_pos,
            point_normals_t=nrm_t,
            f_t=f_t,
            v_t=v_t,
            uv_t=uv_t,
            face_normals_t=face_normals_t,
            normal_dot_min=normal_dot_min,
        )
        uv_neg, face_neg, t_neg, valid_neg = ray_result_to_uv(
            ray_result=ray_neg,
            point_normals_t=-nrm_t,
            f_t=f_t,
            v_t=v_t,
            uv_t=uv_t,
            face_normals_t=face_normals_t,
            normal_dot_min=normal_dot_min,
        )

        choose_pos_pre = valid_pos & (~valid_neg | (t_pos <= t_neg))
        choose_neg_pre = valid_neg & (~valid_pos | (t_neg < t_pos))
        if guard_enabled and expected_t is not None and high_face_island_t is not None:
            pos_compat = island_compatible_mask_torch(
                face_ids_t=face_pos,
                expected_island_t=expected_t,
                high_face_island_t=high_face_island_t,
                allow_unknown=guard_allow_unknown,
            )
            neg_compat = island_compatible_mask_torch(
                face_ids_t=face_neg,
                expected_island_t=expected_t,
                high_face_island_t=high_face_island_t,
                allow_unknown=guard_allow_unknown,
            )
            valid_pos = valid_pos & pos_compat
            valid_neg = valid_neg & neg_compat
        choose_pos = valid_pos & (~valid_neg | (t_pos <= t_neg))
        choose_neg = valid_neg & (~valid_pos | (t_neg < t_pos))
        chunk_primary = choose_pos | choose_neg
        chunk_valid = chunk_primary.clone()
        chunk_fallback = torch.zeros_like(chunk_primary)

        chunk_uv = torch.zeros((end - start, 2), dtype=torch.float32, device=v_t.device)
        chunk_face = torch.full((end - start,), -1, dtype=torch.long, device=v_t.device)
        if torch.any(choose_pos):
            chunk_uv[choose_pos] = uv_pos[choose_pos]
            chunk_face[choose_pos] = face_pos[choose_pos]
        if torch.any(choose_neg):
            chunk_uv[choose_neg] = uv_neg[choose_neg]
            chunk_face[choose_neg] = face_neg[choose_neg]

        if guard_enabled:
            primary_reject = (choose_pos_pre | choose_neg_pre) & (~chunk_primary)
            guard_reject_mask[start:end] |= primary_reject.detach().cpu().numpy().astype(np.bool_)

        if guard_enabled and not guard_use_fallback:
            fb_need = torch.zeros_like(chunk_primary)
        else:
            fb_need = ~chunk_primary
        if torch.any(fb_need):
            fb_pts = pts_t[fb_need]
            fb_nrm = nrm_t[fb_need]
            fb_res = bvh.udf(fb_pts, return_closest=True, return_uvw=True, return_face_ids=True)

            fb_face = torch.clamp(fb_res.face_ids.long(), 0, f_t.shape[0] - 1)
            fb_tri_vidx = f_t[fb_face]
            fb_tri_uv = uv_t[fb_tri_vidx]
            if fb_res.uvw is not None:
                fb_uvw = torch.clamp(fb_res.uvw, min=0.0)
                fb_uvw = fb_uvw / torch.clamp(fb_uvw.sum(dim=-1, keepdim=True), min=1e-8)
            else:
                fb_tri_v = v_t[fb_tri_vidx]
                fb_uvw = barycentric_from_points_torch(fb_res.closest_points, fb_tri_v)

            fb_uv = (fb_tri_uv * fb_uvw.unsqueeze(-1)).sum(dim=1)
            fb_fn = face_normals_t[fb_face]
            fb_dot = torch.sum(fb_fn * fb_nrm, dim=1)
            fb_valid = fb_dot >= 0.0
            fb_valid_pre = fb_valid

            if guard_enabled and expected_t is not None and high_face_island_t is not None:
                expected_fb = expected_t[fb_need]
                fb_compat = island_compatible_mask_torch(
                    face_ids_t=fb_face,
                    expected_island_t=expected_fb,
                    high_face_island_t=high_face_island_t,
                    allow_unknown=guard_allow_unknown,
                )
                fb_valid = fb_valid & fb_compat

            fb_idx_local = torch.where(fb_need)[0]
            accept = fb_idx_local[fb_valid]
            if accept.numel() > 0:
                chunk_uv[accept] = fb_uv[fb_valid]
                chunk_face[accept] = fb_face[fb_valid]
                chunk_valid[accept] = True
                chunk_fallback[accept] = True
            if guard_enabled:
                reject_fb_local = fb_idx_local[fb_valid_pre & (~fb_valid)]
                if reject_fb_local.numel() > 0:
                    reject_np = reject_fb_local.detach().cpu().numpy().astype(np.int64)
                    guard_reject_mask[start + reject_np] = True

        target_uv[start:end] = chunk_uv.detach().cpu().numpy().astype(np.float32)
        target_face_ids[start:end] = chunk_face.detach().cpu().numpy().astype(np.int64)
        valid_mask[start:end] = chunk_valid.detach().cpu().numpy().astype(np.bool_)
        primary_mask[start:end] = chunk_primary.detach().cpu().numpy().astype(np.bool_)
        fallback_used_mask[start:end] = chunk_fallback.detach().cpu().numpy().astype(np.bool_)
        if guard_enabled:
            guard_fallback_success_mask[start:end] = (
                chunk_fallback & chunk_valid
            ).detach().cpu().numpy().astype(np.bool_)
            guard_invalid_after_mask[start:end] = (~chunk_valid).detach().cpu().numpy().astype(np.bool_)

    constrained_count = int(np.count_nonzero(guard_constrained_mask)) if guard_enabled else 0
    reject_count = int(np.count_nonzero(guard_reject_mask & guard_constrained_mask)) if guard_enabled else 0
    fallback_success_count = (
        int(np.count_nonzero(guard_fallback_success_mask & guard_constrained_mask)) if guard_enabled else 0
    )
    invalid_after_count = int(np.count_nonzero(guard_invalid_after_mask & guard_constrained_mask)) if guard_enabled else 0

    return {
        "target_uv": target_uv,
        "target_face_ids": target_face_ids,
        "valid_mask": valid_mask,
        "primary_mask": primary_mask,
        "fallback_used_mask": fallback_used_mask,
        "island_guard_stats": {
            "enabled": bool(guard_enabled),
            "constrained_points": constrained_count,
            "reject_count": reject_count,
            "fallback_success_count": fallback_success_count,
            "invalid_after_guard_count": invalid_after_count,
        },
    }


def detect_cross_seam_faces(
    *,
    sample_face_ids: np.ndarray,
    target_uv: np.ndarray,
    valid_mask: np.ndarray,
    n_faces: int,
    uv_span_threshold: float,
    min_valid_samples_per_face: int,
) -> np.ndarray:
    cross = np.zeros(n_faces, dtype=np.bool_)
    if len(sample_face_ids) == 0:
        return cross

    valid_ids = np.where(valid_mask)[0]
    if valid_ids.size == 0:
        return cross

    face_valid = sample_face_ids[valid_ids].astype(np.int64, copy=False)
    uv_valid = np.asarray(target_uv[valid_ids], dtype=np.float64)
    if face_valid.size == 0:
        return cross

    order = np.argsort(face_valid, kind="mergesort")
    face_sorted = face_valid[order]
    uv_sorted = uv_valid[order]

    split = np.flatnonzero(np.diff(face_sorted)) + 1
    starts = np.concatenate(([0], split))
    ends = np.concatenate((split, [face_sorted.size]))
    counts = ends - starts

    face_group = face_sorted[starts]
    valid_face = (face_group >= 0) & (face_group < int(n_faces)) & (counts >= int(min_valid_samples_per_face))
    if not np.any(valid_face):
        return cross

    u = uv_sorted[:, 0]
    v = uv_sorted[:, 1]
    u_min = np.minimum.reduceat(u, starts)
    u_max = np.maximum.reduceat(u, starts)
    v_min = np.minimum.reduceat(v, starts)
    v_max = np.maximum.reduceat(v, starts)
    span = np.hypot(u_max - u_min, v_max - v_min)

    threshold = float(uv_span_threshold)
    hit = valid_face & (span > threshold)
    if np.any(hit):
        cross[face_group[hit].astype(np.int64, copy=False)] = True
    return cross


def major_face_island_labels(
    *,
    sample_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    valid_mask: np.ndarray,
    high_face_island: np.ndarray,
    n_low_faces: int,
    min_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_label = np.full((n_low_faces,), -1, dtype=np.int64)
    face_conflict = np.zeros((n_low_faces,), dtype=np.bool_)
    face_confidence = np.zeros((n_low_faces,), dtype=np.float32)

    if len(sample_face_ids) == 0:
        return face_label, face_conflict, face_confidence

    valid = valid_mask & (target_face_ids >= 0) & (target_face_ids < len(high_face_island))
    if not np.any(valid):
        return face_label, face_conflict, face_confidence

    low_f = sample_face_ids[valid].astype(np.int64, copy=False)
    high_f = target_face_ids[valid].astype(np.int64, copy=False)
    islands = high_face_island[high_f].astype(np.int64, copy=False)

    if low_f.size == 0:
        return face_label, face_conflict, face_confidence

    pair = np.stack([low_f, islands], axis=1)
    uniq_pair, pair_cnt = np.unique(pair, axis=0, return_counts=True)
    if uniq_pair.size == 0:
        return face_label, face_conflict, face_confidence

    pair_face = uniq_pair[:, 0].astype(np.int64, copy=False)
    pair_island = uniq_pair[:, 1].astype(np.int64, copy=False)
    split = np.flatnonzero(np.diff(pair_face)) + 1
    starts = np.concatenate(([0], split))
    ends = np.concatenate((split, [pair_face.size]))

    face_group = pair_face[starts]
    group_sizes = ends - starts
    group_total = np.add.reduceat(pair_cnt, starts)
    group_max = np.maximum.reduceat(pair_cnt, starts)

    valid_face = (face_group >= 0) & (face_group < int(n_low_faces)) & (group_total >= int(min_samples))
    if not np.any(valid_face):
        return face_label, face_conflict, face_confidence

    dominant_island = np.full((face_group.shape[0],), -1, dtype=np.int64)
    for gi, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        if s >= e:
            continue
        # Ties resolve to first max in sorted-by-island order; deterministic.
        k = s + int(np.argmax(pair_cnt[s:e]))
        dominant_island[gi] = int(pair_island[k])

    gid = np.where(valid_face)[0]
    dst_face = face_group[gid].astype(np.int64, copy=False)
    face_label[dst_face] = dominant_island[gid]
    face_conflict[dst_face] = group_sizes[gid] > 1
    face_confidence[dst_face] = (
        group_max[gid].astype(np.float64) / np.maximum(group_total[gid].astype(np.float64), 1.0)
    ).astype(np.float32)

    return face_label, face_conflict, face_confidence


def bvh_project_points(
    *,
    points: np.ndarray,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    device: str,
    chunk_size: int,
    return_dist: bool = False,
    return_face_normals: bool = False,
) -> Dict[str, Any]:
    runtime_diag: Dict[str, Any] = {}
    if str(device).strip().lower() == "cuda":
        runtime_diag = ensure_atom3d_cuda_runtime(device, strict=True, require_cuda=True)

    import torch
    from atom3d import MeshBVH

    high_v = np.asarray(high_mesh.vertices, dtype=np.float32)
    high_f = np.asarray(high_mesh.faces, dtype=np.int64)
    high_face_normals = np.asarray(high_mesh.face_normals, dtype=np.float32)

    v_t = torch.tensor(high_v, dtype=torch.float32, device=device)
    f_t = torch.tensor(high_f, dtype=torch.long, device=device)
    uv_t = torch.tensor(np.asarray(high_uv, dtype=np.float32), dtype=torch.float32, device=device)

    bvh = MeshBVH(v_t, f_t, device=device)

    mapped_list = []
    face_ids_list = []
    dist_list = []

    n = len(points)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        pts_t = torch.tensor(points[start:end], dtype=torch.float32, device=device)
        res = bvh.udf(pts_t, return_closest=True, return_uvw=True, return_face_ids=True)

        face_ids = torch.clamp(res.face_ids.long(), 0, f_t.shape[0] - 1)
        uvw = torch.clamp(res.uvw, min=0.0)
        uvw = uvw / torch.clamp(uvw.sum(dim=-1, keepdim=True), min=1e-8)

        tri_vidx = f_t[face_ids]
        tri_uv = uv_t[tri_vidx]
        mapped = (tri_uv * uvw.unsqueeze(-1)).sum(dim=1)

        mapped_list.append(mapped.detach().cpu().numpy())
        face_ids_list.append(face_ids.detach().cpu().numpy())

        if return_dist:
            closest = res.closest_points
            dist = torch.linalg.norm(pts_t - closest, dim=1)
            dist_list.append(dist.detach().cpu().numpy())

    mapped_uv = np.concatenate(mapped_list, axis=0).astype(np.float32)
    face_ids_np = np.concatenate(face_ids_list, axis=0).astype(np.int64)

    out: Dict[str, Any] = {
        "mapped_uv": mapped_uv,
        "face_ids": face_ids_np,
        "runtime_diag": runtime_diag,
    }
    if return_dist:
        out["distance"] = np.concatenate(dist_list, axis=0).astype(np.float32)
    if return_face_normals:
        out["face_normals"] = high_face_normals[np.clip(face_ids_np, 0, len(high_face_normals) - 1)]
    return out


__all__ = [
    "barycentric_from_points_torch",
    "build_high_cuda_context",
    "bvh_project_points",
    "correspond_points_hybrid",
    "detect_cross_seam_faces",
    "island_compatible_mask_torch",
    "major_face_island_labels",
    "ray_result_to_uv",
]
