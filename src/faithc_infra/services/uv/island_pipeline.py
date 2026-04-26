from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

from ..halfedge_topology import compute_high_face_uv_islands, split_vertices_along_cut_edges
from .mesh_sanitizer import ensure_halfedge_external_dependencies, sanitize_mesh_for_halfedge
from .openmesh_seams import SeamExtractionResult, extract_seam_edges_openmesh, validate_face_partition_by_seams
from .options import SeamValidationSettings, resolve_seam_validation_settings
from .semantic_transfer import (
    _build_weighted_face_adjacency,
    _face_label_confidence,
    transfer_face_semantics_by_projection,
)

_HIGH_ISLAND_CACHE: Dict[Tuple[int, int, int, int, int, float, float], Tuple[np.ndarray, Dict[str, int]]] = {}
_HIGH_ISLAND_CACHE_MAX = 8


@dataclass
class HighIslandResult:
    face_labels: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False


@dataclass
class LowSemanticResult:
    face_labels: np.ndarray
    face_conflict: np.ndarray
    face_confidence: np.ndarray
    pre_bfs_labels: np.ndarray
    pre_bfs_confidence: np.ndarray
    pre_bfs_state: np.ndarray
    pre_cleanup_labels: Optional[np.ndarray] = None
    pre_cleanup_conflict: Optional[np.ndarray] = None
    pre_cleanup_confidence: Optional[np.ndarray] = None
    soft_top1_label: Optional[np.ndarray] = None
    soft_top1_prob: Optional[np.ndarray] = None
    soft_top2_label: Optional[np.ndarray] = None
    soft_top2_prob: Optional[np.ndarray] = None
    soft_entropy: Optional[np.ndarray] = None
    soft_candidate_count: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeamTopologyResult:
    seam_edges: np.ndarray
    seam_loops: list[np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)
    partition_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitTopologyResult:
    mesh: trimesh.Trimesh
    split_vertices: Optional[np.ndarray]
    split_faces: Optional[np.ndarray]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IslandPipelineResult:
    low_mesh: trimesh.Trimesh
    solve_mesh: trimesh.Trimesh
    validation: SeamValidationSettings
    high: Optional[HighIslandResult] = None
    semantic: Optional[LowSemanticResult] = None
    seam: Optional[SeamTopologyResult] = None
    split: Optional[SplitTopologyResult] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    validation_ok: bool = False
    validation_error: Optional[str] = None


def _optional_face_array(raw: Any, *, n_faces: int, dtype: Any) -> Optional[np.ndarray]:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=dtype).reshape(-1)
    if int(arr.shape[0]) != int(n_faces):
        return None
    return arr


def _same_label_components(labels: np.ndarray, neighbors: List[List[int]]) -> List[Tuple[int, np.ndarray]]:
    lbl = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_faces = int(lbl.shape[0])
    seen = np.zeros((n_faces,), dtype=np.bool_)
    components: List[Tuple[int, np.ndarray]] = []
    for fid in range(n_faces):
        if seen[fid]:
            continue
        label = int(lbl[fid])
        q: deque[int] = deque([int(fid)])
        seen[fid] = True
        comp_faces: List[int] = []
        while q:
            cur = q.popleft()
            comp_faces.append(cur)
            for nb in neighbors[cur]:
                if seen[nb] or int(lbl[nb]) != label:
                    continue
                seen[nb] = True
                q.append(int(nb))
        components.append((label, np.asarray(comp_faces, dtype=np.int64)))
    return components


def _semantic_stage_summary(
    *,
    labels: np.ndarray,
    neighbors: List[List[int]],
    prefix: str,
    main_ratio_threshold: float,
    tiny_abs_threshold: int,
    tiny_ratio_threshold: float,
    tiny_max_components: int,
) -> Dict[str, Any]:
    lbl = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid_ids = np.unique(lbl[lbl >= 0]).astype(np.int64, copy=False)
    unknown_faces = int(np.count_nonzero(lbl < 0))
    fragmented_labels = 0
    severe_labels = 0
    intrusion_labels = 0

    for lid in valid_ids.tolist():
        label_mask = lbl == int(lid)
        seen = np.zeros((lbl.shape[0],), dtype=np.bool_)
        comp_sizes: List[int] = []
        for fid in np.where(label_mask)[0].tolist():
            if seen[fid]:
                continue
            q: deque[int] = deque([int(fid)])
            seen[fid] = True
            size = 0
            while q:
                cur = q.popleft()
                size += 1
                for nb in neighbors[cur]:
                    if label_mask[nb] and (not seen[nb]):
                        seen[nb] = True
                        q.append(int(nb))
            comp_sizes.append(int(size))
        comp_sizes.sort(reverse=True)
        total = int(sum(comp_sizes))
        main = int(comp_sizes[0]) if len(comp_sizes) > 0 else 0
        comp_count = int(len(comp_sizes))
        main_ratio = float(main / max(1, total))
        leaked_ratio = float(max(0, total - main) / max(1, total))
        tiny_threshold = int(
            max(
                2,
                min(
                    int(max(2, tiny_abs_threshold)),
                    int(max(2, round(total * max(0.0, tiny_ratio_threshold)))),
                ),
            )
        )
        tiny_components = int(sum(1 for s in comp_sizes if s <= tiny_threshold))
        non_tiny_components = int(max(0, comp_count - tiny_components))
        is_normal = bool(
            main_ratio >= float(main_ratio_threshold)
            and non_tiny_components <= 1
            and tiny_components <= int(tiny_max_components)
        )
        if not is_normal:
            fragmented_labels += 1
        if (main_ratio < 0.80) or (comp_count >= 10) or (leaked_ratio >= 0.20):
            severe_labels += 1
        if leaked_ratio >= 0.05:
            intrusion_labels += 1

    return {
        f"{prefix}label_count": int(valid_ids.size),
        f"{prefix}unknown_faces": int(unknown_faces),
        f"{prefix}fragmented_label_count": int(fragmented_labels),
        f"{prefix}severe_label_count": int(severe_labels),
        f"{prefix}intrusion_like_label_count": int(intrusion_labels),
        f"{prefix}normal_overall": bool(fragmented_labels == 0),
    }


def _absorb_small_semantic_components(
    *,
    labels: np.ndarray,
    neighbors: List[List[int]],
    edge_weights: List[List[float]],
    min_faces: int,
    max_iters: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = np.asarray(labels, dtype=np.int64).copy()
    min_faces_i = int(max(0, min_faces))
    max_iters_i = int(max(0, max_iters))
    if min_faces_i <= 1 or max_iters_i <= 0 or out.size == 0:
        return out, {
            "uv_semantic_transfer_component_merge_merged_components": 0,
            "uv_semantic_transfer_component_merge_merged_faces": 0,
            "uv_semantic_transfer_component_merge_iterations": 0,
            "uv_semantic_transfer_component_merge_remaining_small_components": 0,
            "uv_semantic_transfer_component_merge_remaining_small_faces": 0,
        }

    total_merged_components = 0
    total_merged_faces = 0
    iters_used = 0

    for iter_idx in range(max_iters_i):
        label_ids, label_counts = np.unique(out[out >= 0], return_counts=True)
        label_size = {int(k): int(v) for k, v in zip(label_ids.tolist(), label_counts.tolist())}
        plans: List[Tuple[np.ndarray, int]] = []

        for src_label, comp_faces in _same_label_components(out, neighbors):
            if int(comp_faces.size) >= min_faces_i:
                continue
            boundary_weight: Dict[int, float] = {}
            for fid in comp_faces.tolist():
                for nb, weight in zip(neighbors[fid], edge_weights[fid]):
                    dst_label = int(out[nb])
                    if dst_label < 0 or dst_label == int(src_label):
                        continue
                    boundary_weight[dst_label] = boundary_weight.get(dst_label, 0.0) + float(max(weight, 1e-8))
            if not boundary_weight:
                continue
            target = sorted(
                boundary_weight.items(),
                key=lambda kv: (-kv[1], -label_size.get(int(kv[0]), 0), int(kv[0])),
            )[0][0]
            plans.append((comp_faces, int(target)))

        if len(plans) == 0:
            break

        merged_this_iter = 0
        merged_faces_this_iter = 0
        for comp_faces, target in plans:
            source = int(out[int(comp_faces[0])])
            if source == int(target):
                continue
            out[comp_faces] = int(target)
            merged_this_iter += 1
            merged_faces_this_iter += int(comp_faces.size)
        if merged_this_iter == 0:
            break
        total_merged_components += int(merged_this_iter)
        total_merged_faces += int(merged_faces_this_iter)
        iters_used = int(iter_idx + 1)

    remaining_small_components = 0
    remaining_small_faces = 0
    for _, comp_faces in _same_label_components(out, neighbors):
        if int(comp_faces.size) < min_faces_i:
            remaining_small_components += 1
            remaining_small_faces += int(comp_faces.size)

    return out, {
        "uv_semantic_transfer_component_merge_merged_components": int(total_merged_components),
        "uv_semantic_transfer_component_merge_merged_faces": int(total_merged_faces),
        "uv_semantic_transfer_component_merge_iterations": int(iters_used),
        "uv_semantic_transfer_component_merge_remaining_small_components": int(remaining_small_components),
        "uv_semantic_transfer_component_merge_remaining_small_faces": int(remaining_small_faces),
    }


def _component_count_per_label_from_components(labels: np.ndarray, neighbors: List[List[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for label, _ in _same_label_components(labels, neighbors):
        if int(label) < 0:
            continue
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def _label_face_counts(labels: np.ndarray) -> Dict[int, int]:
    lbl = np.asarray(labels, dtype=np.int64).reshape(-1)
    if lbl.size == 0:
        return {}
    uniq, counts = np.unique(lbl[lbl >= 0], return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def _absorb_nonmain_semantic_components(
    *,
    labels: np.ndarray,
    neighbors: List[List[int]],
    edge_weights: List[List[float]],
    main_labels: set[int],
    max_iters: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = np.asarray(labels, dtype=np.int64).copy()
    max_iters_i = int(max(0, max_iters))
    if max_iters_i <= 0 or out.size == 0 or len(main_labels) == 0:
        return out, {
            "uv_semantic_transfer_micro_shell_absorb_merged_components": 0,
            "uv_semantic_transfer_micro_shell_absorb_merged_faces": 0,
            "uv_semantic_transfer_micro_shell_absorb_iterations": 0,
            "uv_semantic_transfer_micro_shell_absorb_remaining_components": 0,
            "uv_semantic_transfer_micro_shell_absorb_remaining_faces": 0,
        }

    total_merged_components = 0
    total_merged_faces = 0
    iters_used = 0

    for iter_idx in range(max_iters_i):
        label_size = _label_face_counts(out)
        plans: List[Tuple[np.ndarray, int]] = []
        for src_label, comp_faces in _same_label_components(out, neighbors):
            if int(src_label) < 0 or int(src_label) in main_labels:
                continue
            boundary_weight_main: Dict[int, float] = {}
            boundary_weight_any: Dict[int, float] = {}
            for fid in comp_faces.tolist():
                for nb, weight in zip(neighbors[fid], edge_weights[fid]):
                    dst_label = int(out[nb])
                    if dst_label < 0 or dst_label == int(src_label):
                        continue
                    boundary_weight_any[dst_label] = boundary_weight_any.get(dst_label, 0.0) + float(max(weight, 1e-8))
                    if dst_label in main_labels:
                        boundary_weight_main[dst_label] = boundary_weight_main.get(dst_label, 0.0) + float(
                            max(weight, 1e-8)
                        )
            candidates = boundary_weight_main if boundary_weight_main else boundary_weight_any
            if not candidates:
                continue
            target = sorted(
                candidates.items(),
                key=lambda kv: (-kv[1], -label_size.get(int(kv[0]), 0), int(kv[0])),
            )[0][0]
            plans.append((comp_faces, int(target)))

        if len(plans) == 0:
            break

        merged_this_iter = 0
        merged_faces_this_iter = 0
        for comp_faces, target in plans:
            source = int(out[int(comp_faces[0])])
            if source == int(target):
                continue
            out[comp_faces] = int(target)
            merged_this_iter += 1
            merged_faces_this_iter += int(comp_faces.size)
        if merged_this_iter == 0:
            break
        total_merged_components += int(merged_this_iter)
        total_merged_faces += int(merged_faces_this_iter)
        iters_used = int(iter_idx + 1)

    remaining_components = 0
    remaining_faces = 0
    for src_label, comp_faces in _same_label_components(out, neighbors):
        if int(src_label) < 0 or int(src_label) in main_labels:
            continue
        remaining_components += 1
        remaining_faces += int(comp_faces.size)

    return out, {
        "uv_semantic_transfer_micro_shell_absorb_merged_components": int(total_merged_components),
        "uv_semantic_transfer_micro_shell_absorb_merged_faces": int(total_merged_faces),
        "uv_semantic_transfer_micro_shell_absorb_iterations": int(iters_used),
        "uv_semantic_transfer_micro_shell_absorb_remaining_components": int(remaining_components),
        "uv_semantic_transfer_micro_shell_absorb_remaining_faces": int(remaining_faces),
    }


def compute_cached_high_face_uv_islands(
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


def run_halfedge_island_pipeline(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    low_mesh: trimesh.Trimesh,
    high_ctx: Dict[str, Any],
    seam_cfg: Dict[str, Any],
    corr_cfg: Dict[str, Any],
    use_high_island_cache: bool = True,
) -> IslandPipelineResult:
    validation = resolve_seam_validation_settings(seam_cfg)
    meta: Dict[str, Any] = {
        "uv_island_pipeline_used": True,
        "uv_island_validation_mode": validation.mode,
        "uv_seam_validation_strict": bool(validation.strict),
        "uv_seam_validation_require_closed_loops": bool(validation.require_closed_loops),
        "uv_seam_validation_require_pure_components": bool(validation.require_pure_components),
        "uv_halfedge_split_requested": True,
        "uv_halfedge_split_topology_applied": False,
        "uv_halfedge_split_fallback_to_legacy": False,
    }

    ensure_halfedge_external_dependencies()
    work_low_mesh, sanitize_meta = sanitize_mesh_for_halfedge(low_mesh=low_mesh, seam_cfg=seam_cfg)
    meta.update(sanitize_meta)

    result = IslandPipelineResult(
        low_mesh=work_low_mesh,
        solve_mesh=work_low_mesh,
        validation=validation,
        meta=meta,
        validation_ok=False,
        validation_error=None,
    )

    try:
        high_labels, high_meta, cache_hit = compute_cached_high_face_uv_islands(
            high_mesh=high_mesh,
            high_uv=high_uv,
            position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
            uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
            use_cache=bool(use_high_island_cache),
        )
    except Exception as exc:
        err = f"unable to compute high-face UV islands: {exc}"
        meta["uv_high_island_error"] = str(exc)
        meta["uv_island_validation_ok"] = False
        meta["uv_island_validation_error"] = err
        result.validation_error = err
        return result

    result.high = HighIslandResult(
        face_labels=np.asarray(high_labels, dtype=np.int64, copy=False),
        meta=dict(high_meta),
        cache_hit=bool(cache_hit),
    )
    meta["uv_m2_perf_high_island_cache_enabled"] = bool(use_high_island_cache)
    meta["uv_m2_perf_high_island_cache_hit"] = bool(cache_hit)
    meta["uv_high_island_count"] = int(high_meta.get("high_island_count", 0))
    meta["uv_high_seam_edges"] = int(high_meta.get("high_seam_edges", 0))
    meta["uv_high_boundary_edges"] = int(high_meta.get("high_boundary_edges", 0))
    meta["uv_high_nonmanifold_edges"] = int(high_meta.get("high_nonmanifold_edges", 0))
    high_face_count_per_label = _label_face_counts(result.high.face_labels)
    main_shell_min_faces = int(max(1, seam_cfg.get("main_shell_min_faces", 16)))
    main_high_labels = {
        int(label) for label, count in high_face_count_per_label.items() if int(count) >= int(main_shell_min_faces)
    }
    micro_high_labels = set(high_face_count_per_label.keys()) - set(main_high_labels)
    meta["uv_high_main_shell_min_faces"] = int(main_shell_min_faces)
    meta["uv_high_main_shell_count"] = int(len(main_high_labels))
    meta["uv_high_micro_shell_count"] = int(len(micro_high_labels))
    meta["uv_high_micro_shell_faces_total"] = int(
        sum(int(high_face_count_per_label.get(label, 0)) for label in micro_high_labels)
    )

    semantic_seam_cfg = dict(seam_cfg)
    semantic_mode = str(semantic_seam_cfg.get("transfer_sampling_mode", "")).strip().lower()
    if semantic_mode in {"", "single_point_projection"}:
        semantic_seam_cfg["transfer_sampling_mode"] = "four_point_soft_flood"
    if "transfer_max_dist_ratio" not in semantic_seam_cfg:
        semantic_seam_cfg["transfer_max_dist_ratio"] = 0.005
    semantic_seam_cfg["transfer_main_shell_labels"] = sorted(int(x) for x in main_high_labels)
    semantic_seam_cfg["transfer_micro_shell_labels"] = sorted(int(x) for x in micro_high_labels)

    semantic_raw = transfer_face_semantics_by_projection(
        high_ctx=high_ctx,
        high_face_island=result.high.face_labels,
        low_mesh=work_low_mesh,
        seam_cfg=semantic_seam_cfg,
        corr_cfg=corr_cfg,
    )
    n_faces = int(len(work_low_mesh.faces))
    face_labels = np.asarray(semantic_raw.get("low_face_island"), dtype=np.int64, copy=False).reshape(-1)
    if int(face_labels.shape[0]) != n_faces:
        face_labels = np.full((n_faces,), -1, dtype=np.int64)
    face_conflict = np.asarray(
        semantic_raw.get("low_face_conflict", np.ones((n_faces,), dtype=np.bool_)),
        dtype=np.bool_,
        copy=False,
    ).reshape(-1)
    if int(face_conflict.shape[0]) != n_faces:
        face_conflict = np.ones((n_faces,), dtype=np.bool_)
    face_confidence = np.asarray(
        semantic_raw.get("low_face_confidence", np.zeros((n_faces,), dtype=np.float32)),
        dtype=np.float32,
        copy=False,
    ).reshape(-1)
    if int(face_confidence.shape[0]) != n_faces:
        face_confidence = np.zeros((n_faces,), dtype=np.float32)
    pre_bfs_labels = np.asarray(
        semantic_raw.get("low_face_pre_bfs_label", face_labels),
        dtype=np.int64,
        copy=False,
    ).reshape(-1)
    if int(pre_bfs_labels.shape[0]) != n_faces:
        pre_bfs_labels = np.full((n_faces,), -1, dtype=np.int64)
    pre_bfs_confidence = np.asarray(
        semantic_raw.get("low_face_pre_bfs_confidence", np.zeros((n_faces,), dtype=np.float32)),
        dtype=np.float32,
        copy=False,
    ).reshape(-1)
    if int(pre_bfs_confidence.shape[0]) != n_faces:
        pre_bfs_confidence = np.zeros((n_faces,), dtype=np.float32)
    pre_bfs_state = np.asarray(
        semantic_raw.get("low_face_pre_bfs_state", np.zeros((n_faces,), dtype=np.uint8)),
        dtype=np.uint8,
        copy=False,
    ).reshape(-1)
    if int(pre_bfs_state.shape[0]) != n_faces:
        pre_bfs_state = np.zeros((n_faces,), dtype=np.uint8)

    soft_top1_label = _optional_face_array(
        semantic_raw.get("low_face_soft_top1_label"), n_faces=n_faces, dtype=np.int64
    )
    soft_top1_prob = _optional_face_array(
        semantic_raw.get("low_face_soft_top1_prob"), n_faces=n_faces, dtype=np.float32
    )
    soft_top2_label = _optional_face_array(
        semantic_raw.get("low_face_soft_top2_label"), n_faces=n_faces, dtype=np.int64
    )
    soft_top2_prob = _optional_face_array(
        semantic_raw.get("low_face_soft_top2_prob"), n_faces=n_faces, dtype=np.float32
    )
    soft_entropy = _optional_face_array(semantic_raw.get("low_face_soft_entropy"), n_faces=n_faces, dtype=np.float32)
    soft_candidate_count = _optional_face_array(
        semantic_raw.get("low_face_soft_candidate_count"), n_faces=n_faces, dtype=np.int32
    )

    neighbors, edge_weights = _build_weighted_face_adjacency(work_low_mesh)
    pre_cleanup_labels = face_labels.astype(np.int64, copy=True)
    pre_cleanup_conflict = face_conflict.astype(np.bool_, copy=True)
    pre_cleanup_confidence = face_confidence.astype(np.float32, copy=True)
    semantic_meta = dict(semantic_raw.get("meta", {}))

    summary_main_ratio = float(semantic_seam_cfg.get("semantic_summary_main_ratio_threshold", 0.95))
    summary_tiny_abs = int(semantic_seam_cfg.get("semantic_summary_tiny_abs_threshold", 16))
    summary_tiny_ratio = float(semantic_seam_cfg.get("semantic_summary_tiny_ratio_threshold", 0.005))
    summary_tiny_max = int(semantic_seam_cfg.get("semantic_summary_tiny_max_components", 2))
    semantic_meta.update(
        _semantic_stage_summary(
            labels=pre_bfs_labels,
            neighbors=neighbors,
            prefix="uv_semantic_transfer_pre_bfs_",
            main_ratio_threshold=summary_main_ratio,
            tiny_abs_threshold=summary_tiny_abs,
            tiny_ratio_threshold=summary_tiny_ratio,
            tiny_max_components=summary_tiny_max,
        )
    )
    semantic_meta.update(
        _semantic_stage_summary(
            labels=pre_cleanup_labels,
            neighbors=neighbors,
            prefix="uv_semantic_transfer_pre_cleanup_",
            main_ratio_threshold=summary_main_ratio,
            tiny_abs_threshold=summary_tiny_abs,
            tiny_ratio_threshold=summary_tiny_ratio,
            tiny_max_components=summary_tiny_max,
        )
    )

    component_merge_enabled = bool(semantic_seam_cfg.get("component_merge_enabled", True))
    component_merge_min_faces = int(semantic_seam_cfg.get("component_merge_min_faces", 4))
    component_merge_max_iters = int(semantic_seam_cfg.get("component_merge_max_iters", 8))
    semantic_meta["uv_semantic_transfer_component_merge_enabled"] = bool(component_merge_enabled)
    semantic_meta["uv_semantic_transfer_component_merge_min_faces"] = int(component_merge_min_faces)
    semantic_meta["uv_semantic_transfer_component_merge_max_iters"] = int(component_merge_max_iters)
    semantic_meta["uv_semantic_transfer_component_merge_changed"] = False
    if component_merge_enabled:
        merged_labels, merge_meta = _absorb_small_semantic_components(
            labels=face_labels,
            neighbors=neighbors,
            edge_weights=edge_weights,
            min_faces=component_merge_min_faces,
            max_iters=component_merge_max_iters,
        )
        semantic_meta.update(merge_meta)
        if np.any(merged_labels != face_labels):
            face_labels = merged_labels
            conf_min = float(semantic_seam_cfg.get("uv_island_guard_confidence_min", 0.55))
            face_confidence = _face_label_confidence(face_labels, neighbors)
            face_conflict = (face_labels < 0) | (face_confidence < conf_min)
            semantic_meta["uv_semantic_transfer_component_merge_changed"] = True
    else:
        semantic_meta.update(
            {
                "uv_semantic_transfer_component_merge_merged_components": 0,
                "uv_semantic_transfer_component_merge_merged_faces": 0,
                "uv_semantic_transfer_component_merge_iterations": 0,
                "uv_semantic_transfer_component_merge_remaining_small_components": 0,
                "uv_semantic_transfer_component_merge_remaining_small_faces": 0,
            }
        )

    micro_shell_absorb_enabled = bool(semantic_seam_cfg.get("micro_shell_absorb_enabled", True))
    micro_shell_absorb_max_iters = int(semantic_seam_cfg.get("micro_shell_absorb_max_iters", 8))
    semantic_meta["uv_semantic_transfer_micro_shell_absorb_enabled"] = bool(micro_shell_absorb_enabled)
    semantic_meta["uv_semantic_transfer_micro_shell_absorb_max_iters"] = int(micro_shell_absorb_max_iters)
    semantic_meta["uv_semantic_transfer_main_shell_min_faces"] = int(main_shell_min_faces)
    semantic_meta["uv_semantic_transfer_main_shell_count"] = int(len(main_high_labels))
    semantic_meta["uv_semantic_transfer_micro_shell_count"] = int(len(micro_high_labels))
    semantic_meta["uv_semantic_transfer_micro_shell_absorb_changed"] = False
    if micro_shell_absorb_enabled and len(main_high_labels) > 0 and len(micro_high_labels) > 0:
        absorbed_labels, micro_absorb_meta = _absorb_nonmain_semantic_components(
            labels=face_labels,
            neighbors=neighbors,
            edge_weights=edge_weights,
            main_labels=main_high_labels,
            max_iters=micro_shell_absorb_max_iters,
        )
        semantic_meta.update(micro_absorb_meta)
        if np.any(absorbed_labels != face_labels):
            face_labels = absorbed_labels
            conf_min = float(semantic_seam_cfg.get("uv_island_guard_confidence_min", 0.55))
            face_confidence = _face_label_confidence(face_labels, neighbors)
            face_conflict = (face_labels < 0) | (face_confidence < conf_min)
            semantic_meta["uv_semantic_transfer_micro_shell_absorb_changed"] = True
    else:
        semantic_meta.update(
            {
                "uv_semantic_transfer_micro_shell_absorb_merged_components": 0,
                "uv_semantic_transfer_micro_shell_absorb_merged_faces": 0,
                "uv_semantic_transfer_micro_shell_absorb_iterations": 0,
                "uv_semantic_transfer_micro_shell_absorb_remaining_components": 0,
                "uv_semantic_transfer_micro_shell_absorb_remaining_faces": 0,
            }
        )

    semantic_meta.update(
        _semantic_stage_summary(
            labels=face_labels,
            neighbors=neighbors,
            prefix="uv_semantic_transfer_final_",
            main_ratio_threshold=summary_main_ratio,
            tiny_abs_threshold=summary_tiny_abs,
            tiny_ratio_threshold=summary_tiny_ratio,
            tiny_max_components=summary_tiny_max,
        )
    )
    valid_final = face_labels >= 0
    final_comp_per_id = _component_count_per_label_from_components(face_labels, neighbors)
    semantic_meta["uv_semantic_transfer_unknown_faces"] = int(np.count_nonzero(~valid_final))
    semantic_meta["uv_semantic_transfer_mapped_faces"] = int(np.count_nonzero(valid_final))
    semantic_meta["uv_semantic_transfer_final_label_count"] = int(np.unique(face_labels[valid_final]).size)
    semantic_meta["uv_semantic_transfer_conflict_faces"] = int(np.count_nonzero(face_conflict))
    semantic_meta["uv_semantic_transfer_confidence_mean"] = (
        float(np.mean(face_confidence[valid_final])) if np.any(valid_final) else 0.0
    )
    semantic_meta["uv_semantic_transfer_component_count_per_id"] = {
        str(k): int(v) for k, v in sorted(final_comp_per_id.items())
    }
    semantic_meta["uv_semantic_transfer_component_count_max"] = (
        int(max(final_comp_per_id.values())) if len(final_comp_per_id) > 0 else 0
    )

    semantic = LowSemanticResult(
        face_labels=face_labels.astype(np.int64, copy=False),
        face_conflict=face_conflict.astype(np.bool_, copy=False),
        face_confidence=face_confidence.astype(np.float32, copy=False),
        pre_bfs_labels=pre_bfs_labels.astype(np.int64, copy=False),
        pre_bfs_confidence=pre_bfs_confidence.astype(np.float32, copy=False),
        pre_bfs_state=pre_bfs_state.astype(np.uint8, copy=False),
        pre_cleanup_labels=pre_cleanup_labels.astype(np.int64, copy=False),
        pre_cleanup_conflict=pre_cleanup_conflict.astype(np.bool_, copy=False),
        pre_cleanup_confidence=pre_cleanup_confidence.astype(np.float32, copy=False),
        soft_top1_label=soft_top1_label,
        soft_top1_prob=soft_top1_prob,
        soft_top2_label=soft_top2_label,
        soft_top2_prob=soft_top2_prob,
        soft_entropy=soft_entropy,
        soft_candidate_count=soft_candidate_count,
        meta=semantic_meta,
    )
    result.semantic = semantic
    meta.update(semantic.meta)
    meta["uv_island_conflict_faces"] = int(np.count_nonzero(semantic.face_conflict))
    meta["uv_island_unknown_faces"] = int(np.count_nonzero(semantic.face_labels < 0))
    meta["uv_island_conflict_faces_excluded"] = 0

    seam_result = extract_seam_edges_openmesh(
        low_mesh=work_low_mesh,
        face_labels=semantic.face_labels,
        include_boundary_as_seam=bool(seam_cfg.get("include_boundary_as_seam", False)),
        validation_min_component_faces=int(validation.min_component_faces),
        validation_allow_open_on_boundary=bool(validation.allow_open_on_boundary),
    )
    partition_meta = validate_face_partition_by_seams(
        low_mesh=work_low_mesh,
        face_labels=semantic.face_labels,
        seam_edges=np.asarray(seam_result.seam_edges, dtype=np.int64, copy=False),
        min_component_faces=int(validation.min_component_faces),
    )
    result.seam = SeamTopologyResult(
        seam_edges=np.asarray(seam_result.seam_edges, dtype=np.int64, copy=False),
        seam_loops=list(seam_result.seam_loops),
        meta=dict(seam_result.meta),
        partition_meta=dict(partition_meta),
    )
    meta.update(seam_result.meta)
    meta["uv_halfedge_backend"] = str(seam_result.meta.get("uv_seam_extraction_backend", "openmesh"))
    meta.update(partition_meta)

    topology_ok = bool(seam_result.meta.get("uv_seam_topology_valid", False)) or (
        not validation.require_closed_loops
    )
    partition_ok = bool(partition_meta.get("uv_seam_partition_is_valid", False)) or (
        not validation.require_pure_components
    )
    validation_ok = bool(topology_ok and partition_ok)
    validation_error = None
    if not validation_ok:
        validation_error = (
            f"topology_ok={topology_ok}, partition_ok={partition_ok}, "
            f"components_open={int(seam_result.meta.get('uv_seam_components_open', 0))}, "
            f"mixed_components={int(partition_meta.get('uv_seam_partition_mixed_components', 0))}"
        )
    meta["uv_seam_validation_ok"] = bool(validation_ok)
    if validation_error is not None:
        meta["uv_seam_validation_error"] = validation_error
    meta["uv_island_validation_ok"] = bool(validation_ok)
    if validation_error is not None:
        meta["uv_island_validation_error"] = validation_error

    split_vertices, split_faces, split_meta = split_vertices_along_cut_edges(
        vertices=np.asarray(work_low_mesh.vertices, dtype=np.float32),
        faces=np.asarray(work_low_mesh.faces, dtype=np.int64),
        cut_edges=np.asarray(seam_result.seam_edges, dtype=np.int64, copy=False),
    )
    split_vertices_out: Optional[np.ndarray] = None
    split_faces_out: Optional[np.ndarray] = None
    solve_mesh = work_low_mesh
    if int(split_vertices.shape[0]) > int(len(work_low_mesh.vertices)):
        solve_mesh = trimesh.Trimesh(vertices=split_vertices, faces=split_faces, process=False)
        split_vertices_out = split_vertices.astype(np.float32, copy=False)
        split_faces_out = split_faces.astype(np.int64, copy=False)

    result.solve_mesh = solve_mesh
    result.split = SplitTopologyResult(
        mesh=solve_mesh,
        split_vertices=split_vertices_out,
        split_faces=split_faces_out,
        meta=dict(split_meta),
    )
    meta["uv_low_cut_edges"] = int(split_meta.get("cut_edges", 0))
    meta["uv_low_split_vertices"] = int(split_meta.get("split_vertices_added", 0))
    meta["uv_low_split_faces"] = int(split_meta.get("split_faces", int(len(work_low_mesh.faces))))
    meta["uv_halfedge_split_topology_applied"] = bool(solve_mesh is not work_low_mesh)

    result.validation_ok = bool(validation_ok)
    result.validation_error = validation_error
    return result


__all__ = [
    "HighIslandResult",
    "LowSemanticResult",
    "SeamTopologyResult",
    "SplitTopologyResult",
    "IslandPipelineResult",
    "compute_cached_high_face_uv_islands",
    "run_halfedge_island_pipeline",
]
