from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import trimesh


@dataclass
class RoutedSeamResult:
    cut_edges: np.ndarray
    low_face_island: np.ndarray
    low_face_expected_high_island: np.ndarray
    low_face_conflict: np.ndarray
    low_face_confidence: np.ndarray
    split_vertices: np.ndarray
    split_faces: np.ndarray
    split_meta: Dict[str, Any]
    route_meta: Dict[str, Any]
    route_edge_p0: np.ndarray
    route_edge_p1: np.ndarray
    route_edge_cost: np.ndarray
    high_seam_p0: np.ndarray
    high_seam_p1: np.ndarray
    projected_points: np.ndarray
    failed_pair_p0: np.ndarray
    failed_pair_p1: np.ndarray
    outlier_pair_p0: np.ndarray
    outlier_pair_p1: np.ndarray
    conn_bad_pair_p0: np.ndarray
    conn_bad_pair_p1: np.ndarray


def _majority_face_labels(
    *,
    n_low_faces: int,
    sample_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    valid_mask: np.ndarray,
    high_face_island: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low_face_label = np.full((n_low_faces,), -1, dtype=np.int64)
    low_face_conflict = np.zeros((n_low_faces,), dtype=np.bool_)
    low_face_conf = np.zeros((n_low_faces,), dtype=np.float32)

    sf = np.asarray(sample_face_ids, dtype=np.int64)
    tf = np.asarray(target_face_ids, dtype=np.int64)
    vm = np.asarray(valid_mask, dtype=np.bool_)
    hi = np.asarray(high_face_island, dtype=np.int64)

    valid = (
        vm
        & (sf >= 0)
        & (sf < n_low_faces)
        & (tf >= 0)
        & (tf < hi.shape[0])
    )
    if not np.any(valid):
        return low_face_label, low_face_conflict, low_face_conf

    sfv = sf[valid]
    hiv = hi[tf[valid]]
    pair_valid = hiv >= 0
    if not np.any(pair_valid):
        return low_face_label, low_face_conflict, low_face_conf

    sfv = sfv[pair_valid]
    hiv = hiv[pair_valid]
    pair = np.stack([sfv, hiv], axis=1)
    uniq_pair, cnt_pair = np.unique(pair, axis=0, return_counts=True)
    low_ids = uniq_pair[:, 0].astype(np.int64, copy=False)
    high_ids = uniq_pair[:, 1].astype(np.int64, copy=False)
    cnt_pair = cnt_pair.astype(np.int64, copy=False)

    order = np.lexsort((high_ids, -cnt_pair, low_ids))
    low_ids = low_ids[order]
    high_ids = high_ids[order]
    cnt_pair = cnt_pair[order]
    uniq_low, first_idx = np.unique(low_ids, return_index=True)
    total_by_low = np.bincount(low_ids, weights=cnt_pair, minlength=max(int(np.max(low_ids)) + 1, 1)).astype(np.float64)

    for li, idx in zip(uniq_low.tolist(), first_idx.tolist()):
        li_i = int(li)
        hi_i = int(high_ids[idx])
        cnt = float(cnt_pair[idx])
        total = float(total_by_low[li_i]) if li_i < total_by_low.shape[0] else cnt
        conf = float(cnt / max(total, 1.0))
        low_face_label[li_i] = hi_i
        low_face_conf[li_i] = float(conf)
        low_face_conflict[li_i] = bool(conf < 0.55)

    return low_face_label, low_face_conflict, low_face_conf


def route_low_mesh_seams_by_dijkstra(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    low_mesh: trimesh.Trimesh,
    sample_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    valid_mask: np.ndarray,
    high_face_island: np.ndarray,
    seam_cfg: Dict[str, Any],
) -> RoutedSeamResult:
    # Compatibility shell: legacy fallback only.
    del high_mesh, high_uv, seam_cfg

    n_faces = int(len(low_mesh.faces))
    low_face_island, low_face_conflict, low_face_conf = _majority_face_labels(
        n_low_faces=n_faces,
        sample_face_ids=np.asarray(sample_face_ids, dtype=np.int64),
        target_face_ids=np.asarray(target_face_ids, dtype=np.int64),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        high_face_island=np.asarray(high_face_island, dtype=np.int64),
    )
    low_expected = low_face_island.astype(np.int64, copy=False)
    low_vertices = np.asarray(low_mesh.vertices, dtype=np.float32)
    low_faces = np.asarray(low_mesh.faces, dtype=np.int64)

    meta: Dict[str, Any] = {
        "uv_route_mode": "legacy_compat_fallback",
        "uv_route_deprecated": True,
        "uv_route_chain_count": 0,
        "uv_route_cut_edges": 0,
        "uv_low_cut_edges": 0,
        "uv_low_split_vertices": 0,
        "uv_low_split_faces": int(low_faces.shape[0]),
        "uv_low_island_count": int(np.unique(low_face_island[low_face_island >= 0]).size),
        "uv_low_island_mapped_count": int(np.count_nonzero(low_expected >= 0)),
        "uv_low_island_conflict_count": int(np.count_nonzero(low_face_conflict)),
        "uv_island_unknown_faces": int(np.count_nonzero(low_expected < 0)),
    }

    return RoutedSeamResult(
        cut_edges=np.zeros((0, 2), dtype=np.int64),
        low_face_island=np.asarray(low_face_island, dtype=np.int64),
        low_face_expected_high_island=np.asarray(low_expected, dtype=np.int64),
        low_face_conflict=np.asarray(low_face_conflict, dtype=np.bool_),
        low_face_confidence=np.asarray(low_face_conf, dtype=np.float32),
        split_vertices=np.asarray(low_vertices, dtype=np.float32),
        split_faces=np.asarray(low_faces, dtype=np.int64),
        split_meta={"cut_edges": 0, "split_vertices_added": 0, "split_faces": int(low_faces.shape[0])},
        route_meta=meta,
        route_edge_p0=np.zeros((0, 3), dtype=np.float32),
        route_edge_p1=np.zeros((0, 3), dtype=np.float32),
        route_edge_cost=np.zeros((0,), dtype=np.float32),
        high_seam_p0=np.zeros((0, 3), dtype=np.float32),
        high_seam_p1=np.zeros((0, 3), dtype=np.float32),
        projected_points=np.zeros((0, 3), dtype=np.float32),
        failed_pair_p0=np.zeros((0, 3), dtype=np.float32),
        failed_pair_p1=np.zeros((0, 3), dtype=np.float32),
        outlier_pair_p0=np.zeros((0, 3), dtype=np.float32),
        outlier_pair_p1=np.zeros((0, 3), dtype=np.float32),
        conn_bad_pair_p0=np.zeros((0, 3), dtype=np.float32),
        conn_bad_pair_p1=np.zeros((0, 3), dtype=np.float32),
    )


__all__ = [
    "RoutedSeamResult",
    "route_low_mesh_seams_by_dijkstra",
]

