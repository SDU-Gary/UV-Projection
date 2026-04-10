from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..halfedge_topology import build_halfedge_mesh, split_vertices_along_cut_edges


@dataclass
class HighSeamData:
    weld_positions: np.ndarray
    seam_edges_weld: np.ndarray
    seam_chains: List[np.ndarray]
    seam_chain_closed: List[bool]
    seam_segments_p0: np.ndarray
    seam_segments_p1: np.ndarray
    seam_segments_dir: np.ndarray
    seam_segments_len: np.ndarray
    meta: Dict[str, Any]


@dataclass
class LowGraphData:
    edge_vertices: np.ndarray
    edge_lengths: np.ndarray
    edge_midpoints: np.ndarray
    edge_dirs_unit: np.ndarray
    edge_flatness: np.ndarray
    adjacency: List[List[Tuple[int, int]]]
    edge_key_to_id: Dict[Tuple[int, int], int]


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


def _quantize_positions(vertices: np.ndarray, eps: float) -> np.ndarray:
    q = np.round(np.asarray(vertices, dtype=np.float64) / max(float(eps), 1e-12)).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _weld_centroids(vertices: np.ndarray, weld_ids: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float64)
    weld = np.asarray(weld_ids, dtype=np.int64)
    if weld.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    n_weld = int(np.max(weld)) + 1
    acc = np.zeros((n_weld, 3), dtype=np.float64)
    cnt = np.zeros((n_weld,), dtype=np.float64)
    np.add.at(acc[:, 0], weld, verts[:, 0])
    np.add.at(acc[:, 1], weld, verts[:, 1])
    np.add.at(acc[:, 2], weld, verts[:, 2])
    np.add.at(cnt, weld, 1.0)
    acc /= np.maximum(cnt[:, None], 1.0)
    return acc


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _extract_high_seam_edges_and_chains(
    *,
    high_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    position_eps: float,
    uv_eps: float,
) -> HighSeamData:
    faces = np.asarray(high_mesh.faces, dtype=np.int64)
    verts = np.asarray(high_mesh.vertices, dtype=np.float64)
    uv = np.asarray(high_uv, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("high mesh faces must be [F,3]")
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != verts.shape[0]:
        raise ValueError("high uv must be [V,2] and match vertices")

    weld = _quantize_positions(verts, float(position_eps))
    weld_pos = _weld_centroids(verts, weld)

    v0 = faces[:, [0, 1, 2]].reshape(-1)
    v1 = faces[:, [1, 2, 0]].reshape(-1)
    fidx = np.repeat(np.arange(faces.shape[0], dtype=np.int64), 3)

    g0 = weld[v0]
    g1 = weld[v1]
    e0 = np.minimum(g0, g1)
    e1 = np.maximum(g0, g1)
    edge_key_arr = np.stack([e0, e1], axis=1).astype(np.int64, copy=False)
    uniq, inv, counts = np.unique(edge_key_arr, axis=0, return_inverse=True, return_counts=True)
    counts = counts.astype(np.int64, copy=False)

    uv0 = uv[v0]
    uv1 = uv[v1]
    swap = g0 > g1
    uv_a = np.where(swap[:, None], uv1, uv0)
    uv_b = np.where(swap[:, None], uv0, uv1)

    seam_group = np.zeros((uniq.shape[0],), dtype=np.bool_)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    seam_group |= counts != 2

    interior_mask = counts == 2
    seam_interior = 0
    if np.any(interior_mask):
        edge_ids = np.arange(edge_key_arr.shape[0], dtype=np.int64)
        interior_edge_ids = edge_ids[interior_mask[inv]]
        interior_grp_ids = inv[interior_edge_ids]
        order = np.argsort(interior_grp_ids, kind="mergesort")
        pair_idx = interior_edge_ids[order]
        if pair_idx.size % 2 != 0:
            pair_idx = pair_idx[: pair_idx.size - 1]
        i0 = pair_idx[0::2]
        i1 = pair_idx[1::2]
        f0 = fidx[i0]
        f1 = fidx[i1]
        uv_da = uv_a[i0] - uv_a[i1]
        uv_db = uv_b[i0] - uv_b[i1]
        uv_eps2 = float(max(0.0, uv_eps) ** 2)
        same_a = np.einsum("ij,ij->i", uv_da, uv_da, optimize=True) <= uv_eps2
        same_b = np.einsum("ij,ij->i", uv_db, uv_db, optimize=True) <= uv_eps2
        can_link = (f0 != f1) & same_a & same_b
        seam_pair = ~can_link
        seam_interior = int(np.count_nonzero(seam_pair))
        if np.any(seam_pair):
            seam_group[interior_grp_ids[order][0::2][seam_pair]] = True

    seam_edges_weld = uniq[seam_group]
    if seam_edges_weld.size == 0:
        return HighSeamData(
            weld_positions=weld_pos,
            seam_edges_weld=np.zeros((0, 2), dtype=np.int64),
            seam_chains=[],
            seam_chain_closed=[],
            seam_segments_p0=np.zeros((0, 3), dtype=np.float64),
            seam_segments_p1=np.zeros((0, 3), dtype=np.float64),
            seam_segments_dir=np.zeros((0, 3), dtype=np.float64),
            seam_segments_len=np.zeros((0,), dtype=np.float64),
            meta={
                "high_island_count": 0,
                "high_seam_edges": 0,
                "high_boundary_edges": boundary_edges,
                "high_nonmanifold_edges": nonmanifold_edges,
                "high_interior_seam_edges": 0,
                "high_seam_chain_count": 0,
            },
        )

    # Build seam chains from seam-edge graph.
    adj: Dict[int, List[int]] = {}
    for a, b in seam_edges_weld.tolist():
        ia = int(a)
        ib = int(b)
        adj.setdefault(ia, []).append(ib)
        adj.setdefault(ib, []).append(ia)
    for k in list(adj.keys()):
        adj[k] = sorted(set(adj[k]))

    used: set[Tuple[int, int]] = set()
    chains_vid: List[np.ndarray] = []
    chains_closed: List[bool] = []

    def _walk(start: int, nxt: int) -> Tuple[List[int], bool]:
        chain = [start, nxt]
        used.add(_edge_key(start, nxt))
        prev = start
        cur = nxt
        closed = False
        while True:
            deg = len(adj.get(cur, []))
            if deg != 2:
                break
            nbs = adj[cur]
            cand = nbs[0] if nbs[1] == prev else nbs[1]
            ek = _edge_key(cur, cand)
            if ek in used:
                if cand == start:
                    chain.append(start)
                    closed = True
                break
            used.add(ek)
            chain.append(cand)
            prev, cur = cur, cand
            if cur == start:
                closed = True
                break
        return chain, closed

    endpoint_nodes = sorted([v for v, nbs in adj.items() if len(nbs) != 2])
    for s in endpoint_nodes:
        for nb in adj.get(s, []):
            if _edge_key(s, nb) in used:
                continue
            c, is_closed = _walk(s, nb)
            if len(c) >= 2:
                chains_vid.append(np.asarray(c, dtype=np.int64))
                chains_closed.append(bool(is_closed))

    # Remaining edges are cycles.
    for a, b in seam_edges_weld.tolist():
        ek = _edge_key(int(a), int(b))
        if ek in used:
            continue
        c, is_closed = _walk(int(a), int(b))
        if len(c) >= 2:
            chains_vid.append(np.asarray(c, dtype=np.int64))
            chains_closed.append(bool(is_closed))

    seam_chains: List[np.ndarray] = [weld_pos[c] for c in chains_vid]

    seg_p0 = weld_pos[seam_edges_weld[:, 0]]
    seg_p1 = weld_pos[seam_edges_weld[:, 1]]
    seg_vec = seg_p1 - seg_p0
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid_seg = seg_len > 1e-12
    seg_p0 = seg_p0[valid_seg]
    seg_p1 = seg_p1[valid_seg]
    seg_len = seg_len[valid_seg]
    seg_dir = seg_vec[valid_seg] / seg_len[:, None]

    return HighSeamData(
        weld_positions=weld_pos,
        seam_edges_weld=seam_edges_weld.astype(np.int64, copy=False),
        seam_chains=seam_chains,
        seam_chain_closed=chains_closed,
        seam_segments_p0=seg_p0.astype(np.float64, copy=False),
        seam_segments_p1=seg_p1.astype(np.float64, copy=False),
        seam_segments_dir=seg_dir.astype(np.float64, copy=False),
        seam_segments_len=seg_len.astype(np.float64, copy=False),
        meta={
            "high_seam_edges": int(seam_edges_weld.shape[0]),
            "high_boundary_edges": int(boundary_edges),
            "high_nonmanifold_edges": int(nonmanifold_edges),
            "high_interior_seam_edges": int(seam_interior),
            "high_seam_chain_count": int(len(seam_chains)),
        },
    )


def _build_low_graph(mesh: trimesh.Trimesh) -> LowGraphData:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_vertices = int(verts.shape[0])
    if faces.size == 0:
        return LowGraphData(
            edge_vertices=np.zeros((0, 2), dtype=np.int64),
            edge_lengths=np.zeros((0,), dtype=np.float64),
            edge_midpoints=np.zeros((0, 3), dtype=np.float64),
            edge_dirs_unit=np.zeros((0, 3), dtype=np.float64),
            edge_flatness=np.zeros((0,), dtype=np.float64),
            adjacency=[[] for _ in range(n_vertices)],
            edge_key_to_id={},
        )

    edge_dir = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    edge_key = np.sort(edge_dir, axis=1)
    uniq, inv = np.unique(edge_key, axis=0, return_inverse=True)
    edge_vertices = uniq.astype(np.int64, copy=False)
    n_edges = int(edge_vertices.shape[0])

    p0 = verts[edge_vertices[:, 0]]
    p1 = verts[edge_vertices[:, 1]]
    vec = p1 - p0
    edge_len = np.linalg.norm(vec, axis=1)
    edge_dir_unit = np.zeros_like(vec)
    ok = edge_len > 1e-12
    edge_dir_unit[ok] = vec[ok] / edge_len[ok, None]
    edge_mid = 0.5 * (p0 + p1)

    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]
    edge_key_to_id: Dict[Tuple[int, int], int] = {}
    for eid, (a, b) in enumerate(edge_vertices.tolist()):
        ia = int(a)
        ib = int(b)
        adjacency[ia].append((ib, eid))
        adjacency[ib].append((ia, eid))
        edge_key_to_id[_edge_key(ia, ib)] = eid
    for vi in range(n_vertices):
        adjacency[vi].sort(key=lambda x: (x[0], x[1]))

    # Optional geometric preference: flatter edges get higher penalty.
    flatness = np.zeros((n_edges,), dtype=np.float64)
    try:
        adj_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        adj_angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
        if adj_edges.ndim == 2 and adj_edges.shape[1] == 2 and adj_angles.shape[0] == adj_edges.shape[0]:
            for (ea, eb), ang in zip(adj_edges.tolist(), adj_angles.tolist()):
                eid = edge_key_to_id.get(_edge_key(int(ea), int(eb)))
                if eid is None:
                    continue
                angle = float(max(0.0, min(np.pi, ang)))
                flatness[eid] = 1.0 - min(angle / (0.5 * np.pi), 1.0)
    except Exception:
        flatness[:] = 0.0

    return LowGraphData(
        edge_vertices=edge_vertices,
        edge_lengths=edge_len.astype(np.float64, copy=False),
        edge_midpoints=edge_mid.astype(np.float64, copy=False),
        edge_dirs_unit=edge_dir_unit.astype(np.float64, copy=False),
        edge_flatness=flatness.astype(np.float64, copy=False),
        adjacency=adjacency,
        edge_key_to_id=edge_key_to_id,
    )


def _point_to_segment_distance(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    v = p1 - p0
    w = points - p0
    vv = np.einsum("ij,ij->i", v, v, optimize=True)
    t = np.zeros((points.shape[0],), dtype=np.float64)
    ok = vv > 1e-18
    if np.any(ok):
        t[ok] = np.einsum("ij,ij->i", w[ok], v[ok], optimize=True) / vv[ok]
    t = np.clip(t, 0.0, 1.0)
    c = p0 + t[:, None] * v
    d = points - c
    return np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d, optimize=True), 0.0))


def _compute_edge_attraction_cost(
    *,
    low_graph: LowGraphData,
    seg_p0: np.ndarray,
    seg_p1: np.ndarray,
    seg_dir: np.ndarray,
    mean_len: float,
    dist_weight: float,
    align_weight: float,
    knn: int,
) -> np.ndarray:
    n_edges = int(low_graph.edge_vertices.shape[0])
    if n_edges == 0:
        return np.zeros((0,), dtype=np.float64)

    if seg_p0.shape[0] == 0:
        return np.zeros((n_edges,), dtype=np.float64)

    seg_mid = 0.5 * (seg_p0 + seg_p1)
    tree = cKDTree(seg_mid)
    k = min(max(1, int(knn)), seg_mid.shape[0])
    _, idx = tree.query(low_graph.edge_midpoints, k=k)
    if idx.ndim == 1:
        idx = idx[:, None]

    best = np.full((n_edges,), np.inf, dtype=np.float64)
    for j in range(idx.shape[1]):
        ids = idx[:, j].astype(np.int64, copy=False)
        sp0 = seg_p0[ids]
        sp1 = seg_p1[ids]
        sdir = seg_dir[ids]
        dist = _point_to_segment_distance(low_graph.edge_midpoints, sp0, sp1)
        dist_norm = dist / max(mean_len, 1e-12)
        align = 1.0 - np.abs(np.einsum("ij,ij->i", low_graph.edge_dirs_unit, sdir, optimize=True))
        c = float(dist_weight) * dist_norm + float(align_weight) * align
        best = np.minimum(best, c)

    best[~np.isfinite(best)] = 0.0
    return np.maximum(best, 0.0)


def _compute_low_edge_costs(
    *,
    low_graph: LowGraphData,
    high_seam: HighSeamData,
    seam_cfg: Dict[str, Any],
) -> np.ndarray:
    n_edges = int(low_graph.edge_vertices.shape[0])
    if n_edges == 0:
        return np.zeros((0,), dtype=np.float64)

    edge_len = low_graph.edge_lengths
    mean_len = float(np.mean(edge_len[edge_len > 0])) if np.any(edge_len > 0) else 1.0
    mean_len = max(mean_len, 1e-8)
    len_norm = edge_len / mean_len

    w1 = float(seam_cfg.get("routing_weight_dist", 3.0))
    w2 = float(seam_cfg.get("routing_weight_align", 1.5))
    w3 = float(seam_cfg.get("routing_weight_length", 1.0))
    w4 = float(seam_cfg.get("routing_weight_dihedral", 0.25))
    knn = max(1, int(seam_cfg.get("routing_knn_segments", 8)))

    attraction = _compute_edge_attraction_cost(
        low_graph=low_graph,
        seg_p0=high_seam.seam_segments_p0,
        seg_p1=high_seam.seam_segments_p1,
        seg_dir=high_seam.seam_segments_dir,
        mean_len=mean_len,
        dist_weight=w1,
        align_weight=w2,
        knn=knn,
    )
    cost = attraction + w3 * len_norm + w4 * np.maximum(low_graph.edge_flatness, 0.0)
    return np.maximum(cost, 1e-8)


def _sample_polyline(points: np.ndarray, n_samples: int, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 1:
        return pts.copy()
    if closed and np.linalg.norm(pts[0] - pts[-1]) <= 1e-10:
        pts = pts[:-1]
    if pts.shape[0] <= 1:
        return pts.copy()

    if closed:
        nxt = np.vstack([pts[1:], pts[:1]])
        seg = nxt - pts
    else:
        seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-12
    if not np.any(keep):
        return np.repeat(pts[:1], max(1, n_samples), axis=0)

    if closed:
        p0 = pts
        p1 = nxt
    else:
        p0 = pts[:-1]
        p1 = pts[1:]
    p0 = p0[keep]
    p1 = p1[keep]
    seg_len = seg_len[keep]
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], max(1, n_samples), axis=0)

    n = max(1, int(n_samples))
    if closed:
        ts = np.linspace(0.0, total, n + 1, dtype=np.float64)[:-1]
    else:
        ts = np.linspace(0.0, total, n, dtype=np.float64)

    seg_idx = np.searchsorted(cum, ts, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_len) - 1)
    local_t = (ts - cum[seg_idx]) / np.maximum(seg_len[seg_idx], 1e-12)
    out = p0[seg_idx] * (1.0 - local_t[:, None]) + p1[seg_idx] * local_t[:, None]
    return out


def _polyline_segments(points: np.ndarray, closed: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    if closed:
        if np.linalg.norm(pts[0] - pts[-1]) <= 1e-10:
            pts = pts[:-1]
        if pts.shape[0] < 2:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
            )
        p0 = pts
        p1 = np.vstack([pts[1:], pts[:1]])
    else:
        p0 = pts[:-1]
        p1 = pts[1:]

    vec = p1 - p0
    seg_len = np.linalg.norm(vec, axis=1)
    keep = seg_len > 1e-12
    if not np.any(keep):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    p0 = p0[keep]
    p1 = p1[keep]
    seg_dir = vec[keep] / seg_len[keep, None]
    return p0.astype(np.float64, copy=False), p1.astype(np.float64, copy=False), seg_dir.astype(np.float64, copy=False)


def _dijkstra_path(
    *,
    adjacency: Sequence[Sequence[Tuple[int, int]]],
    edge_cost: np.ndarray,
    src: int,
    dst: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(adjacency)
    if src < 0 or src >= n or dst < 0 or dst >= n:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    if src == dst:
        return np.asarray([src], dtype=np.int64), np.zeros((0,), dtype=np.int64)

    dist = np.full((n,), np.inf, dtype=np.float64)
    prev_v = np.full((n,), -1, dtype=np.int64)
    prev_e = np.full((n,), -1, dtype=np.int64)
    dist[src] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, int(src))]
    visited = np.zeros((n,), dtype=np.bool_)

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, eid in adjacency[u]:
            nd = d + float(edge_cost[eid])
            if nd + 1e-15 < dist[v]:
                dist[v] = nd
                prev_v[v] = u
                prev_e[v] = eid
                heapq.heappush(pq, (nd, int(v)))

    if not np.isfinite(dist[dst]):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    path_v: List[int] = [int(dst)]
    path_e: List[int] = []
    cur = int(dst)
    while cur != src:
        pe = int(prev_e[cur])
        pv = int(prev_v[cur])
        if pe < 0 or pv < 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        path_e.append(pe)
        path_v.append(pv)
        cur = pv
    path_v.reverse()
    path_e.reverse()
    return np.asarray(path_v, dtype=np.int64), np.asarray(path_e, dtype=np.int64)


def _route_cut_edges(
    *,
    low_mesh: trimesh.Trimesh,
    low_graph: LowGraphData,
    high_seam: HighSeamData,
    seam_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if low_graph.edge_vertices.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), {
            "uv_route_chain_count": 0,
            "uv_route_anchor_total": 0,
            "uv_route_segment_total": 0,
            "uv_route_segment_failed": 0,
            "uv_route_cut_edges": 0,
        }

    base_edge_cost = _compute_low_edge_costs(low_graph=low_graph, high_seam=high_seam, seam_cfg=seam_cfg)
    low_vertices = np.asarray(low_mesh.vertices, dtype=np.float64)
    vtree = cKDTree(low_vertices) if low_vertices.shape[0] > 0 else None

    mean_low_len = float(np.mean(low_graph.edge_lengths[low_graph.edge_lengths > 0])) if np.any(low_graph.edge_lengths > 0) else 1.0
    mean_low_len = max(mean_low_len, 1e-8)
    spacing_ratio = max(0.5, float(seam_cfg.get("routing_anchor_spacing_ratio", 8.0)))
    spacing = spacing_ratio * mean_low_len
    min_anchors = max(2, int(seam_cfg.get("routing_min_anchors", 8)))
    max_anchors = max(min_anchors, int(seam_cfg.get("routing_max_anchors", 128)))
    local_scale = float(seam_cfg.get("routing_chain_local_scale", 1.5))
    local_w1 = float(seam_cfg.get("routing_chain_weight_dist", seam_cfg.get("routing_weight_dist", 3.0)))
    local_w2 = float(seam_cfg.get("routing_chain_weight_align", seam_cfg.get("routing_weight_align", 1.5)))
    local_knn = max(1, int(seam_cfg.get("routing_chain_knn_segments", seam_cfg.get("routing_knn_segments", 8))))
    reuse_penalty = float(seam_cfg.get("routing_edge_reuse_penalty", 2.0))
    reuse_power = float(seam_cfg.get("routing_edge_reuse_power", 1.0))

    routed_edge_ids: set[int] = set()
    edge_usage = np.zeros((low_graph.edge_vertices.shape[0],), dtype=np.int32)
    anchor_total = 0
    seg_total = 0
    seg_failed = 0

    for chain_pts, chain_closed in zip(high_seam.seam_chains, high_seam.seam_chain_closed):
        pts = np.asarray(chain_pts, dtype=np.float64)
        if pts.shape[0] < 2 or vtree is None:
            continue
        cp0, cp1, cdir = _polyline_segments(pts, bool(chain_closed))
        chain_edge_cost = base_edge_cost
        if cp0.shape[0] > 0 and local_scale > 0.0:
            local_attr = _compute_edge_attraction_cost(
                low_graph=low_graph,
                seg_p0=cp0,
                seg_p1=cp1,
                seg_dir=cdir,
                mean_len=mean_low_len,
                dist_weight=local_w1,
                align_weight=local_w2,
                knn=local_knn,
            )
            chain_edge_cost = base_edge_cost + local_scale * local_attr

        seg = pts[1:] - pts[:-1]
        chain_len = float(np.sum(np.linalg.norm(seg, axis=1)))
        if chain_closed and pts.shape[0] > 2 and np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
            chain_len += float(np.linalg.norm(pts[0] - pts[-1]))
        target_n = int(np.ceil(chain_len / max(spacing, 1e-8)))
        target_n = max(min_anchors, min(max_anchors, target_n))
        if chain_closed:
            target_n = max(3, target_n)

        anchor_points = _sample_polyline(pts, target_n, chain_closed)
        if anchor_points.shape[0] < 2:
            continue
        anchor_ids = vtree.query(anchor_points, k=1)[1].astype(np.int64, copy=False)
        dedup: List[int] = []
        for vid in anchor_ids.tolist():
            if not dedup or vid != dedup[-1]:
                dedup.append(int(vid))
        if chain_closed and len(dedup) > 1 and dedup[0] == dedup[-1]:
            dedup.pop()
        if len(dedup) < 2:
            continue

        anchor_total += len(dedup)
        pairs: List[Tuple[int, int]] = [(dedup[i], dedup[i + 1]) for i in range(len(dedup) - 1)]
        if chain_closed:
            pairs.append((dedup[-1], dedup[0]))
        seg_total += len(pairs)

        for s, t in pairs:
            if reuse_penalty > 0.0:
                dyn_cost = chain_edge_cost + reuse_penalty * np.power(edge_usage.astype(np.float64), reuse_power)
            else:
                dyn_cost = chain_edge_cost
            _, path_e = _dijkstra_path(
                adjacency=low_graph.adjacency,
                edge_cost=dyn_cost,
                src=int(s),
                dst=int(t),
            )
            if path_e.size == 0 and s != t:
                seg_failed += 1
                continue
            routed_edge_ids.update(path_e.tolist())
            edge_usage[path_e] += 1

    reused_edge_count = int(np.count_nonzero(edge_usage > 1))
    max_usage = int(np.max(edge_usage)) if edge_usage.size > 0 else 0
    mean_usage = float(np.mean(edge_usage[edge_usage > 0])) if np.any(edge_usage > 0) else 0.0
    if len(routed_edge_ids) == 0:
        return np.zeros((0, 2), dtype=np.int64), {
            "uv_route_chain_count": int(len(high_seam.seam_chains)),
            "uv_route_anchor_total": int(anchor_total),
            "uv_route_segment_total": int(seg_total),
            "uv_route_segment_failed": int(seg_failed),
            "uv_route_cut_edges": 0,
            "uv_route_reused_edges": 0,
            "uv_route_reuse_ratio": 0.0,
            "uv_route_max_edge_usage": int(max_usage),
            "uv_route_mean_edge_usage": float(mean_usage),
        }

    routed = np.asarray(sorted(routed_edge_ids), dtype=np.int64)
    cut_edges = low_graph.edge_vertices[routed]
    return cut_edges.astype(np.int64, copy=False), {
        "uv_route_chain_count": int(len(high_seam.seam_chains)),
        "uv_route_anchor_total": int(anchor_total),
        "uv_route_segment_total": int(seg_total),
        "uv_route_segment_failed": int(seg_failed),
        "uv_route_cut_edges": int(cut_edges.shape[0]),
        "uv_route_reused_edges": int(reused_edge_count),
        "uv_route_reuse_ratio": float(reused_edge_count / max(1, int(cut_edges.shape[0]))),
        "uv_route_max_edge_usage": int(max_usage),
        "uv_route_mean_edge_usage": float(mean_usage),
    }


def _flood_fill_face_islands_from_cut_edges(faces: np.ndarray, cut_edges: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    n_faces = int(tri.shape[0])
    if n_faces == 0:
        return np.zeros((0,), dtype=np.int64)
    hem = build_halfedge_mesh(tri)

    cut = np.asarray(cut_edges, dtype=np.int64)
    cut_set: set[Tuple[int, int]] = set()
    if cut.size > 0:
        cut = np.sort(cut, axis=1)
        for a, b in cut.tolist():
            cut_set.add((int(a), int(b)))

    face_label = np.full((n_faces,), -1, dtype=np.int64)
    face_half = np.arange(hem.n_halfedges, dtype=np.int64).reshape(n_faces, 3)
    next_h = hem.he_next
    twin = hem.he_twin
    origin = hem.he_origin
    dest = hem.he_dest
    he_face = hem.he_face

    cur_label = 0
    for f0 in range(n_faces):
        if face_label[f0] >= 0:
            continue
        stack = [int(f0)]
        face_label[f0] = cur_label
        while stack:
            f = stack.pop()
            hs = face_half[f]
            for h0 in hs.tolist():
                h = int(h0)
                ek = _edge_key(int(origin[h]), int(dest[h]))
                if ek in cut_set:
                    continue
                tw = int(twin[h])
                if tw < 0:
                    continue
                fn = int(he_face[tw])
                if face_label[fn] >= 0:
                    continue
                face_label[fn] = cur_label
                stack.append(fn)
        cur_label += 1
    return face_label


def _map_low_island_to_high_island(
    *,
    low_face_island: np.ndarray,
    sample_face_ids: np.ndarray,
    target_face_ids: np.ndarray,
    valid_mask: np.ndarray,
    high_face_island: np.ndarray,
    conflict_confidence_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n_faces = int(low_face_island.shape[0])
    low_expected = np.full((n_faces,), -1, dtype=np.int64)
    low_conflict = np.zeros((n_faces,), dtype=np.bool_)
    low_conf = np.zeros((n_faces,), dtype=np.float32)

    valid = (
        np.asarray(valid_mask, dtype=np.bool_)
        & (sample_face_ids >= 0)
        & (sample_face_ids < n_faces)
        & (target_face_ids >= 0)
        & (target_face_ids < len(high_face_island))
    )
    if not np.any(valid):
        return low_expected, low_conflict, low_conf, {
            "uv_low_island_count": int(np.unique(low_face_island[low_face_island >= 0]).size),
            "uv_low_island_mapped_count": 0,
            "uv_low_island_conflict_count": 0,
            "uv_island_unknown_faces": int(np.count_nonzero(low_expected < 0)),
        }

    sf = np.asarray(sample_face_ids, dtype=np.int64)[valid]
    tf = np.asarray(target_face_ids, dtype=np.int64)[valid]
    low_is = low_face_island[sf]
    high_is = np.asarray(high_face_island, dtype=np.int64)[tf]
    pair_valid = (low_is >= 0) & (high_is >= 0)
    if not np.any(pair_valid):
        return low_expected, low_conflict, low_conf, {
            "uv_low_island_count": int(np.unique(low_face_island[low_face_island >= 0]).size),
            "uv_low_island_mapped_count": 0,
            "uv_low_island_conflict_count": 0,
            "uv_island_unknown_faces": int(np.count_nonzero(low_expected < 0)),
        }

    low_is = low_is[pair_valid]
    high_is = high_is[pair_valid]
    pair = np.stack([low_is, high_is], axis=1)
    uniq_pair, pair_cnt = np.unique(pair, axis=0, return_counts=True)
    low_ids = uniq_pair[:, 0].astype(np.int64, copy=False)
    high_ids = uniq_pair[:, 1].astype(np.int64, copy=False)
    pair_cnt = pair_cnt.astype(np.int64, copy=False)

    order = np.lexsort((high_ids, -pair_cnt, low_ids))
    low_ids = low_ids[order]
    high_ids = high_ids[order]
    pair_cnt = pair_cnt[order]

    uniq_low, first_idx = np.unique(low_ids, return_index=True)
    total_by_low = np.bincount(low_ids, weights=pair_cnt, minlength=int(np.max(low_ids)) + 1).astype(np.float64)
    map_low_to_high: Dict[int, int] = {}
    conf_by_low: Dict[int, float] = {}
    conflict_low: Dict[int, bool] = {}

    for li, start in zip(uniq_low.tolist(), first_idx.tolist()):
        li_i = int(li)
        hi = int(high_ids[start])
        cnt = float(pair_cnt[start])
        total = float(total_by_low[li_i]) if li_i < total_by_low.shape[0] else cnt
        conf = float(cnt / max(total, 1.0))
        map_low_to_high[li_i] = hi
        conf_by_low[li_i] = conf
        conflict_low[li_i] = bool(conf < float(conflict_confidence_min))

    for f in range(n_faces):
        li = int(low_face_island[f])
        if li < 0:
            continue
        hi = map_low_to_high.get(li, -1)
        low_expected[f] = int(hi)
        low_conf[f] = float(conf_by_low.get(li, 0.0))
        low_conflict[f] = bool(conflict_low.get(li, False))

    return low_expected, low_conflict, low_conf, {
        "uv_low_island_count": int(np.unique(low_face_island[low_face_island >= 0]).size),
        "uv_low_island_mapped_count": int(len(map_low_to_high)),
        "uv_low_island_conflict_count": int(sum(1 for v in conflict_low.values() if v)),
        "uv_island_unknown_faces": int(np.count_nonzero(low_expected < 0)),
    }


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
    high_seam = _extract_high_seam_edges_and_chains(
        high_mesh=high_mesh,
        high_uv=high_uv,
        position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
        uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
    )
    low_graph = _build_low_graph(low_mesh)
    cut_edges, route_meta = _route_cut_edges(
        low_mesh=low_mesh,
        low_graph=low_graph,
        high_seam=high_seam,
        seam_cfg=seam_cfg,
    )

    low_face_island = _flood_fill_face_islands_from_cut_edges(np.asarray(low_mesh.faces, dtype=np.int64), cut_edges)
    low_expected, low_conflict, low_conf, map_meta = _map_low_island_to_high_island(
        low_face_island=low_face_island,
        sample_face_ids=np.asarray(sample_face_ids, dtype=np.int64),
        target_face_ids=np.asarray(target_face_ids, dtype=np.int64),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        high_face_island=np.asarray(high_face_island, dtype=np.int64),
        conflict_confidence_min=float(seam_cfg.get("routing_island_confidence_min", 0.55)),
    )

    split_vertices, split_faces, split_meta = split_vertices_along_cut_edges(
        vertices=np.asarray(low_mesh.vertices, dtype=np.float32),
        faces=np.asarray(low_mesh.faces, dtype=np.int64),
        cut_edges=np.asarray(cut_edges, dtype=np.int64),
    )

    meta: Dict[str, Any] = {}
    meta.update(high_seam.meta)
    meta.update(route_meta)
    meta.update(map_meta)
    meta["uv_low_cut_edges"] = int(cut_edges.shape[0])
    meta["uv_low_split_vertices"] = int(split_meta.get("split_vertices_added", 0))
    meta["uv_low_split_faces"] = int(split_meta.get("split_faces", int(len(low_mesh.faces))))
    meta["uv_low_island_count"] = int(np.unique(low_face_island[low_face_island >= 0]).size)

    return RoutedSeamResult(
        cut_edges=np.asarray(cut_edges, dtype=np.int64),
        low_face_island=np.asarray(low_face_island, dtype=np.int64),
        low_face_expected_high_island=np.asarray(low_expected, dtype=np.int64),
        low_face_conflict=np.asarray(low_conflict, dtype=np.bool_),
        low_face_confidence=np.asarray(low_conf, dtype=np.float32),
        split_vertices=np.asarray(split_vertices, dtype=np.float32),
        split_faces=np.asarray(split_faces, dtype=np.int64),
        split_meta=dict(split_meta),
        route_meta=meta,
    )


__all__ = [
    "RoutedSeamResult",
    "route_low_mesh_seams_by_dijkstra",
]
