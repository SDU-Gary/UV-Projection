from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class HalfEdgeMesh:
    faces: np.ndarray
    n_vertices: int
    he_origin: np.ndarray
    he_dest: np.ndarray
    he_face: np.ndarray
    he_corner: np.ndarray
    he_next: np.ndarray
    he_twin: np.ndarray

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_halfedges(self) -> int:
        return int(self.he_origin.shape[0])


def build_halfedge_mesh(faces: np.ndarray, n_vertices: int | None = None) -> HalfEdgeMesh:
    faces_i64 = np.asarray(faces, dtype=np.int64)
    if faces_i64.ndim != 2 or faces_i64.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")

    f_count = int(faces_i64.shape[0])
    if n_vertices is None:
        n_vertices = int(faces_i64.max()) + 1 if f_count > 0 else 0
    n_vertices = int(n_vertices)

    h_count = f_count * 3
    he_face = np.repeat(np.arange(f_count, dtype=np.int64), 3)
    he_corner = np.tile(np.arange(3, dtype=np.int64), f_count)
    he_origin = faces_i64.reshape(-1)
    he_dest = faces_i64[:, [1, 2, 0]].reshape(-1)
    he_next = (np.arange(h_count, dtype=np.int64).reshape(f_count, 3)[:, [1, 2, 0]]).reshape(-1)
    he_twin = np.full(h_count, -1, dtype=np.int64)

    directed: Dict[Tuple[int, int], List[int]] = {}
    for hid in range(h_count):
        key = (int(he_origin[hid]), int(he_dest[hid]))
        directed.setdefault(key, []).append(hid)

    visited: set[Tuple[int, int]] = set()
    for key, hids in directed.items():
        if key in visited:
            continue
        rev = (key[1], key[0])
        visited.add(key)
        visited.add(rev)
        rev_hids = directed.get(rev, [])
        if not rev_hids:
            continue
        pair_count = min(len(hids), len(rev_hids))
        for idx in range(pair_count):
            h0 = int(hids[idx])
            h1 = int(rev_hids[idx])
            he_twin[h0] = h1
            he_twin[h1] = h0

    return HalfEdgeMesh(
        faces=faces_i64,
        n_vertices=n_vertices,
        he_origin=he_origin,
        he_dest=he_dest,
        he_face=he_face,
        he_corner=he_corner,
        he_next=he_next,
        he_twin=he_twin,
    )


def compute_high_face_uv_islands(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    position_eps: float = 1e-6,
    uv_eps: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, int]]:
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    uv_v = np.asarray(uv, dtype=np.float64)

    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if uv_v.ndim != 2 or uv_v.shape[1] != 2:
        raise ValueError("uv must have shape [V,2]")
    if verts.shape[0] != uv_v.shape[0]:
        raise ValueError("vertices/uv length mismatch")

    f_count = int(tri.shape[0])
    if f_count == 0:
        return np.zeros((0,), dtype=np.int64), {
            "high_island_count": 0,
            "high_seam_edges": 0,
            "high_boundary_edges": 0,
            "high_nonmanifold_edges": 0,
        }

    pos_eps = max(float(position_eps), 1e-12)
    q = np.round(verts / pos_eps).astype(np.int64)
    _, weld = np.unique(q, axis=0, return_inverse=True)

    v0 = tri[:, [0, 1, 2]].reshape(-1)
    v1 = tri[:, [1, 2, 0]].reshape(-1)
    fidx = np.repeat(np.arange(f_count, dtype=np.int64), 3)

    g0 = weld[v0]
    g1 = weld[v1]
    e0 = np.minimum(g0, g1)
    e1 = np.maximum(g0, g1)

    uv0 = uv_v[v0]
    uv1 = uv_v[v1]
    swap = g0 > g1
    uv_a = np.where(swap[:, None], uv1, uv0)
    uv_b = np.where(swap[:, None], uv0, uv1)

    # Group all directed edge records by welded-undirected edge key.
    edge_key = np.stack([e0, e1], axis=1).astype(np.int64, copy=False)
    _, inv, counts = np.unique(edge_key, axis=0, return_inverse=True, return_counts=True)
    counts = counts.astype(np.int64, copy=False)

    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))

    adj_row_np = np.zeros((0,), dtype=np.int64)
    adj_col_np = np.zeros((0,), dtype=np.int64)

    interior_mask = counts == 2
    interior_n = int(np.count_nonzero(interior_mask))
    if interior_n > 0:
        edge_ids = np.arange(edge_key.shape[0], dtype=np.int64)
        interior_edge_ids = edge_ids[interior_mask[inv]]
        interior_group_ids = inv[interior_edge_ids]

        order2 = np.argsort(interior_group_ids, kind="mergesort")
        pair_idx = interior_edge_ids[order2]
        if pair_idx.size % 2 != 0:
            pair_idx = pair_idx[: pair_idx.size - 1]

        i0 = pair_idx[0::2]
        i1 = pair_idx[1::2]
        f0 = fidx[i0]
        f1 = fidx[i1]

        uv_da = uv_a[i0] - uv_a[i1]
        uv_db = uv_b[i0] - uv_b[i1]
        uv_eps2 = float(max(uv_eps, 0.0) ** 2)
        same_a = np.einsum("ij,ij->i", uv_da, uv_da, optimize=True) <= uv_eps2
        same_b = np.einsum("ij,ij->i", uv_db, uv_db, optimize=True) <= uv_eps2

        can_link = (f0 != f1) & same_a & same_b
        if np.any(can_link):
            rf0 = f0[can_link].astype(np.int64, copy=False)
            rf1 = f1[can_link].astype(np.int64, copy=False)
            adj_row_np = np.concatenate([rf0, rf1]).astype(np.int64, copy=False)
            adj_col_np = np.concatenate([rf1, rf0]).astype(np.int64, copy=False)

        seam_interior = int(interior_n - np.count_nonzero(can_link))
    else:
        seam_interior = 0

    seam_edges = int(boundary_edges + nonmanifold_edges + seam_interior)

    if adj_row_np.size > 0:
        data = np.ones(adj_row_np.size, dtype=np.int8)
        graph = coo_matrix((data, (adj_row_np, adj_col_np)), shape=(f_count, f_count)).tocsr()
    else:
        graph = coo_matrix((f_count, f_count), dtype=np.int8).tocsr()

    n_comp, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), {
        "high_island_count": int(n_comp),
        "high_seam_edges": int(seam_edges),
        "high_boundary_edges": int(boundary_edges),
        "high_nonmanifold_edges": int(nonmanifold_edges),
    }


def detect_cut_edges_from_face_labels(faces: np.ndarray, face_labels: np.ndarray) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    hem = build_halfedge_mesh(tri)

    hid = np.arange(hem.n_halfedges, dtype=np.int64)
    twin = hem.he_twin
    interior = (twin >= 0) & (hid < twin)
    if not np.any(interior):
        return np.zeros((0, 2), dtype=np.int64)

    h0 = hid[interior]
    h1 = twin[interior]
    f0 = hem.he_face[h0]
    f1 = hem.he_face[h1]
    l0 = labels[f0]
    l1 = labels[f1]
    cut = (l0 >= 0) & (l1 >= 0) & (l0 != l1)
    if not np.any(cut):
        return np.zeros((0, 2), dtype=np.int64)

    a = np.minimum(hem.he_origin[h0[cut]], hem.he_dest[h0[cut]])
    b = np.maximum(hem.he_origin[h0[cut]], hem.he_dest[h0[cut]])
    return np.unique(np.stack([a, b], axis=1).astype(np.int64), axis=0)


def split_vertices_along_cut_edges(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    cut_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    verts = np.asarray(vertices, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int64)
    n_vertices = int(verts.shape[0])
    f_count = int(tri.shape[0])

    cut_e = np.asarray(cut_edges, dtype=np.int64)
    if cut_e.size == 0:
        return verts.copy(), tri.copy(), {
            "cut_edges": 0,
            "split_vertices_added": 0,
            "split_faces": int(f_count),
        }
    cut_e = np.unique(np.sort(cut_e, axis=1), axis=0)
    cut_set = {tuple(map(int, e)) for e in cut_e}

    hem = build_halfedge_mesh(tri, n_vertices=n_vertices)

    incident: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]
    for fid in range(f_count):
        for c in range(3):
            v = int(tri[fid, c])
            incident[v].append((fid, c))

    v_pairs: List[List[Tuple[int, int]]] = [[] for _ in range(n_vertices)]
    for hid in range(hem.n_halfedges):
        t = int(hem.he_twin[hid])
        if t < 0:
            continue
        v = int(hem.he_origin[hid])
        u = int(hem.he_dest[hid])
        ek = (v, u) if v < u else (u, v)
        if ek in cut_set:
            continue
        f0 = int(hem.he_face[hid])
        f1 = int(hem.he_face[t])
        if f0 != f1:
            v_pairs[v].append((f0, f1))

    faces_out = tri.copy()
    vertices_out: List[np.ndarray] = [verts[i].copy() for i in range(n_vertices)]
    split_added = 0

    for v in range(n_vertices):
        corners = incident[v]
        if len(corners) <= 1:
            continue

        face_ids = np.asarray([fc[0] for fc in corners], dtype=np.int64)
        uniq_faces = np.unique(face_ids)
        if uniq_faces.size <= 1:
            continue

        local_index = {int(fid): idx for idx, fid in enumerate(uniq_faces.tolist())}
        parent = np.arange(uniq_faces.size, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for f0, f1 in v_pairs[v]:
            i0 = local_index.get(int(f0))
            i1 = local_index.get(int(f1))
            if i0 is not None and i1 is not None:
                union(int(i0), int(i1))

        roots = np.asarray([find(i) for i in range(uniq_faces.size)], dtype=np.int64)

        root_order: List[int] = []
        for fid in face_ids.tolist():
            rid = int(roots[local_index[int(fid)]])
            if rid not in root_order:
                root_order.append(rid)
        if len(root_order) <= 1:
            continue

        root_to_vid: Dict[int, int] = {int(root_order[0]): v}
        for rid in root_order[1:]:
            new_vid = len(vertices_out)
            vertices_out.append(verts[v].copy())
            root_to_vid[int(rid)] = new_vid
            split_added += 1

        for fid, corner in corners:
            rid = int(roots[local_index[int(fid)]])
            faces_out[int(fid), int(corner)] = int(root_to_vid[rid])

    verts_out_np = np.asarray(vertices_out, dtype=np.float32)
    return verts_out_np, faces_out.astype(np.int64), {
        "cut_edges": int(cut_e.shape[0]),
        "split_vertices_added": int(split_added),
        "split_faces": int(faces_out.shape[0]),
    }
