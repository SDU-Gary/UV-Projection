from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import trimesh


@dataclass
class SeamExtractionResult:
    seam_edges: np.ndarray
    seam_loops: List[np.ndarray]
    meta: Dict[str, Any]


def _build_openmesh(vertices: np.ndarray, faces: np.ndarray):
    import openmesh as om

    # Prefer direct ndarray constructor; if this fails we fall back to numpy halfedge logic.
    return om.TriMesh(vertices.astype(np.float64, copy=False), faces.astype(np.int32, copy=False))


def _extract_seam_edges_numpy(
    *,
    faces: np.ndarray,
    labels: np.ndarray,
    include_boundary_as_seam: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    f_count = int(faces.shape[0])
    for fid in range(f_count):
        tri = faces[fid]
        e01 = (int(min(tri[0], tri[1])), int(max(tri[0], tri[1])))
        e12 = (int(min(tri[1], tri[2])), int(max(tri[1], tri[2])))
        e20 = (int(min(tri[2], tri[0])), int(max(tri[2], tri[0])))
        edge_faces[e01].append(fid)
        edge_faces[e12].append(fid)
        edge_faces[e20].append(fid)

    seam_set: Set[Tuple[int, int]] = set()
    boundary_edges = 0
    nonmanifold_edges = 0
    semantic_edges = 0
    for ek, flist in edge_faces.items():
        if len(flist) == 1:
            boundary_edges += 1
            if include_boundary_as_seam:
                seam_set.add(ek)
            continue
        if len(flist) > 2:
            nonmanifold_edges += 1
        f0 = int(flist[0])
        f1 = int(flist[1])
        if (
            0 <= f0 < labels.shape[0]
            and 0 <= f1 < labels.shape[0]
            and int(labels[f0]) >= 0
            and int(labels[f1]) >= 0
            and int(labels[f0]) != int(labels[f1])
        ):
            seam_set.add(ek)
            semantic_edges += 1

    seam_edges = (
        np.asarray(sorted(seam_set), dtype=np.int64) if len(seam_set) > 0 else np.zeros((0, 2), dtype=np.int64)
    )
    meta = {
        "uv_seam_edges_semantic": int(semantic_edges),
        "uv_seam_edges_boundary": int(boundary_edges),
        "uv_seam_nonmanifold_edges": int(nonmanifold_edges),
    }
    return seam_edges, meta


def _extract_seam_edges_openmesh_impl(
    *,
    low_mesh: trimesh.Trimesh,
    labels: np.ndarray,
    include_boundary_as_seam: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    vertices = np.asarray(low_mesh.vertices, dtype=np.float64)
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    mesh = _build_openmesh(vertices, faces)

    seam_set: Set[Tuple[int, int]] = set()
    boundary_edges = 0
    semantic_edges = 0
    for eh in mesh.edges():
        he0 = mesh.halfedge_handle(eh, 0)
        v0 = int(mesh.from_vertex_handle(he0).idx())
        v1 = int(mesh.to_vertex_handle(he0).idx())
        ek = (v0, v1) if v0 < v1 else (v1, v0)

        if mesh.is_boundary(eh):
            boundary_edges += 1
            if include_boundary_as_seam:
                seam_set.add(ek)
            continue

        he1 = mesh.halfedge_handle(eh, 1)
        fh0 = mesh.face_handle(he0)
        fh1 = mesh.face_handle(he1)
        if not fh0.is_valid() or not fh1.is_valid():
            continue
        f0 = int(fh0.idx())
        f1 = int(fh1.idx())
        if (
            0 <= f0 < labels.shape[0]
            and 0 <= f1 < labels.shape[0]
            and int(labels[f0]) >= 0
            and int(labels[f1]) >= 0
            and int(labels[f0]) != int(labels[f1])
        ):
            seam_set.add(ek)
            semantic_edges += 1

    seam_edges = (
        np.asarray(sorted(seam_set), dtype=np.int64) if len(seam_set) > 0 else np.zeros((0, 2), dtype=np.int64)
    )
    meta = {
        "uv_seam_edges_semantic": int(semantic_edges),
        "uv_seam_edges_boundary": int(boundary_edges),
        "uv_seam_nonmanifold_edges": 0,
    }
    return seam_edges, meta


def _recover_closed_loop(start: int, adj: Dict[int, List[int]], max_steps: int) -> np.ndarray | None:
    if len(adj.get(start, [])) != 2:
        return None
    prev = -1
    curr = int(start)
    loop = [curr]
    seen = {curr}
    for _ in range(max_steps):
        nbs = adj.get(curr, [])
        if len(nbs) != 2:
            return None
        nxt = nbs[0] if nbs[0] != prev else nbs[1]
        nxt = int(nxt)
        if nxt == start:
            loop.append(start)
            return np.asarray(loop, dtype=np.int64)
        if nxt in seen:
            return None
        loop.append(nxt)
        seen.add(nxt)
        prev, curr = curr, nxt
    return None


def _summarize_seam_topology(
    seam_edges: np.ndarray,
    *,
    label_count: int,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    seam_loops: List[np.ndarray] = []
    if seam_edges.size == 0:
        empty_with_multi_labels = bool(label_count >= 2)
        meta = {
            "uv_low_cut_edges": 0,
            "uv_seam_components": 0,
            "uv_seam_loops_closed": 0,
            "uv_seam_components_open": 0,
            "uv_seam_branch_vertices": 0,
            "uv_seam_topology_valid": not empty_with_multi_labels,
            "uv_seam_empty_with_multi_labels": empty_with_multi_labels,
        }
        return seam_loops, meta

    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in seam_edges.tolist():
        ia = int(a)
        ib = int(b)
        if ia == ib:
            continue
        adj[ia].append(ib)
        adj[ib].append(ia)

    seen = set()
    comp_count = 0
    open_count = 0
    branch_vertices = 0
    for s in list(adj.keys()):
        if s in seen:
            continue
        comp_count += 1
        q = deque([s])
        seen.add(s)
        comp_nodes: List[int] = []
        while q:
            u = q.popleft()
            comp_nodes.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        degrees = [len(adj[n]) for n in comp_nodes]
        branch_vertices += int(sum(1 for d in degrees if d > 2))
        is_closed = bool(len(comp_nodes) >= 3 and all(d == 2 for d in degrees))
        if not is_closed:
            open_count += 1
            continue
        loop = _recover_closed_loop(int(comp_nodes[0]), adj, max_steps=max(8, len(comp_nodes) + 2))
        if loop is None or loop.shape[0] < 4:
            open_count += 1
            continue
        seam_loops.append(loop)

    closed_count = int(len(seam_loops))
    topology_valid = bool(open_count == 0)
    meta = {
        "uv_low_cut_edges": int(seam_edges.shape[0]),
        "uv_seam_components": int(comp_count),
        "uv_seam_loops_closed": int(closed_count),
        "uv_seam_components_open": int(open_count),
        "uv_seam_branch_vertices": int(branch_vertices),
        "uv_seam_topology_valid": bool(topology_valid),
        "uv_seam_empty_with_multi_labels": False,
    }
    return seam_loops, meta


def validate_face_partition_by_seams(
    *,
    low_mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    seam_edges: np.ndarray,
) -> Dict[str, Any]:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    f_count = int(faces.shape[0])
    if labels.shape[0] != f_count:
        raise ValueError("face_labels size mismatch")

    seam_set: Set[Tuple[int, int]] = {
        (int(min(a, b)), int(max(a, b))) for a, b in np.asarray(seam_edges, dtype=np.int64).tolist()
    }
    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fid in range(f_count):
        tri = faces[fid]
        e01 = (int(min(tri[0], tri[1])), int(max(tri[0], tri[1])))
        e12 = (int(min(tri[1], tri[2])), int(max(tri[1], tri[2])))
        e20 = (int(min(tri[2], tri[0])), int(max(tri[2], tri[0])))
        edge_faces[e01].append(fid)
        edge_faces[e12].append(fid)
        edge_faces[e20].append(fid)

    valid_face = labels >= 0
    adj: List[List[int]] = [[] for _ in range(f_count)]
    for ek, flist in edge_faces.items():
        if ek in seam_set or len(flist) != 2:
            continue
        f0 = int(flist[0])
        f1 = int(flist[1])
        if valid_face[f0] and valid_face[f1]:
            adj[f0].append(f1)
            adj[f1].append(f0)

    seen = np.zeros((f_count,), dtype=np.bool_)
    component_count = 0
    mixed_components = 0
    label_comp_count: Dict[int, int] = defaultdict(int)
    valid_face_count = int(np.count_nonzero(valid_face))
    for fid in np.where(valid_face)[0].tolist():
        if seen[fid]:
            continue
        component_count += 1
        q = deque([int(fid)])
        seen[fid] = True
        comp_faces: List[int] = []
        while q:
            u = q.popleft()
            comp_faces.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        comp_labels = np.unique(labels[np.asarray(comp_faces, dtype=np.int64)])
        comp_labels = comp_labels[comp_labels >= 0]
        if comp_labels.size > 1:
            mixed_components += 1
        if comp_labels.size > 0:
            for lb in comp_labels.tolist():
                label_comp_count[int(lb)] += 1

    label_split_count = int(sum(1 for c in label_comp_count.values() if c > 1))
    is_valid = bool(mixed_components == 0)
    return {
        "uv_seam_partition_components": int(component_count),
        "uv_seam_partition_mixed_components": int(mixed_components),
        "uv_seam_partition_label_split_count": int(label_split_count),
        "uv_seam_partition_valid_faces": int(valid_face_count),
        "uv_seam_partition_is_valid": bool(is_valid),
    }


def extract_seam_edges_openmesh(
    *,
    low_mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    include_boundary_as_seam: bool = False,
) -> SeamExtractionResult:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if labels.shape[0] != faces.shape[0]:
        raise ValueError("face_labels size mismatch")

    backend = "openmesh"
    try:
        seam_edges, base_meta = _extract_seam_edges_openmesh_impl(
            low_mesh=low_mesh,
            labels=labels,
            include_boundary_as_seam=include_boundary_as_seam,
        )
    except Exception:
        seam_edges, base_meta = _extract_seam_edges_numpy(
            faces=faces,
            labels=labels,
            include_boundary_as_seam=include_boundary_as_seam,
        )
        backend = "halfedge_numpy_fallback"

    label_count = int(np.unique(labels[labels >= 0]).size)
    seam_loops, topo_meta = _summarize_seam_topology(
        seam_edges,
        label_count=label_count,
    )
    meta: Dict[str, Any] = {}
    meta.update(base_meta)
    meta.update(topo_meta)
    meta["uv_seam_extraction_backend"] = backend
    return SeamExtractionResult(seam_edges=seam_edges, seam_loops=seam_loops, meta=meta)


__all__ = [
    "SeamExtractionResult",
    "extract_seam_edges_openmesh",
    "validate_face_partition_by_seams",
]
