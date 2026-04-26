from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import trimesh

from .mesh_sanitizer import _mesh_orientation_counters


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
    loaded_faces = int(mesh.n_faces())
    if loaded_faces != int(faces.shape[0]):
        raise RuntimeError(f"openmesh_face_drop loaded={loaded_faces} expected={int(faces.shape[0])}")

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
        "uv_openmesh_loaded_faces": int(mesh.n_faces()),
        "uv_openmesh_loaded_edges": int(mesh.n_edges()),
        "uv_openmesh_loaded_vertices": int(mesh.n_vertices()),
    }
    return seam_edges, meta


def _build_edge_faces_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
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
    return edge_faces


def _face_component_sizes(low_mesh: trimesh.Trimesh) -> np.ndarray:
    n_faces = int(len(low_mesh.faces))
    if n_faces <= 0:
        return np.zeros((0,), dtype=np.int32)

    adj = np.asarray(low_mesh.face_adjacency, dtype=np.int64)
    neigh: List[List[int]] = [[] for _ in range(n_faces)]
    if adj.ndim == 2 and adj.shape[1] == 2:
        for a, b in adj.tolist():
            ia = int(a)
            ib = int(b)
            if 0 <= ia < n_faces and 0 <= ib < n_faces and ia != ib:
                neigh[ia].append(ib)
                neigh[ib].append(ia)

    seen = np.zeros((n_faces,), dtype=np.bool_)
    comp_size_per_face = np.ones((n_faces,), dtype=np.int32)
    for fid in range(n_faces):
        if seen[fid]:
            continue
        q = deque([int(fid)])
        seen[fid] = True
        comp_faces: List[int] = []
        while q:
            u = q.popleft()
            comp_faces.append(u)
            for v in neigh[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        csz = int(len(comp_faces))
        comp_size_per_face[np.asarray(comp_faces, dtype=np.int64)] = int(csz)
    return comp_size_per_face


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
    boundary_vertices: Set[int] | None = None,
    allow_open_on_boundary: bool = False,
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
    boundary_open_count = 0
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
            allow_boundary_path = False
            if allow_open_on_boundary and boundary_vertices is not None and len(comp_nodes) >= 2:
                deg1_nodes = [int(n) for n, d in zip(comp_nodes, degrees) if d == 1]
                if len(deg1_nodes) == 2 and all(d in {1, 2} for d in degrees):
                    if (deg1_nodes[0] in boundary_vertices) and (deg1_nodes[1] in boundary_vertices):
                        allow_boundary_path = True
            if allow_boundary_path:
                boundary_open_count += 1
            else:
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
        "uv_seam_components_open_on_boundary": int(boundary_open_count),
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
    min_component_faces: int = 0,
) -> Dict[str, Any]:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    f_count = int(faces.shape[0])
    if labels.shape[0] != f_count:
        raise ValueError("face_labels size mismatch")

    seam_set: Set[Tuple[int, int]] = {
        (int(min(a, b)), int(max(a, b))) for a, b in np.asarray(seam_edges, dtype=np.int64).tolist()
    }
    edge_faces = _build_edge_faces_map(faces)

    valid_face = labels >= 0
    ignored_small_faces = 0
    min_component_faces_i = int(max(0, min_component_faces))
    if min_component_faces_i > 1 and f_count > 0:
        comp_sizes = _face_component_sizes(low_mesh)
        keep_face = comp_sizes >= min_component_faces_i
        ignored_small_faces = int(np.count_nonzero(valid_face & (~keep_face)))
        valid_face = valid_face & keep_face

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
        "uv_seam_partition_min_component_faces": int(min_component_faces_i),
        "uv_seam_partition_ignored_small_faces": int(ignored_small_faces),
        "uv_seam_partition_is_valid": bool(is_valid),
    }


def extract_seam_edges_openmesh(
    *,
    low_mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    include_boundary_as_seam: bool = False,
    validation_min_component_faces: int = 0,
    validation_allow_open_on_boundary: bool = True,
) -> SeamExtractionResult:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    labels = np.asarray(face_labels, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if labels.shape[0] != faces.shape[0]:
        raise ValueError("face_labels size mismatch")

    orient = _mesh_orientation_counters(low_mesh)
    openmesh_meta = {
        "uv_openmesh_input_winding_consistent": bool(orient["winding_consistent"]),
        "uv_openmesh_input_same_direction_adjacency_edges": int(orient["same_direction_adjacency_edges"]),
        "uv_openmesh_input_body_count": int(orient["body_count"]),
        "uv_openmesh_attempted": False,
        "uv_openmesh_fallback_reason": "",
    }

    backend = "openmesh"
    base_meta: Dict[str, Any]
    fallback_reason: str | None = None
    if not bool(orient["winding_consistent"]):
        fallback_reason = (
            "input_winding_inconsistent "
            f"same_dir_adj={int(orient['same_direction_adjacency_edges'])} "
            f"bodies={int(orient['body_count'])}"
        )
    else:
        try:
            openmesh_meta["uv_openmesh_attempted"] = True
            seam_edges, base_meta = _extract_seam_edges_openmesh_impl(
                low_mesh=low_mesh,
                labels=labels,
                include_boundary_as_seam=include_boundary_as_seam,
            )
        except Exception as exc:
            fallback_reason = str(exc)

    if fallback_reason is not None:
        seam_edges, base_meta = _extract_seam_edges_numpy(
            faces=faces,
            labels=labels,
            include_boundary_as_seam=include_boundary_as_seam,
        )
        backend = "halfedge_numpy_fallback"
        openmesh_meta["uv_openmesh_fallback_reason"] = str(fallback_reason)

    label_count = int(np.unique(labels[labels >= 0]).size)
    # Derive boundary vertex set from mesh topology (edge with single incident face).
    edge_faces_full = _build_edge_faces_map(faces)
    boundary_vertices: Set[int] = set()
    for (a, b), flist in edge_faces_full.items():
        if len(flist) == 1:
            boundary_vertices.add(int(a))
            boundary_vertices.add(int(b))
    seam_edges_topology = seam_edges
    min_component_faces_i = int(max(0, validation_min_component_faces))
    dropped_small_component_edges = 0
    if min_component_faces_i > 1 and seam_edges.shape[0] > 0:
        comp_sizes = _face_component_sizes(low_mesh)
        keep_face = comp_sizes >= min_component_faces_i
        edge_faces = edge_faces_full
        keep_edges: List[Tuple[int, int]] = []
        for a, b in seam_edges.tolist():
            ek = (int(min(a, b)), int(max(a, b)))
            flist = edge_faces.get(ek, [])
            # If edge has no incident face record (unexpected), keep it for safety.
            if len(flist) == 0:
                keep_edges.append(ek)
                continue
            if any(bool(keep_face[int(fid)]) for fid in flist):
                keep_edges.append(ek)
            else:
                dropped_small_component_edges += 1
        seam_edges_topology = (
            np.asarray(sorted(set(keep_edges)), dtype=np.int64)
            if len(keep_edges) > 0
            else np.zeros((0, 2), dtype=np.int64)
        )

    seam_loops, topo_meta = _summarize_seam_topology(
        seam_edges_topology,
        label_count=label_count,
        boundary_vertices=boundary_vertices,
        allow_open_on_boundary=bool(validation_allow_open_on_boundary and (not include_boundary_as_seam)),
    )
    meta: Dict[str, Any] = {}
    meta.update(openmesh_meta)
    meta.update(base_meta)
    meta.update(topo_meta)
    meta["uv_seam_validation_min_component_faces"] = int(min_component_faces_i)
    meta["uv_seam_validation_edges_dropped_small_components"] = int(dropped_small_component_edges)
    meta["uv_seam_extraction_backend"] = backend
    return SeamExtractionResult(seam_edges=seam_edges, seam_loops=seam_loops, meta=meta)


__all__ = [
    "SeamExtractionResult",
    "extract_seam_edges_openmesh",
    "validate_face_partition_by_seams",
]
