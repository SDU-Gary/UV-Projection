from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import trimesh


def ensure_halfedge_external_dependencies() -> None:
    try:
        import pymeshlab  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "halfedge_island requires hard dependency 'pymeshlab'. "
            "Install with: pip install pymeshlab"
        ) from exc
    try:
        import openmesh  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "halfedge_island requires hard dependency 'openmesh'. "
            "Install with: pip install openmesh"
        ) from exc


def _apply_first_available_filter(ms, candidates: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
    last_exc: Exception | None = None
    for name, kwargs in candidates:
        try:
            ms.apply_filter(name, **kwargs)
            return
        except Exception as exc:  # pragma: no cover - backend-dependent failures.
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc


def _try_first_available_filter(ms, candidates: Iterable[Tuple[str, Dict[str, Any]]]) -> str | None:
    for name, kwargs in candidates:
        try:
            ms.apply_filter(name, **kwargs)
            return str(name)
        except Exception:
            continue
    return None


def _mesh_topology_counters(mesh: trimesh.Trimesh, *, area_eps: float) -> Dict[str, int]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return {
            "nonmanifold_edges": 0,
            "nonmanifold_vertices": 0,
            "degenerate_faces": 0,
        }

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    nm_vertices = _nonmanifold_vertex_count(faces, n_vertices=int(len(verts)))

    tri = verts[faces]
    e01 = tri[:, 1] - tri[:, 0]
    e02 = tri[:, 2] - tri[:, 0]
    area2 = np.linalg.norm(np.cross(e01, e02), axis=1)
    degenerate_faces = int(np.count_nonzero(area2 <= 2.0 * max(float(area_eps), 0.0)))

    return {
        "nonmanifold_edges": int(nonmanifold_edges),
        "nonmanifold_vertices": int(nm_vertices),
        "degenerate_faces": int(degenerate_faces),
    }


def _nonmanifold_vertex_count(faces: np.ndarray, *, n_vertices: int) -> int:
    if n_vertices <= 0 or faces.size == 0:
        return 0

    incident: list[list[int]] = [[] for _ in range(n_vertices)]
    for fid, (a, b, c) in enumerate(faces.tolist()):
        if 0 <= a < n_vertices:
            incident[a].append(fid)
        if 0 <= b < n_vertices:
            incident[b].append(fid)
        if 0 <= c < n_vertices:
            incident[c].append(fid)

    nonmanifold_vertices = 0
    for v in range(n_vertices):
        if len(incident[v]) <= 1:
            continue

        neigh_to_faces: Dict[int, list[int]] = {}
        for fid in incident[v]:
            tri = faces[fid]
            for u in tri.tolist():
                if u == v:
                    continue
                neigh_to_faces.setdefault(int(u), []).append(int(fid))

        face_adj: Dict[int, list[int]] = {fid: [] for fid in incident[v]}
        for fids in neigh_to_faces.values():
            if len(fids) <= 1:
                continue
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    a = int(fids[i])
                    b = int(fids[j])
                    face_adj[a].append(b)
                    face_adj[b].append(a)

        seen: set[int] = set()
        comp = 0
        for fid in incident[v]:
            if fid in seen:
                continue
            comp += 1
            stack = [fid]
            seen.add(fid)
            while stack:
                cur = stack.pop()
                for nb in face_adj.get(cur, []):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if comp > 1:
            nonmanifold_vertices += 1

    return int(nonmanifold_vertices)


def _mesh_orientation_counters(mesh: trimesh.Trimesh) -> Dict[str, int | bool]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return {
            "winding_consistent": True,
            "same_direction_adjacency_edges": 0,
            "body_count": 0,
        }

    same_dir_edges = 0
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adj_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    if adj.ndim == 2 and adj.shape[1] == 2 and adj_edges.ndim == 2 and adj_edges.shape[1] == 2:
        faces_l = faces.tolist()

        def _edge_dir(tri: list[int], a: int, b: int) -> int:
            for i in range(3):
                x = int(tri[i])
                y = int(tri[(i + 1) % 3])
                if x == a and y == b:
                    return 1
                if x == b and y == a:
                    return -1
            return 0

        for (_, _), (a, b), (f0, f1) in zip(adj, adj_edges, adj):
            ia = int(a)
            ib = int(b)
            d0 = _edge_dir(faces_l[int(f0)], ia, ib)
            d1 = _edge_dir(faces_l[int(f1)], ia, ib)
            if d0 != 0 and d0 == d1:
                same_dir_edges += 1

    return {
        "winding_consistent": bool(mesh.is_winding_consistent),
        "same_direction_adjacency_edges": int(same_dir_edges),
        "body_count": int(mesh.body_count),
    }


def sanitize_mesh_for_halfedge(
    *,
    low_mesh: trimesh.Trimesh,
    seam_cfg: Dict[str, Any],
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    import pymeshlab as ml

    area_eps = float(seam_cfg.get("sanitize_area_eps", 1e-12))
    do_sanitize = bool(seam_cfg.get("sanitize_enabled", True))

    in_vertices = np.asarray(low_mesh.vertices, dtype=np.float64)
    in_faces = np.asarray(low_mesh.faces, dtype=np.int64)
    mesh_in = trimesh.Trimesh(vertices=in_vertices, faces=in_faces, process=False)
    before = _mesh_topology_counters(mesh_in, area_eps=area_eps)
    orient_before = _mesh_orientation_counters(mesh_in)

    if not do_sanitize:
        out = trimesh.Trimesh(vertices=in_vertices.astype(np.float32), faces=in_faces.astype(np.int64), process=False)
        meta = {
            "uv_sanitize_enabled": False,
            "uv_sanitize_nonmanifold_edges_before": int(before["nonmanifold_edges"]),
            "uv_sanitize_nonmanifold_vertices_before": int(before["nonmanifold_vertices"]),
            "uv_sanitize_degenerate_faces_before": int(before["degenerate_faces"]),
            "uv_sanitize_nonmanifold_edges_after": int(before["nonmanifold_edges"]),
            "uv_sanitize_nonmanifold_vertices_after": int(before["nonmanifold_vertices"]),
            "uv_sanitize_degenerate_faces_after": int(before["degenerate_faces"]),
            "uv_sanitize_winding_consistent_before": bool(orient_before["winding_consistent"]),
            "uv_sanitize_winding_consistent_after": bool(orient_before["winding_consistent"]),
            "uv_sanitize_same_direction_adjacency_edges_before": int(orient_before["same_direction_adjacency_edges"]),
            "uv_sanitize_same_direction_adjacency_edges_after": int(orient_before["same_direction_adjacency_edges"]),
            "uv_sanitize_body_count_before": int(orient_before["body_count"]),
            "uv_sanitize_body_count_after": int(orient_before["body_count"]),
        }
        return out, meta

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=in_vertices, face_matrix=in_faces))

    _apply_first_available_filter(
        ms,
        [
            ("repair_non_manifold_edges_by_splitting_vertices", {}),
            ("meshing_repair_non_manifold_edges", {"method": 0}),
            ("meshing_repair_non_manifold_edges", {}),
        ],
    )
    _apply_first_available_filter(
        ms,
        [
            ("repair_non_manifold_vertices_by_splitting_vertices", {}),
            ("meshing_repair_non_manifold_vertices", {"vertdispratio": 0.0}),
            ("meshing_repair_non_manifold_vertices", {}),
        ],
    )
    # Re-orient faces before recomputing normals. The previous implementation
    # stopped after per-face normal computation and never actually fixed winding.
    reorient_filter = _try_first_available_filter(
        ms,
        [
            ("meshing_re_orient_faces_coherently", {}),
            ("meshing_re_orient_all_faces_coherently", {}),
        ],
    )
    normal_filter = _try_first_available_filter(
        ms,
        [
            ("per_face_normal_computation", {}),
            ("compute_normal_per_face", {}),
        ],
    )

    m = ms.current_mesh()
    out_vertices = np.asarray(m.vertex_matrix(), dtype=np.float32)
    out_faces = np.asarray(m.face_matrix(), dtype=np.int64)
    mesh_out = trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False)

    # Drop degenerate faces after PyMeshLab repair.
    valid_faces = mesh_out.nondegenerate_faces(height=area_eps)
    if valid_faces.sum() < len(mesh_out.faces):
        mesh_out.update_faces(valid_faces)
        mesh_out.remove_unreferenced_vertices()
    # Trimesh can normalize face winding per connected patch more reliably than
    # OpenMesh constructor fallback behavior for non-watertight outputs.
    trimesh.repair.fix_normals(mesh_out, multibody=True)
    after = _mesh_topology_counters(mesh_out, area_eps=area_eps)
    orient_after = _mesh_orientation_counters(mesh_out)

    meta = {
        "uv_sanitize_enabled": True,
        "uv_sanitize_nonmanifold_edges_before": int(before["nonmanifold_edges"]),
        "uv_sanitize_nonmanifold_vertices_before": int(before["nonmanifold_vertices"]),
        "uv_sanitize_degenerate_faces_before": int(before["degenerate_faces"]),
        "uv_sanitize_nonmanifold_edges_after": int(after["nonmanifold_edges"]),
        "uv_sanitize_nonmanifold_vertices_after": int(after["nonmanifold_vertices"]),
        "uv_sanitize_degenerate_faces_after": int(after["degenerate_faces"]),
        "uv_sanitize_vertex_count_after": int(len(mesh_out.vertices)),
        "uv_sanitize_face_count_after": int(len(mesh_out.faces)),
        "uv_sanitize_winding_consistent_before": bool(orient_before["winding_consistent"]),
        "uv_sanitize_winding_consistent_after": bool(orient_after["winding_consistent"]),
        "uv_sanitize_same_direction_adjacency_edges_before": int(orient_before["same_direction_adjacency_edges"]),
        "uv_sanitize_same_direction_adjacency_edges_after": int(orient_after["same_direction_adjacency_edges"]),
        "uv_sanitize_body_count_before": int(orient_before["body_count"]),
        "uv_sanitize_body_count_after": int(orient_after["body_count"]),
        "uv_sanitize_reorient_filter_used": reorient_filter or "",
        "uv_sanitize_normal_filter_used": normal_filter or "",
    }
    return mesh_out, meta


__all__ = [
    "ensure_halfedge_external_dependencies",
    "sanitize_mesh_for_halfedge",
]
