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
    nm_vertices = np.unique(uniq[counts > 2].reshape(-1)).size if nonmanifold_edges > 0 else 0

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
    # Face normal consistency for robust semantic projection.
    _apply_first_available_filter(
        ms,
        [
            ("per_face_normal_computation", {}),
            ("compute_normal_per_face", {}),
            ("meshing_re_orient_faces_coherently", {}),
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
    after = _mesh_topology_counters(mesh_out, area_eps=area_eps)

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
    }
    return mesh_out, meta


__all__ = [
    "ensure_halfedge_external_dependencies",
    "sanitize_mesh_for_halfedge",
]

