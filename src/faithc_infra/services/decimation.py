from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from .uv.mesh_sanitizer import _mesh_orientation_counters, _mesh_topology_counters


@dataclass
class DecimationArtifact:
    mesh_low: trimesh.Trimesh
    stats: Dict[str, Any] = field(default_factory=dict)


def decimate_with_pymeshlab_qem(
    mesh_high: trimesh.Trimesh,
    *,
    target_face_count: int = 0,
    target_face_ratio: float = 0.05,
    quality_threshold: float = 0.3,
    preserve_boundary: bool = False,
    boundary_weight: float = 2.0,
    preserve_normal: bool = False,
    preserve_topology: bool = False,
    optimal_placement: bool = True,
    planar_quadric: bool = False,
    planar_weight: float = 1e-3,
    quality_weight: bool = False,
    autoclean: bool = True,
) -> DecimationArtifact:
    import pymeshlab as ml

    faces_in = int(len(mesh_high.faces))
    verts_in = int(len(mesh_high.vertices))
    if faces_in <= 0:
        raise ValueError("pymeshlab_qem requires mesh with faces")

    requested_count = int(target_face_count)
    requested_ratio = float(target_face_ratio)
    if requested_count <= 0:
        if not (0.0 < requested_ratio < 1.0):
            raise ValueError(
                "pymeshlab_qem requires target_face_count > 0 or target_face_ratio in (0,1)"
            )
        requested_count = int(round(faces_in * requested_ratio))
    requested_count = max(4, min(faces_in, int(requested_count)))
    effective_ratio = float(requested_count / max(1, faces_in))

    verts = np.asarray(mesh_high.vertices, dtype=np.float64)
    faces = np.asarray(mesh_high.faces, dtype=np.int64)

    t0 = time.perf_counter()
    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertex_matrix=verts, face_matrix=faces))
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=int(requested_count),
        targetperc=0.0,
        qualitythr=float(quality_threshold),
        preserveboundary=bool(preserve_boundary),
        boundaryweight=float(boundary_weight),
        preservenormal=bool(preserve_normal),
        preservetopology=bool(preserve_topology),
        optimalplacement=bool(optimal_placement),
        planarquadric=bool(planar_quadric),
        planarweight=float(planar_weight),
        qualityweight=bool(quality_weight),
        autoclean=bool(autoclean),
        selected=False,
    )
    m_raw = ms.current_mesh()
    raw_mesh = trimesh.Trimesh(
        vertices=np.asarray(m_raw.vertex_matrix(), dtype=np.float32),
        faces=np.asarray(m_raw.face_matrix(), dtype=np.int64),
        process=False,
    )
    raw_topo = _mesh_topology_counters(raw_mesh, area_eps=1e-12)
    raw_orient = _mesh_orientation_counters(raw_mesh)

    repair_error: Optional[str] = None
    vertex_repair_iters = 0
    try:
        ms.apply_filter("meshing_repair_non_manifold_edges", method="Remove Faces")
        for iter_idx in range(5):
            ms.apply_filter("meshing_repair_non_manifold_vertices", vertdispratio=0.0)
            vertex_repair_iters = iter_idx + 1
            probe = ms.current_mesh()
            probe_mesh = trimesh.Trimesh(
                vertices=np.asarray(probe.vertex_matrix(), dtype=np.float32),
                faces=np.asarray(probe.face_matrix(), dtype=np.int64),
                process=False,
            )
            probe_topo = _mesh_topology_counters(probe_mesh, area_eps=1e-12)
            if int(probe_topo["nonmanifold_vertices"]) == 0:
                break
    except Exception as exc:
        repair_error = str(exc)

    m = ms.current_mesh()
    low_mesh = trimesh.Trimesh(
        vertices=np.asarray(m.vertex_matrix(), dtype=np.float32),
        faces=np.asarray(m.face_matrix(), dtype=np.int64),
        process=False,
    )

    valid_faces = low_mesh.nondegenerate_faces()
    if int(np.count_nonzero(valid_faces)) < int(len(low_mesh.faces)):
        low_mesh.update_faces(valid_faces)
        low_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(low_mesh, multibody=True)
    elapsed = float(time.perf_counter() - t0)
    faces_out = int(len(low_mesh.faces))
    reduction_ratio = float(faces_out / max(1, faces_in))
    target_achieved = bool(faces_out <= max(int(requested_count * 1.1), requested_count + 8))

    topo = _mesh_topology_counters(low_mesh, area_eps=1e-12)
    orient = _mesh_orientation_counters(low_mesh)
    stats: Dict[str, Any] = {
        "reconstruction_backend_requested": "pymeshlab_qem",
        "reconstruction_backend_used": "pymeshlab_qem",
        "reconstruction_pymeshlab_target_face_count_requested": int(requested_count),
        "reconstruction_pymeshlab_target_face_ratio_requested": float(requested_ratio),
        "reconstruction_pymeshlab_target_face_ratio_effective": float(effective_ratio),
        "reconstruction_pymeshlab_quality_threshold": float(quality_threshold),
        "reconstruction_pymeshlab_preserve_boundary": bool(preserve_boundary),
        "reconstruction_pymeshlab_boundary_weight": float(boundary_weight),
        "reconstruction_pymeshlab_preserve_normal": bool(preserve_normal),
        "reconstruction_pymeshlab_preserve_topology": bool(preserve_topology),
        "reconstruction_pymeshlab_optimal_placement": bool(optimal_placement),
        "reconstruction_pymeshlab_planar_quadric": bool(planar_quadric),
        "reconstruction_pymeshlab_planar_weight": float(planar_weight),
        "reconstruction_pymeshlab_quality_weight": bool(quality_weight),
        "reconstruction_pymeshlab_autoclean": bool(autoclean),
        "reconstruction_pymeshlab_runtime_seconds": round(elapsed, 6),
        "reconstruction_pymeshlab_nonmanifold_repair_error": repair_error,
        "reconstruction_pymeshlab_vertex_repair_iters": int(vertex_repair_iters),
        "reconstruction_pymeshlab_raw_faces_after_decimation": int(len(raw_mesh.faces)),
        "reconstruction_pymeshlab_raw_vertices_after_decimation": int(len(raw_mesh.vertices)),
        "reconstruction_pymeshlab_raw_nonmanifold_edges_after_decimation": int(raw_topo["nonmanifold_edges"]),
        "reconstruction_pymeshlab_raw_nonmanifold_vertices_after_decimation": int(raw_topo["nonmanifold_vertices"]),
        "reconstruction_pymeshlab_raw_degenerate_faces_after_decimation": int(raw_topo["degenerate_faces"]),
        "reconstruction_pymeshlab_raw_winding_consistent_after_decimation": bool(raw_orient["winding_consistent"]),
        "reconstruction_pymeshlab_faces_out": faces_out,
        "reconstruction_pymeshlab_face_reduction_ratio": reduction_ratio,
        "reconstruction_pymeshlab_target_achieved": target_achieved,
        "num_input_vertices": int(verts_in),
        "num_input_faces": int(faces_in),
        "num_low_vertices": int(len(low_mesh.vertices)),
        "num_low_faces": faces_out,
        "active_voxels": 0,
        "reconstruction_nonmanifold_edges_after": int(topo["nonmanifold_edges"]),
        "reconstruction_nonmanifold_vertices_after": int(topo["nonmanifold_vertices"]),
        "reconstruction_degenerate_faces_after": int(topo["degenerate_faces"]),
        "reconstruction_winding_consistent_after": bool(orient["winding_consistent"]),
        "reconstruction_same_direction_adjacency_edges_after": int(orient["same_direction_adjacency_edges"]),
        "reconstruction_body_count_after": int(orient["body_count"]),
    }
    return DecimationArtifact(mesh_low=low_mesh, stats=stats)


__all__ = [
    "DecimationArtifact",
    "decimate_with_pymeshlab_qem",
]
