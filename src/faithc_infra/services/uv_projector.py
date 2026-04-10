from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .uv import DEFAULT_OPTIONS, METHOD_ALIASES, deep_merge_dict
from .uv.correspondence import (
    bvh_project_points,
)
from .uv.quality import compute_uv_quality
from .uv.texture_io import extract_uv, resolve_basecolor_image, resolve_device
from .uv.hybrid_pipeline import run_hybrid_global_opt
from .uv.method2_pipeline import run_method2_gradient_poisson
from .uv.method4_pipeline import run_method4_jacobian_injective
from ..mesh_io import MeshIO
from ..types import UVArtifact

class UVProjector:
    def project(
        self,
        sample_name: str,
        high_mesh_path: Path,
        low_mesh_path: Path,
        output_dir: Path,
        *,
        method: str = "nearest_vertex",
        device: str = "auto",
        texture_source_path: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> UVArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)

        high_mesh = MeshIO.load_mesh(high_mesh_path, process=False)
        low_mesh = MeshIO.load_mesh(low_mesh_path, process=False)

        mapped_uv, image, stats, export_payload = self.map_uv(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            method=method,
            device=device,
            texture_source_path=texture_source_path,
            options=options,
            return_export_payload=True,
        )

        uv_mesh = self.build_uv_mesh(
            low_mesh=low_mesh,
            mapped_uv=mapped_uv,
            image=image,
            export_payload=export_payload,
        )
        stats["uv_export_vertices"] = int(len(uv_mesh.vertices))
        stats["uv_export_faces"] = int(len(uv_mesh.faces))
        stats["uv_local_vertex_splits"] = int(len(uv_mesh.vertices) - len(low_mesh.vertices))

        uv_mesh_path = output_dir / "mesh_low_uv.glb"
        try:
            MeshIO.export_mesh(uv_mesh, uv_mesh_path)
        except Exception:
            uv_mesh_path = output_dir / "mesh_low_uv.obj"
            MeshIO.export_mesh(uv_mesh, uv_mesh_path)

        uv_map_path = output_dir / "uv_map.npy"
        saved_uv = np.asarray(getattr(uv_mesh.visual, "uv", mapped_uv), dtype=np.float32)
        np.save(uv_map_path, saved_uv)

        with (output_dir / "uv_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        return UVArtifact(
            sample_name=sample_name,
            low_mesh_uv_path=uv_mesh_path,
            uv_map_path=uv_map_path,
            stats=stats,
        )

    def project_nearest(
        self,
        sample_name: str,
        high_mesh_path: Path,
        low_mesh_path: Path,
        output_dir: Path,
    ) -> UVArtifact:
        return self.project(
            sample_name=sample_name,
            high_mesh_path=high_mesh_path,
            low_mesh_path=low_mesh_path,
            output_dir=output_dir,
            method="nearest_vertex",
        )

    def map_uv(
        self,
        *,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        method: str,
        device: str = "auto",
        texture_source_path: Optional[Path] = None,
        options: Optional[Dict[str, Any]] = None,
        return_export_payload: bool = False,
    ):
        method_name = METHOD_ALIASES.get(method, method)
        if method_name not in {
            "nearest_vertex",
            "barycentric_closest_point",
            "hybrid_global_opt",
            "method2_gradient_poisson",
            "method4_jacobian_injective",
        }:
            raise ValueError(f"Unsupported UV method: {method}")

        cfg = deep_merge_dict(DEFAULT_OPTIONS, options or {})

        high_uv = extract_uv(high_mesh)
        if high_uv is None:
            raise RuntimeError("High mesh has no valid per-vertex UV coordinates")

        image, image_source = resolve_basecolor_image(high_mesh, texture_source_path)
        image_size = list(image.size) if image is not None and hasattr(image, "size") else None

        t0 = time.perf_counter()
        export_payload = None
        if method_name == "nearest_vertex":
            mapped_uv, method_stats = self._map_nearest_vertex(high_mesh, low_mesh, high_uv)
        elif method_name == "barycentric_closest_point":
            mapped_uv, method_stats = self._map_barycentric_closest(high_mesh, low_mesh, high_uv, device, cfg)
        elif method_name == "hybrid_global_opt":
            mapped_uv, method_stats, export_payload = self._map_hybrid_global_opt(
                high_mesh,
                low_mesh,
                high_uv,
                image,
                device,
                cfg,
            )
        elif method_name == "method2_gradient_poisson":
            mapped_uv, method_stats, export_payload = self._map_method2_gradient_poisson(
                high_mesh,
                low_mesh,
                high_uv,
                image,
                device,
                cfg,
            )
        else:
            mapped_uv, method_stats, export_payload = self._map_method4_jacobian_injective(
                high_mesh,
                low_mesh,
                high_uv,
                image,
                device,
                cfg,
            )

        quality_mesh = low_mesh
        if export_payload is not None:
            payload_quality_mesh = export_payload.get("quality_mesh")
            if isinstance(payload_quality_mesh, trimesh.Trimesh):
                quality_mesh = payload_quality_mesh

        quality = compute_uv_quality(quality_mesh, mapped_uv)
        stats: Dict[str, Any] = {
            "uv_mode_used": method_name,
            "uv_method": method_name,
            "uv_projected": True,
            "uv_source_has_uv": True,
            "uv_source_vertices": int(len(high_mesh.vertices)),
            "uv_target_vertices": int(len(quality_mesh.vertices)),
            "uv_has_basecolor_image": image is not None,
            "uv_source_image_source": image_source,
            "uv_source_image_size": image_size,
            "uv_projection_seconds": round(time.perf_counter() - t0, 6),
            **quality,
            **method_stats,
        }
        stats.setdefault("uv_correspondence_success_ratio", 1.0)
        stats.setdefault("uv_color_reproj_l1", None)
        stats.setdefault("uv_color_reproj_l2", None)

        if return_export_payload:
            return mapped_uv, image, stats, export_payload
        return mapped_uv, image, stats

    def build_uv_mesh(
        self,
        *,
        low_mesh: trimesh.Trimesh,
        mapped_uv: np.ndarray,
        image,
        export_payload: Optional[Dict[str, Any]] = None,
    ) -> trimesh.Trimesh:
        if export_payload is not None and bool(export_payload.get("halfedge_split_topology", False)):
            split_vertices = np.asarray(export_payload.get("split_vertices", []), dtype=np.float32)
            split_faces = np.asarray(export_payload.get("split_faces", []), dtype=np.int64)
            if (
                split_vertices.ndim == 2
                and split_vertices.shape[1] == 3
                and split_faces.ndim == 2
                and split_faces.shape[1] == 3
                and split_vertices.shape[0] == int(np.asarray(mapped_uv).shape[0])
            ):
                uv_mesh = trimesh.Trimesh(vertices=split_vertices, faces=split_faces, process=False)
                uv_mesh.visual = trimesh.visual.texture.TextureVisuals(
                    uv=np.asarray(mapped_uv, dtype=np.float32),
                    image=image,
                )
                return uv_mesh

        if export_payload is None or not bool(export_payload.get("local_vertex_split_applied", False)):
            uv_mesh = low_mesh.copy()
            uv_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.asarray(mapped_uv, dtype=np.float32), image=image)
            return uv_mesh

        seam_face_ids = np.asarray(export_payload.get("seam_face_ids", []), dtype=np.int64)
        seam_corner_uv = np.asarray(export_payload.get("seam_corner_uv", []), dtype=np.float32)
        if seam_face_ids.size == 0 or seam_corner_uv.size == 0:
            uv_mesh = low_mesh.copy()
            uv_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.asarray(mapped_uv, dtype=np.float32), image=image)
            return uv_mesh

        verts = np.asarray(low_mesh.vertices, dtype=np.float32)
        faces = np.asarray(low_mesh.faces, dtype=np.int64).copy()
        uv_base = np.asarray(mapped_uv, dtype=np.float32)

        vertices_out: List[np.ndarray] = [verts[i].copy() for i in range(len(verts))]
        uv_out: List[np.ndarray] = [uv_base[i].copy() for i in range(len(uv_base))]

        for i, face_id in enumerate(seam_face_ids):
            if face_id < 0 or face_id >= len(faces):
                continue
            face_uv = seam_corner_uv[i]
            for corner in range(3):
                old_vid = int(faces[face_id, corner])
                new_vid = len(vertices_out)
                vertices_out.append(verts[old_vid].copy())
                uv_out.append(face_uv[corner].copy())
                faces[face_id, corner] = new_vid

        uv_mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices_out, dtype=np.float32),
            faces=faces,
            process=False,
        )
        uv_mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.asarray(uv_out, dtype=np.float32),
            image=image,
        )
        return uv_mesh

    def _map_nearest_vertex(
        self,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        high_uv: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        tree = cKDTree(np.asarray(high_mesh.vertices))
        _, nearest_idx = tree.query(np.asarray(low_mesh.vertices), k=1)
        nearest_idx = np.asarray(nearest_idx, dtype=np.int64)
        mapped_uv = high_uv[np.clip(nearest_idx, 0, len(high_uv) - 1)]
        return mapped_uv, {
            "uv_correspondence_success_ratio": 1.0,
            "uv_correspondence_primary_ratio": 1.0,
        }

    def _map_barycentric_closest(
        self,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        high_uv: np.ndarray,
        device: str,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        resolved = resolve_device(device)
        if resolved != "cuda":
            mapped_uv, stats = self._map_nearest_vertex(high_mesh, low_mesh, high_uv)
            stats["uv_mode_used"] = "nearest_vertex_fallback"
            stats["uv_project_error"] = "CUDA unavailable for barycentric mapping; fell back to nearest vertex"
            return mapped_uv, stats

        out = bvh_project_points(
            points=np.asarray(low_mesh.vertices),
            high_mesh=high_mesh,
            high_uv=high_uv,
            device=resolved,
            chunk_size=int(cfg["correspondence"]["bvh_chunk_size"]),
        )
        mapped_uv = out["mapped_uv"]
        return mapped_uv, {
            "uv_correspondence_success_ratio": 1.0,
            "uv_correspondence_primary_ratio": 1.0,
        }

    def _map_hybrid_global_opt(
        self,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        high_uv: np.ndarray,
        image,
        device: str,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        return run_hybrid_global_opt(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            high_uv=high_uv,
            image=image,
            device=device,
            cfg=cfg,
            nearest_mapper=self._map_nearest_vertex,
            barycentric_mapper=self._map_barycentric_closest,
        )

    def _map_method2_gradient_poisson(
        self,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        high_uv: np.ndarray,
        image,
        device: str,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        return run_method2_gradient_poisson(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            high_uv=high_uv,
            image=image,
            device=device,
            cfg=cfg,
            nearest_mapper=self._map_nearest_vertex,
            barycentric_mapper=self._map_barycentric_closest,
        )

    def _map_method4_jacobian_injective(
        self,
        high_mesh: trimesh.Trimesh,
        low_mesh: trimesh.Trimesh,
        high_uv: np.ndarray,
        image,
        device: str,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        return run_method4_jacobian_injective(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            high_uv=high_uv,
            image=image,
            device=device,
            cfg=cfg,
            nearest_mapper=self._map_nearest_vertex,
            barycentric_mapper=self._map_barycentric_closest,
        )
