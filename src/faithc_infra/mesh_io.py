from __future__ import annotations

from pathlib import Path

import trimesh


class MeshIO:
    @staticmethod
    def load_mesh(path: Path, process: bool = False) -> trimesh.Trimesh:
        if not path.exists():
            raise FileNotFoundError(f"Mesh not found: {path}")

        mesh_or_scene = trimesh.load(path, force="mesh", process=process)
        if isinstance(mesh_or_scene, trimesh.Trimesh):
            mesh = mesh_or_scene
        elif isinstance(mesh_or_scene, trimesh.Scene):
            geometries = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                raise ValueError(f"No triangle geometry found in scene: {path}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            raise ValueError(f"Unsupported mesh type from {path}: {type(mesh_or_scene)}")

        if mesh.faces is None or len(mesh.faces) == 0:
            raise ValueError(f"Mesh has no faces: {path}")

        return mesh

    @staticmethod
    def export_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
