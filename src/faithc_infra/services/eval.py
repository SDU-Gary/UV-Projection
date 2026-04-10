from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..mesh_io import MeshIO
from ..types import MetricsArtifact, UVArtifact


class EvalService:
    def evaluate(
        self,
        sample_name: str,
        high_mesh_path: Path,
        low_mesh_path: Path,
        output_dir: Path,
        sample_points: int,
        uv_artifact: Optional[UVArtifact] = None,
    ) -> MetricsArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        high_mesh = MeshIO.load_mesh(high_mesh_path, process=False)
        low_mesh = MeshIO.load_mesh(low_mesh_path, process=False)

        metrics: Dict[str, float] = {
            "high_vertices": int(len(high_mesh.vertices)),
            "high_faces": int(len(high_mesh.faces)),
            "low_vertices": int(len(low_mesh.vertices)),
            "low_faces": int(len(low_mesh.faces)),
            "face_reduction_ratio": 1.0 - float(len(low_mesh.faces)) / max(1, len(high_mesh.faces)),
            "vertex_reduction_ratio": 1.0 - float(len(low_mesh.vertices)) / max(1, len(high_mesh.vertices)),
        }

        chamfer_l1 = self._chamfer_l1(high_mesh, low_mesh, sample_points)
        metrics["chamfer_l1"] = float(chamfer_l1)

        if uv_artifact is not None:
            metrics.update(uv_artifact.stats)

        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        return MetricsArtifact(sample_name=sample_name, metrics_path=metrics_path, metrics=metrics)

    @staticmethod
    def _chamfer_l1(high_mesh: trimesh.Trimesh, low_mesh: trimesh.Trimesh, sample_points: int) -> float:
        n_points = max(1000, int(sample_points))
        high_pts, _ = trimesh.sample.sample_surface(high_mesh, n_points)
        low_pts, _ = trimesh.sample.sample_surface(low_mesh, n_points)

        high_to_low = cKDTree(low_pts).query(high_pts, k=1)[0]
        low_to_high = cKDTree(high_pts).query(low_pts, k=1)[0]
        return float(high_to_low.mean() + low_to_high.mean())
