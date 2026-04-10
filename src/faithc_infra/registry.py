from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ArtifactRegistry:
    @staticmethod
    def write_manifest(
        sample_name: str,
        sample_dir: Path,
        high_mesh_path: Path,
        low_mesh_path: Path,
        low_mesh_uv_path: Optional[Path],
        metrics_path: Optional[Path],
        preset: str,
    ) -> Path:
        manifest: Dict[str, Any] = {
            "schema_version": "1.0",
            "sample_name": sample_name,
            "meshes": {
                "high": str(high_mesh_path),
                "low": str(low_mesh_path),
                "low_uv": str(low_mesh_uv_path) if low_mesh_uv_path else None,
            },
            "artifacts": {
                "metrics": str(metrics_path) if metrics_path else None,
            },
            "render": {
                "preset": preset,
                "output_dir": str(sample_dir / "render"),
            },
        }

        path = sample_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return path
