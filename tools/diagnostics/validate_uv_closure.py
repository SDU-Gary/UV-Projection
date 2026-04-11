#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.uv.closure_validation import run_uv_closure_validation


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError(f"scene has no mesh: {path}")
        return trimesh.util.concatenate(geoms)
    raise RuntimeError(f"unsupported mesh type: {type(loaded)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UV closure loop validation (semantic/boundary/partition/UV checks)")
    p.add_argument("--high", type=Path, required=True, help="High mesh with UV")
    p.add_argument("--low", type=Path, required=True, help="Low mesh with UV")
    p.add_argument("--out-json", type=Path, required=True, help="Output JSON sidecar")
    p.add_argument("--out-png", type=Path, required=True, help="Output PNG report image")
    p.add_argument("--high-position-eps", type=float, default=1e-6)
    p.add_argument("--high-uv-eps", type=float, default=1e-5)
    p.add_argument("--overlap-raster-res", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    high_mesh = _load_mesh(args.high.resolve())
    low_mesh = _load_mesh(args.low.resolve())

    result = run_uv_closure_validation(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_position_eps=float(args.high_position_eps),
        high_uv_eps=float(args.high_uv_eps),
        output_png=args.out_png.resolve(),
        overlap_raster_res=int(args.overlap_raster_res),
    )

    payload: Dict[str, Any] = {
        "semantic_labels": np.asarray(result.low_face_labels, dtype=np.int64).astype(int).tolist(),
        "seam_edges": np.asarray(result.low_seam_edges, dtype=np.int64).astype(int).tolist(),
        "summary": dict(result.metrics),
        "uv_validation_png": result.image_path,
        "uv_validation_png_error": result.image_error,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"validation_json={args.out_json}")
    if result.image_path:
        print(f"validation_png={result.image_path}")
    if result.image_error:
        print(f"validation_png_error={result.image_error}")
    print(
        "partition_leakage="
        f"{bool(result.metrics.get('partition_has_leakage', False))}, "
        f"mixed_components={int(result.metrics.get('partition_mixed_components', -1))}, "
        f"label_split_count={int(result.metrics.get('partition_label_split_count', -1))}"
    )


if __name__ == "__main__":
    main()

