#!/usr/bin/env python3
"""Standalone Mitsuba3 render adapter for FaithC manifests.

Usage (example):
  FAITHC_MANIFEST=/path/to/manifest.json \
  FAITHC_OUTPUT_DIR=/path/to/output \
  FAITHC_PRESET_PATH=renderer/mitsuba3/presets/default.yaml \
  python renderer/mitsuba3/scripts/render_scene.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mitsuba as mi
import trimesh
import yaml


def _load_manifest() -> dict:
    path = os.environ.get("FAITHC_MANIFEST", "").strip()
    if not path:
        raise RuntimeError("FAITHC_MANIFEST is required")
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _load_preset() -> dict:
    preset_path = os.environ.get("FAITHC_PRESET_PATH", "").strip()
    if preset_path:
        p = Path(preset_path).resolve()
    else:
        p = Path("renderer/mitsuba3/presets/default.yaml").resolve()
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _pick_mesh(manifest: dict) -> Path:
    meshes = manifest.get("meshes", {})
    for key in ("low_uv", "low", "high"):
        candidate = meshes.get(key)
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if not path.exists():
            continue
        mesh = trimesh.load(path, force="mesh", process=False)
        if len(mesh.faces) > 0:
            return path
    raise RuntimeError("No valid mesh with faces found in manifest")


def _to_obj_if_needed(mesh_path: Path, output_dir: Path) -> Path:
    if mesh_path.suffix.lower() in {".obj", ".ply"}:
        return mesh_path
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    out = output_dir / "mitsuba_input.obj"
    mesh.export(out)
    return out


def main() -> int:
    manifest = _load_manifest()
    preset = _load_preset()

    output_dir = Path(os.environ.get("FAITHC_OUTPUT_DIR", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_name = manifest.get("sample_name", "faithc_sample")
    variant = os.environ.get("FAITHC_VARIANT", "cuda_ad_rgb")
    spp = max(1, int(os.environ.get("FAITHC_SPP", "64")))

    mesh_path = _to_obj_if_needed(_pick_mesh(manifest), output_dir)

    sensor = preset.get("sensor", {})
    render = preset.get("render", {})

    mi.set_variant(variant)

    scene_dict = {
        "type": "scene",
        "integrator": preset.get("integrator", {"type": "path", "max_depth": 8}),
        "sensor": {
            "type": "perspective",
            "fov": float(sensor.get("fov", 45.0)),
            "to_world": mi.ScalarTransform4f.look_at(
                origin=sensor.get("eye", [2.2, 1.8, 2.2]),
                target=sensor.get("target", [0.0, 0.0, 0.0]),
                up=sensor.get("up", [0.0, 1.0, 0.0]),
            ),
            "sampler": {"type": "independent"},
            "film": {
                "type": "hdrfilm",
                "width": int(render.get("width", 1280)),
                "height": int(render.get("height", 720)),
                "rfilter": {"type": "gaussian"},
                "pixel_format": "rgb",
                "component_format": "float32",
            },
        },
        "shape": {
            "type": "ply" if mesh_path.suffix.lower() == ".ply" else "obj",
            "filename": str(mesh_path),
            "bsdf": preset.get(
                "bsdf",
                {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.7, 0.7, 0.7]}},
            ),
        },
        "emitter": preset.get("emitter", {"type": "constant", "radiance": 1.0}),
    }

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=spp)

    img_path = output_dir / f"{sample_name}.png"
    mi.util.write_bitmap(str(img_path), image)

    (output_dir / "render_scene_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "variant": variant,
                "spp": spp,
                "mesh": str(mesh_path),
                "image": str(img_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Mitsuba3 render complete: {img_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
