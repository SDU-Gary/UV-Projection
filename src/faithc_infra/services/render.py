from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..types import RenderArtifact


class RenderService:
    def render(
        self,
        sample_name: str,
        manifest_path: Path,
        output_dir: Path,
        preset: str,
        backend: str,
        variant: str = "cuda_ad_rgb",
        samples_per_pixel: int = 64,
        preset_path: str = "",
    ) -> RenderArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "render.log.txt"

        if backend.lower() != "mitsuba3":
            reason = f"unsupported backend: {backend}. expected 'mitsuba3'"
            self._write_log(log_path, reason)
            return RenderArtifact(sample_name, output_dir, "skipped", log_path=log_path, reason=reason)

        try:
            result = self._render_mitsuba3(
                sample_name=sample_name,
                manifest_path=manifest_path,
                output_dir=output_dir,
                preset_name=preset,
                variant=variant,
                samples_per_pixel=samples_per_pixel,
                preset_path=preset_path,
            )

            metadata = {
                "status": "success",
                "reason": None,
                "backend": "mitsuba3",
                "variant": result["variant"],
                "samples_per_pixel": result["samples_per_pixel"],
                "mesh_used": result["mesh_used"],
                "image_path": result["image_path"],
                "preset_name": result["preset_name"],
                "preset_path": result.get("preset_path"),
            }
            with (output_dir / "render_status.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            self._write_log(
                log_path,
                (
                    "Mitsuba3 render finished successfully.\n"
                    f"sample={sample_name}\n"
                    f"variant={result['variant']}\n"
                    f"spp={result['samples_per_pixel']}\n"
                    f"mesh={result['mesh_used']}\n"
                    f"image={result['image_path']}\n"
                ),
            )
            return RenderArtifact(sample_name, output_dir, "success", log_path=log_path, reason=None)

        except Exception as exc:
            reason = str(exc)
            metadata = {
                "status": "failed",
                "reason": reason,
                "backend": "mitsuba3",
                "variant": variant,
            }
            with (output_dir / "render_status.json").open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            self._write_log(log_path, reason)
            return RenderArtifact(sample_name, output_dir, "failed", log_path=log_path, reason=reason)

    def _render_mitsuba3(
        self,
        sample_name: str,
        manifest_path: Path,
        output_dir: Path,
        preset_name: str,
        variant: str,
        samples_per_pixel: int,
        preset_path: str,
    ) -> Dict[str, Any]:
        import mitsuba as mi

        samples_per_pixel = max(1, int(samples_per_pixel))

        manifest = self._load_manifest(manifest_path)
        scene_mesh = self._pick_mesh_with_faces(manifest)
        mesh_for_mitsuba = self._ensure_mitsuba_mesh_format(scene_mesh, output_dir)

        resolved_preset_path = self._resolve_preset_path(preset_name, preset_path)
        preset_cfg = self._load_preset(resolved_preset_path)

        mi.set_variant(variant)

        sensor_cfg = preset_cfg.get("sensor", {})
        eye = sensor_cfg.get("eye", [2.2, 1.8, 2.2])
        target = sensor_cfg.get("target", [0.0, 0.0, 0.0])
        up = sensor_cfg.get("up", [0.0, 1.0, 0.0])
        fov = float(sensor_cfg.get("fov", 45.0))

        render_cfg = preset_cfg.get("render", {})
        width = int(render_cfg.get("width", 1280))
        height = int(render_cfg.get("height", 720))

        integrator_cfg = preset_cfg.get("integrator", {"type": "path", "max_depth": 8})
        emitter_cfg = preset_cfg.get("emitter", {"type": "constant", "radiance": 1.0})
        bsdf_cfg = self._normalize_bsdf(preset_cfg.get("bsdf", {"type": "diffuse", "reflectance": [0.7, 0.7, 0.7]}))

        scene_dict: Dict[str, Any] = {
            "type": "scene",
            "integrator": integrator_cfg,
            "sensor": {
                "type": "perspective",
                "fov": fov,
                "to_world": mi.ScalarTransform4f.look_at(origin=eye, target=target, up=up),
                "sampler": {"type": "independent"},
                "film": {
                    "type": "hdrfilm",
                    "width": width,
                    "height": height,
                    "rfilter": {"type": "gaussian"},
                    "pixel_format": "rgb",
                    "component_format": "float32",
                },
            },
            "shape": {
                "type": "ply" if mesh_for_mitsuba.suffix.lower() == ".ply" else "obj",
                "filename": str(mesh_for_mitsuba),
                "bsdf": bsdf_cfg,
            },
            "emitter": emitter_cfg,
        }

        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=samples_per_pixel)

        image_path = output_dir / f"{sample_name}.png"
        mi.util.write_bitmap(str(image_path), image)

        scene_json_path = output_dir / "mitsuba_scene.json"
        with scene_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._json_safe_scene_dict(scene_dict), handle, indent=2)

        return {
            "variant": variant,
            "samples_per_pixel": samples_per_pixel,
            "mesh_used": str(mesh_for_mitsuba),
            "image_path": str(image_path),
            "preset_name": preset_name,
            "preset_path": str(resolved_preset_path) if resolved_preset_path else None,
        }

    @staticmethod
    def _json_safe_scene_dict(scene_dict: Dict[str, Any]) -> Dict[str, Any]:
        data = json.loads(json.dumps(scene_dict, default=str))
        return data

    @staticmethod
    def _load_manifest(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _pick_mesh_with_faces(manifest: Dict[str, Any]) -> Path:
        import trimesh

        meshes = manifest.get("meshes", {})
        candidates = [meshes.get("low_uv"), meshes.get("low"), meshes.get("high")]

        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if not path.exists():
                errors.append(f"missing: {path}")
                continue
            try:
                mesh = trimesh.load(path, force="mesh", process=False)
                if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
                    return path.resolve()
                errors.append(f"no faces: {path}")
            except Exception as exc:
                errors.append(f"failed load {path}: {exc}")

        raise RuntimeError("No valid mesh with faces found for rendering. Details: " + "; ".join(errors))

    @staticmethod
    def _ensure_mitsuba_mesh_format(mesh_path: Path, output_dir: Path) -> Path:
        suffix = mesh_path.suffix.lower()
        if suffix in {".obj", ".ply"}:
            return mesh_path

        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
            raise RuntimeError(f"Mesh has no faces for Mitsuba3: {mesh_path}")

        converted = output_dir / "mitsuba_input.obj"
        mesh.export(converted)
        return converted

    @staticmethod
    def _resolve_preset_path(preset_name: str, preset_path: str) -> Optional[Path]:
        if preset_path:
            path = Path(preset_path)
            if path.exists():
                return path.resolve()

        default_yaml = Path("renderer/mitsuba3/presets") / f"{preset_name}.yaml"
        if default_yaml.exists():
            return default_yaml.resolve()

        default_json = Path("renderer/mitsuba3/presets") / f"{preset_name}.json"
        if default_json.exists():
            return default_json.resolve()

        return None

    @staticmethod
    def _load_preset(path: Optional[Path]) -> Dict[str, Any]:
        if path is None:
            return {}

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _normalize_bsdf(bsdf_cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(bsdf_cfg)
        reflectance = cfg.get("reflectance")
        if isinstance(reflectance, (list, tuple)) and len(reflectance) in {1, 3}:
            cfg["reflectance"] = {"type": "rgb", "value": list(reflectance)}
        return cfg

    @staticmethod
    def _write_log(path: Path, content: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
