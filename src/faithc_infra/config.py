from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from .services.uv.options import DEFAULT_OPTIONS as DEFAULT_UV_OPTIONS

_ALLOWED_SAMPLE_KEYS = {
    "name",
    "high_mesh",
}


def _default_uv_pipeline_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "method": "method2_gradient_poisson",
        **copy.deepcopy(DEFAULT_UV_OPTIONS),
    }


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "faithc-homework",
        "run_prefix": "hw",
    },
    "paths": {
        "runs_dir": "experiments/runs",
    },
    "data": {
        "samples": [],
    },
    "pipeline": {
        "device": "cuda",
        "reconstruction": {
            "enabled": True,
            "backend": "pymeshlab_qem",
            "resolution": 8,
            "margin": 0.05,
            "tri_mode": "auto",
            "compute_flux": True,
            "clamp_anchors": True,
            "save_tokens": True,
            "retry_on_empty_mesh": True,
            "retry_resolutions": [16, 32, 64, 128, 256],
            "min_level": -1,
            "solver_weights": {
                "lambda_n": 1.0,
                "lambda_d": 1e-3,
                "weight_power": 1,
            },
            "pymeshlab": {
                "target_face_count": 0,
                "target_face_ratio": 0.05,
                "quality_threshold": 0.3,
                "preserve_boundary": False,
                "boundary_weight": 2.0,
                "preserve_normal": False,
                "preserve_topology": False,
                "optimal_placement": True,
                "planar_quadric": False,
                "planar_weight": 1e-3,
                "quality_weight": False,
                "autoclean": True,
            },
        },
        "uv": {
            **_default_uv_pipeline_config(),
        },
        "eval": {
            "enabled": True,
            "sample_points": 10000,
        },
        "render": {
            "enabled": False,
            "backend": "mitsuba3",
            "preset": "default",
            "variant": "cuda_ad_rgb",
            "samples_per_pixel": 64,
            "preset_path": "",
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _schema_from_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _schema_from_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return []
    return None


_CONFIG_SCHEMA: Dict[str, Any] = _schema_from_value(DEFAULT_CONFIG)


def _raise_unknown_keys(path: str, unknown_keys: list[str]) -> None:
    joined = ", ".join(sorted(unknown_keys))
    where = path or "<root>"
    raise ValueError(f"Unknown config key(s) at '{where}': {joined}")


def _validate_known_keys(raw: Any, schema: Any, path: str) -> None:
    if not isinstance(raw, dict):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"Config section '{path}' must not contain nested object values")

    unknown = [str(k) for k in raw.keys() if k not in schema]
    if unknown:
        _raise_unknown_keys(path, unknown)

    for key, value in raw.items():
        next_path = f"{path}.{key}" if path else str(key)
        next_schema = schema.get(key)
        if next_path == "data.samples":
            if not isinstance(value, list):
                raise ValueError("'data.samples' must be a list")
            for idx, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(f"'data.samples[{idx}]' must be an object")
                unknown_item = [str(k) for k in item.keys() if k not in _ALLOWED_SAMPLE_KEYS]
                if unknown_item:
                    _raise_unknown_keys(f"data.samples[{idx}]", unknown_item)
            continue
        _validate_known_keys(value, next_schema, next_path)


class ConfigLoader:
    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config root must be a mapping/object")
        _validate_known_keys(raw, _CONFIG_SCHEMA, "")

        config = copy.deepcopy(DEFAULT_CONFIG)
        _deep_merge(config, raw)
        ConfigLoader._validate(config)
        return config

    @staticmethod
    def _validate(config: Dict[str, Any]) -> None:
        for key in ("project", "paths", "data", "pipeline"):
            if not isinstance(config.get(key), dict):
                raise ValueError(f"'{key}' must be an object")

        pipeline = config.get("pipeline", {})
        for key in ("reconstruction", "uv", "eval", "render"):
            if not isinstance(pipeline.get(key), dict):
                raise ValueError(f"'pipeline.{key}' must be an object")

        samples = config.get("data", {}).get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("'data.samples' must be a list")
        for idx, item in enumerate(samples):
            if not isinstance(item, dict):
                raise ValueError(f"'data.samples[{idx}]' must be an object")
            if "high_mesh" not in item:
                raise ValueError(f"'data.samples[{idx}]' must provide 'high_mesh'")
