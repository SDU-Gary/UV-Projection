from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


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
        },
        "uv": {
            "enabled": True,
            "method": "method2_gradient_poisson",
            "sample": {
                "base_per_face": 4,
                "min_per_face": 3,
                "max_per_face": 12,
                "seed": 12345,
            },
            "correspondence": {
                "normal_weight": 0.2,
                "normal_dot_min": 0.7,
                "ray_max_dist_ratio": 0.08,
                "fallback_k": 8,
                "fallback_weight": 0.7,
                "bvh_chunk_size": 200000,
            },
            "seam": {
                "strategy": "legacy",
                "uv_span_threshold": 0.35,
                "min_valid_samples_per_face": 2,
                "exclude_cross_seam_faces": True,
                "local_vertex_split": True,
                "high_position_eps": 1e-6,
                "high_uv_eps": 1e-5,
            },
            "iterative": {
                "enabled": True,
                "min_iters": 2,
                "max_iters": 4,
                "strict_mode_from_iter": 2,
                "label_change_tol": 0.02,
                "energy_rel_tol": 1e-3,
                "patience": 1,
                "unknown_face_policy": "exclude",
            },
            "solve": {
                "backend": "auto",
                "lambda_smooth": 2e-4,
                "pcg_max_iter": 2000,
                "pcg_tol": 1e-6,
                "pcg_check_every": 25,
                "pcg_preconditioner": "jacobi",
                "cg_max_iter": 2000,
                "cg_tol": 1e-6,
                "anchor_weight": 1e2,
                "ridge_eps": 1e-8,
            },
            "texture_weight": {
                "enabled": True,
                "grad_weight_gamma": 1.0,
                "max_weight": 5.0,
            },
            "method2": {
                "outlier_sigma": 4.0,
                "outlier_quantile": 0.95,
                "min_samples_per_face": 2,
                "face_weight_floor": 1e-6,
                "use_island_guard": False,
                "adaptive_anchor_enabled": True,
                "anchor_mode": "component_minimal",
                "anchor_points_per_component": 4,
                "anchor_min_points_per_component": 3,
                "anchor_max_points_per_component": 5,
                "anchor_target_vertices_per_anchor": 8000,
                "anchor_confidence_floor": 0.2,
                "anchor_confidence_power": 1.0,
                "anchor_boundary_boost": 0.5,
                "anchor_curvature_boost": 0.5,
                "hard_anchor_enabled": False,
                "hard_anchor_conf_min": 0.85,
                "hard_anchor_min_per_component": 2,
                "hard_anchor_max_per_component": 4,
                "irls_iters": 2,
                "huber_delta": 3.0,
                "post_align_translation": True,
                "post_align_min_samples": 64,
                "post_align_max_shift": 0.25,
                "adaptive_smooth_enabled": True,
                "adaptive_smooth_beta": 2.0,
                "adaptive_smooth_min_alpha": 0.25,
                "adaptive_smooth_max_alpha": 1.5,
                "laplacian_mode": "cotan",
                "system_cond_estimate": "diag_ratio",
                "solve_per_island": True,
            },
            "method4": {
                "enabled": True,
                "device": "auto",
                "optimizer": "lbfgs",
                "max_iters": 120,
                "lr": 0.25,
                "jacobian_weight": 1.0,
                "smooth_weight": 1e-6,
                "sym_dirichlet_weight": 2e-2,
                "logdet_barrier_weight": 1e-2,
                "flip_barrier_weight": 5e-2,
                "barrier_weight": 0.0,
                "anchor_weight": 5e-4,
                "det_eps": 1e-7,
                "det_softplus_beta": 40.0,
                "area_eps": 1e-10,
                "grad_clip": 5.0,
                "early_stop_rel_tol": 1e-5,
                "early_stop_patience": 10,
                "max_line_search_fail": 16,
                "line_search_alpha": 0.5,
                "line_search_c1": 1e-4,
                "patch_refine_rounds": 3,
                "patch_refine_steps": 80,
                "patch_refine_lr": 0.05,
                "pre_repair_enabled": True,
                "pre_repair_iters": 8,
                "pre_repair_step": 0.25,
                "fallback_to_method2_on_violation": True,
                "fallback_violation_ratio_tol": 0.02,
                "fallback_violation_count_tol": 8,
                "barrier_homotopy_enabled": True,
                "barrier_homotopy_warmup_iters": 40,
            },
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


class ConfigLoader:
    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        config = copy.deepcopy(DEFAULT_CONFIG)
        _deep_merge(config, raw)
        ConfigLoader._validate(config)
        return config

    @staticmethod
    def _validate(config: Dict[str, Any]) -> None:
        samples = config.get("data", {}).get("samples", [])
        if not isinstance(samples, list):
            raise ValueError("'data.samples' must be a list")
        for item in samples:
            if "high_mesh" not in item:
                raise ValueError("Each sample must provide 'high_mesh'")
