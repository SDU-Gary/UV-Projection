from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

METHOD_ALIASES = {
    "nearest": "nearest_vertex",
    "nearest_vertex": "nearest_vertex",
    "barycentric": "barycentric_closest_point",
    "barycentric_closest_point": "barycentric_closest_point",
    "hybrid": "hybrid_global_opt",
    "hybrid_global_opt": "hybrid_global_opt",
    "method2": "method2_gradient_poisson",
    "gradient_poisson": "method2_gradient_poisson",
    "method2_gradient_poisson": "method2_gradient_poisson",
    "method2p": "method2p_projected_gradient_poisson",
    "method2p_projected_gradient_poisson": "method2p_projected_gradient_poisson",
    "method25": "method25_projected_jacobian_injective",
    "method25_projected_jacobian_injective": "method25_projected_jacobian_injective",
    "method4": "method4_jacobian_injective",
    "jacobian_injective": "method4_jacobian_injective",
    "method4_jacobian_injective": "method4_jacobian_injective",
    # Keep auto mapped to Method2 as the maintained default path.
    "auto": "method2_gradient_poisson",
}

DEFAULT_OPTIONS: Dict[str, Any] = {
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
    "solve": {
        "backend": "auto",
        "lambda_smooth": 2e-4,
        "pcg_max_iter": 2000,
        "pcg_tol": 1e-6,
        "pcg_check_every": 25,
        "pcg_preconditioner": "jacobi",
        # Legacy aliases kept for backward compatibility.
        "cg_max_iter": 2000,
        "cg_tol": 1e-6,
        "anchor_weight": 1e2,
        "ridge_eps": 1e-8,
        "constraint_mode": "none",
        "constraint_device": "cpu",
        "constraint_box_weight": 10.0,
        "constraint_box_margin": 0.0,
        "constraint_refine_iters": 80,
        "constraint_refine_lr": 0.05,
        "constraint_grad_clip": 5.0,
        "constraint_early_stop_rel_tol": 1e-5,
        "constraint_early_stop_patience": 10,
    },
    "seam": {
        "strategy": "legacy",
        "uv_span_threshold": 0.35,
        "min_valid_samples_per_face": 2,
        "exclude_cross_seam_faces": True,
        "local_vertex_split": True,
        "high_position_eps": 1e-6,
        "high_uv_eps": 1e-5,
        # halfedge_island pipeline controls (Mesh sanitization + semantic transfer).
        "sanitize_enabled": True,
        "sanitize_area_eps": 1e-12,
        # Semantic transfer mode:
        # - single_point_projection: legacy single centroid raycast
        # - four_point_bfs: 4-point supersampling + topology-guided BFS growth
        "transfer_sampling_mode": "four_point_soft_flood",
        "transfer_max_dist_ratio": 0.005,
        "transfer_four_point_barycentric": [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ],
        "transfer_max_normal_angle_deg": 30.0,
        "transfer_soft_seed_prob_min": 0.95,
        "transfer_soft_seed_margin_min": 0.25,
        "transfer_soft_unary_eps": 1e-4,
        "transfer_soft_other_label_penalty": 2.5,
        "transfer_soft_unknown_label_penalty": 1.25,
        "transfer_soft_smoothness_weight": 0.35,
        "transfer_soft_icm_iters": 2,
        "transfer_soft_prefer_main_shells": True,
        "transfer_soft_micro_shell_penalty": 0.75,
        "transfer_fill_unknown_iters": 2,
        "transfer_majority_vote_iters": 1,
        "transfer_morph_close_iters": 2,
        "component_merge_enabled": True,
        "component_merge_min_faces": 4,
        "component_merge_max_iters": 8,
        "main_shell_min_faces": 16,
        "micro_shell_absorb_enabled": True,
        "micro_shell_absorb_max_iters": 8,
        "semantic_summary_main_ratio_threshold": 0.95,
        "semantic_summary_tiny_abs_threshold": 16,
        "semantic_summary_tiny_ratio_threshold": 0.005,
        "semantic_summary_tiny_max_components": 2,
        "include_boundary_as_seam": False,
        "validation_mode": "hard",
        "validation_strict": True,
        "validation_require_closed_loops": True,
        "validation_require_pure_components": True,
        # Ignore tiny disconnected low-mesh components during seam validation.
        # This helps FaithC outputs that contain many micro floating pieces.
        "validation_ignore_small_components_faces": 32,
        # Allow seam components that terminate at physical mesh boundaries.
        "validation_allow_open_on_boundary": True,
        # Compatibility-only keys below are retained to avoid breaking old configs.
        # Some legacy routing keys are no longer used by the maintained pipeline.
        "routing_mode": "point_cloud_mask",
        # Legacy routing compatibility params (ignored).
        "routing_weight_dist": 3.0,
        "routing_weight_align": 1.5,
        "routing_weight_length": 1.0,
        "routing_weight_dihedral": 0.25,
        "routing_knn_segments": 8,
        # Per-chain local attraction (on top of global seam attraction) to avoid
        # multiple high-seam chains collapsing onto identical low edges.
        "routing_chain_local_scale": 1.5,
        "routing_chain_weight_dist": 3.0,
        "routing_chain_weight_align": 1.5,
        "routing_chain_knn_segments": 8,
        # Penalize reusing routed low edges across chains.
        "routing_edge_reuse_penalty": 2.0,
        "routing_edge_reuse_power": 1.0,
        "routing_path_outlier_ratio": 5.0,
        "routing_anchor_spacing_ratio": 8.0,
        "routing_min_anchors": 8,
        "routing_max_anchors": 128,
        "routing_island_confidence_min": 0.55,
        # Legacy point-cloud mask compatibility params (ignored).
        "mask_seed_knn_edges": 2,
        "mask_seed_sigma_ratio": 1.0,
        "mask_seed_max_dist_ratio": 0.6,
        "mask_seed_min_support": 1,
        "mask_closing_rounds": 2,
        "mask_closing_dist_ratio": 1.25,
        "mask_closing_max_add_ratio": 0.25,
        "mask_band_outside_multiplier": 100.0,
        "mask_min_component_edges": 1,
        "mask_cost_seed_bonus": 2.0,
        "mask_cost_inactive_penalty": 50.0,
        "mask_skeleton_seed_bonus": 4.0,
        "mask_skeleton_anchor_use_junctions": True,
        "mask_skeleton_anchor_spacing_ratio": 8.0,
        "mask_skeleton_min_anchors": 8,
        "mask_skeleton_max_anchors": 128,
        "mask_skeleton_fallback_to_band": True,
        # Guard correspondence against cross-island matches before global UV solve.
        "uv_island_guard_enabled": True,
        "uv_island_guard_mode": "soft",
        "uv_island_guard_confidence_min": 0.55,
        "uv_island_guard_allow_unknown": False,
        "uv_island_guard_fallback": "nearest_same_island_then_udf",
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
        "emit_face_sample_counts": False,
        "solve_per_island": True,
        "perf_fast_island_cache": True,
        "perf_fast_agg_vectorized": True,
        "perf_fast_small_group_threshold": 6,
        "perf_fast_small_group_skip_irls": True,
        "perf_fast_small_group_skip_outlier": True,
        "perf_fast_cut_edge_count": True,
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
        "recovery_mode_enabled": False,
        "recovery_det_improve_eps": 1e-8,
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
    "method25": {
        "samplefit_min_samples": 3,
        "strict_gate": False,
        "lambda_decay": 1.0,
        "lambda_curl": 20.0,
        "ridge_eps": 1e-8,
    },
}


@dataclass(frozen=True)
class SeamValidationSettings:
    mode: str
    strict: bool
    require_closed_loops: bool
    require_pure_components: bool
    allow_open_on_boundary: bool
    min_component_faces: int


def resolve_seam_validation_settings(seam_cfg: Dict[str, Any]) -> SeamValidationSettings:
    raw_mode = str(seam_cfg.get("validation_mode", "")).strip().lower()
    if raw_mode not in {"hard", "diagnostic"}:
        raw_mode = ""

    strict = bool(seam_cfg.get("validation_strict", True))
    require_closed_loops = bool(seam_cfg.get("validation_require_closed_loops", True))
    require_pure_components = bool(seam_cfg.get("validation_require_pure_components", True))

    if raw_mode == "diagnostic":
        strict = False
    elif raw_mode == "hard":
        strict = True

    mode = raw_mode or ("hard" if strict else "diagnostic")
    return SeamValidationSettings(
        mode=mode,
        strict=bool(strict),
        require_closed_loops=bool(require_closed_loops),
        require_pure_components=bool(require_pure_components),
        allow_open_on_boundary=bool(seam_cfg.get("validation_allow_open_on_boundary", True)),
        min_component_faces=int(max(0, seam_cfg.get("validation_ignore_small_components_faces", 32))),
    )


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    stack = [(out, override)]
    while stack:
        dst, src = stack.pop()
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                stack.append((dst[key], value))
            else:
                dst[key] = value
    return out
