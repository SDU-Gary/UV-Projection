#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import trimesh

# Ensure preview bridge uses repository source first (avoid stale site-packages).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    src_str = str(SRC_ROOT)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from faithc_infra.profiler import ExecutionProfiler, ProfilerConfig
from faithc_infra.services.atom3d_runtime import ensure_atom3d_cuda_runtime, merge_runtime_diag
from faithc_infra.services.decimation import decimate_with_pymeshlab_qem
from faithc_infra.services.uv.options import DEFAULT_OPTIONS, METHOD_ALIASES, deep_merge_dict
from faithc_infra.services.uv_projector import UVProjector
from faithc_infra.services.uv.closure_validation import run_uv_closure_validation
from faithc_infra.services.uv.texture_io import resolve_device


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(payload), handle, indent=2, allow_nan=False)


def _write_method2_face_samples_sidecar(status_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    accepted_raw = payload.get("uv_m2_face_accepted_samples", None)
    total_raw = payload.get("uv_m2_face_total_samples", None)
    if accepted_raw is None:
        return {}

    accepted = np.asarray(accepted_raw, dtype=np.int32).reshape(-1)
    if accepted.size == 0:
        return {}
    total = None
    if total_raw is not None:
        total_np = np.asarray(total_raw, dtype=np.int32).reshape(-1)
        if total_np.shape[0] == accepted.shape[0]:
            total = total_np

    sidecar = status_path.parent / f"{status_path.stem}.m2_face_samples.json"
    sidecar_payload: Dict[str, Any] = {
        "accepted": accepted.astype(int).tolist(),
    }
    if total is not None:
        sidecar_payload["total"] = total.astype(int).tolist()
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(sidecar_payload, separators=(",", ":")), encoding="utf-8")

    nonzero = int(np.count_nonzero(accepted))
    vmax = int(np.max(accepted)) if accepted.size > 0 else 0
    return {
        "uv_m2_face_sample_counts_path": str(sidecar),
        "uv_m2_face_sample_faces": int(accepted.shape[0]),
        "uv_m2_face_sample_nonzero": nonzero,
        "uv_m2_face_sample_max": vmax,
    }


def _write_uv_semantic_transfer_sidecar(status_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Keep post-BFS labels for closure sidecar writer; it will pop them later.
    post_labels_raw = payload.get("uv_low_face_semantic_labels", None)

    # These arrays are consumed by semantic sidecar only.
    post_conf_raw = payload.get("uv_low_face_semantic_confidence", None)
    post_conflict_raw = payload.get("uv_low_face_semantic_conflict", None)
    pre_labels_raw = payload.get("uv_low_face_semantic_pre_bfs_labels", None)
    pre_conf_raw = payload.get("uv_low_face_semantic_pre_bfs_confidence", None)
    pre_state_raw = payload.get("uv_low_face_semantic_pre_bfs_state", None)
    pre_cleanup_labels_raw = payload.get("uv_low_face_semantic_pre_cleanup_labels", None)
    pre_cleanup_conf_raw = payload.get("uv_low_face_semantic_pre_cleanup_confidence", None)
    pre_cleanup_conflict_raw = payload.get("uv_low_face_semantic_pre_cleanup_conflict", None)
    soft_top1_label_raw = payload.get("uv_low_face_semantic_soft_top1_label", None)
    soft_top1_prob_raw = payload.get("uv_low_face_semantic_soft_top1_prob", None)
    soft_top2_label_raw = payload.get("uv_low_face_semantic_soft_top2_label", None)
    soft_top2_prob_raw = payload.get("uv_low_face_semantic_soft_top2_prob", None)
    soft_entropy_raw = payload.get("uv_low_face_semantic_soft_entropy", None)
    soft_candidate_count_raw = payload.get("uv_low_face_semantic_soft_candidate_count", None)

    if (
        post_labels_raw is None
        and post_conf_raw is None
        and post_conflict_raw is None
        and pre_labels_raw is None
        and pre_conf_raw is None
        and pre_state_raw is None
        and pre_cleanup_labels_raw is None
        and pre_cleanup_conf_raw is None
        and pre_cleanup_conflict_raw is None
        and soft_top1_label_raw is None
        and soft_top1_prob_raw is None
        and soft_top2_label_raw is None
        and soft_top2_prob_raw is None
        and soft_entropy_raw is None
        and soft_candidate_count_raw is None
    ):
        return {}

    def _as_i64(raw):
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.int64).reshape(-1)

    def _as_f32(raw):
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.float32).reshape(-1)

    def _as_bool(raw):
        if raw is None:
            return None
        return np.asarray(raw, dtype=np.bool_).reshape(-1)

    post_labels = _as_i64(post_labels_raw)
    post_conf = _as_f32(post_conf_raw)
    post_conflict = _as_bool(post_conflict_raw)
    pre_labels = _as_i64(pre_labels_raw)
    pre_conf = _as_f32(pre_conf_raw)
    pre_state = _as_i64(pre_state_raw)
    pre_cleanup_labels = _as_i64(pre_cleanup_labels_raw)
    pre_cleanup_conf = _as_f32(pre_cleanup_conf_raw)
    pre_cleanup_conflict = _as_bool(pre_cleanup_conflict_raw)
    soft_top1_label = _as_i64(soft_top1_label_raw)
    soft_top1_prob = _as_f32(soft_top1_prob_raw)
    soft_top2_label = _as_i64(soft_top2_label_raw)
    soft_top2_prob = _as_f32(soft_top2_prob_raw)
    soft_entropy = _as_f32(soft_entropy_raw)
    soft_candidate_count = _as_i64(soft_candidate_count_raw)

    candidates = [
        arr
        for arr in [
            post_labels,
            post_conf,
            post_conflict,
            pre_labels,
            pre_conf,
            pre_state,
            pre_cleanup_labels,
            pre_cleanup_conf,
            pre_cleanup_conflict,
            soft_top1_label,
            soft_top1_prob,
            soft_top2_label,
            soft_top2_prob,
            soft_entropy,
            soft_candidate_count,
        ]
        if arr is not None
    ]
    if len(candidates) == 0:
        return {}
    face_count = int(candidates[0].shape[0])

    def _match_or_none(arr):
        if arr is None:
            return None
        return arr if int(arr.shape[0]) == face_count else None

    post_labels = _match_or_none(post_labels)
    post_conf = _match_or_none(post_conf)
    post_conflict = _match_or_none(post_conflict)
    pre_labels = _match_or_none(pre_labels)
    pre_conf = _match_or_none(pre_conf)
    pre_state = _match_or_none(pre_state)
    pre_cleanup_labels = _match_or_none(pre_cleanup_labels)
    pre_cleanup_conf = _match_or_none(pre_cleanup_conf)
    pre_cleanup_conflict = _match_or_none(pre_cleanup_conflict)
    soft_top1_label = _match_or_none(soft_top1_label)
    soft_top1_prob = _match_or_none(soft_top1_prob)
    soft_top2_label = _match_or_none(soft_top2_label)
    soft_top2_prob = _match_or_none(soft_top2_prob)
    soft_entropy = _match_or_none(soft_entropy)
    soft_candidate_count = _match_or_none(soft_candidate_count)

    sidecar = status_path.parent / f"{status_path.stem}.uv_semantic_transfer.json"
    sidecar_payload: Dict[str, Any] = {
        "summary": {
            "face_count": int(face_count),
            "has_post_bfs_labels": bool(post_labels is not None),
            "has_pre_bfs_confidence": bool(pre_conf is not None),
            "has_pre_cleanup_labels": bool(pre_cleanup_labels is not None),
            "has_soft_evidence": bool(soft_top1_prob is not None),
        }
    }
    if post_labels is not None:
        sidecar_payload["post_bfs_labels"] = post_labels.astype(int).tolist()
        sidecar_payload["summary"]["post_bfs_unknown_faces"] = int(np.count_nonzero(post_labels < 0))
        sidecar_payload["summary"]["post_bfs_unique_labels"] = int(np.unique(post_labels[post_labels >= 0]).size)
    if post_conf is not None:
        sidecar_payload["post_bfs_confidence"] = post_conf.astype(float).tolist()
    if post_conflict is not None:
        sidecar_payload["post_bfs_conflict"] = post_conflict.astype(bool).tolist()
        sidecar_payload["summary"]["post_bfs_conflict_faces"] = int(np.count_nonzero(post_conflict))
    if pre_labels is not None:
        sidecar_payload["pre_bfs_labels"] = pre_labels.astype(int).tolist()
    if pre_conf is not None:
        sidecar_payload["pre_bfs_confidence"] = pre_conf.astype(float).tolist()
        sidecar_payload["summary"]["pre_bfs_confidence_mean"] = float(np.mean(pre_conf))
    if pre_state is not None:
        ps = pre_state.astype(np.int64, copy=False)
        sidecar_payload["pre_bfs_state"] = ps.astype(int).tolist()
        sidecar_payload["summary"]["pre_bfs_strong_faces"] = int(np.count_nonzero(ps == 2))
        sidecar_payload["summary"]["pre_bfs_boundary_faces"] = int(np.count_nonzero(ps == 1))
        sidecar_payload["summary"]["pre_bfs_unknown_faces"] = int(np.count_nonzero(ps == 0))
    if pre_cleanup_labels is not None:
        sidecar_payload["pre_cleanup_labels"] = pre_cleanup_labels.astype(int).tolist()
        sidecar_payload["summary"]["pre_cleanup_unknown_faces"] = int(np.count_nonzero(pre_cleanup_labels < 0))
        sidecar_payload["summary"]["pre_cleanup_unique_labels"] = int(
            np.unique(pre_cleanup_labels[pre_cleanup_labels >= 0]).size
        )
    if pre_cleanup_conf is not None:
        sidecar_payload["pre_cleanup_confidence"] = pre_cleanup_conf.astype(float).tolist()
        sidecar_payload["summary"]["pre_cleanup_confidence_mean"] = float(np.mean(pre_cleanup_conf))
    if pre_cleanup_conflict is not None:
        sidecar_payload["pre_cleanup_conflict"] = pre_cleanup_conflict.astype(bool).tolist()
        sidecar_payload["summary"]["pre_cleanup_conflict_faces"] = int(np.count_nonzero(pre_cleanup_conflict))
    if soft_top1_label is not None:
        sidecar_payload["soft_top1_label"] = soft_top1_label.astype(int).tolist()
    if soft_top1_prob is not None:
        sidecar_payload["soft_top1_prob"] = soft_top1_prob.astype(float).tolist()
        sidecar_payload["summary"]["soft_top1_prob_mean"] = float(np.mean(soft_top1_prob))
    if soft_top2_label is not None:
        sidecar_payload["soft_top2_label"] = soft_top2_label.astype(int).tolist()
    if soft_top2_prob is not None:
        sidecar_payload["soft_top2_prob"] = soft_top2_prob.astype(float).tolist()
    if soft_entropy is not None:
        sidecar_payload["soft_entropy"] = soft_entropy.astype(float).tolist()
        sidecar_payload["summary"]["soft_entropy_mean"] = float(np.mean(soft_entropy))
    if soft_candidate_count is not None:
        sidecar_payload["soft_candidate_count"] = soft_candidate_count.astype(int).tolist()
        sidecar_payload["summary"]["soft_candidate_count_mean"] = float(np.mean(soft_candidate_count))

    pipeline_meta_keys = [
        "uv_semantic_transfer_mode",
        "uv_semantic_transfer_component_merge_enabled",
        "uv_semantic_transfer_component_merge_min_faces",
        "uv_semantic_transfer_component_merge_max_iters",
        "uv_semantic_transfer_component_merge_changed",
        "uv_semantic_transfer_component_merge_merged_components",
        "uv_semantic_transfer_component_merge_merged_faces",
        "uv_semantic_transfer_component_merge_iterations",
        "uv_semantic_transfer_component_merge_remaining_small_components",
        "uv_semantic_transfer_component_merge_remaining_small_faces",
        "uv_semantic_transfer_pre_bfs_fragmented_label_count",
        "uv_semantic_transfer_pre_bfs_severe_label_count",
        "uv_semantic_transfer_pre_cleanup_fragmented_label_count",
        "uv_semantic_transfer_pre_cleanup_severe_label_count",
        "uv_semantic_transfer_final_fragmented_label_count",
        "uv_semantic_transfer_final_severe_label_count",
    ]
    pipeline_meta = {k: payload[k] for k in pipeline_meta_keys if k in payload}
    if pipeline_meta:
        sidecar_payload["pipeline_meta"] = pipeline_meta

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(sidecar_payload, separators=(",", ":")), encoding="utf-8")
    return {
        "uv_semantic_transfer_sidecar_path": str(sidecar),
        "uv_semantic_transfer_sidecar_faces": int(face_count),
        "uv_semantic_transfer_sidecar_has_pre_bfs_confidence": bool(pre_conf is not None),
        "uv_semantic_transfer_sidecar_has_post_bfs_labels": bool(post_labels is not None),
        "uv_semantic_transfer_sidecar_has_pre_cleanup_labels": bool(pre_cleanup_labels is not None),
        "uv_semantic_transfer_sidecar_has_soft_evidence": bool(soft_top1_prob is not None),
    }


def _write_uv_closure_validation_sidecar(
    *,
    status_path: Path,
    payload: Dict[str, Any],
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    uv_options: Dict[str, Any],
) -> Dict[str, Any]:
    low_labels_raw = payload.get("uv_low_face_semantic_labels", None)
    low_seam_edges_raw = payload.get("uv_low_seam_edges", None)

    low_face_labels = None
    if low_labels_raw is not None:
        low_face_labels = np.asarray(low_labels_raw, dtype=np.int64).reshape(-1)
    low_seam_edges = None
    if low_seam_edges_raw is not None:
        low_seam_edges = np.asarray(low_seam_edges_raw, dtype=np.int64).reshape(-1, 2)

    seam_cfg = uv_options.get("seam", {})
    out_png = status_path.parent / f"{status_path.stem}.uv_closure_validation.png"
    out_sidecar = status_path.parent / f"{status_path.stem}.uv_closure_validation.json"

    try:
        result = run_uv_closure_validation(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            low_face_labels=low_face_labels,
            low_seam_edges=low_seam_edges,
            high_position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
            high_uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
            output_png=out_png,
            overlap_raster_res=int(seam_cfg.get("validation_overlap_raster_res", 256)),
        )
    except Exception as exc:
        return {
            "uv_closure_validation_error": str(exc),
        }

    sidecar_payload: Dict[str, Any] = {
        "semantic_labels": np.asarray(result.low_face_labels, dtype=np.int64).astype(int).tolist(),
        "seam_edges": np.asarray(result.low_seam_edges, dtype=np.int64).astype(int).tolist(),
        "summary": dict(result.metrics),
        "uv_validation_png": result.image_path,
        "uv_validation_png_error": result.image_error,
    }
    out_sidecar.parent.mkdir(parents=True, exist_ok=True)
    out_sidecar.write_text(json.dumps(sidecar_payload, separators=(",", ":")), encoding="utf-8")

    summary = result.metrics
    return {
        "uv_closure_validation_sidecar_path": str(out_sidecar),
        "uv_closure_validation_png_path": str(result.image_path) if result.image_path else "",
        "uv_closure_validation_png_error": result.image_error or "",
        "uv_closure_partition_has_leakage": bool(summary.get("partition_has_leakage", False)),
        "uv_closure_partition_mixed_components": int(summary.get("partition_mixed_components", -1)),
        "uv_closure_partition_label_split_count": int(summary.get("partition_label_split_count", -1)),
        "uv_closure_seam_topology_valid": bool(summary.get("seam_topology_valid", False)),
        "uv_closure_seam_components": int(summary.get("seam_components", -1)),
        "uv_closure_seam_loops_closed": int(summary.get("seam_loops_closed", -1)),
        "uv_closure_seam_components_open": int(summary.get("seam_components_open", -1)),
        "uv_closure_low_island_count": int(summary.get("low_island_count", -1)),
        "uv_closure_high_island_count": int(summary.get("high_island_count", -1)),
        "uv_closure_semantic_unknown_faces": int(summary.get("semantic_unknown_faces", -1)),
        "uv_closure_uv_bbox_iou_mean": float(summary.get("uv_bbox_iou_mean", 0.0)),
        "uv_closure_uv_overlap_ratio": float(summary.get("uv_overlap_ratio", 0.0)),
        "uv_closure_uv_stretch_p95": float(summary.get("uv_stretch_p95", 0.0)),
        "uv_closure_uv_stretch_p99": float(summary.get("uv_stretch_p99", 0.0)),
    }


def _strip_large_uv_sidecar_fields(payload: Dict[str, Any]) -> None:
    for key in [
        "uv_m2_face_accepted_samples",
        "uv_m2_face_total_samples",
        "uv_low_face_semantic_labels",
        "uv_low_face_semantic_confidence",
        "uv_low_face_semantic_conflict",
        "uv_low_face_semantic_pre_bfs_labels",
        "uv_low_face_semantic_pre_bfs_confidence",
        "uv_low_face_semantic_pre_bfs_state",
        "uv_low_face_semantic_pre_cleanup_labels",
        "uv_low_face_semantic_pre_cleanup_confidence",
        "uv_low_face_semantic_pre_cleanup_conflict",
        "uv_low_face_semantic_soft_top1_label",
        "uv_low_face_semantic_soft_top1_prob",
        "uv_low_face_semantic_soft_top2_label",
        "uv_low_face_semantic_soft_top2_prob",
        "uv_low_face_semantic_soft_entropy",
        "uv_low_face_semantic_soft_candidate_count",
        "uv_low_seam_edges",
    ]:
        payload.pop(key, None)


def _build_preview_uv_options(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "sample": {
            "base_per_face": int(args.uv_sample_base_per_face),
            "min_per_face": int(args.uv_sample_min_per_face),
            "max_per_face": int(args.uv_sample_max_per_face),
            "seed": int(args.uv_sample_seed),
        },
        "correspondence": {
            "normal_weight": float(args.uv_normal_weight),
            "normal_dot_min": float(args.uv_normal_dot_min),
            "ray_max_dist_ratio": float(args.uv_ray_max_dist_ratio),
            "fallback_k": int(args.uv_fallback_k),
            "fallback_weight": float(args.uv_fallback_weight),
            "bvh_chunk_size": int(args.uv_batch_size),
        },
        "seam": {
            "strategy": str(args.uv_seam_strategy),
            "uv_span_threshold": float(args.uv_seam_uv_span_threshold),
            "min_valid_samples_per_face": int(args.uv_seam_min_valid_samples),
            "exclude_cross_seam_faces": bool(args.uv_exclude_cross_seam_faces),
            "emit_validation_sidecar_data": True,
            "validation_overlap_raster_res": 128,
            "validation_mode": "hard" if bool(args.uv_seam_validation_strict) else "diagnostic",
            "validation_strict": bool(args.uv_seam_validation_strict),
            "validation_require_closed_loops": bool(args.uv_seam_validation_require_closed_loops),
            "validation_require_pure_components": bool(args.uv_seam_validation_require_pure_components),
            "validation_allow_open_on_boundary": bool(args.uv_seam_validation_allow_open_on_boundary),
            "validation_ignore_small_components_faces": int(args.uv_seam_validation_ignore_small_components_faces),
            "local_vertex_split": bool(args.uv_local_vertex_split),
            "uv_island_guard_enabled": bool(args.uv_island_guard),
            "uv_island_guard_mode": str(args.uv_island_guard_mode),
            "uv_island_guard_confidence_min": float(args.uv_island_guard_confidence_min),
            "uv_island_guard_allow_unknown": bool(args.uv_island_guard_allow_unknown),
            "uv_island_guard_fallback": str(args.uv_island_guard_fallback),
        },
        "solve": {
            "backend": str(args.uv_solve_backend),
            "lambda_smooth": float(args.uv_lambda_smooth),
            "pcg_max_iter": int(args.uv_pcg_max_iter),
            "pcg_tol": float(args.uv_pcg_tol),
            "pcg_check_every": int(args.uv_pcg_check_every),
            "pcg_preconditioner": str(args.uv_pcg_preconditioner),
            "cg_max_iter": int(args.uv_cg_max_iter),
            "cg_tol": float(args.uv_cg_tol),
            "anchor_weight": float(args.uv_anchor_weight),
            "ridge_eps": float(args.uv_ridge_eps),
        },
        "texture_weight": {
            "enabled": bool(args.uv_texture_weight),
            "grad_weight_gamma": float(args.uv_grad_weight_gamma),
            "max_weight": float(args.uv_max_texture_weight),
        },
        "iterative": {
            "enabled": bool(args.uv_iterative),
        },
        "method2": {
            "outlier_sigma": float(args.uv_m2_outlier_sigma),
            "outlier_quantile": float(args.uv_m2_outlier_quantile),
            "min_samples_per_face": int(args.uv_m2_min_samples_per_face),
            "face_weight_floor": float(args.uv_m2_face_weight_floor),
            "anchor_mode": str(args.uv_m2_anchor_mode),
            "anchor_points_per_component": int(args.uv_m2_anchor_points_per_component),
            "use_island_guard": bool(args.uv_m2_use_island_guard),
            "irls_iters": int(args.uv_m2_irls_iters),
            "huber_delta": float(args.uv_m2_huber_delta),
            "post_align_translation": bool(args.uv_m2_post_align),
            "post_align_min_samples": int(args.uv_m2_post_align_min_samples),
            "post_align_max_shift": float(args.uv_m2_post_align_max_shift),
            "laplacian_mode": str(args.uv_m2_laplacian_mode),
            "system_cond_estimate": str(args.uv_m2_system_cond_estimate),
            "emit_face_sample_counts": True,
        },
        "method4": {
            "recovery_mode_enabled": bool(args.uv_m4_recovery_mode),
            "recovery_det_improve_eps": float(args.uv_m4_recovery_det_improve_eps),
        },
    }
    return deep_merge_dict(DEFAULT_OPTIONS, overrides)


def _profile_step(profiler: ExecutionProfiler | None, name: str):
    return profiler.step(name) if profiler is not None else nullcontext()


def _write_perf_sidecars(
    *,
    profiler: ExecutionProfiler,
    status_path: Path,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    report = profiler.stop(extra=extra)
    json_path = status_path.parent / f"{status_path.stem}.perf.json"
    txt_path = status_path.parent / f"{status_path.stem}.perf.txt"
    profiler.write_reports(json_path=json_path, text_path=txt_path, report=report)
    return {
        "perf_profile_json": str(json_path),
        "perf_profile_txt": str(txt_path),
        "perf_wall_time_seconds": report.get("wall_time_seconds"),
        "perf_cpu_time_seconds": report.get("cpu_time_seconds"),
        "perf_stage_top": report.get("stage_summary", [])[:10],
    }


def _normalize_mesh(mesh: trimesh.Trimesh, margin: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    mesh.vertices -= center

    target_half_size = 1.0 - margin
    current_half_size = np.abs(mesh.vertices).max()
    if current_half_size > 1e-8:
        mesh.vertices *= target_half_size / current_half_size
    return mesh


def _load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")

    mesh_or_scene = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    elif isinstance(mesh_or_scene, trimesh.Scene):
        geoms = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangle geometry in scene: {path}")
        mesh = trimesh.util.concatenate(geoms)
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")

    valid_faces = mesh.nondegenerate_faces()
    if valid_faces.sum() < len(mesh.faces):
        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no valid faces after cleanup: {path}")

    return mesh


def _validate_runtime(device: str) -> Dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "FaithC preview bridge requires CUDA runtime. "
            f"Resolved device='{device}', torch.cuda.is_available()={torch.cuda.is_available()}."
        )
    return ensure_atom3d_cuda_runtime(device, strict=True, require_cuda=True)


def run_pipeline(
    args: argparse.Namespace,
    *,
    profiler: ExecutionProfiler | None = None,
    status_path: Path | None = None,
) -> Dict[str, Any]:
    start_total = time.time()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    if args.resolution < 2 or (args.resolution & (args.resolution - 1)) != 0:
        raise ValueError(f"resolution must be power of two and >= 2, got {args.resolution}")

    with _profile_step(profiler, "preview:load_mesh"):
        source_mesh = _load_mesh(in_path)
    with _profile_step(profiler, "preview:normalize_mesh"):
        mesh = _normalize_mesh(source_mesh, margin=float(args.margin))

    num_input_faces = int(len(mesh.faces))
    device = resolve_device(args.device)
    reconstruction_backend_requested = str(args.lowpoly_backend).strip().lower()
    reconstruction_backend_used = reconstruction_backend_requested
    kernel_diag: Dict[str, Any] | None = None
    kernel_setup_seconds = 0.0
    cuda_diag: Dict[str, Any] = {}
    encode_seconds = 0.0
    decode_seconds = 0.0
    active_voxels = 0
    min_level_candidates: list[int] = []
    min_level_used = None
    requested_min_level = int(args.min_level)

    if reconstruction_backend_requested == "faithc":
        kernel_setup_t0 = time.time()
        with _profile_step(profiler, "preview:kernel_setup"):
            kernel_diag = _validate_runtime(device)
        kernel_setup_seconds = time.time() - kernel_setup_t0

        cuda_device = torch.device(device)
        if cuda_device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
            props = torch.cuda.get_device_properties(cuda_device)
            cuda_diag["cuda_device_name"] = props.name
            cuda_diag["cuda_device_capability"] = f"{props.major}.{props.minor}"
        from atom3d import MeshBVH
        from atom3d.grid import OctreeIndexer
        from faithcontour import FCTDecoder, FCTEncoder

        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        grid_bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)

        max_level = int(math.log2(args.resolution))
        auto_min_level = min(4, max(1, max_level - 1))
        if requested_min_level > 0:
            if requested_min_level > max_level:
                raise ValueError(
                    f"min_level must be <= max_level ({max_level}) for resolution={args.resolution}, got {requested_min_level}"
                )
            min_level_candidates = [requested_min_level]
        else:
            min_level_candidates = [auto_min_level, 3, 2, 1]
        min_level_candidates = [lv for lv in min_level_candidates if 1 <= lv <= max_level]
        min_level_candidates = list(dict.fromkeys(min_level_candidates))

        start_encode = time.time()
        with _profile_step(profiler, "preview:encode"):
            bvh = MeshBVH(vertices, faces, device=device)
            octree = OctreeIndexer(max_level=max_level, bounds=grid_bounds, device=device)
            encoder = FCTEncoder(bvh=bvh, octree=octree, device=device)
            fct_result = None
            for min_level in min_level_candidates:
                candidate = encoder.encode(
                    min_level=min_level,
                    solver_weights={"lambda_n": 1.0, "lambda_d": 1e-3, "weight_power": 1},
                    compute_flux=True,
                    clamp_anchors=True,
                )
                if int(candidate.active_voxel_indices.shape[0]) > 0:
                    fct_result = candidate
                    min_level_used = min_level
                    break
                if fct_result is None:
                    fct_result = candidate
            if min_level_used is None:
                min_level_used = min_level_candidates[0] if min_level_candidates else auto_min_level
        encode_seconds = time.time() - start_encode

        start_decode = time.time()
        with _profile_step(profiler, "preview:decode"):
            decoder = FCTDecoder(resolution=args.resolution, bounds=grid_bounds, device=device)
            decoded = decoder.decode(
                active_voxel_indices=fct_result.active_voxel_indices,
                anchors=fct_result.anchor,
                edge_flux_sign=fct_result.edge_flux_sign,
                normals=fct_result.normal,
                triangulation_mode=args.tri_mode,
            )
        decode_seconds = time.time() - start_decode

        num_output_faces = int(decoded.faces.shape[0])
        active_voxels = int(fct_result.active_voxel_indices.shape[0])
        if num_output_faces <= 0:
            raise RuntimeError(
                "FaithC produced an empty mesh "
                f"(active_voxels={active_voxels}, output_faces={num_output_faces}). "
                "Try a higher resolution (>=128 or 256) and/or adjust margin. "
                f"min_level tried={min_level_candidates}. "
                "If this persists, verify CUDA + Atom3d kernel setup first."
            )
        recon_mesh = trimesh.Trimesh(
            vertices=decoded.vertices.detach().cpu().numpy(),
            faces=decoded.faces.detach().cpu().numpy(),
            process=False,
        )
    else:
        with _profile_step(profiler, "preview:decimate_pymeshlab_qem"):
            decimated = decimate_with_pymeshlab_qem(
                mesh,
                target_face_count=int(args.lowpoly_target_faces),
                target_face_ratio=float(args.lowpoly_target_ratio),
                quality_threshold=float(args.lowpoly_quality_threshold),
                preserve_boundary=bool(args.lowpoly_preserve_boundary),
                boundary_weight=float(args.lowpoly_boundary_weight),
                preserve_normal=bool(args.lowpoly_preserve_normal),
                preserve_topology=bool(args.lowpoly_preserve_topology),
                optimal_placement=bool(args.lowpoly_optimal_placement),
                planar_quadric=bool(args.lowpoly_planar_quadric),
                planar_weight=float(args.lowpoly_planar_weight),
                quality_weight=bool(args.lowpoly_quality_weight),
                autoclean=bool(args.lowpoly_autoclean),
            )
        recon_mesh = decimated.mesh_low
        num_output_faces = int(len(recon_mesh.faces))
        if num_output_faces <= 0:
            raise RuntimeError("PyMeshLab QEM decimation produced empty mesh")
        uv_diag_extra = dict(decimated.stats)
        uv_diag_extra["reconstruction_backend_requested"] = reconstruction_backend_requested
        uv_diag_extra["reconstruction_backend_used"] = reconstruction_backend_used
    if reconstruction_backend_requested == "faithc":
        uv_diag_extra = {
            "reconstruction_backend_requested": reconstruction_backend_requested,
            "reconstruction_backend_used": reconstruction_backend_used,
        }
    uv_diag: Dict[str, Any] = {
        "uv_projected": False,
        "uv_source_has_uv": False,
    }
    export_mesh = recon_mesh
    if bool(args.project_uv):
        uv_service = UVProjector()
        uv_method = METHOD_ALIASES.get(str(args.uv_mode), str(args.uv_mode))
        uv_cuda_methods = {
            "barycentric_closest_point",
            "hybrid_global_opt",
            "method2_gradient_poisson",
            "method2p_projected_gradient_poisson",
            "method25_projected_jacobian_injective",
            "method4_jacobian_injective",
        }
        if kernel_diag is None and device == "cuda" and uv_method in uv_cuda_methods:
            kernel_setup_t0 = time.time()
            with _profile_step(profiler, "preview:uv_runtime_setup"):
                kernel_diag = _validate_runtime(device)
            kernel_setup_seconds += time.time() - kernel_setup_t0
        uv_options = _build_preview_uv_options(args)

        with _profile_step(profiler, "preview:uv_projection"):
            try:
                mapped_uv, source_image, uv_diag, uv_export_payload = uv_service.map_uv(
                    high_mesh=mesh,
                    low_mesh=recon_mesh,
                    method=uv_method,
                    device=device,
                    texture_source_path=in_path,
                    options=uv_options,
                    return_export_payload=True,
                )
            except Exception as exc:
                if args.uv_mode == "auto":
                    mapped_uv, source_image, uv_diag, uv_export_payload = uv_service.map_uv(
                        high_mesh=mesh,
                        low_mesh=recon_mesh,
                        method="nearest_vertex",
                        device=device,
                        texture_source_path=in_path,
                        options=uv_options,
                        return_export_payload=True,
                    )
                    uv_diag["uv_auto_primary_error"] = str(exc)
                else:
                    raise RuntimeError(f"UV projection failed for mode '{args.uv_mode}': {exc}") from exc

            export_mesh = uv_service.build_uv_mesh(
                low_mesh=export_mesh,
                mapped_uv=mapped_uv,
                image=source_image,
                export_payload=uv_export_payload,
            )
            if status_path is not None:
                uv_diag.update(_write_method2_face_samples_sidecar(status_path, uv_diag))
                uv_diag.update(_write_uv_semantic_transfer_sidecar(status_path, uv_diag))
                uv_diag.update(
                    _write_uv_closure_validation_sidecar(
                        status_path=status_path,
                        payload=uv_diag,
                        high_mesh=mesh,
                        low_mesh=export_mesh,
                        uv_options=uv_options,
                    )
                )
                _strip_large_uv_sidecar_fields(uv_diag)

    with _profile_step(profiler, "preview:export_mesh"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_mesh.export(out_path)

    atom3d_path = None
    faithcontour_path = None
    if kernel_diag is not None:
        import atom3d

        atom3d_path = str(Path(atom3d.__file__).resolve())
    if reconstruction_backend_requested == "faithc":
        import atom3d
        import faithcontour

        atom3d_path = str(Path(atom3d.__file__).resolve())
        faithcontour_path = str(Path(faithcontour.__file__).resolve())
        cuda_device = torch.device(device)
        if cuda_device.type == "cuda":
            torch.cuda.synchronize(cuda_device)
            cuda_diag["torch_cuda_peak_alloc_mb"] = round(
                float(torch.cuda.max_memory_allocated(cuda_device)) / (1024.0 * 1024.0), 3
            )
            cuda_diag["torch_cuda_peak_reserved_mb"] = round(
                float(torch.cuda.max_memory_reserved(cuda_device)) / (1024.0 * 1024.0), 3
            )

    payload = {
        "success": True,
        "input_mesh": str(in_path),
        "output_mesh": str(out_path),
        "reconstruction_backend_requested": reconstruction_backend_requested,
        "reconstruction_backend_used": reconstruction_backend_used,
        "device": device,
        "resolution": int(args.resolution),
        "tri_mode": str(args.tri_mode),
        "margin": float(args.margin),
        "min_level_requested": requested_min_level,
        "min_level_used": int(min_level_used) if min_level_used is not None else None,
        "min_level_tried": min_level_candidates,
        "num_input_faces": num_input_faces,
        "num_output_faces": num_output_faces,
        "face_reduction_ratio": (num_output_faces / num_input_faces) if num_input_faces > 0 else None,
        "active_voxels": active_voxels,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "kernel_setup_seconds": kernel_setup_seconds,
        "total_seconds": time.time() - start_total,
        "faithcontour_path": faithcontour_path,
        "atom3d_path": atom3d_path,
        "kernel_diag": kernel_diag,
        "cuda_diag": cuda_diag,
        **uv_diag_extra,
        **uv_diag,
    }
    merge_runtime_diag(payload, kernel_diag or uv_diag)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FaithC reconstruction for preview decimation.")
    parser.add_argument("--input", required=True, help="Input mesh path")
    parser.add_argument("--output", required=True, help="Output mesh path")
    parser.add_argument("--status", required=True, help="Status JSON output path")
    parser.add_argument("--resolution", type=int, default=128, help="FaithC reconstruction resolution")
    parser.add_argument(
        "--tri-mode",
        type=str,
        default="auto",
        choices=["auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs"],
        help="FaithC triangulation mode",
    )
    parser.add_argument("--margin", type=float, default=0.05, help="Normalization margin")
    parser.add_argument("--min-level", type=int, default=-1, help="Octree min level (-1 = auto with fallback)")
    parser.add_argument("--device", type=str, default="auto", help="cuda/cpu/auto")
    parser.add_argument(
        "--lowpoly-backend",
        type=str,
        default="pymeshlab_qem",
        choices=["faithc", "pymeshlab_qem"],
        help="Low-poly generation backend: FaithC reconstruction or PyMeshLab QEM decimation baseline",
    )
    parser.add_argument(
        "--lowpoly-target-faces",
        type=int,
        default=0,
        help="PyMeshLab QEM: explicit target face count (0 = use ratio)",
    )
    parser.add_argument(
        "--lowpoly-target-ratio",
        type=float,
        default=0.05,
        help="PyMeshLab QEM: target face ratio when target-faces=0",
    )
    parser.add_argument(
        "--lowpoly-quality-threshold",
        type=float,
        default=0.3,
        help="PyMeshLab QEM: quality threshold",
    )
    parser.add_argument(
        "--lowpoly-preserve-boundary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab QEM: preserve mesh boundary",
    )
    parser.add_argument(
        "--lowpoly-boundary-weight",
        type=float,
        default=2.0,
        help="PyMeshLab QEM: boundary preservation weight",
    )
    parser.add_argument(
        "--lowpoly-preserve-normal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab QEM: preserve normals",
    )
    parser.add_argument(
        "--lowpoly-preserve-topology",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab QEM: preserve topology",
    )
    parser.add_argument(
        "--lowpoly-optimal-placement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PyMeshLab QEM: enable optimal vertex placement",
    )
    parser.add_argument(
        "--lowpoly-planar-quadric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab QEM: add planar quadric term",
    )
    parser.add_argument(
        "--lowpoly-planar-weight",
        type=float,
        default=1e-3,
        help="PyMeshLab QEM: planar quadric weight",
    )
    parser.add_argument(
        "--lowpoly-quality-weight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PyMeshLab QEM: use per-vertex quality weighting",
    )
    parser.add_argument(
        "--lowpoly-autoclean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PyMeshLab QEM: auto-clean after decimation",
    )
    parser.add_argument(
        "--project-uv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transfer UV from source mesh to FaithC output",
    )
    parser.add_argument(
        "--uv-mode",
        type=str,
        default="method2",
        choices=["hybrid", "method2", "method2p", "method4", "method25", "barycentric", "nearest", "auto"],
        help="UV transfer mode: hybrid, method2, method2p projected-linear, method4, method25 projected+injective, barycentric, nearest, or auto fallback",
    )
    parser.add_argument(
        "--uv-batch-size",
        type=int,
        default=int(DEFAULT_OPTIONS["correspondence"]["bvh_chunk_size"]),
        help="Chunk size for BVH projection queries",
    )
    parser.add_argument("--uv-sample-base-per-face", type=int, default=int(DEFAULT_OPTIONS["sample"]["base_per_face"]), help="Hybrid: base samples per low-mesh face")
    parser.add_argument("--uv-sample-min-per-face", type=int, default=int(DEFAULT_OPTIONS["sample"]["min_per_face"]), help="Hybrid: min samples per low-mesh face")
    parser.add_argument("--uv-sample-max-per-face", type=int, default=int(DEFAULT_OPTIONS["sample"]["max_per_face"]), help="Hybrid: max samples per low-mesh face")
    parser.add_argument("--uv-sample-seed", type=int, default=int(DEFAULT_OPTIONS["sample"]["seed"]), help="Hybrid: random seed for face sampling")
    parser.add_argument("--uv-normal-weight", type=float, default=float(DEFAULT_OPTIONS["correspondence"]["normal_weight"]), help="Hybrid fallback nearest normal penalty weight")
    parser.add_argument("--uv-normal-dot-min", type=float, default=float(DEFAULT_OPTIONS["correspondence"]["normal_dot_min"]), help="Hybrid primary/fallback normal-dot threshold")
    parser.add_argument("--uv-ray-max-dist-ratio", type=float, default=float(DEFAULT_OPTIONS["correspondence"]["ray_max_dist_ratio"]), help="Hybrid primary max distance ratio to bbox diagonal")
    parser.add_argument("--uv-fallback-k", type=int, default=int(DEFAULT_OPTIONS["correspondence"]["fallback_k"]), help="Hybrid fallback nearest candidate count")
    parser.add_argument("--uv-fallback-weight", type=float, default=float(DEFAULT_OPTIONS["correspondence"]["fallback_weight"]), help="Hybrid fallback correspondence confidence weight")
    parser.add_argument(
        "--uv-solve-backend",
        type=str,
        default=str(DEFAULT_OPTIONS["solve"]["backend"]),
        choices=["auto", "cuda_pcg", "cpu_scipy"],
        help="Hybrid UV solve backend",
    )
    parser.add_argument("--uv-lambda-smooth", type=float, default=float(DEFAULT_OPTIONS["solve"]["lambda_smooth"]), help="Hybrid linear solve Laplacian smooth weight")
    parser.add_argument("--uv-pcg-max-iter", type=int, default=int(DEFAULT_OPTIONS["solve"]["pcg_max_iter"]), help="Hybrid CUDA-PCG max iterations")
    parser.add_argument("--uv-pcg-tol", type=float, default=float(DEFAULT_OPTIONS["solve"]["pcg_tol"]), help="Hybrid CUDA-PCG relative tolerance")
    parser.add_argument("--uv-pcg-check-every", type=int, default=int(DEFAULT_OPTIONS["solve"]["pcg_check_every"]), help="Hybrid CUDA-PCG full residual recompute interval")
    parser.add_argument(
        "--uv-pcg-preconditioner",
        type=str,
        default=str(DEFAULT_OPTIONS["solve"]["pcg_preconditioner"]),
        choices=["jacobi", "none"],
        help="Hybrid CUDA-PCG preconditioner type",
    )
    parser.add_argument("--uv-cg-max-iter", type=int, default=int(DEFAULT_OPTIONS["solve"]["cg_max_iter"]), help="Hybrid linear solve conjugate-gradient max iterations")
    parser.add_argument("--uv-cg-tol", type=float, default=float(DEFAULT_OPTIONS["solve"]["cg_tol"]), help="Hybrid linear solve conjugate-gradient tolerance")
    parser.add_argument("--uv-anchor-weight", type=float, default=float(DEFAULT_OPTIONS["solve"]["anchor_weight"]), help="Method2/Hybrid anchor penalty weight")
    parser.add_argument("--uv-ridge-eps", type=float, default=float(DEFAULT_OPTIONS["solve"]["ridge_eps"]), help="Hybrid linear solve diagonal ridge regularization")
    parser.add_argument(
        "--uv-seam-strategy",
        type=str,
        default=str(DEFAULT_OPTIONS["seam"]["strategy"]),
        choices=["legacy", "halfedge_island"],
        help="Seam handling strategy: legacy sampling heuristic or halfedge UV-island split",
    )
    parser.add_argument(
        "--uv-seam-validation-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable strict seam validation (default: off in preview bridge so diagnostics sidecars can still be emitted)",
    )
    parser.add_argument(
        "--uv-seam-validation-require-closed-loops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When seam validation is enabled, require all seam components to be closed loops",
    )
    parser.add_argument(
        "--uv-seam-validation-require-pure-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When seam validation is enabled, require partition components to be single semantic label",
    )
    parser.add_argument(
        "--uv-seam-validation-allow-open-on-boundary",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["seam"]["validation_allow_open_on_boundary"]),
        help="Treat seam components that are open only on mesh boundary as acceptable",
    )
    parser.add_argument(
        "--uv-seam-validation-ignore-small-components-faces",
        type=int,
        default=int(DEFAULT_OPTIONS["seam"]["validation_ignore_small_components_faces"]),
        help="Ignore partition components smaller than this face count during seam validation",
    )
    parser.add_argument("--uv-seam-uv-span-threshold", type=float, default=float(DEFAULT_OPTIONS["seam"]["uv_span_threshold"]), help="Mark faces as cross-seam when sample UV span exceeds this value")
    parser.add_argument("--uv-seam-min-valid-samples", type=int, default=int(DEFAULT_OPTIONS["seam"]["min_valid_samples_per_face"]), help="Minimum valid samples required to judge one face as cross-seam")
    parser.add_argument(
        "--uv-exclude-cross-seam-faces",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["seam"]["exclude_cross_seam_faces"]),
        help="Exclude detected cross-seam faces from global linear system",
    )
    parser.add_argument(
        "--uv-local-vertex-split",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["seam"]["local_vertex_split"]),
        help="Duplicate cross-seam face vertices during export to preserve per-face UV",
    )
    parser.add_argument(
        "--uv-island-guard",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["seam"]["uv_island_guard_enabled"]),
        help="Guard hybrid correspondence with high-mesh UV-island consistency constraints",
    )
    parser.add_argument(
        "--uv-island-guard-mode",
        type=str,
        default=str(DEFAULT_OPTIONS["seam"]["uv_island_guard_mode"]),
        choices=["soft", "strict"],
        help="Island guard mode: soft (with fallback) or strict (no fallback)",
    )
    parser.add_argument(
        "--uv-island-guard-confidence-min",
        type=float,
        default=float(DEFAULT_OPTIONS["seam"]["uv_island_guard_confidence_min"]),
        help="Minimum face-level island confidence required before enforcing island guard",
    )
    parser.add_argument(
        "--uv-island-guard-allow-unknown",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["seam"]["uv_island_guard_allow_unknown"]),
        help="Allow correspondences landing on unknown high-mesh islands",
    )
    parser.add_argument(
        "--uv-island-guard-fallback",
        type=str,
        default=str(DEFAULT_OPTIONS["seam"]["uv_island_guard_fallback"]),
        help="Diagnostic label for island-guard fallback policy",
    )
    parser.add_argument(
        "--uv-iterative",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable iterative Stage-1 hybrid UV pipeline (default: disabled, use legacy one-pass route)",
    )
    parser.add_argument(
        "--uv-texture-weight",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["texture_weight"]["enabled"]),
        help="Enable texture-gradient correspondence weighting",
    )
    parser.add_argument("--uv-grad-weight-gamma", type=float, default=float(DEFAULT_OPTIONS["texture_weight"]["grad_weight_gamma"]), help="Texture-gradient weighting gamma")
    parser.add_argument("--uv-max-texture-weight", type=float, default=float(DEFAULT_OPTIONS["texture_weight"]["max_weight"]), help="Texture-gradient max sample weight")
    parser.add_argument("--uv-m2-outlier-sigma", type=float, default=float(DEFAULT_OPTIONS["method2"]["outlier_sigma"]), help="Method2 IRLS outlier rejection sigma")
    parser.add_argument(
        "--uv-m2-outlier-quantile",
        type=float,
        default=float(DEFAULT_OPTIONS["method2"]["outlier_quantile"]),
        help="Method2 IRLS outlier rejection quantile",
    )
    parser.add_argument(
        "--uv-m2-min-samples-per-face",
        type=int,
        default=int(DEFAULT_OPTIONS["method2"]["min_samples_per_face"]),
        help="Method2 minimum valid samples per face",
    )
    parser.add_argument(
        "--uv-m2-face-weight-floor",
        type=float,
        default=float(DEFAULT_OPTIONS["method2"]["face_weight_floor"]),
        help="Method2 face weight floor",
    )
    parser.add_argument(
        "--uv-m2-anchor-mode",
        type=str,
        default=str(DEFAULT_OPTIONS["method2"]["anchor_mode"]),
        choices=["component_minimal", "boundary", "none"],
        help="Method2 anchor mode",
    )
    parser.add_argument(
        "--uv-m2-anchor-points-per-component",
        type=int,
        default=int(DEFAULT_OPTIONS["method2"]["anchor_points_per_component"]),
        help="Method2 anchor points per connected component",
    )
    parser.add_argument(
        "--uv-m2-use-island-guard",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["method2"]["use_island_guard"]),
        help="Method2: enable island-guard filtering in gradient-poisson solve",
    )
    parser.add_argument("--uv-m2-irls-iters", type=int, default=int(DEFAULT_OPTIONS["method2"]["irls_iters"]), help="Method2 IRLS iterations")
    parser.add_argument("--uv-m2-huber-delta", type=float, default=float(DEFAULT_OPTIONS["method2"]["huber_delta"]), help="Method2 Huber delta")
    parser.add_argument(
        "--uv-m2-post-align",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["method2"]["post_align_translation"]),
        help="Method2: apply global UV translation alignment from sample residual median",
    )
    parser.add_argument(
        "--uv-m2-post-align-min-samples",
        type=int,
        default=int(DEFAULT_OPTIONS["method2"]["post_align_min_samples"]),
        help="Method2: minimum valid samples required to run post-alignment",
    )
    parser.add_argument(
        "--uv-m2-post-align-max-shift",
        type=float,
        default=float(DEFAULT_OPTIONS["method2"]["post_align_max_shift"]),
        help="Method2: maximum allowed global UV alignment shift",
    )
    parser.add_argument(
        "--uv-m2-laplacian-mode",
        type=str,
        default=str(DEFAULT_OPTIONS["method2"]["laplacian_mode"]),
        choices=["uniform", "cotan"],
        help="Method2 Laplacian mode",
    )
    parser.add_argument(
        "--uv-m2-system-cond-estimate",
        type=str,
        default=str(DEFAULT_OPTIONS["method2"]["system_cond_estimate"]),
        choices=["diag_ratio", "eigsh"],
        help="Method2 system condition-number estimate method",
    )
    parser.add_argument(
        "--uv-m4-recovery-mode",
        action=argparse.BooleanOptionalAction,
        default=bool(DEFAULT_OPTIONS["method4"]["recovery_mode_enabled"]),
        help="Method4/Method2.5: enable recovery-mode line search for infeasible starts",
    )
    parser.add_argument(
        "--uv-m4-recovery-det-improve-eps",
        type=float,
        default=float(DEFAULT_OPTIONS["method4"]["recovery_det_improve_eps"]),
        help="Method4/Method2.5: minimum determinant improvement threshold for same-violation recovery acceptance",
    )
    parser.add_argument(
        "--profiler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable built-in profiler (preferred switch)",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("FAITHC_PREVIEW_PROFILE", False),
        help="(Legacy) Enable built-in profiler (timing/hotspots/memory, default: off)",
    )
    parser.add_argument(
        "--profile-top-k",
        type=int,
        default=_env_int("FAITHC_PREVIEW_PROFILE_TOP_K", 80),
        help="Top-K hotspots kept in profiler report",
    )
    parser.add_argument(
        "--profile-no-cprofile",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("FAITHC_PREVIEW_PROFILE_NO_CPROFILE", False),
        help="Disable cProfile hotspots and keep lightweight profiler metrics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    profiler_enabled = bool(args.profiler) if args.profiler is not None else bool(args.profile)
    status_path = Path(args.status).resolve()
    profiler = ExecutionProfiler(
        name="faithc_preview_bridge",
        config=ProfilerConfig(
            enabled=profiler_enabled,
            cprofile_enabled=not bool(args.profile_no_cprofile),
            top_k=int(args.profile_top_k),
        ),
        metadata={
            "command": "preview_bridge",
            "input": str(Path(args.input).resolve()),
            "output": str(Path(args.output).resolve()),
            "resolution": int(args.resolution),
            "uv_mode": str(args.uv_mode),
        },
    )
    profiler.start()

    try:
        payload = run_pipeline(args, profiler=profiler, status_path=status_path)
        payload.update(
            _write_perf_sidecars(
                profiler=profiler,
                status_path=status_path,
                extra={"success": True},
            )
        )
        _write_status(status_path, payload)
        return 0
    except Exception as exc:
        atom3d_path = None
        faithcontour_path = None
        kernel_diag = None
        try:
            import atom3d

            atom3d_path = str(Path(atom3d.__file__).resolve())
        except Exception:
            pass
        try:
            import faithcontour

            faithcontour_path = str(Path(faithcontour.__file__).resolve())
        except Exception:
            pass
        try:
            resolved_device = resolve_device(str(getattr(args, "device", "auto")))
            kernel_diag = ensure_atom3d_cuda_runtime(resolved_device, strict=False)
        except Exception:
            kernel_diag = None

        payload = {
            "success": False,
            "error": str(exc),
            "uv_project_error": str(exc),
            "input_mesh": str(Path(args.input).resolve()),
            "output_mesh": str(Path(args.output).resolve()),
            "reconstruction_backend_requested": str(getattr(args, "lowpoly_backend", "faithc")),
            "resolution": int(args.resolution),
            "tri_mode": str(args.tri_mode),
            "margin": float(args.margin),
            "min_level_requested": int(args.min_level),
            "project_uv": bool(args.project_uv),
            "uv_mode": str(args.uv_mode),
            "uv_seam_strategy_requested": str(getattr(args, "uv_seam_strategy", "")),
            "uv_island_validation_mode": "hard" if bool(getattr(args, "uv_seam_validation_strict", False)) else "diagnostic",
            "uv_island_validation_ok": False,
            "uv_batch_size": int(args.uv_batch_size),
            "faithcontour_path": faithcontour_path,
            "atom3d_path": atom3d_path,
            "kernel_diag": kernel_diag,
        }
        merge_runtime_diag(payload, kernel_diag)
        err_str = str(exc)
        marker = "halfedge seam validation failed:"
        if marker in err_str:
            validation_err = err_str.split(marker, 1)[1].strip()
            payload["uv_seam_validation_error"] = validation_err
            payload["uv_island_validation_error"] = validation_err
        payload.update(
            _write_perf_sidecars(
                profiler=profiler,
                status_path=status_path,
                extra={"success": False, "error": str(exc)},
            )
        )
        _write_status(status_path, payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
