#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.uv_projector import UVProjector


EXPERIMENTS: List[Dict[str, str]] = [
    {
        "name": "g1_method2_current_linear",
        "method": "method2",
        "note": "current Method2 target field + Method2 linear solver",
    },
    {
        "name": "g2_method2p_exp13_linear",
        "method": "method2p",
        "note": "Exp13 residual projected field + Method2 linear solver",
    },
    {
        "name": "g3_method4_current_nonlinear",
        "method": "method4",
        "note": "current Method2 target field + Method4 nonlinear solver",
    },
    {
        "name": "g4_method25_exp13_nonlinear",
        "method": "method25",
        "note": "Exp13 residual projected field + Method4 nonlinear solver",
    },
]

SUMMARY_KEYS: Tuple[str, ...] = (
    "uv_projection_seconds",
    "uv_solver_stage",
    "uv_stretch_p95",
    "uv_stretch_p99",
    "uv_bad_tri_ratio",
    "uv_flip_ratio",
    "uv_out_of_bounds_ratio",
    "uv_color_reproj_l1",
    "uv_color_reproj_l2",
    "uv_m4_refine_status",
    "uv_m4_energy_init",
    "uv_m4_energy_final",
    "uv_m4_barrier_violations",
    "uv_m4_det_min",
    "uv_m4_accepted_step_count",
    "uv_m4_recovery_accepted_step_count",
    "uv_m2p_field_source",
    "uv_m25_field_source",
    "uv_m2p_linear_init_color_reproj_l1",
    "uv_m25_linear_init_color_reproj_l1",
)

DELTA_KEYS: Tuple[str, ...] = (
    "uv_stretch_p95",
    "uv_stretch_p99",
    "uv_bad_tri_ratio",
    "uv_flip_ratio",
    "uv_out_of_bounds_ratio",
    "uv_color_reproj_l1",
    "uv_color_reproj_l2",
    "uv_m4_barrier_violations",
    "uv_m4_det_min",
)

PAIRWISE_COMPARISONS: List[Tuple[str, str, str]] = [
    ("g2_minus_g1", "g2_method2p_exp13_linear", "g1_method2_current_linear"),
    ("g3_minus_g1", "g3_method4_current_nonlinear", "g1_method2_current_linear"),
    ("g4_minus_g3", "g4_method25_exp13_nonlinear", "g3_method4_current_nonlinear"),
    ("g4_minus_g2", "g4_method25_exp13_nonlinear", "g2_method2p_exp13_linear"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the 4-way Method2/2p/4/25 comparison suite.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--high", type=Path, required=True, help="High mesh with UV")
    p.add_argument("--low", type=Path, default=None, help="Low mesh; defaults to high mesh for self-transfer")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--solve-backend", type=str, default="auto", choices=["auto", "cuda_pcg", "cpu_scipy"])
    p.add_argument(
        "--uv-seam-strategy",
        type=str,
        default="halfedge_island",
        choices=["legacy", "halfedge_island"],
        help="Shared seam strategy",
    )
    p.add_argument(
        "--sanitize-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-mesh sanitization inside the UV pipeline",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="scientific",
        choices=["scientific", "engineering"],
        help="scientific: isolate field/solver effects; engineering: enable more post-processing",
    )
    p.add_argument("--method4-max-iters", type=int, default=80)
    p.add_argument(
        "--method4-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Method4 recovery-mode line search for the nonlinear groups",
    )
    p.add_argument("--method4-recovery-det-improve-eps", type=float, default=1e-8)
    p.add_argument("--method25-samplefit-min-samples", type=int, default=3)
    p.add_argument("--method25-lambda-decay", type=float, default=1.0)
    p.add_argument("--method25-lambda-curl", type=float, default=20.0)
    p.add_argument(
        "--method25-strict-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use strict >=4-sample low-fallback gate in the projected-field builder",
    )
    return p.parse_args()


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _shared_options(args: argparse.Namespace) -> Dict[str, Any]:
    scientific = str(args.mode) == "scientific"
    return {
        "solve": {
            "backend": str(args.solve_backend),
        },
        "seam": {
            "strategy": str(args.uv_seam_strategy),
            "sanitize_enabled": bool(args.sanitize_low),
        },
        "method2": {
            "post_align_translation": False if scientific else True,
        },
        "method4": {
            "enabled": True,
            "device": str(args.device),
            "max_iters": int(args.method4_max_iters),
            "fallback_to_method2_on_violation": False if scientific else True,
            "patch_refine_rounds": 0 if scientific else 3,
            "recovery_mode_enabled": bool(args.method4_recovery),
            "recovery_det_improve_eps": float(args.method4_recovery_det_improve_eps),
        },
        "method25": {
            "samplefit_min_samples": int(args.method25_samplefit_min_samples),
            "strict_gate": bool(args.method25_strict_gate),
            "lambda_decay": float(args.method25_lambda_decay),
            "lambda_curl": float(args.method25_lambda_curl),
        },
    }


def _select_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {k: stats.get(k) for k in SUMMARY_KEYS}


def _pairwise_delta(candidate: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in DELTA_KEYS:
        ca = candidate.get(key)
        re = reference.get(key)
        if isinstance(ca, (int, float)) and isinstance(re, (int, float)):
            out[key] = float(ca) - float(re)
        else:
            out[key] = None
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    high_path = args.high.resolve()
    low_path = args.low.resolve() if args.low is not None else high_path
    shared_options = _shared_options(args)

    projector = UVProjector()
    suite: Dict[str, Any] = {
        "high": str(high_path),
        "low": str(low_path),
        "device": str(args.device),
        "solve_backend": str(args.solve_backend),
        "mode": str(args.mode),
        "shared_options": _sanitize_json(shared_options),
        "experiments": {},
        "pairwise": {},
    }

    for exp in EXPERIMENTS:
        name = str(exp["name"])
        method = str(exp["method"])
        exp_dir = out_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            artifact = projector.project(
                sample_name=name,
                high_mesh_path=high_path,
                low_mesh_path=low_path,
                output_dir=exp_dir,
                method=method,
                device=str(args.device),
                texture_source_path=high_path,
                options=shared_options,
            )
            elapsed = float(time.perf_counter() - t0)
            stats = dict(artifact.stats)
            suite["experiments"][name] = {
                "status": "ok",
                "method": method,
                "note": str(exp["note"]),
                "runtime_seconds": elapsed,
                "summary": _sanitize_json(_select_summary(stats)),
                "stats_json": str((exp_dir / "uv_stats.json").resolve()),
                "uv_mesh": str(artifact.low_mesh_uv_path.resolve()),
                "uv_map": str(artifact.uv_map_path.resolve()),
            }
            print(
                "[fourway] "
                f"{name} method={method} "
                f"stretch_p95={stats.get('uv_stretch_p95')} "
                f"flip={stats.get('uv_flip_ratio')} "
                f"oob={stats.get('uv_out_of_bounds_ratio')} "
                f"solver_stage={stats.get('uv_solver_stage')}"
            )
        except Exception as exc:
            suite["experiments"][name] = {
                "status": "error",
                "method": method,
                "note": str(exp["note"]),
                "error": str(exc),
            }
            print(f"[fourway][error] {name} method={method} error={exc}")

    for label, cand_name, ref_name in PAIRWISE_COMPARISONS:
        cand = suite["experiments"].get(cand_name, {})
        ref = suite["experiments"].get(ref_name, {})
        cand_summary = cand.get("summary") if isinstance(cand, dict) else None
        ref_summary = ref.get("summary") if isinstance(ref, dict) else None
        suite["pairwise"][label] = {
            "candidate": cand_name,
            "reference": ref_name,
            "delta": _pairwise_delta(cand_summary or {}, ref_summary or {}),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_sanitize_json(suite), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
