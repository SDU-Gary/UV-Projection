#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_method2_internal import _load_mesh, _sanitize_json, run_method2_internal_audit_on_meshes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Method2 massive-only diagnostic suite with 15 experiments")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--uv-seam-strategy",
        type=str,
        default="halfedge_island",
        choices=["legacy", "halfedge_island"],
        help="Method2 seam strategy",
    )
    p.add_argument(
        "--sanitize-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-mesh sanitization",
    )
    p.add_argument("--massive-high", type=Path, required=True, help="High mesh with UV")
    p.add_argument("--massive-low", type=Path, required=True, help="Low mesh to audit")
    return p.parse_args()


def _experiment_statuses(experiments: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in experiments.items():
        if isinstance(value, dict):
            out[key] = {
                "status": value.get("status", "unknown"),
                "error": value.get("error"),
            }
        else:
            out[key] = {"status": "invalid_payload", "error": None}
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = out_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    high_path = args.massive_high.resolve()
    low_path = args.massive_low.resolve()
    high_mesh = _load_mesh(high_path)
    low_mesh = _load_mesh(low_path)

    baseline_json = out_dir / "baseline.json"
    t0 = time.perf_counter()
    report = run_method2_internal_audit_on_meshes(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_path=high_path,
        low_path=low_path,
        device=str(args.device),
        seam_strategy=str(args.uv_seam_strategy),
        sanitize_low=bool(args.sanitize_low),
        options_overrides=None,
        out_json=baseline_json,
        case_name="massive:diagnostic_suite",
    )
    runtime = float(time.perf_counter() - t0)

    experiments = dict(report.get("experiments", {}))
    for exp_name, payload in experiments.items():
        exp_path = experiments_dir / f"{exp_name}.json"
        exp_path.write_text(
            json.dumps(_sanitize_json(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    summary = {
        "device": str(args.device),
        "uv_seam_strategy": str(args.uv_seam_strategy),
        "sanitize_low": bool(args.sanitize_low),
        "high": str(high_path),
        "low": str(low_path),
        "runtime_seconds": runtime,
        "baseline_json": str(baseline_json),
        "solve_mesh_quality": {
            k: report.get("solve_mesh_quality", {}).get(k)
            for k in [
                "uv_stretch_p95",
                "uv_stretch_p99",
                "uv_bad_tri_ratio_stretch_only",
                "uv_out_of_bounds_ratio",
            ]
        },
        "jacobian_summary": {
            k: report.get("jacobian_summary", {}).get(k)
            for k in ["frob_rel_error_p95", "cosine_p05", "log_area_ratio_p95"]
        },
        "support_summary": {
            k: report.get("support_summary", {}).get(k)
            for k in [
                "valid_face_ratio",
                "accepted_face_ratio",
                "unsupported_valid_face_ratio",
                "sample_acceptance_ratio",
            ]
        },
        "target_dispersion_summary": {
            k: report.get("target_dispersion_summary", {}).get(k)
            for k in ["cov_norm_p95", "high_dispersion_face_ratio", "smooth_alpha_p50"]
        },
        "experiment_statuses": _experiment_statuses(experiments),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_sanitize_json(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        "[method2_massive_suite] "
        f"stretch_p95={summary['solve_mesh_quality'].get('uv_stretch_p95')} "
        f"jac_rel_p95={summary['jacobian_summary'].get('frob_rel_error_p95')} "
        f"cov_norm_p95={summary['target_dispersion_summary'].get('cov_norm_p95')}"
    )
    print(f"baseline_json={baseline_json}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
