#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.mesh_io import MeshIO
from faithc_infra.services.decimation import decimate_with_pymeshlab_qem

from audit_method2_internal import _load_mesh, _sanitize_json, run_method2_internal_audit_on_meshes


DEFAULT_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    ("baseline", {}),
    ("global_solve", {"method2": {"solve_per_island": False}}),
    ("anchor_w500", {"solve": {"anchor_weight": 500.0}}),
    (
        "anchor_w1000_anchor8",
        {
            "solve": {"anchor_weight": 1000.0},
            "method2": {
                "anchor_points_per_component": 8,
                "anchor_min_points_per_component": 6,
                "anchor_max_points_per_component": 10,
                "anchor_target_vertices_per_anchor": 2000,
            },
        },
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run four Method2 internal-audit experiment groups on corgi and massive")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for per-run reports and summary")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--uv-seam-strategy",
        type=str,
        default="halfedge_island",
        choices=["legacy", "halfedge_island"],
        help="Shared seam strategy for all audit runs",
    )
    p.add_argument(
        "--sanitize-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-mesh sanitization for audit runs",
    )
    p.add_argument(
        "--skip-corgi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip corgi case",
    )
    p.add_argument(
        "--skip-massive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip massive case",
    )
    p.add_argument("--corgi-high", type=Path, default=REPO_ROOT / "assets/examples/corgi_traveller.glb")
    p.add_argument("--corgi-low", type=Path, default=None)
    p.add_argument("--massive-high", type=Path, default=REPO_ROOT / "assets/massive_nordic_coastal_cliff_vdssailfa_raw.glb")
    p.add_argument("--massive-low", type=Path, default=None)
    p.add_argument("--lowpoly-target-faces", type=int, default=0, help="If generating low mesh, explicit target face count")
    p.add_argument("--lowpoly-target-ratio", type=float, default=0.05, help="If generating low mesh, target ratio")
    p.add_argument("--lowpoly-quality-threshold", type=float, default=0.3)
    p.add_argument(
        "--reuse-generated-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse generated low meshes if they already exist in out-dir/generated_low",
    )
    return p.parse_args()


def _variant_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    solve_q = report.get("solve_mesh_quality", {})
    jac = report.get("jacobian_summary", {})
    sel = report.get("selected_method2_stats", {})
    support = report.get("support_summary", {})
    dispersion = report.get("target_dispersion_summary", {})
    return {
        "solve_stretch_p95": solve_q.get("uv_stretch_p95"),
        "solve_stretch_p99": solve_q.get("uv_stretch_p99"),
        "solve_bad_stretch_ratio": solve_q.get("uv_bad_tri_ratio_stretch_only"),
        "solve_out_of_bounds_ratio": solve_q.get("uv_out_of_bounds_ratio"),
        "jac_rel_p95": jac.get("frob_rel_error_p95"),
        "jac_cos_p05": jac.get("cosine_p05"),
        "valid_face_ratio": support.get("valid_face_ratio"),
        "accepted_face_ratio": support.get("accepted_face_ratio"),
        "unsupported_valid_face_ratio": support.get("unsupported_valid_face_ratio"),
        "target_cov_p95": dispersion.get("cov_trace_p95"),
        "target_cov_norm_p95": dispersion.get("cov_norm_p95"),
        "high_dispersion_face_ratio": dispersion.get("high_dispersion_face_ratio"),
        "sample_reproj_l1": sel.get("uv_color_reproj_l1"),
        "sample_reproj_l2": sel.get("uv_color_reproj_l2"),
        "anchor_count_total": sel.get("uv_m2_anchor_count_total"),
        "island_count": sel.get("uv_m2_island_count"),
        "solve_per_island_enabled": sel.get("uv_m2_solve_per_island_enabled"),
        "post_align_applied": sel.get("uv_m2_post_align_applied"),
    }


def _ensure_low_mesh(
    *,
    case_name: str,
    high_path: Path,
    low_path: Optional[Path],
    out_dir: Path,
    target_faces: int,
    target_ratio: float,
    quality_threshold: float,
    reuse_generated_low: bool,
) -> Tuple[Path, Dict[str, Any]]:
    if low_path is not None:
        return low_path.resolve(), {"source": "provided"}

    gen_dir = out_dir / "generated_low"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_low = gen_dir / f"{case_name}_low.glb"
    meta_path = gen_dir / f"{case_name}_low.meta.json"
    if reuse_generated_low and out_low.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["source"] = "generated_cached"
        return out_low.resolve(), meta

    high_mesh = _load_mesh(high_path.resolve())
    artifact = decimate_with_pymeshlab_qem(
        high_mesh,
        target_face_count=int(target_faces),
        target_face_ratio=float(target_ratio),
        quality_threshold=float(quality_threshold),
        preserve_boundary=False,
        boundary_weight=2.0,
        preserve_normal=False,
        preserve_topology=False,
        optimal_placement=True,
        planar_quadric=False,
        planar_weight=1e-3,
        quality_weight=False,
        autoclean=True,
    )
    MeshIO.export_mesh(artifact.mesh_low, out_low)
    meta = dict(artifact.stats)
    meta["source"] = "generated"
    meta["generated_low_path"] = str(out_low.resolve())
    meta_path.write_text(json.dumps(_sanitize_json(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_low.resolve(), meta


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Tuple[str, Path, Optional[Path]]] = []
    if not bool(args.skip_corgi):
        cases.append(("corgi", args.corgi_high.resolve(), args.corgi_low.resolve() if args.corgi_low is not None else None))
    if not bool(args.skip_massive):
        cases.append(("massive", args.massive_high.resolve(), args.massive_low.resolve() if args.massive_low is not None else None))

    suite_report: Dict[str, Any] = {
        "device": str(args.device),
        "uv_seam_strategy": str(args.uv_seam_strategy),
        "sanitize_low": bool(args.sanitize_low),
        "variants": [{"name": name, "overrides": overrides} for name, overrides in DEFAULT_VARIANTS],
        "cases": {},
    }

    for case_name, high_path, low_path_in in cases:
        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        low_path, low_meta = _ensure_low_mesh(
            case_name=case_name,
            high_path=high_path,
            low_path=low_path_in,
            out_dir=out_dir,
            target_faces=int(args.lowpoly_target_faces),
            target_ratio=float(args.lowpoly_target_ratio),
            quality_threshold=float(args.lowpoly_quality_threshold),
            reuse_generated_low=bool(args.reuse_generated_low),
        )
        high_mesh = _load_mesh(high_path)
        low_mesh = _load_mesh(low_path)

        case_report: Dict[str, Any] = {
            "high": str(high_path),
            "low": str(low_path),
            "low_meta": _sanitize_json(low_meta),
            "variants": {},
        }
        suite_report["cases"][case_name] = case_report

        for variant_name, overrides in DEFAULT_VARIANTS:
            report_path = case_dir / f"{variant_name}.json"
            t0 = time.perf_counter()
            try:
                report = run_method2_internal_audit_on_meshes(
                    high_mesh=high_mesh,
                    low_mesh=low_mesh,
                    high_path=high_path,
                    low_path=low_path,
                    device=str(args.device),
                    seam_strategy=str(args.uv_seam_strategy),
                    sanitize_low=bool(args.sanitize_low),
                    options_overrides=overrides,
                    out_json=report_path,
                    case_name=f"{case_name}:{variant_name}",
                )
                elapsed = float(time.perf_counter() - t0)
                summary = _variant_summary(report)
                summary["runtime_seconds"] = elapsed
                summary["report_json"] = str(report_path)
                case_report["variants"][variant_name] = summary
                print(
                    "[method2_suite] "
                    f"case={case_name} variant={variant_name} "
                    f"stretch_p95={summary.get('solve_stretch_p95')} "
                    f"bad_stretch={summary.get('solve_bad_stretch_ratio')} "
                    f"jac_rel_p95={summary.get('jac_rel_p95')} "
                    f"reproj_l1={summary.get('sample_reproj_l1')} "
                    f"anchors={summary.get('anchor_count_total')}"
                )
            except Exception as exc:
                case_report["variants"][variant_name] = {
                    "error": str(exc),
                    "report_json": str(report_path),
                }
                print(f"[method2_suite][error] case={case_name} variant={variant_name} error={exc}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_sanitize_json(suite_report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
