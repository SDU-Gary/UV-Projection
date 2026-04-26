from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import RunLogger
from .pathing import PathManager
from .profiler import ExecutionProfiler, ProfilerConfig
from .registry import ArtifactRegistry
from .types import SampleRecord, UVArtifact


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_run_index(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_perf_reports(
    *,
    profiler: ExecutionProfiler,
    out_dir: Path,
    prefix: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = profiler.stop(extra=extra)
    json_path = out_dir / f"{prefix}.json"
    txt_path = out_dir / f"{prefix}.txt"
    profiler.write_reports(json_path=json_path, text_path=txt_path, report=report)
    return {
        "json": str(json_path),
        "txt": str(txt_path),
        "wall_time_seconds": report.get("wall_time_seconds"),
        "cpu_time_seconds": report.get("cpu_time_seconds"),
    }


def _resolve_run_dir(runs_dir: Path, run_id: str) -> Path:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    return run_dir


def _resolve_profiler_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "profiler", None)
    if explicit is not None:
        return bool(explicit)
    return bool(getattr(args, "profile", True))


def _sample_name(sample_cfg: Dict[str, Any]) -> str:
    if sample_cfg.get("name"):
        return str(sample_cfg["name"])
    return Path(sample_cfg["high_mesh"]).stem


def cmd_run(args: argparse.Namespace) -> int:
    from .config import ConfigLoader

    config_path = Path(args.config).resolve()
    config = ConfigLoader.load(config_path)
    repo_root = Path.cwd()

    runs_dir = PathManager.resolve_path(config["paths"]["runs_dir"], repo_root)
    path_manager = PathManager(runs_dir)
    run_dir = path_manager.create_run_dir(config["project"]["run_prefix"], run_id=args.run_id)

    logger = RunLogger(run_dir / "run.log.jsonl")
    logger.info("run_started", run_dir=str(run_dir), config=str(config_path), dry_run=args.dry_run)

    profiler_enabled = _resolve_profiler_enabled(args)
    run_profiler = ExecutionProfiler(
        name="faithc_cli_run",
        config=ProfilerConfig(
            enabled=profiler_enabled,
            cprofile_enabled=not bool(args.profile_no_cprofile),
            top_k=int(args.profile_top_k),
        ),
        metadata={
            "command": "run",
            "run_id": run_dir.name,
            "config_path": str(config_path),
            "dry_run": bool(args.dry_run),
        },
    )
    run_profiler.start()

    _write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_dir.name,
            "created_at": _now_iso(),
            "config_path": str(config_path),
            "config": config,
        },
    )

    recon_service = None
    uv_service = None
    eval_service = None
    render_service = None
    if not args.dry_run:
        from .services.eval import EvalService
        from .services.reconstruction import ReconstructionService
        from .services.render import RenderService
        from .services.uv_projector import UVProjector

        recon_service = ReconstructionService()
        uv_service = UVProjector()
        eval_service = EvalService()
        render_service = RenderService()

    pipeline_cfg = config["pipeline"]
    recon_cfg = pipeline_cfg["reconstruction"]
    uv_cfg = pipeline_cfg["uv"]
    eval_cfg = pipeline_cfg["eval"]
    render_cfg = pipeline_cfg["render"]

    sample_records: List[SampleRecord] = []
    summary_rows: List[Dict[str, Any]] = []

    for sample_cfg in config["data"]["samples"]:
        sample_name = _sample_name(sample_cfg)
        if args.only_sample and sample_name != args.only_sample:
            continue

        sample_dir = path_manager.create_sample_dir(run_dir, sample_name)
        high_mesh_path = PathManager.resolve_path(sample_cfg["high_mesh"], repo_root)

        paths: Dict[str, str] = {"sample_dir": str(sample_dir), "high_mesh_input": str(high_mesh_path)}
        stats: Dict[str, Any] = {}
        status = "success"
        error: Optional[str] = None

        logger.info("sample_started", sample_name=sample_name, high_mesh=str(high_mesh_path))
        with run_profiler.step(f"sample:{sample_name}:total"):
            try:
                if args.dry_run:
                    status = "dry-run"
                else:
                    if not recon_cfg.get("enabled", True):
                        raise RuntimeError("pipeline.reconstruction.enabled=false is not supported in v1 infra")

                    with run_profiler.step(f"sample:{sample_name}:reconstruction"):
                        rec = recon_service.reconstruct(
                            sample_name=sample_name,
                            input_mesh_path=high_mesh_path,
                            output_dir=sample_dir,
                            resolution=int(recon_cfg["resolution"]),
                            tri_mode=str(recon_cfg["tri_mode"]),
                            margin=float(recon_cfg["margin"]),
                            device=str(pipeline_cfg.get("device", "auto")),
                            compute_flux=bool(recon_cfg.get("compute_flux", True)),
                            clamp_anchors=bool(recon_cfg.get("clamp_anchors", True)),
                            save_tokens=bool(recon_cfg.get("save_tokens", True)),
                            retry_on_empty_mesh=bool(recon_cfg.get("retry_on_empty_mesh", True)),
                            retry_resolutions=recon_cfg.get("retry_resolutions"),
                            min_level=int(recon_cfg.get("min_level", -1)),
                            solver_weights=recon_cfg.get("solver_weights"),
                            backend=str(recon_cfg.get("backend", "faithc")),
                            decimation_options=recon_cfg.get("pymeshlab"),
                        )
                    paths["high_mesh_normalized"] = str(rec.high_mesh_normalized_path)
                    paths["low_mesh"] = str(rec.low_mesh_path)
                    if rec.tokens_path:
                        paths["fct_tokens"] = str(rec.tokens_path)
                    stats.update(rec.stats)

                    uv_art = None
                    if uv_cfg.get("enabled", True):
                        uv_method = str(uv_cfg.get("method", "hybrid_global_opt"))
                        uv_options = {k: v for k, v in uv_cfg.items() if k not in {"enabled", "method"}}
                        with run_profiler.step(f"sample:{sample_name}:uv"):
                            uv_art = uv_service.project(
                                sample_name=sample_name,
                                high_mesh_path=rec.high_mesh_normalized_path,
                                low_mesh_path=rec.low_mesh_path,
                                output_dir=sample_dir,
                                method=uv_method,
                                device=str(pipeline_cfg.get("device", "auto")),
                                texture_source_path=high_mesh_path,
                                options=uv_options or None,
                            )
                        paths["low_mesh_uv"] = str(uv_art.low_mesh_uv_path)
                        paths["uv_map"] = str(uv_art.uv_map_path)
                        stats.update(uv_art.stats)

                    metrics_art = None
                    if eval_cfg.get("enabled", True):
                        with run_profiler.step(f"sample:{sample_name}:eval"):
                            metrics_art = eval_service.evaluate(
                                sample_name=sample_name,
                                high_mesh_path=rec.high_mesh_normalized_path,
                                low_mesh_path=rec.low_mesh_path,
                                output_dir=sample_dir,
                                sample_points=int(eval_cfg.get("sample_points", 10000)),
                                uv_artifact=uv_art,
                            )
                        paths["metrics"] = str(metrics_art.metrics_path)
                        stats.update(metrics_art.metrics)

                    with run_profiler.step(f"sample:{sample_name}:manifest"):
                        manifest_path = ArtifactRegistry.write_manifest(
                            sample_name=sample_name,
                            sample_dir=sample_dir,
                            high_mesh_path=rec.high_mesh_normalized_path,
                            low_mesh_path=rec.low_mesh_path,
                            low_mesh_uv_path=uv_art.low_mesh_uv_path if uv_art else None,
                            metrics_path=metrics_art.metrics_path if metrics_art else None,
                            preset=str(render_cfg.get("preset", "default")),
                        )
                    paths["manifest"] = str(manifest_path)

                    if render_cfg.get("enabled", False):
                        preset_name = str(render_cfg.get("preset", "default"))
                        preset_path = Path("renderer/mitsuba3/presets") / f"{preset_name}.yaml"
                        with run_profiler.step(f"sample:{sample_name}:render"):
                            render_art = render_service.render(
                                sample_name=sample_name,
                                manifest_path=manifest_path,
                                output_dir=sample_dir / "render",
                                preset=preset_name,
                                backend=str(render_cfg.get("backend", "mitsuba3")),
                                variant=str(render_cfg.get("variant", "cuda_ad_rgb")),
                                samples_per_pixel=int(render_cfg.get("samples_per_pixel", 64)),
                                preset_path=str(
                                    render_cfg.get("preset_path", str(preset_path) if preset_path.exists() else "")
                                ),
                            )
                        paths["render_dir"] = str(render_art.render_dir)
                        if render_art.log_path:
                            paths["render_log"] = str(render_art.log_path)
                        stats["render_status"] = render_art.status
                        if render_art.reason:
                            stats["render_reason"] = render_art.reason
                        if render_art.status == "failed":
                            raise RuntimeError(f"Render failed: {render_art.reason}")

            except Exception as exc:
                status = "failed"
                error = str(exc)
                logger.error("sample_failed", sample_name=sample_name, error=error)

        record = SampleRecord(
            sample_name=sample_name,
            status=status,
            paths=paths,
            stats=stats,
            error=error,
        )
        sample_records.append(record)

        summary_rows.append(
            {
                "sample_name": sample_name,
                "status": status,
                "low_faces": stats.get("num_low_faces"),
                "active_voxels": stats.get("active_voxels"),
                "chamfer_l1": stats.get("chamfer_l1"),
                "uv_out_of_bounds_ratio": stats.get("uv_out_of_bounds_ratio"),
                "error": error,
            }
        )
        logger.info("sample_finished", sample_name=sample_name, status=status)

    n_ok = sum(1 for s in sample_records if s.status == "success")
    n_fail = sum(1 for s in sample_records if s.status == "failed")

    perf_paths = _write_perf_reports(
        profiler=run_profiler,
        out_dir=run_dir / "perf",
        prefix="run_profile",
        extra={
            "run_id": run_dir.name,
            "samples_total": len(sample_records),
            "samples_success": n_ok,
            "samples_failed": n_fail,
        },
    )
    logger.info("run_profile_written", json=perf_paths["json"], text=perf_paths["txt"])

    run_index = {
        "run_id": run_dir.name,
        "created_at": _now_iso(),
        "config_path": str(config_path),
        "samples": [record.__dict__ for record in sample_records],
        "performance": perf_paths,
    }
    _write_json(run_dir / "run_index.json", run_index)
    _write_csv(run_dir / "summary.csv", summary_rows)

    logger.info("run_finished", run_id=run_dir.name, success=n_ok, failed=n_fail)

    print(f"run_id={run_dir.name}")
    print(f"run_dir={run_dir}")
    print(f"success={n_ok} failed={n_fail}")
    return 0 if n_fail == 0 else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from .services.eval import EvalService

    runs_dir = Path(args.runs_dir).resolve()
    run_dir = _resolve_run_dir(runs_dir, args.run_id)

    run_index_path = run_dir / "run_index.json"
    run_index = _load_run_index(run_index_path)

    eval_service = EvalService()
    summary_rows: List[Dict[str, Any]] = []

    for sample in run_index["samples"]:
        if sample["status"] != "success":
            continue

        sample_name = sample["sample_name"]
        paths = sample["paths"]

        uv_artifact = None
        if paths.get("uv_map") and paths.get("low_mesh_uv"):
            uv_artifact = UVArtifact(
                sample_name=sample_name,
                low_mesh_uv_path=Path(paths["low_mesh_uv"]),
                uv_map_path=Path(paths["uv_map"]),
                stats={},
            )

        metrics = eval_service.evaluate(
            sample_name=sample_name,
            high_mesh_path=Path(paths["high_mesh_normalized"]),
            low_mesh_path=Path(paths["low_mesh"]),
            output_dir=Path(paths["sample_dir"]),
            sample_points=int(args.sample_points),
            uv_artifact=uv_artifact,
        )

        paths["metrics"] = str(metrics.metrics_path)
        sample["stats"].update(metrics.metrics)

        summary_rows.append(
            {
                "sample_name": sample_name,
                "status": sample["status"],
                "chamfer_l1": sample["stats"].get("chamfer_l1"),
                "face_reduction_ratio": sample["stats"].get("face_reduction_ratio"),
            }
        )

    _write_json(run_index_path, run_index)
    _write_csv(run_dir / "summary_eval.csv", summary_rows)
    print(f"evaluation updated: {run_dir}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .services.render import RenderService

    runs_dir = Path(args.runs_dir).resolve()
    run_dir = _resolve_run_dir(runs_dir, args.run_id)

    run_index_path = run_dir / "run_index.json"
    run_meta_path = run_dir / "run_meta.json"
    run_index = _load_run_index(run_index_path)
    run_meta = _load_run_index(run_meta_path)

    render_cfg = run_meta.get("config", {}).get("pipeline", {}).get("render", {})
    render_service = RenderService()

    summary_rows: List[Dict[str, Any]] = []

    for sample in run_index["samples"]:
        if sample["status"] != "success":
            continue

        sample_name = sample["sample_name"]
        paths = sample["paths"]

        if not paths.get("manifest"):
            sample["stats"]["render_status"] = "skipped"
            sample["stats"]["render_reason"] = "manifest missing"
            continue

        preset_name = str(args.preset or render_cfg.get("preset", "default"))
        preset_path = Path("renderer/mitsuba3/presets") / f"{preset_name}.yaml"

        render_art = render_service.render(
            sample_name=sample_name,
            manifest_path=Path(paths["manifest"]),
            output_dir=Path(paths["sample_dir"]) / "render",
            preset=preset_name,
            backend=str(render_cfg.get("backend", "mitsuba3")),
            variant=str(render_cfg.get("variant", "cuda_ad_rgb")),
            samples_per_pixel=int(render_cfg.get("samples_per_pixel", 64)),
            preset_path=str(render_cfg.get("preset_path", str(preset_path) if preset_path.exists() else "")),
        )

        paths["render_dir"] = str(render_art.render_dir)
        if render_art.log_path:
            paths["render_log"] = str(render_art.log_path)
        sample["stats"]["render_status"] = render_art.status
        if render_art.reason:
            sample["stats"]["render_reason"] = render_art.reason

        summary_rows.append(
            {
                "sample_name": sample_name,
                "render_status": render_art.status,
                "render_reason": render_art.reason,
            }
        )

    _write_json(run_index_path, run_index)
    _write_csv(run_dir / "summary_render.csv", summary_rows)
    print(f"render pass finished: {run_dir}")
    return 0


def _resolve_viewer_bin(repo_root: Path, requested: Optional[str]) -> Optional[Path]:
    if requested:
        candidate = Path(requested).expanduser().resolve()
        return candidate if candidate.exists() else None

    env_bin_value = os.environ.get("FAITHC_VIEWER_BIN", "").strip()
    if env_bin_value:
        env_bin = Path(env_bin_value).expanduser()
        if env_bin.exists():
            return env_bin.resolve()

    candidates = [
        repo_root / "viewer/opengl_previewer/build/faithc_viewer",
        repo_root / "viewer/opengl_previewer/build/Release/faithc_viewer",
        repo_root / "viewer/opengl_previewer/build/Debug/faithc_viewer",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def cmd_preview(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()
    viewer_bin = _resolve_viewer_bin(repo_root, args.viewer_bin)
    if viewer_bin is None:
        print(
            "preview viewer binary not found. Build it first:\n"
            "  cd viewer/opengl_previewer\n"
            "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build -j"
        )
        return 1

    bridge_script = Path(args.bridge_script).expanduser().resolve() if args.bridge_script else (
        repo_root / "tools/preview/run_faithc_preview.py"
    ).resolve()
    if not bridge_script.exists():
        print(f"preview bridge script not found: {bridge_script}")
        return 1

    python_bin = str(Path(args.python_bin).expanduser().resolve()) if args.python_bin else sys.executable
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    profiler_enabled = _resolve_profiler_enabled(args)
    preview_profiler = ExecutionProfiler(
        name="faithc_cli_preview_launcher",
        config=ProfilerConfig(
            enabled=profiler_enabled,
            cprofile_enabled=not bool(args.profile_no_cprofile),
            top_k=int(args.profile_top_k),
        ),
        metadata={
            "command": "preview",
            "repo_root": str(repo_root),
            "viewer_bin": str(viewer_bin),
            "bridge_script": str(bridge_script),
            "python_bin": python_bin,
        },
    )
    preview_profiler.start()

    cmd = [
        str(viewer_bin),
        "--repo-root",
        str(repo_root),
        "--python-bin",
        python_bin,
        "--bridge-script",
        str(bridge_script),
        "--work-dir",
        str(work_dir),
    ]

    if args.mesh:
        mesh_path = Path(args.mesh).expanduser().resolve()
        cmd.extend(["--mesh", str(mesh_path)])

    viewer_env = os.environ.copy()
    viewer_env["FAITHC_PREVIEW_PROFILE"] = "1" if profiler_enabled else "0"
    viewer_env["FAITHC_PREVIEW_PROFILE_NO_CPROFILE"] = "1" if bool(args.profile_no_cprofile) else "0"
    viewer_env["FAITHC_PREVIEW_PROFILE_TOP_K"] = str(int(args.profile_top_k))

    with preview_profiler.step("preview:launch_viewer"):
        proc = subprocess.run(cmd, check=False, env=viewer_env)

    perf_paths = _write_perf_reports(
        profiler=preview_profiler,
        out_dir=work_dir / "perf",
        prefix=f"preview_launcher_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        extra={"return_code": int(proc.returncode)},
    )
    print(f"preview_perf_json={perf_paths['json']}")
    print(f"preview_perf_txt={perf_paths['txt']}")
    return int(proc.returncode)


def cmd_report_stage2(args: argparse.Namespace) -> int:
    from .reporting_stage2 import build_stage2_report, render_stage2_markdown

    runs_dir = Path(args.runs_dir).resolve()
    hard_samples = [s.strip() for s in str(args.hard_samples).split(",") if s.strip()]
    report = build_stage2_report(
        runs_dir=runs_dir,
        baseline_run_id=str(args.baseline_run),
        method2_run_id=str(args.method2_run),
        method4_run_id=str(args.method4_run),
        hard_samples=hard_samples,
        improve_threshold=float(args.improve_threshold),
        hard_flip_threshold=float(args.hard_flip_threshold),
    )

    reports_dir = runs_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = Path(args.output_md).resolve() if args.output_md else (reports_dir / f"stage2_{stamp}.md")
    json_path = Path(args.output_json).resolve() if args.output_json else (reports_dir / f"stage2_{stamp}.json")

    _write_json(json_path, report)
    md_text = render_stage2_markdown(report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"stage2_report_json={json_path}")
    print(f"stage2_report_md={md_path}")
    print(f"stage2_any_passed={report['stage2_any_passed']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faithc-exp",
        description="FaithC experiment infrastructure CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute reconstruction/uv/eval/render pipeline")
    run_parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    run_parser.add_argument("--run-id", default=None, help="Optional fixed run id")
    run_parser.add_argument("--only-sample", default=None, help="Execute only one sample name")
    run_parser.add_argument("--dry-run", action="store_true", help="Validate config and generate run layout only")
    run_parser.add_argument(
        "--profiler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable built-in profiler (preferred switch)",
    )
    run_parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(Legacy) Enable built-in profiler (timing/hotspots/memory) for this run",
    )
    run_parser.add_argument(
        "--profile-top-k",
        type=int,
        default=60,
        help="Top-K hotspot functions to keep in profiler reports",
    )
    run_parser.add_argument(
        "--profile-no-cprofile",
        action="store_true",
        help="Disable cProfile hotspots and keep lightweight stage/memory metrics only",
    )
    run_parser.set_defaults(func=cmd_run)

    eval_parser = subparsers.add_parser("eval", help="Recompute metrics for an existing run")
    eval_parser.add_argument("-r", "--run-id", required=True, help="Run id under runs-dir")
    eval_parser.add_argument("--runs-dir", default="experiments/runs", help="Base runs directory")
    eval_parser.add_argument("--sample-points", type=int, default=10000, help="Surface samples for chamfer")
    eval_parser.set_defaults(func=cmd_eval)

    render_parser = subparsers.add_parser("render", help="Run render backend for existing manifests")
    render_parser.add_argument("-r", "--run-id", required=True, help="Run id under runs-dir")
    render_parser.add_argument("--runs-dir", default="experiments/runs", help="Base runs directory")
    render_parser.add_argument("--preset", default=None, help="Override preset name")
    render_parser.set_defaults(func=cmd_render)

    preview_parser = subparsers.add_parser("preview", help="Launch OpenGL interactive previewer")
    preview_parser.add_argument("--mesh", default=None, help="Optional initial mesh path")
    preview_parser.add_argument(
        "--viewer-bin",
        default=None,
        help="Path to compiled viewer binary (defaults to viewer/opengl_previewer/build/*)",
    )
    preview_parser.add_argument("--python-bin", default=None, help="Python executable used by FaithC bridge")
    preview_parser.add_argument(
        "--bridge-script",
        default=None,
        help="Path to tools/preview/run_faithc_preview.py",
    )
    preview_parser.add_argument(
        "--work-dir",
        default="experiments/runs/preview_tmp",
        help="Directory for preview temporary outputs",
    )
    preview_parser.add_argument(
        "--profiler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable built-in profiler for preview launcher (preferred switch)",
    )
    preview_parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(Legacy) Enable built-in profiler for preview launcher (default: off)",
    )
    preview_parser.add_argument(
        "--profile-top-k",
        type=int,
        default=40,
        help="Top-K hotspot functions for preview launcher profiler",
    )
    preview_parser.add_argument(
        "--profile-no-cprofile",
        action="store_true",
        help="Disable cProfile hotspots for preview launcher profiler",
    )
    preview_parser.set_defaults(func=cmd_preview)

    report_parser = subparsers.add_parser(
        "report-stage2",
        help="Compare Stage2 runs against Stage1 baseline and evaluate roadmap gate",
    )
    report_parser.add_argument(
        "--runs-dir",
        default="experiments/runs",
        help="Base runs directory",
    )
    report_parser.add_argument(
        "--baseline-run",
        required=True,
        help="Baseline run id (typically Stage1 best)",
    )
    report_parser.add_argument(
        "--method2-run",
        required=True,
        help="Stage2 Method2 run id",
    )
    report_parser.add_argument(
        "--method4-run",
        required=True,
        help="Stage2 Method4 run id",
    )
    report_parser.add_argument(
        "--hard-samples",
        default="aksfbx,rotarycannon,massive_nordic_coastal_cliff_vdssailfa_raw",
        help="Comma-separated hard sample names used for flip-ratio gate",
    )
    report_parser.add_argument(
        "--improve-threshold",
        type=float,
        default=0.10,
        help="Required relative improvement threshold for bad-tri and color L1",
    )
    report_parser.add_argument(
        "--hard-flip-threshold",
        type=float,
        default=0.05,
        help="Hard-sample mean flip-ratio threshold",
    )
    report_parser.add_argument(
        "--output-md",
        default=None,
        help="Optional output markdown path",
    )
    report_parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path",
    )
    report_parser.set_defaults(func=cmd_report_stage2)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
