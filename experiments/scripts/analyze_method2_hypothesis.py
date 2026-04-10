#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

from faithc_infra.services.uv.linear_solver import interpolate_sample_uv
from faithc_infra.services.uv.method2_pipeline import run_method2_gradient_poisson
from faithc_infra.services.uv.options import DEFAULT_OPTIONS, deep_merge_dict
from faithc_infra.services.uv.quality import compute_uv_quality, texture_reprojection_error
from faithc_infra.services.uv.texture_io import extract_uv, resolve_basecolor_image
from faithc_infra.services.uv_projector import UVProjector


@dataclass
class RunRow:
    resolution: int
    baseline_run: str
    method2_run: str
    method4_run: str
    report_json: str
    report_md: str


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_image_rgb_bilinear(image, uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0

    h, w = arr.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = u * (w - 1)
    y = (1.0 - v) * (h - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    c00 = arr[y0, x0]
    c10 = arr[y0, x1]
    c01 = arr[y1, x0]
    c11 = arr[y1, x1]

    c0 = c00 * (1.0 - wx)[:, None] + c10 * wx[:, None]
    c1 = c01 * (1.0 - wx)[:, None] + c11 * wx[:, None]
    c = c0 * (1.0 - wy)[:, None] + c1 * wy[:, None]
    return c.astype(np.float32)


def _texture_reprojection_error_bilinear(
    image,
    target_uv: np.ndarray,
    pred_uv: np.ndarray,
) -> Tuple[float | None, float | None]:
    if image is None or target_uv.size == 0 or pred_uv.size == 0:
        return None, None
    tgt = _sample_image_rgb_bilinear(image, target_uv)
    pred = _sample_image_rgb_bilinear(image, pred_uv)
    diff = tgt - pred
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    return l1, l2


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0.0):
        return 0.0
    xs = np.sort(np.clip(x, 0.0, None))
    n = xs.size
    cum = np.cumsum(xs)
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def _compute_face_uv_jacobian(mesh: trimesh.Trimesh, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tri = verts[faces]

    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]

    g11 = np.sum(e1 * e1, axis=1)
    g12 = np.sum(e1 * e2, axis=1)
    g22 = np.sum(e2 * e2, axis=1)

    det = g11 * g22 - g12 * g12
    valid = np.isfinite(det) & (det > 1e-18)

    pinv = np.zeros((len(faces), 2, 3), dtype=np.float64)
    if np.any(valid):
        inv_det = 1.0 / det[valid]
        m00 = g22[valid] * inv_det
        m01 = -g12[valid] * inv_det
        m10 = -g12[valid] * inv_det
        m11 = g11[valid] * inv_det

        e1v = e1[valid]
        e2v = e2[valid]
        pinv[valid, 0] = m00[:, None] * e1v + m01[:, None] * e2v
        pinv[valid, 1] = m10[:, None] * e1v + m11[:, None] * e2v

    tri_uv = np.asarray(uv, dtype=np.float64)[faces]
    du1 = tri_uv[:, 1] - tri_uv[:, 0]
    du2 = tri_uv[:, 2] - tri_uv[:, 0]
    uv_grad = np.stack([du1, du2], axis=2)

    jac = np.zeros((len(faces), 2, 3), dtype=np.float64)
    if np.any(valid):
        jac[valid] = np.einsum("fij,fjk->fik", uv_grad[valid], pinv[valid], optimize=True)
    return jac, valid


def _load_csv_rows(path: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                RunRow(
                    resolution=int(row["resolution"]),
                    baseline_run=row["baseline_run"],
                    method2_run=row["method2_run"],
                    method4_run=row["method4_run"],
                    report_json=row["report_json"],
                    report_md=row["report_md"],
                )
            )
    return rows


def _run_method2_once(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    cfg: Dict[str, Any],
    image,
    device: str,
    return_internal: bool,
):
    projector = UVProjector()
    high_uv = extract_uv(high_mesh)
    if high_uv is None:
        raise RuntimeError("High mesh has no per-vertex UV")
    return run_method2_gradient_poisson(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        high_uv=high_uv,
        image=image,
        device=device,
        cfg=cfg,
        nearest_mapper=projector._map_nearest_vertex,
        barycentric_mapper=projector._map_barycentric_closest,
        return_internal=return_internal,
    )


def _resolve_asset_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _build_static_summary(repo_root: Path, row: RunRow) -> Dict[str, Any]:
    m2_meta = _read_json(repo_root / "experiments" / "runs" / row.method2_run / "run_meta.json")
    base_stats = _read_json(repo_root / "experiments" / "runs" / row.baseline_run / "massive_cliff" / "uv_stats.json")
    m2_stats = _read_json(repo_root / "experiments" / "runs" / row.method2_run / "massive_cliff" / "uv_stats.json")
    m4_stats = _read_json(repo_root / "experiments" / "runs" / row.method4_run / "massive_cliff" / "uv_stats.json")
    report = _read_json(Path(row.report_json))

    sample_cfg = m2_meta["config"]["data"]["samples"][0]
    uv_cfg = m2_meta["config"]["pipeline"]["uv"]
    recon_cfg = m2_meta["config"]["pipeline"]["reconstruction"]

    return {
        "resolution": row.resolution,
        "run_ids": {
            "baseline": row.baseline_run,
            "method2": row.method2_run,
            "method4": row.method4_run,
        },
        "asset": {
            "sample_name": sample_cfg["name"],
            "high_mesh_input": sample_cfg["high_mesh"],
        },
        "reconstruction": {
            "resolution": recon_cfg.get("resolution"),
            "margin": recon_cfg.get("margin"),
            "tri_mode": recon_cfg.get("tri_mode"),
            "min_level": recon_cfg.get("min_level"),
            "retry_resolutions": recon_cfg.get("retry_resolutions"),
        },
        "uv_method": {
            "baseline": "hybrid_global_opt",
            "method2": uv_cfg.get("method"),
            "method4": "method4_jacobian_injective",
            "sample": uv_cfg.get("sample"),
            "correspondence": uv_cfg.get("correspondence"),
            "solve": uv_cfg.get("solve"),
            "method2_cfg": uv_cfg.get("method2"),
            "method4_cfg": uv_cfg.get("method4"),
        },
        "reported_metrics": {
            "baseline": {
                "uv_color_reproj_l1": base_stats.get("uv_color_reproj_l1"),
                "uv_color_reproj_l2": base_stats.get("uv_color_reproj_l2"),
                "uv_flip_ratio": base_stats.get("uv_flip_ratio"),
                "uv_bad_tri_ratio": base_stats.get("uv_bad_tri_ratio"),
            },
            "method2": {
                "uv_color_reproj_l1": m2_stats.get("uv_color_reproj_l1"),
                "uv_color_reproj_l2": m2_stats.get("uv_color_reproj_l2"),
                "uv_flip_ratio": m2_stats.get("uv_flip_ratio"),
                "uv_bad_tri_ratio": m2_stats.get("uv_bad_tri_ratio"),
                "uv_correspondence_success_ratio": m2_stats.get("uv_correspondence_success_ratio"),
                "uv_correspondence_primary_ratio": m2_stats.get("uv_correspondence_primary_ratio"),
                "uv_m2_anchor_count_total": m2_stats.get("uv_m2_anchor_count_total"),
                "uv_m2_anchor_mode_used": m2_stats.get("uv_m2_anchor_mode_used"),
            },
            "method4": {
                "uv_solver_stage": m4_stats.get("uv_solver_stage"),
                "uv_m4_refine_status": m4_stats.get("uv_m4_refine_status"),
                "uv_m4_barrier_violation_ratio": m4_stats.get("uv_m4_barrier_violation_ratio"),
            },
        },
        "stage2_report": report,
    }


def _build_cuda_diagnostics(
    *,
    repo_root: Path,
    row: RunRow,
    run_sweeps: bool,
) -> Dict[str, Any]:
    m2_run_root = repo_root / "experiments" / "runs" / row.method2_run
    m2_meta = _read_json(m2_run_root / "run_meta.json")
    sample_cfg = m2_meta["config"]["data"]["samples"][0]
    sample_name = sample_cfg["name"]
    sample_dir = m2_run_root / sample_name

    high_mesh = trimesh.load(sample_dir / "mesh_high_normalized.glb", force="mesh", process=False)
    low_mesh = trimesh.load(sample_dir / "mesh_low.glb", force="mesh", process=False)

    source_path = _resolve_asset_path(repo_root, str(sample_cfg["high_mesh"]))
    uv_cfg = deep_merge_dict(DEFAULT_OPTIONS, m2_meta["config"]["pipeline"]["uv"])
    image, image_source = resolve_basecolor_image(high_mesh, source_path)

    mapped_uv, stats, _, internal = _run_method2_once(
        high_mesh=high_mesh,
        low_mesh=low_mesh,
        cfg=uv_cfg,
        image=image,
        device="cuda",
        return_internal=True,
    )
    if internal is None:
        raise RuntimeError("method2 internal diagnostics unavailable (likely CUDA fallback path)")

    sf = np.asarray(internal.solve_sample_face_ids, dtype=np.int64)
    sb = np.asarray(internal.solve_sample_bary, dtype=np.float64)
    target_uv = np.asarray(internal.solve_target_uv, dtype=np.float64)
    pred_uv = interpolate_sample_uv(np.asarray(low_mesh.faces, dtype=np.int64), sf, sb, mapped_uv).astype(np.float64)

    delta = target_uv - pred_uv
    delta_norm = np.linalg.norm(delta, axis=1)
    mean_delta = delta.mean(axis=0)
    median_delta = np.median(delta, axis=0)
    pred_shift_mean = pred_uv + mean_delta[None, :]
    pred_shift_median = pred_uv + median_delta[None, :]

    nn_l1, nn_l2 = texture_reprojection_error(image, target_uv, pred_uv)
    nn_l1_shift_mean, nn_l2_shift_mean = texture_reprojection_error(image, target_uv, pred_shift_mean)
    nn_l1_shift_median, nn_l2_shift_median = texture_reprojection_error(image, target_uv, pred_shift_median)

    bl_l1, bl_l2 = _texture_reprojection_error_bilinear(image, target_uv, pred_uv)
    bl_l1_shift_mean, bl_l2_shift_mean = _texture_reprojection_error_bilinear(image, target_uv, pred_shift_mean)
    bl_l1_shift_median, bl_l2_shift_median = _texture_reprojection_error_bilinear(image, target_uv, pred_shift_median)

    n_faces = int(len(low_mesh.faces))
    counts = np.bincount(sf, minlength=n_faces)
    covered = counts > 0

    jac_pred, jac_geom_valid = _compute_face_uv_jacobian(low_mesh, mapped_uv)
    jac_target = np.asarray(internal.face_target_jacobian, dtype=np.float64)
    jac_target_valid = np.asarray(internal.face_target_valid_mask, dtype=np.bool_)
    jac_valid = jac_target_valid & jac_geom_valid
    jac_residual = np.linalg.norm((jac_pred - jac_target).reshape(len(jac_target), -1), axis=1)
    jac_residual_valid = jac_residual[jac_valid]

    anchors = np.asarray(internal.anchor_vertex_ids, dtype=np.int64)
    anchor_uv = np.asarray(internal.anchor_uv, dtype=np.float64)
    if anchors.size > 0:
        anchor_dev = np.linalg.norm(mapped_uv[anchors] - anchor_uv, axis=1)
        anchor_dev_mean = float(np.mean(anchor_dev))
        anchor_dev_max = float(np.max(anchor_dev))
    else:
        anchor_dev_mean = 0.0
        anchor_dev_max = 0.0

    oob = np.logical_or(pred_uv < 0.0, pred_uv > 1.0).any(axis=1)

    out: Dict[str, Any] = {
        "image_available": image is not None,
        "image_source": image_source,
        "uv_offset": {
            "mean_delta_uv": mean_delta.tolist(),
            "median_delta_uv": median_delta.tolist(),
            "mean_delta_norm": float(np.mean(delta_norm)),
            "median_delta_norm": float(np.median(delta_norm)),
            "p95_delta_norm": float(np.percentile(delta_norm, 95)),
        },
        "color_error_recheck": {
            "nearest": {
                "original": {"l1": nn_l1, "l2": nn_l2},
                "shift_mean": {"l1": nn_l1_shift_mean, "l2": nn_l2_shift_mean},
                "shift_median": {"l1": nn_l1_shift_median, "l2": nn_l2_shift_median},
            },
            "bilinear": {
                "original": {"l1": bl_l1, "l2": bl_l2},
                "shift_mean": {"l1": bl_l1_shift_mean, "l2": bl_l2_shift_mean},
                "shift_median": {"l1": bl_l1_shift_median, "l2": bl_l2_shift_median},
            },
        },
        "sample_distribution": {
            "num_faces": n_faces,
            "num_valid_samples": int(len(sf)),
            "covered_faces": int(np.count_nonzero(covered)),
            "covered_face_ratio": float(np.mean(covered)),
            "samples_per_face_mean": float(np.mean(counts)),
            "samples_per_face_median": float(np.median(counts)),
            "samples_per_face_p90": float(np.percentile(counts, 90)),
            "samples_per_face_p99": float(np.percentile(counts, 99)),
            "samples_per_face_max": int(np.max(counts) if counts.size > 0 else 0),
            "samples_per_face_gini": _gini(counts.astype(np.float64)),
        },
        "jacobian_fit": {
            "target_valid_face_ratio": float(np.mean(jac_target_valid)),
            "effective_valid_face_ratio": float(np.mean(jac_valid)),
            "residual_mean": float(np.mean(jac_residual_valid)) if jac_residual_valid.size > 0 else 0.0,
            "residual_median": float(np.median(jac_residual_valid)) if jac_residual_valid.size > 0 else 0.0,
            "residual_p90": float(np.percentile(jac_residual_valid, 90)) if jac_residual_valid.size > 0 else 0.0,
            "residual_p95": float(np.percentile(jac_residual_valid, 95)) if jac_residual_valid.size > 0 else 0.0,
        },
        "anchor_consistency": {
            "anchor_count": int(anchors.size),
            "anchor_uv_deviation_mean": anchor_dev_mean,
            "anchor_uv_deviation_max": anchor_dev_max,
        },
        "pred_uv_oob": {
            "sample_oob_ratio": float(np.mean(oob)) if oob.size > 0 else 0.0,
        },
        "method2_recompute_stats": {
            "uv_color_reproj_l1": stats.get("uv_color_reproj_l1"),
            "uv_color_reproj_l2": stats.get("uv_color_reproj_l2"),
            "uv_correspondence_success_ratio": stats.get("uv_correspondence_success_ratio"),
            "uv_correspondence_primary_ratio": stats.get("uv_correspondence_primary_ratio"),
            "uv_island_conflict_faces": stats.get("uv_island_conflict_faces"),
            "uv_island_unknown_faces": stats.get("uv_island_unknown_faces"),
            "uv_cross_seam_faces": stats.get("uv_cross_seam_faces"),
            "uv_m2_anchor_count_total": stats.get("uv_m2_anchor_count_total"),
            "uv_m2_anchor_mode_used": stats.get("uv_m2_anchor_mode_used"),
        },
    }

    if run_sweeps:
        out["sweeps"] = {
            "huber": [],
            "anchor_points": [],
        }
        for huber_delta in [0.5, 1.0, 1.5, 2.5, 4.0]:
            cfg_huber = deep_merge_dict(uv_cfg, {"method2": {"huber_delta": huber_delta}})
            uv_h, st_h, _ = _run_method2_once(
                high_mesh=high_mesh,
                low_mesh=low_mesh,
                cfg=cfg_huber,
                image=image,
                device="cuda",
                return_internal=False,
            )
            q_h = compute_uv_quality(low_mesh, uv_h)
            out["sweeps"]["huber"].append(
                {
                    "huber_delta": huber_delta,
                    "uv_color_reproj_l1": st_h.get("uv_color_reproj_l1"),
                    "uv_color_reproj_l2": st_h.get("uv_color_reproj_l2"),
                    "uv_flip_ratio": q_h.get("uv_flip_ratio"),
                    "uv_bad_tri_ratio": q_h.get("uv_bad_tri_ratio"),
                }
            )

        for k in [2, 4, 8, 16]:
            cfg_anchor = deep_merge_dict(
                uv_cfg,
                {"method2": {"anchor_mode": "component_minimal", "anchor_points_per_component": k}},
            )
            uv_a, st_a, _ = _run_method2_once(
                high_mesh=high_mesh,
                low_mesh=low_mesh,
                cfg=cfg_anchor,
                image=image,
                device="cuda",
                return_internal=False,
            )
            q_a = compute_uv_quality(low_mesh, uv_a)
            out["sweeps"]["anchor_points"].append(
                {
                    "anchor_points_per_component": k,
                    "uv_m2_anchor_count_total": st_a.get("uv_m2_anchor_count_total"),
                    "uv_color_reproj_l1": st_a.get("uv_color_reproj_l1"),
                    "uv_color_reproj_l2": st_a.get("uv_color_reproj_l2"),
                    "uv_flip_ratio": q_a.get("uv_flip_ratio"),
                    "uv_bad_tri_ratio": q_a.get("uv_bad_tri_ratio"),
                }
            )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Method2 hypotheses on UV reset runs")
    parser.add_argument("--run-ids-csv", required=True, help="Path to run_ids.csv from run_uv_reset_massive.sh")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--run-cuda", action="store_true", help="Run CUDA-dependent diagnostics")
    parser.add_argument("--run-sweeps", action="store_true", help="Run huber/anchor sensitivity sweeps (requires --run-cuda)")
    parser.add_argument("--only-resolution", default="", help="Comma-separated list, e.g. 64,128")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    rows = _load_csv_rows(Path(args.run_ids_csv).resolve())
    if args.only_resolution.strip():
        allowed = {int(x.strip()) for x in args.only_resolution.split(",") if x.strip()}
        rows = [r for r in rows if r.resolution in allowed]

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "run_ids_csv": str(Path(args.run_ids_csv).resolve()),
        "entries": [],
    }

    if args.run_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable in current runtime. "
                    "Run this script in your GPU environment with the same command."
                )
        except Exception as exc:
            raise RuntimeError(
                "CUDA precheck failed before diagnostics. "
                "Please run in your GPU environment."
            ) from exc

    for row in rows:
        entry: Dict[str, Any] = _build_static_summary(repo_root, row)
        if args.run_cuda:
            entry["cuda_diagnostics"] = _build_cuda_diagnostics(
                repo_root=repo_root,
                row=row,
                run_sweeps=args.run_sweeps,
            )
        report["entries"].append(entry)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = (repo_root / "experiments" / "reports" / f"method2_hypothesis_diag_{ts}.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"method2_diag_json={output_path}")


if __name__ == "__main__":
    main()
