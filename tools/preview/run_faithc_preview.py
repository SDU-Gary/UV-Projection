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

from faithcontour import FCTDecoder, FCTEncoder
from faithc_infra.profiler import ExecutionProfiler, ProfilerConfig
from faithc_infra.services.uv_projector import UVProjector
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
    accepted_raw = payload.pop("uv_m2_face_accepted_samples", None)
    total_raw = payload.pop("uv_m2_face_total_samples", None)
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
    sidecar.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

    nonzero = int(np.count_nonzero(accepted))
    vmax = int(np.max(accepted)) if accepted.size > 0 else 0
    return {
        "uv_m2_face_sample_counts_path": str(sidecar),
        "uv_m2_face_sample_faces": int(accepted.shape[0]),
        "uv_m2_face_sample_nonzero": nonzero,
        "uv_m2_face_sample_max": vmax,
    }


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


def _patch_atom3d_kernels_for_current_gpu() -> Dict[str, Any]:
    from torch.utils.cpp_extension import load

    import atom3d
    from atom3d.core import mesh_bvh as mesh_bvh_mod
    from atom3d.kernels import bvh as bvh_kernels_mod
    import atom3d.kernels as kernels_mod

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}{minor}"

    atom3d_root = Path(atom3d.__file__).resolve().parent
    kernels_root = atom3d_root / "kernels"
    cumtv_src = kernels_root / "cumtv_kernels.cu"
    bvh_src = kernels_root / "bvh_kernels.cu"
    if not cumtv_src.exists() or not bvh_src.exists():
        raise RuntimeError(
            "Atom3d CUDA source files are missing. Expected files:\n"
            f"- {cumtv_src}\n"
            f"- {bvh_src}"
        )

    gencode = f"-gencode=arch=compute_{arch},code=sm_{arch}"
    gencode_ptx = f"-gencode=arch=compute_{arch},code=compute_{arch}"

    cumtv_build = kernels_root / "build"
    bvh_build = kernels_root / "build" / "bvh"
    cumtv_build.mkdir(parents=True, exist_ok=True)
    bvh_build.mkdir(parents=True, exist_ok=True)

    cumtv_name = f"cumtv_cuda_sm{arch}"
    bvh_name = f"bvh_cuda_sm{arch}"

    try:
        cumtv_cuda = load(
            name=cumtv_name,
            sources=[str(cumtv_src)],
            build_directory=str(cumtv_build),
            extra_cuda_cflags=["-O3", "--use_fast_math", gencode, gencode_ptx],
            verbose=False,
        )
        bvh_cuda = load(
            name=bvh_name,
            sources=[str(bvh_src)],
            build_directory=str(bvh_build),
            extra_cuda_cflags=["-O3", gencode, gencode_ptx],
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to compile Atom3d CUDA kernels for current GPU arch "
            f"compute capability {major}.{minor} (gencode {gencode}). "
            f"Original error: {exc}"
        ) from exc

    kernels_mod._cumtv_cuda = cumtv_cuda
    kernels_mod._kernel_loaded = True
    kernels_mod.get_cuda_kernels = lambda: cumtv_cuda

    bvh_kernels_mod._bvh_cuda = bvh_cuda
    bvh_kernels_mod.get_bvh_kernels = lambda: bvh_cuda

    # Ensure MeshBVH path sees CUDA/BVH as available in this process.
    mesh_bvh_mod.HAS_CUDA = True
    mesh_bvh_mod.HAS_BVH = True
    mesh_bvh_mod.BVHAccelerator = bvh_kernels_mod.BVHAccelerator

    return {
        "gpu_compute_capability": f"{major}.{minor}",
        "atom3d_arch": arch,
        "atom3d_cumtv_module": cumtv_name,
        "atom3d_bvh_module": bvh_name,
    }


def _smoke_test_atom3d_kernels() -> None:
    import atom3d.kernels as kernels_mod

    # A single triangle crossing a unit square on z=0 plane.
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    aabb_min = torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float32, device="cuda")
    aabb_max = torch.tensor([[1.0, 1.0, 0.1]], dtype=torch.float32, device="cuda")

    hit_mask, _, _ = kernels_mod.triangle_aabb_intersect(vertices, faces, aabb_min, aabb_max)
    if hit_mask.numel() == 0 or not bool(hit_mask[0].item()):
        raise RuntimeError(
            "Atom3d kernel smoke test failed: triangle_aabb_intersect expected a hit but got miss."
        )


def _validate_runtime(device: str) -> Dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "FaithC preview bridge requires CUDA runtime. "
            f"Resolved device='{device}', torch.cuda.is_available()={torch.cuda.is_available()}."
        )

    import atom3d

    atom3d_root = Path(atom3d.__file__).resolve().parent
    kernel_file = atom3d_root / "kernels" / "cumtv_kernels.cu"
    if not kernel_file.exists():
        raise RuntimeError(
            "Atom3d CUDA kernel source is missing: "
            f"{kernel_file}. Reinstall Atom3d from source according to README."
        )

    kernel_diag = _patch_atom3d_kernels_for_current_gpu()
    _smoke_test_atom3d_kernels()
    kernel_diag["atom3d_smoke_test"] = "passed"
    return kernel_diag


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
    kernel_setup_t0 = time.time()
    with _profile_step(profiler, "preview:kernel_setup"):
        kernel_diag = _validate_runtime(device)
    kernel_setup_seconds = time.time() - kernel_setup_t0

    cuda_diag: Dict[str, Any] = {}
    cuda_device = torch.device(device)
    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        props = torch.cuda.get_device_properties(cuda_device)
        cuda_diag["cuda_device_name"] = props.name
        cuda_diag["cuda_device_capability"] = f"{props.major}.{props.minor}"
    from atom3d import MeshBVH
    from atom3d.grid import OctreeIndexer
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    grid_bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)

    max_level = int(math.log2(args.resolution))
    auto_min_level = min(4, max(1, max_level - 1))
    requested_min_level = int(args.min_level)

    if requested_min_level > 0:
        if requested_min_level > max_level:
            raise ValueError(
                f"min_level must be <= max_level ({max_level}) for resolution={args.resolution}, got {requested_min_level}"
            )
        min_level_candidates = [requested_min_level]
    else:
        min_level_candidates = [auto_min_level, 3, 2, 1]
    min_level_candidates = [lv for lv in min_level_candidates if 1 <= lv <= max_level]
    # preserve order while deduplicating
    min_level_candidates = list(dict.fromkeys(min_level_candidates))

    start_encode = time.time()
    with _profile_step(profiler, "preview:encode"):
        bvh = MeshBVH(vertices, faces, device=device)
        octree = OctreeIndexer(max_level=max_level, bounds=grid_bounds, device=device)
        encoder = FCTEncoder(bvh=bvh, octree=octree, device=device)
        fct_result = None
        min_level_used = None
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
    uv_diag: Dict[str, Any] = {
        "uv_projected": False,
        "uv_source_has_uv": False,
    }
    export_mesh = recon_mesh
    if bool(args.project_uv):
        uv_service = UVProjector()
        uv_mode_map = {
            "nearest": "nearest_vertex",
            "barycentric": "barycentric_closest_point",
            "hybrid": "hybrid_global_opt",
            "method2": "method2_gradient_poisson",
            "method4": "method4_jacobian_injective",
            "auto": "auto",
        }
        uv_method = uv_mode_map[str(args.uv_mode)]
        uv_options = {
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
        }

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

    with _profile_step(profiler, "preview:export_mesh"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_mesh.export(out_path)

    import atom3d
    import faithcontour

    if cuda_device.type == "cuda":
        torch.cuda.synchronize(cuda_device)
        cuda_diag["torch_cuda_peak_alloc_mb"] = round(
            float(torch.cuda.max_memory_allocated(cuda_device)) / (1024.0 * 1024.0), 3
        )
        cuda_diag["torch_cuda_peak_reserved_mb"] = round(
            float(torch.cuda.max_memory_reserved(cuda_device)) / (1024.0 * 1024.0), 3
        )

    return {
        "success": True,
        "input_mesh": str(in_path),
        "output_mesh": str(out_path),
        "device": device,
        "resolution": int(args.resolution),
        "tri_mode": str(args.tri_mode),
        "margin": float(args.margin),
        "min_level_requested": requested_min_level,
        "min_level_used": int(min_level_used),
        "min_level_tried": min_level_candidates,
        "num_input_faces": num_input_faces,
        "num_output_faces": num_output_faces,
        "face_reduction_ratio": (num_output_faces / num_input_faces) if num_input_faces > 0 else None,
        "active_voxels": active_voxels,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "kernel_setup_seconds": kernel_setup_seconds,
        "total_seconds": time.time() - start_total,
        "faithcontour_path": str(Path(faithcontour.__file__).resolve()),
        "atom3d_path": str(Path(atom3d.__file__).resolve()),
        "kernel_diag": kernel_diag,
        "cuda_diag": cuda_diag,
        **uv_diag,
    }


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
        "--project-uv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transfer UV from source mesh to FaithC output",
    )
    parser.add_argument(
        "--uv-mode",
        type=str,
        default="method2",
        choices=["hybrid", "method2", "method4", "barycentric", "nearest", "auto"],
        help="UV transfer mode: hybrid, method2 gradient-poisson, method4 jacobian-injective, barycentric, nearest, or auto fallback",
    )
    parser.add_argument(
        "--uv-batch-size",
        type=int,
        default=200000,
        help="Chunk size for BVH projection queries",
    )
    parser.add_argument("--uv-sample-base-per-face", type=int, default=4, help="Hybrid: base samples per low-mesh face")
    parser.add_argument("--uv-sample-min-per-face", type=int, default=3, help="Hybrid: min samples per low-mesh face")
    parser.add_argument("--uv-sample-max-per-face", type=int, default=12, help="Hybrid: max samples per low-mesh face")
    parser.add_argument("--uv-sample-seed", type=int, default=12345, help="Hybrid: random seed for face sampling")
    parser.add_argument("--uv-normal-weight", type=float, default=0.2, help="Hybrid fallback nearest normal penalty weight")
    parser.add_argument("--uv-normal-dot-min", type=float, default=0.7, help="Hybrid primary/fallback normal-dot threshold")
    parser.add_argument("--uv-ray-max-dist-ratio", type=float, default=0.08, help="Hybrid primary max distance ratio to bbox diagonal")
    parser.add_argument("--uv-fallback-k", type=int, default=8, help="Hybrid fallback nearest candidate count")
    parser.add_argument("--uv-fallback-weight", type=float, default=0.7, help="Hybrid fallback correspondence confidence weight")
    parser.add_argument(
        "--uv-solve-backend",
        type=str,
        default="auto",
        choices=["auto", "cuda_pcg", "cpu_scipy"],
        help="Hybrid UV solve backend",
    )
    parser.add_argument("--uv-lambda-smooth", type=float, default=2e-4, help="Hybrid linear solve Laplacian smooth weight")
    parser.add_argument("--uv-pcg-max-iter", type=int, default=2000, help="Hybrid CUDA-PCG max iterations")
    parser.add_argument("--uv-pcg-tol", type=float, default=1e-6, help="Hybrid CUDA-PCG relative tolerance")
    parser.add_argument("--uv-pcg-check-every", type=int, default=25, help="Hybrid CUDA-PCG full residual recompute interval")
    parser.add_argument(
        "--uv-pcg-preconditioner",
        type=str,
        default="jacobi",
        choices=["jacobi", "none"],
        help="Hybrid CUDA-PCG preconditioner type",
    )
    parser.add_argument("--uv-cg-max-iter", type=int, default=2000, help="Hybrid linear solve conjugate-gradient max iterations")
    parser.add_argument("--uv-cg-tol", type=float, default=1e-6, help="Hybrid linear solve conjugate-gradient tolerance")
    parser.add_argument("--uv-anchor-weight", type=float, default=1e2, help="Method2/Hybrid anchor penalty weight")
    parser.add_argument("--uv-ridge-eps", type=float, default=1e-8, help="Hybrid linear solve diagonal ridge regularization")
    parser.add_argument(
        "--uv-seam-strategy",
        type=str,
        default="legacy",
        choices=["legacy", "halfedge_island"],
        help="Seam handling strategy: legacy sampling heuristic or halfedge UV-island split",
    )
    parser.add_argument("--uv-seam-uv-span-threshold", type=float, default=0.35, help="Mark faces as cross-seam when sample UV span exceeds this value")
    parser.add_argument("--uv-seam-min-valid-samples", type=int, default=2, help="Minimum valid samples required to judge one face as cross-seam")
    parser.add_argument(
        "--uv-exclude-cross-seam-faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude detected cross-seam faces from global linear system",
    )
    parser.add_argument(
        "--uv-local-vertex-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate cross-seam face vertices during export to preserve per-face UV",
    )
    parser.add_argument(
        "--uv-island-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guard hybrid correspondence with high-mesh UV-island consistency constraints",
    )
    parser.add_argument(
        "--uv-island-guard-mode",
        type=str,
        default="soft",
        choices=["soft", "strict"],
        help="Island guard mode: soft (with fallback) or strict (no fallback)",
    )
    parser.add_argument(
        "--uv-island-guard-confidence-min",
        type=float,
        default=0.55,
        help="Minimum face-level island confidence required before enforcing island guard",
    )
    parser.add_argument(
        "--uv-island-guard-allow-unknown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow correspondences landing on unknown high-mesh islands",
    )
    parser.add_argument(
        "--uv-island-guard-fallback",
        type=str,
        default="nearest_same_island_then_udf",
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
        default=True,
        help="Enable texture-gradient correspondence weighting",
    )
    parser.add_argument("--uv-grad-weight-gamma", type=float, default=1.0, help="Texture-gradient weighting gamma")
    parser.add_argument("--uv-max-texture-weight", type=float, default=5.0, help="Texture-gradient max sample weight")
    parser.add_argument("--uv-m2-outlier-sigma", type=float, default=4.0, help="Method2 IRLS outlier rejection sigma")
    parser.add_argument(
        "--uv-m2-outlier-quantile",
        type=float,
        default=0.95,
        help="Method2 IRLS outlier rejection quantile",
    )
    parser.add_argument(
        "--uv-m2-min-samples-per-face",
        type=int,
        default=2,
        help="Method2 minimum valid samples per face",
    )
    parser.add_argument(
        "--uv-m2-face-weight-floor",
        type=float,
        default=1e-6,
        help="Method2 face weight floor",
    )
    parser.add_argument(
        "--uv-m2-anchor-mode",
        type=str,
        default="component_minimal",
        choices=["component_minimal", "boundary", "none"],
        help="Method2 anchor mode",
    )
    parser.add_argument(
        "--uv-m2-anchor-points-per-component",
        type=int,
        default=4,
        help="Method2 anchor points per connected component",
    )
    parser.add_argument(
        "--uv-m2-use-island-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Method2: enable island-guard filtering in gradient-poisson solve",
    )
    parser.add_argument("--uv-m2-irls-iters", type=int, default=2, help="Method2 IRLS iterations")
    parser.add_argument("--uv-m2-huber-delta", type=float, default=3.0, help="Method2 Huber delta")
    parser.add_argument(
        "--uv-m2-post-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Method2: apply global UV translation alignment from sample residual median",
    )
    parser.add_argument(
        "--uv-m2-post-align-min-samples",
        type=int,
        default=64,
        help="Method2: minimum valid samples required to run post-alignment",
    )
    parser.add_argument(
        "--uv-m2-post-align-max-shift",
        type=float,
        default=0.25,
        help="Method2: maximum allowed global UV alignment shift",
    )
    parser.add_argument(
        "--uv-m2-laplacian-mode",
        type=str,
        default="cotan",
        choices=["uniform", "cotan"],
        help="Method2 Laplacian mode",
    )
    parser.add_argument(
        "--uv-m2-system-cond-estimate",
        type=str,
        default="diag_ratio",
        choices=["diag_ratio", "eigsh"],
        help="Method2 system condition-number estimate method",
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

        payload = {
            "success": False,
            "error": str(exc),
            "input_mesh": str(Path(args.input).resolve()),
            "output_mesh": str(Path(args.output).resolve()),
            "resolution": int(args.resolution),
            "tri_mode": str(args.tri_mode),
            "margin": float(args.margin),
            "min_level_requested": int(args.min_level),
            "project_uv": bool(args.project_uv),
            "uv_mode": str(args.uv_mode),
            "uv_batch_size": int(args.uv_batch_size),
            "faithcontour_path": faithcontour_path,
            "atom3d_path": atom3d_path,
            "kernel_diag": None,
        }
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
