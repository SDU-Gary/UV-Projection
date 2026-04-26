from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import trimesh

from .atom3d_runtime import ensure_atom3d_cuda_runtime, merge_runtime_diag
from .decimation import decimate_with_pymeshlab_qem
from ..mesh_io import MeshIO
from ..types import ReconstructionArtifact


class ReconstructionService:
    def reconstruct(
        self,
        sample_name: str,
        input_mesh_path: Path,
        output_dir: Path,
        resolution: int,
        tri_mode: str,
        margin: float,
        device: str,
        compute_flux: bool,
        clamp_anchors: bool,
        save_tokens: bool,
        retry_on_empty_mesh: bool = True,
        retry_resolutions: Optional[Iterable[int]] = None,
        min_level: int = -1,
        solver_weights: Optional[Dict[str, float]] = None,
        backend: str = "faithc",
        decimation_options: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionArtifact:
        if resolution <= 0 or (resolution & (resolution - 1)) != 0:
            raise ValueError(f"resolution must be power of two, got: {resolution}")

        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_high = MeshIO.load_mesh(input_mesh_path, process=False)
        mesh_high = self._normalize_mesh(mesh_high, margin=margin)
        self._drop_degenerate_faces(mesh_high)

        high_mesh_normalized_path = output_dir / "mesh_high_normalized.glb"
        MeshIO.export_mesh(mesh_high, high_mesh_normalized_path)

        backend_name = str(backend).strip().lower()
        if backend_name not in {"faithc", "pymeshlab_qem"}:
            raise ValueError(f"Unsupported reconstruction backend: {backend}")

        if backend_name == "pymeshlab_qem":
            t0 = time.perf_counter()
            decimated = decimate_with_pymeshlab_qem(
                mesh_high,
                **dict(decimation_options or {}),
            )
            low_mesh_path = output_dir / "mesh_low.glb"
            MeshIO.export_mesh(decimated.mesh_low, low_mesh_path)
            total_seconds = round(time.perf_counter() - t0, 6)
            stats: Dict[str, Any] = {
                "reconstruction_backend_requested": backend_name,
                "reconstruction_backend_used": backend_name,
                "resolution_requested": int(resolution),
                "resolution": None,
                "resolution_used": None,
                "tri_mode": tri_mode,
                "retry_on_empty_mesh": False,
                "resolution_attempts": [],
                "reconstruction_retry_count": 0,
                "attempts": [],
                "kernel_diag": None,
                "runtime_diag": None,
                "encode_seconds": 0.0,
                "decode_seconds": 0.0,
                "total_seconds": total_seconds,
                **decimated.stats,
            }
            with (output_dir / "reconstruction_stats.json").open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)
            return ReconstructionArtifact(
                sample_name=sample_name,
                high_mesh_normalized_path=high_mesh_normalized_path,
                low_mesh_path=low_mesh_path,
                tokens_path=None,
                stats=stats,
            )

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError(
                "FaithC reconstruction requires CUDA. "
                f"Resolved device='{device}', torch.cuda.is_available()={torch.cuda.is_available()}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FaithC reconstruction requires CUDA, but torch.cuda.is_available() is False "
                "in the current runtime."
            )

        from atom3d.grid import OctreeIndexer
        from faithcontour import FCTDecoder, FCTEncoder

        kernel_diag = ensure_atom3d_cuda_runtime(device=device, strict=True, require_cuda=True)
        from atom3d import MeshBVH  # Import after runtime patching.

        resolution_schedule = self._build_resolution_schedule(
            base_resolution=resolution,
            retry_on_empty_mesh=retry_on_empty_mesh,
            retry_resolutions=retry_resolutions,
        )

        t0 = time.perf_counter()
        vertices = torch.tensor(mesh_high.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh_high.faces, dtype=torch.long, device=device)
        bvh = MeshBVH(vertices, faces, device=device)
        bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)

        attempts: list[Dict[str, Any]] = []
        chosen: Optional[Dict[str, Any]] = None
        for res in resolution_schedule:
            run = self._reconstruct_once(
                resolution=int(res),
                tri_mode=tri_mode,
                device=device,
                compute_flux=compute_flux,
                clamp_anchors=clamp_anchors,
                bvh=bvh,
                bounds=bounds,
                octree_cls=OctreeIndexer,
                encoder_cls=FCTEncoder,
                decoder_cls=FCTDecoder,
                min_level=min_level,
                solver_weights=solver_weights,
            )
            attempts.append(run["stats"])
            if run["stats"]["active_voxels"] > 0 and run["stats"]["num_low_faces"] > 0:
                chosen = run
                break

        total_seconds = round(time.perf_counter() - t0, 6)
        if chosen is None:
            stats: Dict[str, Any] = {
                "resolution_requested": int(resolution),
                "resolution": int(resolution),
                "resolution_used": None,
                "tri_mode": tri_mode,
                "retry_on_empty_mesh": bool(retry_on_empty_mesh),
                "resolution_attempts": [int(r) for r in resolution_schedule],
                "attempts": attempts,
                "kernel_diag": kernel_diag,
                "active_voxels": 0,
                "num_low_vertices": 0,
                "num_low_faces": 0,
                "encode_seconds": round(float(sum(float(a["encode_seconds"]) for a in attempts)), 6),
                "decode_seconds": round(float(sum(float(a["decode_seconds"]) for a in attempts)), 6),
                "total_seconds": total_seconds,
            }
            merge_runtime_diag(stats, kernel_diag)
            with (output_dir / "reconstruction_stats.json").open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)
            raise RuntimeError(
                "FaithC reconstruction produced empty mesh at all attempted resolutions: "
                f"{[int(r) for r in resolution_schedule]}. "
                "Increase resolution and/or margin."
            )

        result = chosen["result"]
        mesh_low = chosen["mesh_low"]
        resolution_used = int(chosen["stats"]["resolution"])

        low_mesh_path = output_dir / "mesh_low.glb"
        MeshIO.export_mesh(mesh_low, low_mesh_path)

        tokens_path = None
        if save_tokens:
            tokens_path = output_dir / "fct_tokens.npz"
            np.savez_compressed(
                tokens_path,
                active_voxel_indices=result.active_voxel_indices.detach().cpu().numpy(),
                anchor=result.anchor.detach().cpu().numpy(),
                normal=result.normal.detach().cpu().numpy(),
                edge_flux_sign=result.edge_flux_sign.detach().cpu().numpy(),
            )

        stats: Dict[str, Any] = {
            "reconstruction_backend_requested": backend_name,
            "reconstruction_backend_used": backend_name,
            "resolution_requested": int(resolution),
            "resolution": int(resolution_used),
            "resolution_used": int(resolution_used),
            "tri_mode": tri_mode,
            "retry_on_empty_mesh": bool(retry_on_empty_mesh),
            "resolution_attempts": [int(r) for r in resolution_schedule],
            "reconstruction_retry_count": int(max(0, len(attempts) - 1)),
            "attempts": attempts,
            "kernel_diag": kernel_diag,
            "active_voxels": int(chosen["stats"]["active_voxels"]),
            "num_low_vertices": int(chosen["stats"]["num_low_vertices"]),
            "num_low_faces": int(chosen["stats"]["num_low_faces"]),
            "encode_seconds": round(float(chosen["stats"]["encode_seconds"]), 6),
            "decode_seconds": round(float(chosen["stats"]["decode_seconds"]), 6),
            "total_seconds": total_seconds,
        }
        merge_runtime_diag(stats, kernel_diag)
        with (output_dir / "reconstruction_stats.json").open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        return ReconstructionArtifact(
            sample_name=sample_name,
            high_mesh_normalized_path=high_mesh_normalized_path,
            low_mesh_path=low_mesh_path,
            tokens_path=tokens_path,
            stats=stats,
        )

    @staticmethod
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

    @staticmethod
    def _drop_degenerate_faces(mesh: trimesh.Trimesh) -> None:
        valid_mask = mesh.nondegenerate_faces()
        if valid_mask.sum() < len(mesh.faces):
            mesh.update_faces(valid_mask)
            mesh.remove_unreferenced_vertices()

    @staticmethod
    def _build_resolution_schedule(
        *,
        base_resolution: int,
        retry_on_empty_mesh: bool,
        retry_resolutions: Optional[Iterable[int]],
    ) -> list[int]:
        schedule: list[int] = [int(base_resolution)]
        if not retry_on_empty_mesh:
            return schedule

        candidates = retry_resolutions
        if candidates is None:
            candidates = [16, 32, 64, 128, 256]

        for value in candidates:
            try:
                res = int(value)
            except Exception:
                continue
            if res <= 0 or (res & (res - 1)) != 0:
                continue
            if res <= int(base_resolution):
                continue
            if res not in schedule:
                schedule.append(res)
        return schedule

    @staticmethod
    def _reconstruct_once(
        *,
        resolution: int,
        tri_mode: str,
        device: str,
        compute_flux: bool,
        clamp_anchors: bool,
        bvh,
        bounds,
        octree_cls,
        encoder_cls,
        decoder_cls,
        min_level: int,
        solver_weights: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        max_level = int(math.log2(resolution))
        auto_min_level = min(4, max(1, max_level - 1))
        requested_min_level = int(min_level)
        if requested_min_level > 0:
            if requested_min_level > max_level:
                raise ValueError(
                    f"min_level must be <= max_level ({max_level}) for resolution={resolution}, "
                    f"got {requested_min_level}"
                )
            min_level_candidates = [requested_min_level]
        else:
            min_level_candidates = [auto_min_level, 3, 2, 1]
        min_level_candidates = [lv for lv in min_level_candidates if 1 <= lv <= max_level]
        min_level_candidates = list(dict.fromkeys(min_level_candidates))

        octree = octree_cls(max_level=max_level, bounds=bounds, device=device)

        encoder = encoder_cls(bvh, octree, device=device)
        t_encode = time.perf_counter()
        result = None
        min_level_used = auto_min_level
        for min_level_try in min_level_candidates:
            candidate = encoder.encode(
                min_level=min_level_try,
                solver_weights=solver_weights if solver_weights is not None else {"lambda_n": 1.0, "lambda_d": 1e-3, "weight_power": 1},
                compute_flux=compute_flux,
                clamp_anchors=clamp_anchors,
            )
            if int(candidate.active_voxel_indices.shape[0]) > 0:
                result = candidate
                min_level_used = int(min_level_try)
                break
            if result is None:
                result = candidate
                min_level_used = int(min_level_try)

        if result is None:
            raise RuntimeError(f"FaithC encode returned no result at resolution={resolution}")

        encode_time = time.perf_counter() - t_encode

        decoder = decoder_cls(resolution=resolution, bounds=bounds, device=device)
        t_decode = time.perf_counter()
        decoded = decoder.decode(
            active_voxel_indices=result.active_voxel_indices,
            anchors=result.anchor,
            edge_flux_sign=result.edge_flux_sign,
            normals=result.normal,
            triangulation_mode=tri_mode,
        )
        decode_time = time.perf_counter() - t_decode

        mesh_low = trimesh.Trimesh(
            vertices=decoded.vertices.detach().cpu().numpy(),
            faces=decoded.faces.detach().cpu().numpy(),
            process=False,
        )

        stats: Dict[str, Any] = {
            "resolution": int(resolution),
            "min_level_requested": int(requested_min_level),
            "min_level_used": int(min_level_used),
            "min_level_candidates": [int(v) for v in min_level_candidates],
            "active_voxels": int(result.active_voxel_indices.shape[0]),
            "num_low_vertices": int(mesh_low.vertices.shape[0]),
            "num_low_faces": int(mesh_low.faces.shape[0]),
            "encode_seconds": round(encode_time, 6),
            "decode_seconds": round(decode_time, 6),
        }
        return {
            "mesh_low": mesh_low,
            "result": result,
            "stats": stats,
        }
