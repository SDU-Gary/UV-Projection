#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.halfedge_topology import compute_high_face_uv_islands
from faithc_infra.services.uv.correspondence import build_high_cuda_context
from faithc_infra.services.uv.mesh_sanitizer import sanitize_mesh_for_halfedge
from faithc_infra.services.uv.options import DEFAULT_OPTIONS, deep_merge_dict
from faithc_infra.services.uv.semantic_transfer import transfer_face_semantics_by_projection


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError(f"scene has no mesh: {path}")
        return trimesh.util.concatenate(geoms)
    raise RuntimeError(f"unsupported mesh type: {type(loaded)}")


def _safe_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        raise RuntimeError("mesh has no UV")
    uv_np = np.asarray(uv, dtype=np.float64)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        raise RuntimeError("mesh UV shape invalid")
    if uv_np.shape[0] != int(len(mesh.vertices)):
        raise RuntimeError("mesh UV/vertex length mismatch")
    return uv_np


def _face_neighbors(mesh: trimesh.Trimesh) -> List[List[int]]:
    n_faces = int(len(mesh.faces))
    neigh: List[List[int]] = [[] for _ in range(n_faces)]
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[1] != 2:
        return neigh
    for a, b in adj.tolist():
        ia = int(a)
        ib = int(b)
        if 0 <= ia < n_faces and 0 <= ib < n_faces and ia != ib:
            neigh[ia].append(ib)
            neigh[ib].append(ia)
    return neigh


def _component_sizes_for_label(
    *,
    label: int,
    labels: np.ndarray,
    neighbors: List[List[int]],
) -> List[int]:
    face_ids = np.where(labels == int(label))[0]
    if face_ids.size == 0:
        return []
    is_label = np.zeros((labels.shape[0],), dtype=np.bool_)
    is_label[face_ids] = True
    seen = np.zeros((labels.shape[0],), dtype=np.bool_)
    comp_sizes: List[int] = []
    for fid in face_ids.tolist():
        if seen[fid]:
            continue
        q: deque[int] = deque([int(fid)])
        seen[fid] = True
        size = 0
        while q:
            cur = q.popleft()
            size += 1
            for nb in neighbors[cur]:
                if is_label[nb] and (not seen[nb]):
                    seen[nb] = True
                    q.append(int(nb))
        comp_sizes.append(int(size))
    comp_sizes.sort(reverse=True)
    return comp_sizes


def _audit_semantic_connectivity(
    *,
    labels: np.ndarray,
    neighbors: List[List[int]],
    main_ratio_threshold: float,
    tiny_abs_threshold: int,
    tiny_ratio_threshold: float,
    tiny_max_components: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    labels_i64 = np.asarray(labels, dtype=np.int64).reshape(-1)
    valid_ids = np.unique(labels_i64[labels_i64 >= 0]).astype(np.int64)
    unknown_faces = int(np.count_nonzero(labels_i64 < 0))

    per_label: List[Dict[str, Any]] = []
    fragmented_labels = 0
    severe_labels = 0
    intrusion_labels = 0

    for lid in valid_ids.tolist():
        comp_sizes = _component_sizes_for_label(label=int(lid), labels=labels_i64, neighbors=neighbors)
        total = int(np.sum(comp_sizes)) if len(comp_sizes) > 0 else 0
        main = int(comp_sizes[0]) if len(comp_sizes) > 0 else 0
        comp_count = int(len(comp_sizes))
        main_ratio = float(main / max(1, total))
        leaked = int(max(0, total - main))
        leaked_ratio = float(leaked / max(1, total))

        tiny_threshold = int(
            max(
                2,
                min(
                    int(max(2, tiny_abs_threshold)),
                    int(max(2, round(total * max(0.0, tiny_ratio_threshold)))),
                ),
            )
        )
        tiny_components = int(sum(1 for s in comp_sizes if s <= tiny_threshold))
        non_tiny_components = int(max(0, comp_count - tiny_components))

        is_normal = bool(
            main_ratio >= float(main_ratio_threshold)
            and non_tiny_components <= 1
            and tiny_components <= int(tiny_max_components)
        )
        is_severe = bool((main_ratio < 0.80) or (comp_count >= 10) or (leaked_ratio >= 0.20))
        is_intrusion = bool(leaked_ratio >= 0.05)

        if not is_normal:
            fragmented_labels += 1
        if is_severe:
            severe_labels += 1
        if is_intrusion:
            intrusion_labels += 1

        per_label.append(
            {
                "label_id": int(lid),
                "face_count": int(total),
                "component_count": int(comp_count),
                "largest_component_faces": int(main),
                "largest_component_ratio": float(main_ratio),
                "non_main_faces": int(leaked),
                "non_main_ratio": float(leaked_ratio),
                "tiny_component_threshold_faces": int(tiny_threshold),
                "tiny_component_count": int(tiny_components),
                "non_tiny_component_count": int(non_tiny_components),
                "component_sizes_top10": [int(v) for v in comp_sizes[:10]],
                "normal_pass": bool(is_normal),
                "severe_fragmentation": bool(is_severe),
                "intrusion_like": bool(is_intrusion),
            }
        )

    per_label.sort(key=lambda x: (x["normal_pass"], x["largest_component_ratio"], -x["face_count"]))

    summary = {
        "face_count_total": int(labels_i64.shape[0]),
        "face_count_unknown": int(unknown_faces),
        "label_count": int(valid_ids.size),
        "fragmented_label_count": int(fragmented_labels),
        "severe_label_count": int(severe_labels),
        "intrusion_like_label_count": int(intrusion_labels),
        "normal_overall": bool(fragmented_labels == 0),
    }
    return summary, per_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit semantic connectivity per low-face semantic ID (without seam cutting)"
    )
    p.add_argument("--high", type=Path, required=True, help="High mesh (with UV)")
    p.add_argument("--low", type=Path, required=True, help="Low mesh")
    p.add_argument("--out-json", type=Path, default=None, help="Optional output JSON report")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for semantic projection raycasting",
    )
    p.add_argument(
        "--sanitize-low",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply low-mesh manifold sanitization before semantic projection",
    )
    p.add_argument("--main-ratio-threshold", type=float, default=0.95)
    p.add_argument("--tiny-abs-threshold", type=int, default=16)
    p.add_argument("--tiny-ratio-threshold", type=float, default=0.005)
    p.add_argument("--tiny-max-components", type=int, default=2)
    return p.parse_args()


def _resolve_device(name: str) -> str:
    if name in {"cpu", "cuda"}:
        return name
    import torch

    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _patch_atom3d_kernels_for_current_gpu() -> Dict[str, Any]:
    import torch
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

    kernels_mod._cumtv_cuda = cumtv_cuda
    kernels_mod._kernel_loaded = True
    kernels_mod.get_cuda_kernels = lambda: cumtv_cuda

    bvh_kernels_mod._bvh_cuda = bvh_cuda
    bvh_kernels_mod.get_bvh_kernels = lambda: bvh_cuda

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
    import torch
    import atom3d.kernels as kernels_mod

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


def _ensure_atom3d_runtime_for_device(device: str) -> Dict[str, Any]:
    if str(device) != "cuda":
        return {"atom3d_runtime_patched": False, "reason": "device_not_cuda"}
    import torch

    if not torch.cuda.is_available():
        return {
            "atom3d_runtime_patched": False,
            "reason": "torch_cuda_unavailable",
        }

    import atom3d

    atom3d_root = Path(atom3d.__file__).resolve().parent
    kernel_file = atom3d_root / "kernels" / "cumtv_kernels.cu"
    if not kernel_file.exists():
        return {
            "atom3d_runtime_patched": False,
            "reason": f"missing_kernel_source:{kernel_file}",
        }

    try:
        diag = _patch_atom3d_kernels_for_current_gpu()
        _smoke_test_atom3d_kernels()
    except Exception as exc:
        return {
            "atom3d_runtime_patched": False,
            "reason": f"runtime_patch_failed:{exc}",
        }

    out: Dict[str, Any] = dict(diag)
    out["atom3d_runtime_patched"] = True
    out["atom3d_smoke_test"] = "passed"
    return out


def main() -> None:
    args = parse_args()
    high_mesh = _load_mesh(args.high.resolve())
    low_mesh_in = _load_mesh(args.low.resolve())
    high_uv = _safe_uv(high_mesh)

    opts = deep_merge_dict(DEFAULT_OPTIONS, {})
    seam_cfg = dict(opts.get("seam", {}))
    corr_cfg = dict(opts.get("correspondence", {}))

    sanitize_meta: Dict[str, Any] = {}
    if bool(args.sanitize_low):
        low_mesh, sanitize_meta = sanitize_mesh_for_halfedge(low_mesh=low_mesh_in, seam_cfg=seam_cfg)
    else:
        low_mesh = trimesh.Trimesh(
            vertices=np.asarray(low_mesh_in.vertices, dtype=np.float32),
            faces=np.asarray(low_mesh_in.faces, dtype=np.int64),
            process=False,
        )

    high_face_island, high_meta = compute_high_face_uv_islands(
        vertices=np.asarray(high_mesh.vertices, dtype=np.float64),
        faces=np.asarray(high_mesh.faces, dtype=np.int64),
        uv=np.asarray(high_uv, dtype=np.float64),
        position_eps=float(seam_cfg.get("high_position_eps", 1e-6)),
        uv_eps=float(seam_cfg.get("high_uv_eps", 1e-5)),
    )

    resolved_device = _resolve_device(str(args.device))
    runtime_diag = _ensure_atom3d_runtime_for_device(resolved_device)
    high_ctx = build_high_cuda_context(
        high_mesh=high_mesh,
        high_uv=high_uv,
        device=resolved_device,
    )
    semantic = transfer_face_semantics_by_projection(
        high_ctx=high_ctx,
        high_face_island=np.asarray(high_face_island, dtype=np.int64),
        low_mesh=low_mesh,
        seam_cfg=seam_cfg,
        corr_cfg=corr_cfg,
    )
    low_labels = np.asarray(semantic["low_face_island"], dtype=np.int64).reshape(-1)
    neighbors = _face_neighbors(low_mesh)
    summary, per_label = _audit_semantic_connectivity(
        labels=low_labels,
        neighbors=neighbors,
        main_ratio_threshold=float(args.main_ratio_threshold),
        tiny_abs_threshold=int(args.tiny_abs_threshold),
        tiny_ratio_threshold=float(args.tiny_ratio_threshold),
        tiny_max_components=int(args.tiny_max_components),
    )

    report: Dict[str, Any] = {
        "high_mesh": str(args.high.resolve()),
        "low_mesh_input": str(args.low.resolve()),
        "low_mesh_used_face_count": int(len(low_mesh.faces)),
        "low_mesh_used_vertex_count": int(len(low_mesh.vertices)),
        "device_used": resolved_device,
        "runtime_diag": runtime_diag,
        "sanitize_low": bool(args.sanitize_low),
        "sanitize_meta": sanitize_meta,
        "high_uv_island_meta": high_meta,
        "semantic_transfer_meta": semantic.get("meta", {}),
        "summary": summary,
        "labels": per_label,
    }

    print(
        "[semantic_connectivity] "
        f"labels={summary['label_count']}, unknown_faces={summary['face_count_unknown']}, "
        f"fragmented_labels={summary['fragmented_label_count']}, severe_labels={summary['severe_label_count']}, "
        f"intrusion_like_labels={summary['intrusion_like_label_count']}, normal_overall={summary['normal_overall']}"
    )

    top = per_label[: min(15, len(per_label))]
    for item in top:
        print(
            "[semantic_connectivity][label] "
            f"id={item['label_id']}, faces={item['face_count']}, comps={item['component_count']}, "
            f"main_ratio={item['largest_component_ratio']:.4f}, non_main_ratio={item['non_main_ratio']:.4f}, "
            f"tiny={item['tiny_component_count']}, normal={item['normal_pass']}"
        )

    if args.out_json is not None:
        out = args.out_json.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report_json={out}")


if __name__ == "__main__":
    main()
