from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_ATOM3D_RUNTIME_CACHE: Optional[Dict[str, Any]] = None
RUNTIME_DIAG_KEYS = (
    "gpu_compute_capability",
    "atom3d_arch",
    "atom3d_cumtv_module",
    "atom3d_bvh_module",
    "atom3d_runtime_patched",
    "atom3d_runtime_reason",
    "atom3d_smoke_test",
    "atom3d_bvh_smoke_test",
)


def _diag_failure(reason: str, **extra: Any) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "atom3d_runtime_patched": False,
        "atom3d_runtime_reason": str(reason),
        "reason": str(reason),
    }
    diag.update(extra)
    return diag


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


def _smoke_test_atom3d_bvh() -> None:
    import torch
    from atom3d import MeshBVH

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device="cuda")
    rays_o = torch.tensor([[0.25, 0.25, 1.0]], dtype=torch.float32, device="cuda")
    rays_d = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device="cuda")

    bvh = MeshBVH(vertices, faces, device="cuda")
    res = bvh.intersect_ray(rays_o, rays_d, max_t=2.0)
    if res.hit.numel() == 0 or not bool(res.hit[0].item()):
        raise RuntimeError("Atom3d BVH smoke test failed: MeshBVH.intersect_ray expected a hit but got miss.")
    if int(res.face_ids[0].item()) != 0:
        raise RuntimeError(
            "Atom3d BVH smoke test failed: MeshBVH.intersect_ray returned unexpected face id "
            f"{int(res.face_ids[0].item())}."
        )


def _patch_atom3d_kernels_for_current_gpu() -> Dict[str, Any]:
    import torch
    from torch.utils.cpp_extension import load

    import atom3d
    import atom3d.kernels as kernels_mod
    from atom3d.core import mesh_bvh as mesh_bvh_mod
    from atom3d.kernels import bvh as bvh_kernels_mod

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

    mesh_bvh_mod.HAS_CUDA = True
    mesh_bvh_mod.HAS_BVH = True
    mesh_bvh_mod.BVHAccelerator = bvh_kernels_mod.BVHAccelerator

    return {
        "gpu_compute_capability": f"{major}.{minor}",
        "atom3d_arch": arch,
        "atom3d_cumtv_module": cumtv_name,
        "atom3d_bvh_module": bvh_name,
    }


def ensure_atom3d_cuda_runtime(
    device: str,
    *,
    strict: bool = False,
    require_cuda: bool = False,
) -> Dict[str, Any]:
    global _ATOM3D_RUNTIME_CACHE

    resolved = str(device).strip().lower()
    if resolved != "cuda":
        diag = _diag_failure("device_not_cuda")
        if strict or require_cuda:
            raise RuntimeError(
                "Atom3d CUDA runtime is required for this path, but resolved device is "
                f"'{device}'."
            )
        return diag

    if _ATOM3D_RUNTIME_CACHE is not None:
        return dict(_ATOM3D_RUNTIME_CACHE)

    try:
        import torch
    except Exception as exc:
        diag = _diag_failure(f"torch_import_failed:{exc}")
        if strict:
            raise RuntimeError(diag["atom3d_runtime_reason"]) from exc
        return diag

    if not torch.cuda.is_available():
        diag = _diag_failure("torch_cuda_unavailable")
        if strict:
            raise RuntimeError(
                "Atom3d CUDA runtime is required, but torch.cuda.is_available() is False."
            )
        return diag

    try:
        import atom3d
    except Exception as exc:
        diag = _diag_failure(f"atom3d_import_failed:{exc}")
        if strict:
            raise RuntimeError("Failed to import atom3d runtime.") from exc
        return diag

    atom3d_root = Path(atom3d.__file__).resolve().parent
    kernel_file = atom3d_root / "kernels" / "cumtv_kernels.cu"
    if not kernel_file.exists():
        diag = _diag_failure(f"missing_kernel_source:{kernel_file}")
        if strict:
            raise RuntimeError(
                "Atom3d CUDA kernel source is missing: "
                f"{kernel_file}. Reinstall Atom3d from source according to README."
            )
        return diag

    try:
        diag = _patch_atom3d_kernels_for_current_gpu()
        _smoke_test_atom3d_kernels()
        _smoke_test_atom3d_bvh()
    except Exception as exc:
        failure = _diag_failure(f"runtime_patch_failed:{exc}")
        if strict:
            raise RuntimeError(failure["atom3d_runtime_reason"]) from exc
        return failure

    diag["atom3d_runtime_patched"] = True
    diag["atom3d_runtime_reason"] = "ok"
    diag["reason"] = "ok"
    diag["atom3d_smoke_test"] = "passed"
    diag["atom3d_bvh_smoke_test"] = "passed"
    _ATOM3D_RUNTIME_CACHE = dict(diag)
    return dict(_ATOM3D_RUNTIME_CACHE)


def compact_runtime_diag(diag: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(diag, dict):
        return None
    src = diag.get("runtime_diag")
    if isinstance(src, dict):
        diag = src
    out = {key: diag[key] for key in RUNTIME_DIAG_KEYS if key in diag}
    return out or None


def merge_runtime_diag(payload: Dict[str, Any], diag: Optional[Dict[str, Any]], *, overwrite: bool = False) -> None:
    runtime_diag = compact_runtime_diag(diag)
    if runtime_diag is None:
        payload.setdefault("runtime_diag", None)
        return
    payload["runtime_diag"] = dict(runtime_diag)
    for key, value in runtime_diag.items():
        if overwrite or key not in payload:
            payload[key] = value


__all__ = ["compact_runtime_diag", "ensure_atom3d_cuda_runtime", "merge_runtime_diag"]
