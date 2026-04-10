from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh


def sample_image_rgb(image, uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]

    h, w = arr.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.clip(np.round(u * (w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip(np.round((1.0 - v) * (h - 1)).astype(np.int64), 0, h - 1)
    return arr[y, x].astype(np.float32) / 255.0


def sample_image_scalar(scalar_image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = scalar_image.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)
    x = np.clip(np.round(u * (w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip(np.round((1.0 - v) * (h - 1)).astype(np.int64), 0, h - 1)
    return scalar_image[y, x].astype(np.float32)


def texture_gradient_weights(image, uv: np.ndarray, gamma: float, max_weight: float) -> np.ndarray:
    if image is None:
        return np.ones(len(uv), dtype=np.float64)

    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32) / 255.0
    else:
        rgb = arr[..., :3].astype(np.float32) / 255.0
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    gy, gx = np.gradient(gray)
    gm = np.sqrt(gx * gx + gy * gy)

    s = sample_image_scalar(gm, uv)
    base = np.mean(gm) + 1e-8
    w = 1.0 + gamma * (s / base)
    return np.clip(w, 1.0, max_weight).astype(np.float64)


def texture_reprojection_error(image, target_uv: np.ndarray, pred_uv: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if image is None or target_uv.size == 0 or pred_uv.size == 0:
        return None, None

    tgt = sample_image_rgb(image, target_uv)
    pred = sample_image_rgb(image, pred_uv)
    diff = tgt - pred
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    return l1, l2


def face_stretch_anisotropy(low_mesh: trimesh.Trimesh, uv: np.ndarray) -> np.ndarray:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    tri3 = np.asarray(low_mesh.vertices)[faces]
    tri2 = uv[faces]

    l3 = np.stack(
        [
            np.linalg.norm(tri3[:, 1] - tri3[:, 0], axis=1),
            np.linalg.norm(tri3[:, 2] - tri3[:, 1], axis=1),
            np.linalg.norm(tri3[:, 0] - tri3[:, 2], axis=1),
        ],
        axis=1,
    )
    l2 = np.stack(
        [
            np.linalg.norm(tri2[:, 1] - tri2[:, 0], axis=1),
            np.linalg.norm(tri2[:, 2] - tri2[:, 1], axis=1),
            np.linalg.norm(tri2[:, 0] - tri2[:, 2], axis=1),
        ],
        axis=1,
    )
    ratio = l2 / np.maximum(l3, 1e-12)
    return np.max(ratio, axis=1) / np.maximum(np.min(ratio, axis=1), 1e-12)


def bad_face_mask(low_mesh: trimesh.Trimesh, uv: np.ndarray, stretch_factor: float) -> np.ndarray:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    tri = uv[faces]
    signed = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (
        tri[:, 1, 1] - tri[:, 0, 1]
    ) * (tri[:, 2, 0] - tri[:, 0, 0])
    flips = signed <= 1e-10

    stretch = face_stretch_anisotropy(low_mesh, uv)
    if stretch.size > 0:
        p95 = np.percentile(stretch, 95)
        bad_stretch = stretch > max(1.0, p95 * stretch_factor)
    else:
        bad_stretch = np.zeros(len(faces), dtype=np.bool_)

    return np.logical_or(flips, bad_stretch)


def compute_uv_quality(low_mesh: trimesh.Trimesh, uv: np.ndarray) -> Dict[str, Any]:
    faces = np.asarray(low_mesh.faces, dtype=np.int64)
    tri_uv = uv[faces]

    e1 = tri_uv[:, 1] - tri_uv[:, 0]
    e2 = tri_uv[:, 2] - tri_uv[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    flip_ratio = float(np.mean(signed <= 1e-10)) if signed.size > 0 else 0.0

    stretch = face_stretch_anisotropy(low_mesh, uv)
    p95 = float(np.percentile(stretch, 95)) if stretch.size > 0 else 1.0
    p99 = float(np.percentile(stretch, 99)) if stretch.size > 0 else 1.0
    bad_ratio = float(np.mean(bad_face_mask(low_mesh, uv, stretch_factor=1.5))) if stretch.size > 0 else 0.0

    out_of_bounds = np.logical_or(uv < 0.0, uv > 1.0)
    return {
        "uv_out_of_bounds_ratio": float(out_of_bounds.any(axis=1).mean()),
        "num_low_vertices": int(len(uv)),
        "uv_flip_ratio": flip_ratio,
        "uv_bad_tri_ratio": bad_ratio,
        "uv_stretch_p95": p95,
        "uv_stretch_p99": p99,
    }


__all__ = [
    "bad_face_mask",
    "compute_uv_quality",
    "face_stretch_anisotropy",
    "sample_image_rgb",
    "sample_image_scalar",
    "texture_gradient_weights",
    "texture_reprojection_error",
]
