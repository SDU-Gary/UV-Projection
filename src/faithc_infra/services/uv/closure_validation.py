from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from ..halfedge_topology import compute_high_face_uv_islands
from .openmesh_seams import extract_seam_edges_openmesh, validate_face_partition_by_seams


def _safe_uv(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return None
    uv_np = np.asarray(uv, dtype=np.float64)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        return None
    if uv_np.shape[0] != int(len(mesh.vertices)):
        return None
    return uv_np


def _tri_area_3d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)


def _tri_area_2d(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = uv[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return 0.5 * np.abs(cross_z)


def _compute_island_bbox_iou_mean(
    *,
    high_uv: np.ndarray,
    high_faces: np.ndarray,
    high_labels: np.ndarray,
    low_uv: np.ndarray,
    low_faces: np.ndarray,
    low_labels: np.ndarray,
) -> float:
    def _bbox_per_label(uv: np.ndarray, faces: np.ndarray, labels: np.ndarray) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
        out: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        valid_ids = np.unique(labels[labels >= 0])
        for lb in valid_ids.tolist():
            mask = labels == int(lb)
            if not np.any(mask):
                continue
            pts = uv[faces[mask].reshape(-1)]
            if pts.size == 0:
                continue
            out[int(lb)] = (np.min(pts, axis=0), np.max(pts, axis=0))
        return out

    hb = _bbox_per_label(high_uv, high_faces, high_labels)
    lb = _bbox_per_label(low_uv, low_faces, low_labels)
    shared = sorted(set(hb.keys()) & set(lb.keys()))
    if len(shared) == 0:
        return 0.0
    ious = []
    for sid in shared:
        hmin, hmax = hb[sid]
        lmin, lmax = lb[sid]
        inter_min = np.maximum(hmin, lmin)
        inter_max = np.minimum(hmax, lmax)
        inter = np.maximum(inter_max - inter_min, 0.0)
        ia = float(inter[0] * inter[1])
        ha = float(max(0.0, (hmax[0] - hmin[0]) * (hmax[1] - hmin[1])))
        la = float(max(0.0, (lmax[0] - lmin[0]) * (lmax[1] - lmin[1])))
        ua = ha + la - ia
        ious.append(ia / ua if ua > 1e-12 else 0.0)
    return float(np.mean(ious))


def _compute_uv_stretch_metrics(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    area_3d = _tri_area_3d(vertices, faces)
    area_uv = _tri_area_2d(uv, faces)
    valid = (area_3d > 1e-12) & np.isfinite(area_3d) & np.isfinite(area_uv)
    if not np.any(valid):
        return {
            "stretch_mean": 0.0,
            "stretch_p95": 0.0,
            "stretch_p99": 0.0,
        }

    ratio = np.zeros_like(area_3d, dtype=np.float64)
    ratio[valid] = area_uv[valid] / area_3d[valid]
    stretch = np.zeros_like(ratio, dtype=np.float64)

    valid_ids = np.unique(labels[(labels >= 0) & valid])
    if valid_ids.size == 0:
        med = float(np.median(ratio[valid]))
        med = max(med, 1e-12)
        rv = ratio[valid] / med
        stretch[valid] = np.maximum(rv, 1.0 / np.maximum(rv, 1e-12))
    else:
        for lid in valid_ids.tolist():
            mask = valid & (labels == int(lid))
            if not np.any(mask):
                continue
            med = float(np.median(ratio[mask]))
            med = max(med, 1e-12)
            rv = ratio[mask] / med
            stretch[mask] = np.maximum(rv, 1.0 / np.maximum(rv, 1e-12))

    sv = stretch[valid]
    return {
        "stretch_mean": float(np.mean(sv)) if sv.size > 0 else 0.0,
        "stretch_p95": float(np.percentile(sv, 95.0)) if sv.size > 0 else 0.0,
        "stretch_p99": float(np.percentile(sv, 99.0)) if sv.size > 0 else 0.0,
    }


def _point_in_triangle_2d(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a = tri[0]
    b = tri[1]
    c = tri[2]
    v0 = c - a
    v1 = b - a
    v2 = points - a[None, :]
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(den)) < 1e-12:
        return np.zeros((points.shape[0],), dtype=np.bool_)
    inv_den = 1.0 / den
    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) * inv_den
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) * inv_den
    return (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)


def _estimate_island_overlap_ratio(
    *,
    uv: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    raster_res: int = 256,
    max_faces: int = 160_000,
) -> Dict[str, float]:
    if faces.shape[0] <= 0 or uv.shape[0] <= 0:
        return {"overlap_ratio": 0.0, "covered_ratio": 0.0}
    if int(faces.shape[0]) > int(max_faces):
        return {"overlap_ratio": -1.0, "covered_ratio": -1.0}

    uv_valid = uv[np.isfinite(uv).all(axis=1)]
    if uv_valid.size == 0:
        return {"overlap_ratio": 0.0, "covered_ratio": 0.0}

    uv_min = np.min(uv_valid, axis=0)
    uv_max = np.max(uv_valid, axis=0)
    span = np.maximum(uv_max - uv_min, 1e-8)

    res = int(max(64, raster_res))
    owner = np.full((res, res), -1, dtype=np.int32)
    overlap_mask = np.zeros((res, res), dtype=np.bool_)

    uv_norm = (uv - uv_min[None, :]) / span[None, :]
    uv_px = uv_norm * float(res - 1)

    for fi in range(int(faces.shape[0])):
        lid = int(labels[fi]) if fi < labels.shape[0] else -1
        if lid < 0:
            continue
        tri = uv_px[faces[fi]]
        if not np.isfinite(tri).all():
            continue

        min_xy = np.floor(np.min(tri, axis=0) - 1.0).astype(np.int64)
        max_xy = np.ceil(np.max(tri, axis=0) + 1.0).astype(np.int64)
        x0 = int(np.clip(min_xy[0], 0, res - 1))
        y0 = int(np.clip(min_xy[1], 0, res - 1))
        x1 = int(np.clip(max_xy[0], 0, res - 1))
        y1 = int(np.clip(max_xy[1], 0, res - 1))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        inside = _point_in_triangle_2d(pts, tri.astype(np.float64))
        if not np.any(inside):
            continue
        inside_pts = pts[inside]
        px = np.clip(np.floor(inside_pts[:, 0]).astype(np.int64), 0, res - 1)
        py = np.clip(np.floor(inside_pts[:, 1]).astype(np.int64), 0, res - 1)
        prev = owner[py, px]
        empty = prev < 0
        owner[py[empty], px[empty]] = lid
        clash = (~empty) & (prev != lid)
        if np.any(clash):
            overlap_mask[py[clash], px[clash]] = True

    covered = owner >= 0
    covered_count = int(np.count_nonzero(covered))
    overlap_count = int(np.count_nonzero(overlap_mask))
    if covered_count == 0:
        return {"overlap_ratio": 0.0, "covered_ratio": 0.0}
    return {
        "overlap_ratio": float(overlap_count / max(1, covered_count)),
        "covered_ratio": float(covered_count / float(res * res)),
    }


def _make_label_face_colors(labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(v) for v in np.unique(labels) if int(v) >= 0)
    if len(unique) == 0:
        return np.tile(np.asarray([[0.75, 0.75, 0.75, 0.80]], dtype=np.float32), (labels.shape[0], 1))
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("tab20", max(20, len(unique)))
        colors = np.zeros((labels.shape[0], 4), dtype=np.float32)
        id_to_idx = {lid: i for i, lid in enumerate(unique)}
        for i in range(labels.shape[0]):
            lid = int(labels[i])
            if lid < 0:
                colors[i] = np.asarray([0.70, 0.70, 0.70, 0.65], dtype=np.float32)
            else:
                rgba = cmap(id_to_idx.get(lid, 0) % cmap.N)
                colors[i] = np.asarray(rgba, dtype=np.float32)
        colors[:, 3] = np.clip(colors[:, 3], 0.50, 0.95)
        return colors
    except Exception:
        # Fallback deterministic pseudo-coloring without matplotlib colormap.
        colors = np.zeros((labels.shape[0], 4), dtype=np.float32)
        for i in range(labels.shape[0]):
            lid = int(labels[i])
            if lid < 0:
                colors[i] = np.asarray([0.70, 0.70, 0.70, 0.65], dtype=np.float32)
                continue
            h = np.uint32((lid + 1) * 2654435761)
            r = float((h & np.uint32(255)) / 255.0)
            g = float(((h >> np.uint32(8)) & np.uint32(255)) / 255.0)
            b = float(((h >> np.uint32(16)) & np.uint32(255)) / 255.0)
            colors[i] = np.asarray([r, g, b, 0.85], dtype=np.float32)
        return colors


def _render_uv_validation_png(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    high_uv: np.ndarray,
    low_uv: np.ndarray,
    high_labels: np.ndarray,
    low_labels: np.ndarray,
    metrics: Dict[str, Any],
    output_png: Path,
    dpi: int = 220,
) -> tuple[Optional[str], Optional[str]]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except Exception:
        return None, "missing_dependency_matplotlib"

    high_faces = np.asarray(high_mesh.faces, dtype=np.int64)
    low_faces = np.asarray(low_mesh.faces, dtype=np.int64)
    high_tri = np.asarray(high_uv[high_faces], dtype=np.float32)
    low_tri = np.asarray(low_uv[low_faces], dtype=np.float32)
    high_colors = _make_label_face_colors(high_labels)
    low_colors = _make_label_face_colors(low_labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_h, ax_l, ax_t = axes

    def _safe_limits(vals: np.ndarray) -> tuple[float, float]:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return 0.0, 1.0
        if abs(vmax - vmin) < 1e-8:
            pad = max(1e-3, abs(vmax) * 1e-3 + 1e-3)
            return vmin - pad, vmax + pad
        return vmin, vmax

    h_coll = PolyCollection(
        high_tri,
        facecolors=high_colors,
        edgecolors=(0.08, 0.08, 0.08, 0.40),
        linewidths=0.05,
    )
    ax_h.add_collection(h_coll)
    ax_h.set_title("High UV Islands")
    ax_h.set_aspect("equal")
    hx0, hx1 = _safe_limits(high_uv[:, 0])
    hy0, hy1 = _safe_limits(high_uv[:, 1])
    ax_h.set_xlim(hx0, hx1)
    ax_h.set_ylim(hy0, hy1)
    ax_h.axis("off")

    l_coll = PolyCollection(
        low_tri,
        facecolors=low_colors,
        edgecolors=(0.08, 0.08, 0.08, 0.40),
        linewidths=0.05,
    )
    ax_l.add_collection(l_coll)
    ax_l.set_title("Low UV Islands")
    ax_l.set_aspect("equal")
    lx0, lx1 = _safe_limits(low_uv[:, 0])
    ly0, ly1 = _safe_limits(low_uv[:, 1])
    ax_l.set_xlim(lx0, lx1)
    ax_l.set_ylim(ly0, ly1)
    ax_l.axis("off")

    ax_t.axis("off")
    text_lines = [
        "Closure Validation Summary",
        "",
        f"High islands: {metrics.get('high_island_count', -1)}",
        f"Low islands: {metrics.get('low_island_count', -1)}",
        f"Semantic unknown faces: {metrics.get('semantic_unknown_faces', -1)}",
        "",
        f"Seam topology valid: {metrics.get('seam_topology_valid', False)}",
        f"Seam components/open/closed: {metrics.get('seam_components', -1)} / "
        f"{metrics.get('seam_components_open', -1)} / {metrics.get('seam_loops_closed', -1)}",
        "",
        f"Partition leakage: {metrics.get('partition_has_leakage', False)}",
        f"Mixed components: {metrics.get('partition_mixed_components', -1)}",
        f"Label split count: {metrics.get('partition_label_split_count', -1)}",
        "",
        f"UV bbox IoU mean: {metrics.get('uv_bbox_iou_mean', 0.0):.4f}",
        f"UV overlap ratio: {metrics.get('uv_overlap_ratio', 0.0):.6f}",
        f"UV stretch p95/p99: {metrics.get('uv_stretch_p95', 0.0):.4f} / {metrics.get('uv_stretch_p99', 0.0):.4f}",
    ]
    ax_t.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=10, family="monospace")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return str(output_png), None


@dataclass
class UVClosureValidationResult:
    low_face_labels: np.ndarray
    low_seam_edges: np.ndarray
    metrics: Dict[str, Any]
    image_path: Optional[str]
    image_error: Optional[str]


def run_uv_closure_validation(
    *,
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    low_face_labels: Optional[np.ndarray] = None,
    low_seam_edges: Optional[np.ndarray] = None,
    high_position_eps: float = 1e-6,
    high_uv_eps: float = 1e-5,
    output_png: Optional[Path] = None,
    overlap_raster_res: int = 256,
) -> UVClosureValidationResult:
    high_uv = _safe_uv(high_mesh)
    low_uv = _safe_uv(low_mesh)
    if high_uv is None:
        raise RuntimeError("closure_validation requires valid UV on high mesh")
    if low_uv is None:
        raise RuntimeError("closure_validation requires valid UV on low mesh")

    high_faces = np.asarray(high_mesh.faces, dtype=np.int64)
    low_faces = np.asarray(low_mesh.faces, dtype=np.int64)
    high_vertices = np.asarray(high_mesh.vertices, dtype=np.float64)
    low_vertices = np.asarray(low_mesh.vertices, dtype=np.float64)

    high_labels, high_meta = compute_high_face_uv_islands(
        vertices=high_vertices,
        faces=high_faces,
        uv=high_uv,
        position_eps=float(high_position_eps),
        uv_eps=float(high_uv_eps),
    )

    if low_face_labels is None or int(np.asarray(low_face_labels).shape[0]) != int(low_faces.shape[0]):
        low_labels, _ = compute_high_face_uv_islands(
            vertices=low_vertices,
            faces=low_faces,
            uv=low_uv,
            position_eps=float(high_position_eps),
            uv_eps=float(high_uv_eps),
        )
    else:
        low_labels = np.asarray(low_face_labels, dtype=np.int64).reshape(-1)

    if low_seam_edges is None:
        seam_res = extract_seam_edges_openmesh(
            low_mesh=low_mesh,
            face_labels=low_labels,
            include_boundary_as_seam=False,
        )
        seam_edges = np.asarray(seam_res.seam_edges, dtype=np.int64)
        seam_meta = dict(seam_res.meta)
    else:
        seam_edges = np.asarray(low_seam_edges, dtype=np.int64).reshape(-1, 2)
        seam_meta = {}

    partition_meta = validate_face_partition_by_seams(
        low_mesh=low_mesh,
        face_labels=low_labels,
        seam_edges=seam_edges,
    )
    partition_has_leakage = bool(int(partition_meta.get("uv_seam_partition_mixed_components", 0)) > 0)

    stretch = _compute_uv_stretch_metrics(
        vertices=low_vertices,
        faces=low_faces,
        uv=low_uv,
        labels=low_labels,
    )
    overlap = _estimate_island_overlap_ratio(
        uv=low_uv,
        faces=low_faces,
        labels=low_labels,
        raster_res=int(overlap_raster_res),
    )
    bbox_iou = _compute_island_bbox_iou_mean(
        high_uv=high_uv,
        high_faces=high_faces,
        high_labels=np.asarray(high_labels, dtype=np.int64),
        low_uv=low_uv,
        low_faces=low_faces,
        low_labels=low_labels,
    )

    metrics: Dict[str, Any] = {
        "high_island_count": int(high_meta.get("high_island_count", np.unique(high_labels[high_labels >= 0]).size)),
        "low_island_count": int(np.unique(low_labels[low_labels >= 0]).size),
        "semantic_unknown_faces": int(np.count_nonzero(low_labels < 0)),
        "seam_topology_valid": bool(seam_meta.get("uv_seam_topology_valid", True)),
        "seam_components": int(seam_meta.get("uv_seam_components", -1)),
        "seam_loops_closed": int(seam_meta.get("uv_seam_loops_closed", -1)),
        "seam_components_open": int(seam_meta.get("uv_seam_components_open", -1)),
        "partition_has_leakage": bool(partition_has_leakage),
        "partition_mixed_components": int(partition_meta.get("uv_seam_partition_mixed_components", -1)),
        "partition_label_split_count": int(partition_meta.get("uv_seam_partition_label_split_count", -1)),
        "partition_components": int(partition_meta.get("uv_seam_partition_components", -1)),
        "uv_bbox_iou_mean": float(bbox_iou),
        "uv_overlap_ratio": float(overlap.get("overlap_ratio", 0.0)),
        "uv_overlap_covered_ratio": float(overlap.get("covered_ratio", 0.0)),
        "uv_stretch_mean": float(stretch.get("stretch_mean", 0.0)),
        "uv_stretch_p95": float(stretch.get("stretch_p95", 0.0)),
        "uv_stretch_p99": float(stretch.get("stretch_p99", 0.0)),
    }

    image_path: Optional[str] = None
    image_error: Optional[str] = None
    if output_png is not None:
        image_path, image_error = _render_uv_validation_png(
            high_mesh=high_mesh,
            low_mesh=low_mesh,
            high_uv=high_uv,
            low_uv=low_uv,
            high_labels=np.asarray(high_labels, dtype=np.int64),
            low_labels=low_labels,
            metrics=metrics,
            output_png=output_png,
        )

    return UVClosureValidationResult(
        low_face_labels=low_labels,
        low_seam_edges=seam_edges,
        metrics=metrics,
        image_path=image_path,
        image_error=image_error,
    )


__all__ = [
    "UVClosureValidationResult",
    "run_uv_closure_validation",
]
