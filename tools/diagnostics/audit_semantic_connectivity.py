#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.uv.correspondence import build_high_cuda_context
from faithc_infra.services.uv.island_pipeline import run_halfedge_island_pipeline
from faithc_infra.services.uv.options import DEFAULT_OPTIONS, deep_merge_dict
from faithc_infra.services.atom3d_runtime import ensure_atom3d_cuda_runtime


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


def _face_component_sizes_from_neighbors(neighbors: List[List[int]]) -> np.ndarray:
    n_faces = int(len(neighbors))
    comp_sizes = np.zeros((n_faces,), dtype=np.int32)
    seen = np.zeros((n_faces,), dtype=np.bool_)
    for fid in range(n_faces):
        if seen[fid]:
            continue
        q: deque[int] = deque([int(fid)])
        seen[fid] = True
        comp_faces: List[int] = []
        while q:
            cur = q.popleft()
            comp_faces.append(cur)
            for nb in neighbors[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(int(nb))
        csz = int(len(comp_faces))
        comp_sizes[np.asarray(comp_faces, dtype=np.int64)] = int(csz)
    return comp_sizes


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


def _cross_semantic_edge_stats(
    *,
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
) -> Dict[str, Any]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if faces.shape[0] != face_labels.shape[0]:
        raise RuntimeError("cross-semantic audit: labels/face mismatch")

    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for fid, tri in enumerate(faces.tolist()):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        edge_faces[(min(a, b), max(a, b))].append(int(fid))
        edge_faces[(min(b, c), max(b, c))].append(int(fid))
        edge_faces[(min(c, a), max(c, a))].append(int(fid))

    boundary_edges = 0
    nonmanifold_edges = 0
    interior_edges_total = 0
    interior_known_edges = 0
    interior_unknown_touch_edges = 0
    cross_semantic_edges = 0
    same_semantic_edges = 0

    for _, flist in edge_faces.items():
        if len(flist) == 1:
            boundary_edges += 1
            continue
        if len(flist) != 2:
            nonmanifold_edges += 1
            continue
        interior_edges_total += 1
        f0, f1 = int(flist[0]), int(flist[1])
        l0 = int(face_labels[f0])
        l1 = int(face_labels[f1])
        if l0 < 0 or l1 < 0:
            interior_unknown_touch_edges += 1
            continue
        interior_known_edges += 1
        if l0 != l1:
            cross_semantic_edges += 1
        else:
            same_semantic_edges += 1

    return {
        "low_edge_boundary_count": int(boundary_edges),
        "low_edge_nonmanifold_count": int(nonmanifold_edges),
        "low_edge_interior_total": int(interior_edges_total),
        "low_edge_interior_known": int(interior_known_edges),
        "low_edge_interior_unknown_touch": int(interior_unknown_touch_edges),
        "low_edge_cross_semantic_count": int(cross_semantic_edges),
        "low_edge_same_semantic_count": int(same_semantic_edges),
        "low_edge_cross_semantic_ratio_in_known": float(cross_semantic_edges / max(1, interior_known_edges)),
    }


def _classify_cross_semantic_pattern(
    *,
    cross_semantic_count: int,
    high_seam_edges: int,
    summary: Dict[str, Any],
    pepper_ratio_threshold: float,
    reasonable_low: float,
    reasonable_high: float,
) -> Dict[str, Any]:
    ratio = float(cross_semantic_count / max(1, int(high_seam_edges)))
    pepper = bool(ratio >= float(pepper_ratio_threshold))
    fragmented = int(summary.get("fragmented_label_count", 0))
    severe = int(summary.get("severe_label_count", 0))
    normal = bool(summary.get("normal_overall", False))
    reasonable = bool(float(reasonable_low) <= ratio <= float(reasonable_high))

    if pepper:
        mode = "pepper_noise_likely"
        reason = "low cross-semantic edges are far higher than high UV seam edges"
    elif (not normal) and reasonable and (fragmented > 0 or severe > 0):
        mode = "coarse_semantic_shift_likely"
        reason = "cross-semantic edge count is in a reasonable range but islands are fragmented/misaligned"
    elif not normal:
        mode = "fragmented_but_inconclusive"
        reason = "islands are fragmented but cross-semantic edge ratio is not clearly diagnostic"
    else:
        mode = "healthy_or_near_healthy"
        reason = "semantic connectivity is mostly coherent"

    return {
        "cross_semantic_vs_high_seam_ratio": float(ratio),
        "diagnosis_mode": mode,
        "diagnosis_reason": reason,
    }


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
    p.add_argument(
        "--pepper-ratio-threshold",
        type=float,
        default=2.0,
        help="If low cross-semantic edge count / high seam edge count exceeds this, flag pepper-noise likely",
    )
    p.add_argument(
        "--reasonable-cross-ratio-low",
        type=float,
        default=0.5,
        help="Lower bound of a reasonable cross/high ratio window",
    )
    p.add_argument(
        "--reasonable-cross-ratio-high",
        type=float,
        default=2.0,
        help="Upper bound of a reasonable cross/high ratio window",
    )
    p.add_argument(
        "--ignore-small-mesh-components-faces",
        type=int,
        default=32,
        help="Ignore faces on low-mesh connected components smaller than this when auditing semantic connectivity",
    )
    p.add_argument(
        "--component-merge-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable small-component semantic absorption before seam extraction",
    )
    p.add_argument(
        "--component-merge-min-faces",
        type=int,
        default=4,
        help="Merge semantic connected components smaller than this face count",
    )
    p.add_argument(
        "--component-merge-max-iters",
        type=int,
        default=8,
        help="Maximum cleanup iterations for small semantic components",
    )
    return p.parse_args()


def _resolve_device(name: str) -> str:
    if name in {"cpu", "cuda"}:
        return name
    import torch

    return "cuda" if bool(torch.cuda.is_available()) else "cpu"


def _ensure_atom3d_runtime_for_device(device: str) -> Dict[str, Any]:
    return ensure_atom3d_cuda_runtime(device, strict=False)


def main() -> None:
    args = parse_args()
    high_mesh = _load_mesh(args.high.resolve())
    low_mesh_in = _load_mesh(args.low.resolve())
    high_uv = _safe_uv(high_mesh)

    opts = deep_merge_dict(DEFAULT_OPTIONS, {})
    seam_cfg = dict(opts.get("seam", {}))
    corr_cfg = dict(opts.get("correspondence", {}))
    seam_cfg["strategy"] = "halfedge_island"
    seam_cfg["sanitize_enabled"] = bool(args.sanitize_low)
    seam_cfg["validation_mode"] = "diagnostic"
    seam_cfg["transfer_sampling_mode"] = "four_point_bfs"
    if "transfer_max_dist_ratio" not in seam_cfg:
        seam_cfg["transfer_max_dist_ratio"] = 0.005
    seam_cfg["component_merge_enabled"] = bool(args.component_merge_enabled)
    seam_cfg["component_merge_min_faces"] = int(args.component_merge_min_faces)
    seam_cfg["component_merge_max_iters"] = int(args.component_merge_max_iters)

    resolved_device = _resolve_device(str(args.device))
    runtime_diag = _ensure_atom3d_runtime_for_device(resolved_device)
    high_ctx = build_high_cuda_context(
        high_mesh=high_mesh,
        high_uv=high_uv,
        device=resolved_device,
    )
    island_result = run_halfedge_island_pipeline(
        high_mesh=high_mesh,
        high_uv=np.asarray(high_uv, dtype=np.float64),
        low_mesh=low_mesh_in,
        high_ctx=high_ctx,
        seam_cfg=seam_cfg,
        corr_cfg=corr_cfg,
    )
    low_mesh = island_result.low_mesh
    sanitize_meta = {k: v for k, v in island_result.meta.items() if str(k).startswith("uv_sanitize_")}
    high_meta = dict(island_result.high.meta) if island_result.high is not None else {}
    semantic_meta = dict(island_result.semantic.meta) if island_result.semantic is not None else {}
    if island_result.semantic is not None:
        low_labels = np.asarray(island_result.semantic.face_labels, dtype=np.int64).reshape(-1)
    else:
        low_labels = np.full((len(low_mesh.faces),), -1, dtype=np.int64)
    neighbors = _face_neighbors(low_mesh)
    stage_summaries_raw: Dict[str, Any] = {}
    stage_summaries_filtered: Dict[str, Any] = {}

    def _stage_summary(stage_name: str, labels_arr: np.ndarray) -> Dict[str, Any]:
        stage_summary, _ = _audit_semantic_connectivity(
            labels=labels_arr,
            neighbors=neighbors,
            main_ratio_threshold=float(args.main_ratio_threshold),
            tiny_abs_threshold=int(args.tiny_abs_threshold),
            tiny_ratio_threshold=float(args.tiny_ratio_threshold),
            tiny_max_components=int(args.tiny_max_components),
        )
        return stage_summary

    if island_result.semantic is not None:
        stage_inputs = {
            "pre_bfs": np.asarray(island_result.semantic.pre_bfs_labels, dtype=np.int64).reshape(-1),
            "pre_cleanup": np.asarray(
                island_result.semantic.pre_cleanup_labels
                if island_result.semantic.pre_cleanup_labels is not None
                else island_result.semantic.face_labels,
                dtype=np.int64,
            ).reshape(-1),
            "final": np.asarray(island_result.semantic.face_labels, dtype=np.int64).reshape(-1),
        }
        for stage_name, labels_stage in stage_inputs.items():
            stage_summaries_raw[stage_name] = _stage_summary(stage_name, labels_stage)

    summary_raw, per_label_raw = _audit_semantic_connectivity(
        labels=low_labels,
        neighbors=neighbors,
        main_ratio_threshold=float(args.main_ratio_threshold),
        tiny_abs_threshold=int(args.tiny_abs_threshold),
        tiny_ratio_threshold=float(args.tiny_ratio_threshold),
        tiny_max_components=int(args.tiny_max_components),
    )
    labels_for_audit = low_labels.copy()
    ignored_small_faces = 0
    min_comp_faces = int(max(0, args.ignore_small_mesh_components_faces))
    if min_comp_faces > 1 and labels_for_audit.size > 0:
        comp_sizes = _face_component_sizes_from_neighbors(neighbors)
        keep_mask = comp_sizes >= min_comp_faces
        ignored_small_faces = int(np.count_nonzero(~keep_mask))
        labels_for_audit[~keep_mask] = -1
        if island_result.semantic is not None:
            for stage_name, labels_stage in stage_inputs.items():
                filtered_labels = np.asarray(labels_stage, dtype=np.int64).copy()
                filtered_labels[~keep_mask] = -1
                stage_summaries_filtered[stage_name] = _stage_summary(stage_name, filtered_labels)
    elif island_result.semantic is not None:
        stage_summaries_filtered = dict(stage_summaries_raw)

    summary, per_label = _audit_semantic_connectivity(
        labels=labels_for_audit,
        neighbors=neighbors,
        main_ratio_threshold=float(args.main_ratio_threshold),
        tiny_abs_threshold=int(args.tiny_abs_threshold),
        tiny_ratio_threshold=float(args.tiny_ratio_threshold),
        tiny_max_components=int(args.tiny_max_components),
    )
    cross_stats = _cross_semantic_edge_stats(mesh=low_mesh, labels=low_labels)
    diagnosis = _classify_cross_semantic_pattern(
        cross_semantic_count=int(cross_stats.get("low_edge_cross_semantic_count", 0)),
        high_seam_edges=int(high_meta.get("high_seam_edges", 0)),
        summary=summary_raw,
        pepper_ratio_threshold=float(args.pepper_ratio_threshold),
        reasonable_low=float(args.reasonable_cross_ratio_low),
        reasonable_high=float(args.reasonable_cross_ratio_high),
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
        "island_pipeline_meta": island_result.meta,
        "semantic_transfer_meta": semantic_meta,
        "audit_filter_min_component_faces": int(min_comp_faces),
        "audit_filter_ignored_faces": int(ignored_small_faces),
        "summary_raw": summary_raw,
        "summary": summary,
        "cross_semantic_edges": cross_stats,
        "cross_semantic_diagnosis": diagnosis,
        "stage_summaries_raw": stage_summaries_raw,
        "stage_summaries": stage_summaries_filtered,
        "labels_raw": per_label_raw,
        "labels": per_label,
    }

    print(
        "[semantic_connectivity] "
        f"labels={summary['label_count']}, unknown_faces={summary['face_count_unknown']}, "
        f"fragmented_labels={summary['fragmented_label_count']}, severe_labels={summary['severe_label_count']}, "
        f"intrusion_like_labels={summary['intrusion_like_label_count']}, normal_overall={summary['normal_overall']}"
    )
    print(
        "[semantic_connectivity][cross_edges] "
        f"low_cross={cross_stats['low_edge_cross_semantic_count']}, "
        f"high_seam={int(high_meta.get('high_seam_edges', 0))}, "
        f"ratio={diagnosis['cross_semantic_vs_high_seam_ratio']:.4f}, "
        f"mode={diagnosis['diagnosis_mode']}"
    )
    if stage_summaries_filtered:
        for stage_name in ["pre_bfs", "pre_cleanup", "final"]:
            if stage_name not in stage_summaries_filtered:
                continue
            ss = stage_summaries_filtered[stage_name]
            print(
                "[semantic_connectivity][stage] "
                f"{stage_name}: labels={ss['label_count']}, unknown={ss['face_count_unknown']}, "
                f"fragmented={ss['fragmented_label_count']}, severe={ss['severe_label_count']}, "
                f"intrusion={ss['intrusion_like_label_count']}, normal={ss['normal_overall']}"
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
