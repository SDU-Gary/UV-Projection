#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    import trimesh
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: trimesh. "
        "Install it in your active environment first (e.g. `pip install trimesh`)."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from faithc_infra.services.halfedge_topology import compute_high_face_uv_islands  # noqa: E402


@dataclass
class MeshUVStats:
    mesh_name: str
    vertices: int
    faces: int
    islands: int
    avg_faces_per_island: float
    median_faces_per_island: float
    p95_faces_per_island: float
    max_faces_per_island: int
    min_faces_per_island: int
    uv_min: List[float]
    uv_max: List[float]
    uv_bbox_area: float
    uv_in_01_ratio: float
    seam_edges: int
    boundary_edges: int
    nonmanifold_edges: int
    unique_edges: int
    seam_edge_ratio: float
    layout_class: str
    risk_level: str
    diagnosis: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mesh_name": self.mesh_name,
            "vertices": self.vertices,
            "faces": self.faces,
            "islands": self.islands,
            "avg_faces_per_island": self.avg_faces_per_island,
            "median_faces_per_island": self.median_faces_per_island,
            "p95_faces_per_island": self.p95_faces_per_island,
            "max_faces_per_island": self.max_faces_per_island,
            "min_faces_per_island": self.min_faces_per_island,
            "uv_min": self.uv_min,
            "uv_max": self.uv_max,
            "uv_bbox_area": self.uv_bbox_area,
            "uv_in_01_ratio": self.uv_in_01_ratio,
            "seam_edges": self.seam_edges,
            "boundary_edges": self.boundary_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "unique_edges": self.unique_edges,
            "seam_edge_ratio": self.seam_edge_ratio,
            "layout_class": self.layout_class,
            "risk_level": self.risk_level,
            "diagnosis": self.diagnosis,
        }


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(name))
    cleaned = cleaned.strip("._")
    return cleaned or "mesh"


def _iter_model_paths(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(path.glob(pattern)):
        if p.is_file() and p.suffix.lower() in {".glb", ".gltf"}:
            yield p


def _classify_layout(islands: int, avg_faces_per_island: float) -> Tuple[str, str, str]:
    if islands <= 0:
        return "no_uv", "high", "模型没有可分析的 UV 岛。"

    if islands < 100 and avg_faces_per_island > 500:
        return "A_ideal_organic", "low", "岛数量少且单岛面积大，通常对 Method2 友好。"
    if 500 <= islands <= 2000 and 50 <= avg_faces_per_island <= 100:
        return "B_mechanical_like", "medium", "岛较多但仍有结构，Method2 可能出现边界伪影。"
    if islands > 5000 and avg_faces_per_island < 10:
        return "C_uv_soup", "high", "UV 岛过碎，属于典型 UV Soup，Method2 高风险失真。"

    if avg_faces_per_island < 5 or islands > 3000:
        return "fragmented_high_risk", "high", "UV 岛碎片化严重，强烈建议先做分岛与过滤。"
    if avg_faces_per_island < 20 or islands > 1000:
        return "fragmented_medium_risk", "medium", "UV 岛偏碎，Method2 需要更强的分岛约束。"
    return "mixed", "medium", "UV 结构中等，需结合实际对应质量判断。"


def _safe_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv_np = np.asarray(uv, dtype=np.float64)
    if uv_np.ndim != 2 or uv_np.shape[1] != 2:
        return None
    if uv_np.shape[0] != int(len(mesh.vertices)):
        return None
    if not np.isfinite(uv_np).all():
        return None
    return uv_np


def _unique_edge_count(faces: np.ndarray) -> int:
    tri = np.asarray(faces, dtype=np.int64)
    if tri.size == 0:
        return 0
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return int(np.unique(edges, axis=0).shape[0])


def _render_uv_layout(
    mesh: trimesh.Trimesh,
    mesh_name: str,
    model_path: Path,
    out_dir: Path,
    dpi: int,
    figsize: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    face_alpha: float,
    edge_alpha: float,
    linewidth: float,
    max_faces: int,
) -> Tuple[Path | None, str | None]:
    uv = _safe_uv(mesh)
    if uv is None:
        return None, "invalid_uv"

    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        return None, "invalid_faces"

    if max_faces > 0 and int(faces.shape[0]) > max_faces:
        stride = int(np.ceil(float(faces.shape[0]) / float(max_faces)))
        faces = faces[::stride]

    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ModuleNotFoundError:
        return None, "missing_dependency_matplotlib"

    uv_triangles = np.asarray(uv[faces], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(float(figsize), float(figsize)))
    collection = PolyCollection(
        uv_triangles,
        facecolors=(0.2, 0.6, 1.0, float(face_alpha)),
        edgecolors=(0.1, 0.1, 0.1, float(edge_alpha)),
        linewidths=float(linewidth),
    )
    ax.add_collection(collection)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal")
    ax.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{_sanitize_name(model_path.stem)}__{_sanitize_name(mesh_name)}_uv.png"
    out_path = out_dir / out_name
    fig.savefig(
        out_path,
        dpi=int(dpi),
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.0,
    )
    plt.close(fig)
    return out_path, None


def _analyze_mesh(
    mesh: trimesh.Trimesh,
    mesh_name: str,
    position_eps: float,
    uv_eps: float,
) -> MeshUVStats | None:
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    if int(len(mesh.faces)) <= 0 or int(len(mesh.vertices)) <= 0:
        return None
    uv = _safe_uv(mesh)
    if uv is None:
        return None

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    labels, meta = compute_high_face_uv_islands(
        vertices=verts,
        faces=faces,
        uv=uv,
        position_eps=position_eps,
        uv_eps=uv_eps,
    )
    islands = int(meta.get("high_island_count", 0))
    counts = np.bincount(labels[labels >= 0]) if labels.size > 0 else np.zeros((0,), dtype=np.int64)
    if counts.size > 0:
        avg_faces = float(np.mean(counts))
        med_faces = float(np.median(counts))
        p95_faces = float(np.percentile(counts, 95))
        max_faces = int(np.max(counts))
        min_faces = int(np.min(counts))
    else:
        avg_faces = 0.0
        med_faces = 0.0
        p95_faces = 0.0
        max_faces = 0
        min_faces = 0

    uv_min = np.min(uv, axis=0)
    uv_max = np.max(uv, axis=0)
    uv_range = uv_max - uv_min
    uv_bbox_area = float(max(0.0, uv_range[0] * uv_range[1]))
    in_01 = ((uv >= 0.0) & (uv <= 1.0)).all(axis=1)
    uv_in_01_ratio = float(np.mean(in_01)) if in_01.size > 0 else 0.0

    unique_edges = _unique_edge_count(faces)
    seam_edges = int(meta.get("high_seam_edges", 0))
    seam_edge_ratio = float(seam_edges / max(1, unique_edges))

    layout_class, risk_level, diagnosis = _classify_layout(islands, avg_faces)
    return MeshUVStats(
        mesh_name=mesh_name,
        vertices=int(len(verts)),
        faces=int(len(faces)),
        islands=islands,
        avg_faces_per_island=avg_faces,
        median_faces_per_island=med_faces,
        p95_faces_per_island=p95_faces,
        max_faces_per_island=max_faces,
        min_faces_per_island=min_faces,
        uv_min=[float(uv_min[0]), float(uv_min[1])],
        uv_max=[float(uv_max[0]), float(uv_max[1])],
        uv_bbox_area=uv_bbox_area,
        uv_in_01_ratio=uv_in_01_ratio,
        seam_edges=seam_edges,
        boundary_edges=int(meta.get("high_boundary_edges", 0)),
        nonmanifold_edges=int(meta.get("high_nonmanifold_edges", 0)),
        unique_edges=unique_edges,
        seam_edge_ratio=seam_edge_ratio,
        layout_class=layout_class,
        risk_level=risk_level,
        diagnosis=diagnosis,
    )


def _load_meshes(path: Path) -> Tuple[List[Tuple[str, trimesh.Trimesh]], str | None]:
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as exc:
        return [], f"load_failed: {exc}"

    meshes: List[Tuple[str, trimesh.Trimesh]] = []
    if isinstance(loaded, trimesh.Trimesh):
        meshes.append(("mesh0", loaded))
        return meshes, None

    if isinstance(loaded, trimesh.Scene):
        idx = 0
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append((str(name) if name else f"mesh{idx}", geom))
                idx += 1
        if len(meshes) == 0:
            return [], "scene_has_no_trimesh_geometry"
        return meshes, None

    return [], f"unsupported_type: {type(loaded).__name__}"


def _aggregate_file_stats(mesh_stats: List[MeshUVStats]) -> Dict[str, Any]:
    faces = int(sum(m.faces for m in mesh_stats))
    vertices = int(sum(m.vertices for m in mesh_stats))
    islands = int(sum(m.islands for m in mesh_stats))
    avg_faces_per_island = float(faces / max(1, islands))
    seam_edges = int(sum(m.seam_edges for m in mesh_stats))
    unique_edges = int(sum(m.unique_edges for m in mesh_stats))
    seam_edge_ratio = float(seam_edges / max(1, unique_edges))

    uv_min = np.array([np.inf, np.inf], dtype=np.float64)
    uv_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    for m in mesh_stats:
        uv_min = np.minimum(uv_min, np.asarray(m.uv_min, dtype=np.float64))
        uv_max = np.maximum(uv_max, np.asarray(m.uv_max, dtype=np.float64))
    if not np.isfinite(uv_min).all() or not np.isfinite(uv_max).all():
        uv_min = np.zeros((2,), dtype=np.float64)
        uv_max = np.zeros((2,), dtype=np.float64)
    uv_range = uv_max - uv_min
    uv_bbox_area = float(max(0.0, uv_range[0] * uv_range[1]))

    in01_weighted = 0.0
    for m in mesh_stats:
        in01_weighted += m.uv_in_01_ratio * m.vertices
    uv_in_01_ratio = float(in01_weighted / max(1, vertices))

    layout_class, risk_level, diagnosis = _classify_layout(islands, avg_faces_per_island)
    return {
        "vertices": vertices,
        "faces": faces,
        "islands": islands,
        "avg_faces_per_island": avg_faces_per_island,
        "uv_min": [float(uv_min[0]), float(uv_min[1])],
        "uv_max": [float(uv_max[0]), float(uv_max[1])],
        "uv_bbox_area": uv_bbox_area,
        "uv_in_01_ratio": uv_in_01_ratio,
        "seam_edges": seam_edges,
        "unique_edges": unique_edges,
        "seam_edge_ratio": seam_edge_ratio,
        "layout_class": layout_class,
        "risk_level": risk_level,
        "diagnosis": diagnosis,
    }


def analyze_model(
    path: Path,
    position_eps: float,
    uv_eps: float,
    *,
    visualize: bool = False,
    viz_out_dir: Path | None = None,
    viz_dpi: int = 300,
    viz_figsize: float = 24.0,
    viz_xmin: float = 0.0,
    viz_xmax: float = 1.0,
    viz_ymin: float = 0.0,
    viz_ymax: float = 1.0,
    viz_face_alpha: float = 0.4,
    viz_edge_alpha: float = 0.8,
    viz_linewidth: float = 0.02,
    viz_max_faces: int = 0,
) -> Dict[str, Any]:
    meshes, error = _load_meshes(path)
    if error is not None:
        return {"path": str(path), "ok": False, "error": error, "mesh_reports": []}

    mesh_reports: List[MeshUVStats] = []
    visualizations: List[Dict[str, Any]] = []
    skipped_meshes: List[str] = []
    for name, mesh in meshes:
        report = _analyze_mesh(mesh, name, position_eps=position_eps, uv_eps=uv_eps)
        if report is None:
            skipped_meshes.append(name)
            continue
        mesh_reports.append(report)
        if visualize and viz_out_dir is not None:
            out_path, viz_error = _render_uv_layout(
                mesh=mesh,
                mesh_name=name,
                model_path=path,
                out_dir=viz_out_dir,
                dpi=int(viz_dpi),
                figsize=float(viz_figsize),
                x_min=float(viz_xmin),
                x_max=float(viz_xmax),
                y_min=float(viz_ymin),
                y_max=float(viz_ymax),
                face_alpha=float(viz_face_alpha),
                edge_alpha=float(viz_edge_alpha),
                linewidth=float(viz_linewidth),
                max_faces=int(viz_max_faces),
            )
            visualizations.append(
                {
                    "mesh_name": name,
                    "ok": viz_error is None,
                    "path": str(out_path) if out_path is not None else None,
                    "error": viz_error,
                }
            )

    if len(mesh_reports) == 0:
        return {
            "path": str(path),
            "ok": False,
            "error": "no_mesh_with_valid_uv",
            "mesh_reports": [],
            "skipped_meshes": skipped_meshes,
            "visualizations": visualizations,
        }

    aggregate = _aggregate_file_stats(mesh_reports)
    return {
        "path": str(path),
        "ok": True,
        "error": None,
        "mesh_count_total": len(meshes),
        "mesh_count_analyzed": len(mesh_reports),
        "skipped_meshes": skipped_meshes,
        "aggregate": aggregate,
        "mesh_reports": [m.to_dict() for m in mesh_reports],
        "visualizations": visualizations,
    }


def _print_report(rep: Dict[str, Any]) -> None:
    print("=" * 96)
    print(f"Model: {rep['path']}")
    if not rep.get("ok", False):
        print(f"Status: FAILED | reason={rep.get('error')}")
        return

    agg = rep["aggregate"]
    print(
        "Status: OK | "
        f"meshes={rep['mesh_count_analyzed']}/{rep['mesh_count_total']} | "
        f"faces={agg['faces']} | islands={agg['islands']} | "
        f"avg_faces_per_island={agg['avg_faces_per_island']:.2f}"
    )
    print(
        "UV range: "
        f"{agg['uv_min']} -> {agg['uv_max']} | "
        f"bbox_area={agg['uv_bbox_area']:.6f} | in_01_ratio={agg['uv_in_01_ratio']:.4f}"
    )
    print(
        "Topology: "
        f"seam_edges={agg['seam_edges']} | unique_edges={agg['unique_edges']} | "
        f"seam_edge_ratio={agg['seam_edge_ratio']:.4f}"
    )
    print(
        "Classify: "
        f"class={agg['layout_class']} | risk={agg['risk_level']} | diagnosis={agg['diagnosis']}"
    )
    if rep.get("skipped_meshes"):
        print(f"Skipped meshes (no valid UV): {rep['skipped_meshes']}")
    viz_items = rep.get("visualizations", [])
    if viz_items:
        ok_items = [v for v in viz_items if v.get("ok", False)]
        fail_items = [v for v in viz_items if not v.get("ok", False)]
        print(f"Visualizations: ok={len(ok_items)} failed={len(fail_items)}")
        for item in ok_items:
            print(f"  uv_png[{item.get('mesh_name')}]: {item.get('path')}")
        for item in fail_items:
            print(f"  uv_png[{item.get('mesh_name')}]: FAILED ({item.get('error')})")


def _write_csv(path: Path, reports: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for rep in reports:
        row: Dict[str, Any] = {
            "path": rep.get("path"),
            "ok": rep.get("ok", False),
            "error": rep.get("error"),
        }
        if rep.get("ok", False):
            agg = rep["aggregate"]
            row.update(
                {
                    "faces": agg["faces"],
                    "vertices": agg["vertices"],
                    "islands": agg["islands"],
                    "avg_faces_per_island": agg["avg_faces_per_island"],
                    "uv_bbox_area": agg["uv_bbox_area"],
                    "uv_in_01_ratio": agg["uv_in_01_ratio"],
                    "seam_edge_ratio": agg["seam_edge_ratio"],
                    "layout_class": agg["layout_class"],
                    "risk_level": agg["risk_level"],
                    "diagnosis": agg["diagnosis"],
                }
            )
        rows.append(row)

    fields = [
        "path",
        "ok",
        "error",
        "faces",
        "vertices",
        "islands",
        "avg_faces_per_island",
        "uv_bbox_area",
        "uv_in_01_ratio",
        "seam_edge_ratio",
        "layout_class",
        "risk_level",
        "diagnosis",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _batch_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_reports = [r for r in reports if r.get("ok", False)]
    class_count: Dict[str, int] = {}
    risk_count: Dict[str, int] = {}
    for rep in ok_reports:
        c = rep["aggregate"]["layout_class"]
        rv = rep["aggregate"]["risk_level"]
        class_count[c] = class_count.get(c, 0) + 1
        risk_count[rv] = risk_count.get(rv, 0) + 1

    islands = [r["aggregate"]["islands"] for r in ok_reports]
    avg_faces_island = [r["aggregate"]["avg_faces_per_island"] for r in ok_reports]
    return {
        "models_total": len(reports),
        "models_ok": len(ok_reports),
        "models_failed": len(reports) - len(ok_reports),
        "islands_mean": float(np.mean(islands)) if islands else 0.0,
        "islands_median": float(np.median(islands)) if islands else 0.0,
        "avg_faces_per_island_mean": float(np.mean(avg_faces_island)) if avg_faces_island else 0.0,
        "avg_faces_per_island_median": float(np.median(avg_faces_island)) if avg_faces_island else 0.0,
        "layout_class_count": class_count,
        "risk_count": risk_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze GLB/GLTF UV island topology health for Method2-style pipelines."
    )
    parser.add_argument("input", type=str, help="GLB/GLTF file path, or a directory containing models")
    parser.add_argument("--recursive", action="store_true", help="Recursively search directory inputs")
    parser.add_argument("--position-eps", type=float, default=1e-6, help="Position weld epsilon")
    parser.add_argument("--uv-eps", type=float, default=1e-5, help="UV continuity epsilon on shared edges")
    parser.add_argument("--json-out", type=str, default="", help="Write full report JSON to path")
    parser.add_argument("--csv-out", type=str, default="", help="Write aggregate CSV to path")
    parser.add_argument("--quiet", action="store_true", help="Do not print per-model report lines")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Render UV layout PNG with matplotlib PolyCollection for each analyzed mesh",
    )
    parser.add_argument(
        "--viz-out-dir",
        type=str,
        default="",
        help="Directory for UV visualization PNG outputs (default: <json_out_dir>/uv_viz or ./uv_viz)",
    )
    parser.add_argument("--viz-dpi", type=int, default=300, help="DPI for UV visualization output")
    parser.add_argument("--viz-figsize", type=float, default=24.0, help="Figure size (inch) for UV visualization")
    parser.add_argument("--viz-xmin", type=float, default=0.0, help="UV plot x min")
    parser.add_argument("--viz-xmax", type=float, default=1.0, help="UV plot x max")
    parser.add_argument("--viz-ymin", type=float, default=0.0, help="UV plot y min")
    parser.add_argument("--viz-ymax", type=float, default=1.0, help="UV plot y max")
    parser.add_argument("--viz-face-alpha", type=float, default=0.4, help="Face alpha in UV visualization")
    parser.add_argument("--viz-edge-alpha", type=float, default=0.8, help="Edge alpha in UV visualization")
    parser.add_argument("--viz-linewidth", type=float, default=0.02, help="Edge linewidth in UV visualization")
    parser.add_argument(
        "--viz-max-faces",
        type=int,
        default=0,
        help="Max number of faces to render per mesh (0 means render all faces)",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    model_paths = list(_iter_model_paths(in_path, recursive=bool(args.recursive)))
    if in_path.is_file() and in_path.suffix.lower() not in {".glb", ".gltf"}:
        model_paths = [in_path]
    if len(model_paths) == 0:
        print(f"No model files found: {in_path}")
        sys.exit(2)

    reports: List[Dict[str, Any]] = []
    viz_out_dir: Path | None = None
    if args.visualize:
        if args.viz_out_dir:
            viz_out_dir = Path(args.viz_out_dir).expanduser().resolve()
        elif args.json_out:
            viz_out_dir = Path(args.json_out).expanduser().resolve().parent / "uv_viz"
        else:
            viz_out_dir = Path.cwd() / "uv_viz"

    for p in model_paths:
        rep = analyze_model(
            p,
            position_eps=float(args.position_eps),
            uv_eps=float(args.uv_eps),
            visualize=bool(args.visualize),
            viz_out_dir=viz_out_dir,
            viz_dpi=int(args.viz_dpi),
            viz_figsize=float(args.viz_figsize),
            viz_xmin=float(args.viz_xmin),
            viz_xmax=float(args.viz_xmax),
            viz_ymin=float(args.viz_ymin),
            viz_ymax=float(args.viz_ymax),
            viz_face_alpha=float(args.viz_face_alpha),
            viz_edge_alpha=float(args.viz_edge_alpha),
            viz_linewidth=float(args.viz_linewidth),
            viz_max_faces=int(args.viz_max_faces),
        )
        reports.append(rep)
        if not args.quiet:
            _print_report(rep)

    summary = _batch_summary(reports)
    print("-" * 96)
    print("Batch summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = {
        "input": str(in_path),
        "position_eps": float(args.position_eps),
        "uv_eps": float(args.uv_eps),
        "summary": summary,
        "reports": reports,
    }
    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json_report={out_path}")
    if args.csv_out:
        out_csv = Path(args.csv_out).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out_csv, reports)
        print(f"csv_report={out_csv}")
    if args.visualize and viz_out_dir is not None:
        print(f"uv_visualization_dir={viz_out_dir}")


if __name__ == "__main__":
    main()
