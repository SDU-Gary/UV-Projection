#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class MeshAudit:
    mesh_path: str
    vertex_count: int
    face_count: int
    non_finite_vertex_count: int
    repeated_index_face_count: int
    degenerate_face_count: int
    boundary_edge_count: int
    manifold_edge_count: int
    nonmanifold_edge_count: int
    nonmanifold_vertex_count: int
    vertices_touching_nonmanifold_edge_count: int
    halfedge_safe: bool


@dataclass
class SpatialAudit:
    high_mesh_path: str
    low_mesh_path: str
    high_mesh_empty: bool
    low_mesh_empty: bool
    high_bbox_center: List[float]
    low_bbox_center: List[float]
    center_delta_l2: float
    center_delta_ratio_to_high_diag: float
    high_bbox_extent: List[float]
    low_bbox_extent: List[float]
    extent_ratio_xyz: List[float]
    high_bbox_diag: float
    low_bbox_diag: float
    diag_ratio_low_over_high: float
    high_max_abs_coord: float
    low_max_abs_coord: float
    likely_normalized_high: bool
    likely_normalized_low: bool
    nn_low_to_high_p50: float
    nn_low_to_high_p95: float
    nn_low_to_high_p99: float
    nn_low_to_high_max: float
    nn_high_to_low_p50: float
    nn_high_to_low_p95: float
    nn_high_to_low_p99: float
    nn_high_to_low_max: float
    potential_transform_mismatch: bool


def _load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise FileNotFoundError(f"mesh not found: {path}")

    obj = trimesh.load(path, force="mesh", process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj

    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"scene has no mesh geometry: {path}")
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise ValueError(f"unsupported mesh type for {path}: {type(obj)}")


def _sorted_edges_and_counts(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).astype(np.int64, copy=False)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq, counts


def _degenerate_face_count(vertices: np.ndarray, faces: np.ndarray, area_eps: float) -> int:
    if faces.size == 0:
        return 0
    tri = vertices[faces]
    v01 = tri[:, 1] - tri[:, 0]
    v02 = tri[:, 2] - tri[:, 0]
    cross = np.cross(v01, v02)
    area2 = np.linalg.norm(cross, axis=1)
    return int(np.count_nonzero(area2 <= float(area_eps) * 2.0))


def _repeated_index_face_count(faces: np.ndarray) -> int:
    if faces.size == 0:
        return 0
    rep = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return int(np.count_nonzero(rep))


def _nonmanifold_vertex_count(faces: np.ndarray, n_vertices: int) -> int:
    if n_vertices <= 0 or faces.size == 0:
        return 0

    incident: List[List[int]] = [[] for _ in range(n_vertices)]
    for fid, (a, b, c) in enumerate(faces.tolist()):
        if 0 <= a < n_vertices:
            incident[a].append(fid)
        if 0 <= b < n_vertices:
            incident[b].append(fid)
        if 0 <= c < n_vertices:
            incident[c].append(fid)

    nonmanifold_vertices = 0
    for v in range(n_vertices):
        if len(incident[v]) <= 1:
            continue

        # Build adjacency in the one-ring face graph around vertex v.
        neigh_to_faces: Dict[int, List[int]] = {}
        for fid in incident[v]:
            tri = faces[fid]
            ids = tri.tolist()
            if v not in ids:
                continue
            for u in ids:
                if u == v:
                    continue
                neigh_to_faces.setdefault(int(u), []).append(int(fid))

        face_adj: Dict[int, List[int]] = {fid: [] for fid in incident[v]}
        for fids in neigh_to_faces.values():
            if len(fids) <= 1:
                continue
            # In a manifold neighborhood, each (v,u) links at most two faces.
            for i in range(len(fids)):
                for j in range(i + 1, len(fids)):
                    a = fids[i]
                    b = fids[j]
                    face_adj[a].append(b)
                    face_adj[b].append(a)

        # Count connected components in incident faces around this vertex.
        seen = set()
        comp = 0
        for fid in incident[v]:
            if fid in seen:
                continue
            comp += 1
            stack = [fid]
            seen.add(fid)
            while stack:
                cur = stack.pop()
                for nb in face_adj.get(cur, []):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)

        # More than one local fan => bow-tie / nonmanifold vertex.
        if comp > 1:
            nonmanifold_vertices += 1

    return int(nonmanifold_vertices)


def audit_mesh_topology(mesh: trimesh.Trimesh, mesh_path: Path, area_eps: float) -> MeshAudit:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    non_finite_vertex_count = int(np.count_nonzero(~np.isfinite(verts).all(axis=1))) if verts.size > 0 else 0
    repeated_index_face_count = _repeated_index_face_count(faces)
    degenerate_face_count = _degenerate_face_count(verts, faces, area_eps=area_eps)

    uniq_edges, edge_counts = _sorted_edges_and_counts(faces) if faces.size > 0 else (np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    boundary_edge_count = int(np.count_nonzero(edge_counts == 1))
    manifold_edge_count = int(np.count_nonzero(edge_counts == 2))
    nonmanifold_edge_count = int(np.count_nonzero(edge_counts > 2))

    nonmanifold_edge_vertices = np.zeros((0,), dtype=np.int64)
    if nonmanifold_edge_count > 0:
        nonmanifold_edge_vertices = np.unique(uniq_edges[edge_counts > 2].reshape(-1))

    nonmanifold_vertex_count = _nonmanifold_vertex_count(faces, n_vertices=len(verts))

    halfedge_safe = (
        non_finite_vertex_count == 0
        and repeated_index_face_count == 0
        and degenerate_face_count == 0
        and nonmanifold_edge_count == 0
        and nonmanifold_vertex_count == 0
    )

    return MeshAudit(
        mesh_path=str(mesh_path),
        vertex_count=int(len(verts)),
        face_count=int(len(faces)),
        non_finite_vertex_count=non_finite_vertex_count,
        repeated_index_face_count=repeated_index_face_count,
        degenerate_face_count=degenerate_face_count,
        boundary_edge_count=boundary_edge_count,
        manifold_edge_count=manifold_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        nonmanifold_vertex_count=nonmanifold_vertex_count,
        vertices_touching_nonmanifold_edge_count=int(len(nonmanifold_edge_vertices)),
        halfedge_safe=bool(halfedge_safe),
    )


def _bbox_stats(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if vertices.size == 0:
        z = np.zeros((3,), dtype=np.float64)
        return z, z, z, 0.0, 0.0
    bb_min = vertices.min(axis=0)
    bb_max = vertices.max(axis=0)
    extent = bb_max - bb_min
    center = 0.5 * (bb_min + bb_max)
    diag = float(np.linalg.norm(extent))
    max_abs = float(np.max(np.abs(vertices))) if vertices.size > 0 else 0.0
    return bb_min, bb_max, center, diag, max_abs


def _sample_vertices(vertices: np.ndarray, sample_count: int, seed: int) -> np.ndarray:
    n = int(vertices.shape[0])
    if n <= sample_count:
        return vertices
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_count, replace=False)
    return vertices[idx]


def _percentiles(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    p50, p95, p99 = np.percentile(x, [50.0, 95.0, 99.0])
    return float(p50), float(p95), float(p99), float(np.max(x))


def audit_spatial_alignment(
    high_mesh: trimesh.Trimesh,
    low_mesh: trimesh.Trimesh,
    high_path: Path,
    low_path: Path,
    sample_count: int,
    seed: int,
    center_ratio_warn: float,
    diag_ratio_warn: float,
) -> SpatialAudit:
    hv = np.asarray(high_mesh.vertices, dtype=np.float64)
    lv = np.asarray(low_mesh.vertices, dtype=np.float64)
    high_empty = hv.size == 0
    low_empty = lv.size == 0

    _, _, hc, hdiag, hmax = _bbox_stats(hv)
    _, _, lc, ldiag, lmax = _bbox_stats(lv)

    h_extent = (hv.max(axis=0) - hv.min(axis=0)) if not high_empty else np.zeros((3,), dtype=np.float64)
    l_extent = (lv.max(axis=0) - lv.min(axis=0)) if not low_empty else np.zeros((3,), dtype=np.float64)
    extent_ratio = np.divide(l_extent, np.maximum(h_extent, 1e-12))

    center_delta = lc - hc
    center_delta_l2 = float(np.linalg.norm(center_delta))
    center_delta_ratio = float(center_delta_l2 / max(hdiag, 1e-12))

    diag_ratio = float(ldiag / max(hdiag, 1e-12))

    if high_empty or low_empty:
        l2h_p50 = l2h_p95 = l2h_p99 = l2h_max = float("nan")
        h2l_p50 = h2l_p95 = h2l_p99 = h2l_max = float("nan")
    else:
        hs = _sample_vertices(hv, sample_count=sample_count, seed=seed)
        ls = _sample_vertices(lv, sample_count=sample_count, seed=seed + 1)

        # Use vertex-vertex nearest neighbor as a cheap overlap proxy.
        tree_h = cKDTree(hv)
        tree_l = cKDTree(lv)
        d_l2h, _ = tree_h.query(ls, k=1)
        d_h2l, _ = tree_l.query(hs, k=1)
        l2h_p50, l2h_p95, l2h_p99, l2h_max = _percentiles(np.asarray(d_l2h, dtype=np.float64))
        h2l_p50, h2l_p95, h2l_p99, h2l_max = _percentiles(np.asarray(d_h2l, dtype=np.float64))

    likely_norm_high = bool(hmax <= 1.25)
    likely_norm_low = bool(lmax <= 1.25)

    potential_transform_mismatch = bool(
        high_empty
        or low_empty
        or center_delta_ratio > float(center_ratio_warn)
        or abs(diag_ratio - 1.0) > float(diag_ratio_warn)
    )

    return SpatialAudit(
        high_mesh_path=str(high_path),
        low_mesh_path=str(low_path),
        high_mesh_empty=bool(high_empty),
        low_mesh_empty=bool(low_empty),
        high_bbox_center=[float(v) for v in hc.tolist()],
        low_bbox_center=[float(v) for v in lc.tolist()],
        center_delta_l2=center_delta_l2,
        center_delta_ratio_to_high_diag=center_delta_ratio,
        high_bbox_extent=[float(v) for v in h_extent.tolist()],
        low_bbox_extent=[float(v) for v in l_extent.tolist()],
        extent_ratio_xyz=[float(v) for v in extent_ratio.tolist()],
        high_bbox_diag=float(hdiag),
        low_bbox_diag=float(ldiag),
        diag_ratio_low_over_high=diag_ratio,
        high_max_abs_coord=float(hmax),
        low_max_abs_coord=float(lmax),
        likely_normalized_high=likely_norm_high,
        likely_normalized_low=likely_norm_low,
        nn_low_to_high_p50=l2h_p50,
        nn_low_to_high_p95=l2h_p95,
        nn_low_to_high_p99=l2h_p99,
        nn_low_to_high_max=l2h_max,
        nn_high_to_low_p50=h2l_p50,
        nn_high_to_low_p95=h2l_p95,
        nn_high_to_low_p99=h2l_p99,
        nn_high_to_low_max=h2l_max,
        potential_transform_mismatch=potential_transform_mismatch,
    )


def _find_method_args(tree: ast.AST, class_name: str, method_name: str) -> List[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    args = [a.arg for a in child.args.args]
                    kwonly = [a.arg for a in child.args.kwonlyargs]
                    merged = args + kwonly
                    return [a for a in merged if a != "self"]
    return []


def audit_faithc_api(repo_root: Path) -> Dict[str, Any]:
    encoder_path = repo_root / "src" / "faithcontour" / "encoder.py"
    decoder_path = repo_root / "src" / "faithcontour" / "decoder.py"

    enc_src = encoder_path.read_text(encoding="utf-8")
    dec_src = decoder_path.read_text(encoding="utf-8")
    enc_tree = ast.parse(enc_src)
    dec_tree = ast.parse(dec_src)

    encode_args = _find_method_args(enc_tree, "FCTEncoder", "encode")
    decode_args = _find_method_args(dec_tree, "FCTDecoder", "decode")

    all_args = [s.lower() for s in (encode_args + decode_args)]
    tokens = "\n".join([enc_src.lower(), dec_src.lower()])

    has_boundary_lock_arg = any(k in a for a in all_args for k in ["boundary", "lock", "seam"]) \
        or ("boundary" in tokens and "lock" in tokens)
    has_attribute_arg = any(k in a for a in all_args for k in ["attribute", "uv", "label", "id", "metadata"])

    # Keyword footprint for human audit.
    keyword_hits = {
        "attribute": tokens.count("attribute"),
        "uv": tokens.count("uv"),
        "seam": tokens.count("seam"),
        "boundary": tokens.count("boundary"),
        "lock": tokens.count("lock"),
        "preserve": tokens.count("preserve"),
        "weight": tokens.count("weight"),
    }

    return {
        "encoder_path": str(encoder_path),
        "decoder_path": str(decoder_path),
        "encoder_encode_args": encode_args,
        "decoder_decode_args": decode_args,
        "supports_boundary_lock_api": bool(has_boundary_lock_arg),
        "supports_attribute_passthrough_api": bool(has_attribute_arg),
        "keyword_hits": keyword_hits,
    }


def audit_halfedge_infra(repo_root: Path) -> Dict[str, Any]:
    he_path = repo_root / "src" / "faithc_infra" / "services" / "halfedge_topology.py"
    seam_path = repo_root / "src" / "faithc_infra" / "services" / "uv" / "openmesh_seams.py"
    sanitizer_path = repo_root / "src" / "faithc_infra" / "services" / "uv" / "mesh_sanitizer.py"
    method2_path = repo_root / "src" / "faithc_infra" / "services" / "uv" / "method2_pipeline.py"
    hybrid_path = repo_root / "src" / "faithc_infra" / "services" / "uv" / "hybrid_pipeline.py"

    he_text = he_path.read_text(encoding="utf-8").lower()
    seam_text = seam_path.read_text(encoding="utf-8").lower() if seam_path.exists() else ""
    sanitizer_text = sanitizer_path.read_text(encoding="utf-8").lower() if sanitizer_path.exists() else ""
    method2_text = method2_path.read_text(encoding="utf-8").lower() if method2_path.exists() else ""
    hybrid_text = hybrid_path.read_text(encoding="utf-8").lower() if hybrid_path.exists() else ""

    uses_openmesh = ("openmesh" in seam_text) or ("openmesh" in he_text)
    uses_pymeshlab = "pymeshlab" in sanitizer_text
    uses_cgal = "cgal" in he_text or "cgal" in seam_text
    uses_geometry_central = (
        "geometry-central" in he_text
        or "geometrycentral" in he_text
        or "geometry-central" in seam_text
        or "geometrycentral" in seam_text
    )
    seam_pipeline_uses_openmesh = "extract_seam_edges_openmesh" in method2_text and "extract_seam_edges_openmesh" in hybrid_text

    if seam_pipeline_uses_openmesh:
        impl_mode = "openmesh_seam_extraction_plus_numpy_split"
    elif uses_openmesh:
        impl_mode = "openmesh_mixed"
    else:
        impl_mode = "custom_numpy_halfedge"

    return {
        "halfedge_impl_path": str(he_path),
        "seam_extraction_path": str(seam_path),
        "mesh_sanitizer_path": str(sanitizer_path),
        "halfedge_impl_mode": impl_mode,
        "uses_openmesh": bool(uses_openmesh),
        "uses_pymeshlab": bool(uses_pymeshlab),
        "seam_pipeline_uses_openmesh": bool(seam_pipeline_uses_openmesh),
        "uses_cgal": bool(uses_cgal),
        "uses_geometry_central": bool(uses_geometry_central),
        "uses_scipy_connected_components": "connected_components" in he_text,
    }


def _resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        high = run_dir / "mesh_high_normalized.glb"
        low = run_dir / "mesh_low.glb"
        return high, low
    if args.high is None or args.low is None:
        raise ValueError("Either --run-dir or both --high and --low must be provided")
    return args.high.resolve(), args.low.resolve()


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _print_summary(report: Dict[str, Any]) -> None:
    topo = report["topology_audit_low"]
    spatial = report["spatial_audit"]
    halfedge = report["halfedge_infra_audit"]
    api = report["faithc_api_audit"]

    print("=== Issue 1: Topology Purity (FaithC low mesh) ===")
    print(
        "faces={face_count}, verts={vertex_count}, degenerate={degenerate_face_count}, "
        "nonmanifold_edges={nonmanifold_edge_count}, nonmanifold_vertices={nonmanifold_vertex_count}, "
        "halfedge_safe={halfedge_safe}".format(**topo)
    )

    print("\n=== Issue 2: Spatial Anchoring (high vs low) ===")
    print(
        f"center_delta_ratio={spatial['center_delta_ratio_to_high_diag']:.6f}, "
        f"diag_ratio={spatial['diag_ratio_low_over_high']:.6f}, "
        f"potential_transform_mismatch={spatial['potential_transform_mismatch']}"
    )
    print(
        f"nn low->high p95={spatial['nn_low_to_high_p95']:.6f}, "
        f"high->low p95={spatial['nn_high_to_low_p95']:.6f}"
    )

    print("\n=== Issue 3: Half-edge Infrastructure ===")
    print(
        f"impl_mode={halfedge['halfedge_impl_mode']}, "
        f"OpenMesh={halfedge['uses_openmesh']}, PyMeshLab={halfedge['uses_pymeshlab']}, "
        f"seam_pipeline_uses_openmesh={halfedge['seam_pipeline_uses_openmesh']}, "
        f"CGAL={halfedge['uses_cgal']}, "
        f"geometry-central={halfedge['uses_geometry_central']}"
    )

    print("\n=== Issue 4: FaithC API Capability ===")
    print(
        f"supports_boundary_lock_api={api['supports_boundary_lock_api']}, "
        f"supports_attribute_passthrough_api={api['supports_attribute_passthrough_api']}"
    )
    print(f"encoder.encode args: {api['encoder_encode_args']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit FaithC low-mesh topology and high/low alignment")
    p.add_argument("--run-dir", type=Path, default=None, help="Run dir containing mesh_high_normalized.glb + mesh_low.glb")
    p.add_argument("--high", type=Path, default=None, help="Path to high mesh")
    p.add_argument("--low", type=Path, default=None, help="Path to low mesh")
    p.add_argument("--out-json", type=Path, default=None, help="Write full report JSON")
    p.add_argument("--sample-count", type=int, default=20000, help="NN sampling count per mesh")
    p.add_argument("--seed", type=int, default=12345, help="Random seed")
    p.add_argument("--area-eps", type=float, default=1e-12, help="Degenerate triangle area threshold")
    p.add_argument(
        "--center-ratio-warn",
        type=float,
        default=0.02,
        help="Warn when center shift exceeds this ratio of high bbox diagonal",
    )
    p.add_argument(
        "--diag-ratio-warn",
        type=float,
        default=0.02,
        help="Warn when |diag_ratio-1| exceeds this threshold",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    high_path, low_path = _resolve_paths(args)
    high_mesh = _load_mesh(high_path)
    low_mesh = _load_mesh(low_path)

    report = {
        "high_mesh": str(high_path),
        "low_mesh": str(low_path),
        "topology_audit_low": _to_dict(audit_mesh_topology(low_mesh, low_path, area_eps=float(args.area_eps))),
        "spatial_audit": _to_dict(
            audit_spatial_alignment(
                high_mesh,
                low_mesh,
                high_path=high_path,
                low_path=low_path,
                sample_count=int(max(1000, args.sample_count)),
                seed=int(args.seed),
                center_ratio_warn=float(args.center_ratio_warn),
                diag_ratio_warn=float(args.diag_ratio_warn),
            )
        ),
        "halfedge_infra_audit": audit_halfedge_infra(repo_root),
        "faithc_api_audit": audit_faithc_api(repo_root),
    }

    _print_summary(report)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport_json={args.out_json}")


if __name__ == "__main__":
    main()
