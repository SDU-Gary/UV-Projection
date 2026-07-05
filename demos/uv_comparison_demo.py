#!/usr/bin/env python3
"""
UV Projection Comparison Demo

Demonstrates different UV mapping methods on example meshes,
comparing their quality and performance.

Usage:
    python uv_comparison_demo.py --mesh assets/examples/corgi_traveller.glb
    python uv_comparison_demo.py --mesh assets/examples/pirateship.glb --method-only method4
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import trimesh

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.uv_projector import UVProjector
from faithc_infra.mesh_io import MeshIO


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_metrics(metrics: Dict[str, Any], method_name: str):
    """Print quality metrics for a method."""
    print(f"\n📊 {method_name} Results:")
    print("-" * 40)

    # Core UV metrics
    if 'uv_flip_ratio' in metrics:
        print(f"  Flip Ratio:      {metrics['uv_flip_ratio']:.4f}")
    if 'uv_bad_tri_ratio' in metrics:
        print(f"  Bad Tri Ratio:   {metrics['uv_bad_tri_ratio']:.4f}")
    if 'uv_stretch_p95' in metrics:
        print(f"  Stretch (P95):   {metrics['uv_stretch_p95']:.4f}")
    if 'uv_stretch_p99' in metrics:
        print(f"  Stretch (P99):   {metrics['uv_stretch_p99']:.4f}")

    # Color reprojection error
    if 'uv_color_reproj_l1' in metrics and metrics['uv_color_reproj_l1']:
        print(f"  Color L1 Error:  {metrics['uv_color_reproj_l1']:.4f}")
    if 'uv_color_reproj_l2' in metrics and metrics['uv_color_reproj_l2']:
        print(f"  Color L2 Error:  {metrics['uv_color_reproj_l2']:.4f}")

    # Performance
    if 'uv_projection_seconds' in metrics:
        print(f"  Projection Time: {metrics['uv_projection_seconds']:.3f}s")

    # Solver info
    if 'uv_solver_stage' in metrics:
        print(f"  Solver Stage:    {metrics['uv_solver_stage']}")

    # Method-specific stats
    if 'uv_m4_nonlinear_iters' in metrics:
        print(f"  Nonlinear Iters:  {metrics['uv_m4_nonlinear_iters']}")


def run_comparison(
    mesh_path: Path,
    methods: List[str],
    output_dir: Path,
    texture_path: Path = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run UV projection comparison across multiple methods.

    Args:
        mesh_path: Path to input mesh file
        methods: List of method names to test
        output_dir: Output directory for results
        texture_path: Optional texture source path

    Returns:
        Dictionary mapping method names to their results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load high mesh for reconstruction
    high_mesh = MeshIO.load_mesh(mesh_path, process=False)

    print_section("Mesh Analysis")
    print(f"  Input Mesh: {mesh_path.name}")
    print(f"  Vertices: {len(high_mesh.vertices):,}")
    print(f"  Faces: {len(high_mesh.faces):,}")

    # For this demo, we'll use the same mesh as both high and low
    # In practice, you'd have separate high and low poly versions
    low_mesh = high_mesh

    projector = UVProjector()
    results = {}

    for method in methods:
        print_section(f"Testing {method}")

        method_dir = output_dir / method
        method_dir.mkdir(exist_ok=True)

        try:
            t0 = time.time()

            result = projector.project(
                sample_name=mesh_path.stem,
                high_mesh_path=mesh_path,
                low_mesh_path=mesh_path,  # Using same mesh for demo
                output_dir=method_dir,
                method=method,
                texture_source_path=texture_path
            )

            elapsed = time.time() - t0

            print(f"  ✅ Completed in {elapsed:.2f}s")
            print(f"  📁 Output: {result.low_mesh_uv_path}")

            # Load and print stats
            stats_path = method_dir / "uv_stats.json"
            if stats_path.exists():
                import json
                with open(stats_path) as f:
                    stats = json.load(f)
                    print_metrics(stats, method)
                    results[method] = stats
            else:
                print(f"  ⚠️  Stats file not found")
                results[method] = result.stats

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[method] = {"error": str(e)}

    return results


def generate_summary_report(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate a summary comparison report."""
    print_section("Summary Comparison")

    # Collect metrics for comparison
    metrics_comparison = {}

    for method, metrics in results.items():
        if "error" in metrics:
            continue

        metrics_comparison[method] = {
            "flip_ratio": metrics.get("uv_flip_ratio", None),
            "bad_tri_ratio": metrics.get("uv_bad_tri_ratio", None),
            "stretch_p95": metrics.get("uv_stretch_p95", None),
            "time": metrics.get("uv_projection_seconds", None)
        }

    # Print comparison table
    print("\nMethod Comparison:")
    print("-" * 80)
    print(f"{'Method':<25} {'Flip Ratio':<12} {'Bad Tri':<12} {'Stretch':<12} {'Time(s)':<10}")
    print("-" * 80)

    for method, metrics in metrics_comparison.items():
        flip = f"{metrics['flip_ratio']:.4f}" if metrics["flip_ratio"] else "N/A"
        bad = f"{metrics['bad_tri_ratio']:.4f}" if metrics["bad_tri_ratio"] else "N/A"
        stretch = f"{metrics['stretch_p95']:.4f}" if metrics["stretch_p95"] else "N/A"
        time_val = f"{metrics['time']:.3f}" if metrics["time"] else "N/A"

        print(f"{method:<25} {flip:<12} {bad:<12} {stretch:<12} {time_val:<10}")

    # Save report to file
    import json
    with open(output_path, "w") as f:
        json.dump({
            "comparison": metrics_comparison,
            "raw_results": results
        }, f, indent=2)

    print(f"\n📄 Report saved to: {output_path}")


def main():
    """Main entry point for UV comparison demo."""
    parser = argparse.ArgumentParser(
        description="Compare UV projection methods on example meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--mesh",
        type=str,
        default="assets/examples/corgi_traveller.glb",
        help="Path to mesh file for testing"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/uv_comparison",
        help="Output directory for results"
    )

    parser.add_argument(
        "--texture",
        type=str,
        default=None,
        help="Optional texture source path"
    )

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["nearest_vertex", "method2_gradient_poisson", "method4_jacobian_injective"],
        help="Methods to compare"
    )

    parser.add_argument(
        "--method-only",
        type=str,
        default=None,
        help="Run only this specific method"
    )

    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        # Try relative to repo root
        mesh_path = REPO_ROOT / args.mesh
        if not mesh_path.exists():
            print(f"❌ Mesh not found: {args.mesh}")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    texture_path = Path(args.texture) if args.texture else None

    # Determine methods to test
    if args.method_only:
        methods = [args.method_only]
    else:
        methods = args.methods

    print_section("UV Projection Comparison Demo")
    print(f"  Mesh: {mesh_path}")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Output: {output_dir}")

    # Run comparison
    results = run_comparison(
        mesh_path=mesh_path,
        methods=methods,
        output_dir=output_dir,
        texture_path=texture_path
    )

    # Generate summary
    report_path = output_dir / "comparison_report.json"
    generate_summary_report(results, report_path)

    print_section("Demo Complete")
    print(f"  ✅ All results saved to: {output_dir}")
    print(f"  📊 Report: {report_path}")


if __name__ == "__main__":
    main()
