#!/usr/bin/env python3
"""
Quick Start Demo - UV Projection Basic Usage

Simple demonstration of UV projection functionality.
"""

import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faithc_infra.services.uv_projector import UVProjector


def main():
    """Run quick start demo."""
    print("=" * 60)
    print("  UV Projection Quick Start Demo")
    print("=" * 60)

    # Example mesh paths (using included examples)
    mesh_path = REPO_ROOT / "assets/examples/corgi_traveller.glb"
    output_dir = REPO_ROOT / "outputs/quick_start"

    if not mesh_path.exists():
        print(f"❌ Example mesh not found: {mesh_path}")
        print("\nTo run this demo, ensure example assets are available.")
        return

    print(f"\n📁 Input Mesh: {mesh_path.name}")
    print(f"📁 Output Dir: {output_dir}")

    # Initialize projector
    projector = UVProjector()

    # Run Method2 UV projection
    print("\n🔄 Running Method2 Gradient-Poisson UV projection...")

    try:
        result = projector.project(
            sample_name="quick_start_demo",
            high_mesh_path=mesh_path,
            low_mesh_path=mesh_path,  # Using same mesh for demo
            output_dir=output_dir,
            method="method2_gradient_poisson"
        )

        print(f"\n✅ UV projection completed!")
        print(f"   Output mesh: {result.low_mesh_uv_path}")
        print(f"   UV map: {result.uv_map_path}")

        # Print key metrics
        print("\n📊 Key Quality Metrics:")
        stats = result.stats
        if 'uv_flip_ratio' in stats:
            print(f"   Flip Ratio: {stats['uv_flip_ratio']:.4f}")
        if 'uv_bad_tri_ratio' in stats:
            print(f"   Bad Triangle Ratio: {stats['uv_bad_tri_ratio']:.4f}")
        if 'uv_projection_seconds' in stats:
            print(f"   Processing Time: {stats['uv_projection_seconds']:.3f}s")

        print("\n💡 Next Steps:")
        print("   - View the output mesh in a 3D viewer")
        print("   - Try different methods: method4_jacobian_injective")
        print("   - Run full comparison: python uv_comparison_demo.py")

    except Exception as e:
        print(f"\n❌ Error during UV projection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
