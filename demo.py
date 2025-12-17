#!/usr/bin/env python3
"""
FaithContour Demo Script

Demonstrates the FCT encoding and decoding pipeline using FaithContour.
This is a refactored implementation built on Atom3d primitives.

Usage:
    python demo.py -p /path/to/mesh.obj -r 512 -o output/reconstructed.glb
    python demo.py  # Uses default icosphere
"""

import os
import sys
import math
import time
import argparse
import contextlib

import torch
import trimesh
import numpy as np

# FaithContour imports
from faithcontour import FCTEncoder, FCTDecoder

# Atom3d imports
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer


@contextlib.contextmanager
def SuppressPrint(turn_stdout: bool = True):
    """
    A context manager to temporarily suppress print statements.

    Usage:
        with SuppressPrint():
            noisy_function()
    """
    if not turn_stdout:
        yield
        return
    original_stdout = sys.stdout
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout


def normalize_mesh(mesh: trimesh.Trimesh, margin: float = 0.05) -> trimesh.Trimesh:
    """
    Normalize mesh to fit within [-1+margin, 1-margin]^3 centered at origin.
    
    Args:
        mesh: Input trimesh
        margin: Margin from grid boundary (0.0-1.0), e.g., 0.05 means 5% margin on each side
    
    Returns:
        Normalized mesh
    """
    # Get current bounding box
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    
    # Center the mesh at the bounding box center (not centroid)
    bbox_center = (bbox_min + bbox_max) / 2.0
    mesh.vertices -= bbox_center
    
    # Scale to fit in [-1+margin, 1-margin]^3
    # Target half-size: 1.0 - margin
    target_half_size = 1.0 - margin
    current_half_size = np.abs(mesh.vertices).max()
    
    if current_half_size > 1e-8:
        scale = target_half_size / current_half_size
        mesh.vertices *= scale
    
    return mesh


def main():
    """
    Main function to run the FaithContour FCT encoding and decoding pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the FaithContour FCT encoding and decoding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p", "--mesh_path",
        type=str,
        default="",
        help="Path to the input mesh file (e.g., /path/to/model.obj)."
    )
    parser.add_argument(
        "-r", "--res",
        type=int,
        default=128,
        help="Final grid resolution. Must be a power of two."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/reconstructed_mesh.glb",
        help="Path for the output reconstructed mesh file."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margin from grid boundary (0.0-1.0). E.g., 0.05 = 5%% margin, mesh fits in [-0.95, 0.95]^3."
    )
    parser.add_argument(
        "--compute_flux",
        action="store_true",
        default=True,
        help="Whether to compute edge flux signs (required for mesh reconstruction)."
    )
    parser.add_argument(
        "--clamp_anchors",
        action="store_true",
        default=True,
        help="Clamp anchors to voxel bounds and project to surface for improved quality."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output."
    )
    parser.add_argument(
        "--tri_mode",
        type=str,
        default="auto",
        choices=['auto', 'simple_02', 'simple_13', 'length', 'angle', 'normal', 'normal_abs'],
        help="Triangulation mode for quad-to-triangle conversion."
    )
    args = parser.parse_args()

    # Validate resolution is a power of two
    if (args.res & (args.res - 1)) != 0 or args.res == 0:
        print(f"❌ Error: Resolution --res must be a power of two, but got {args.res}.")
        sys.exit(1)

    # Compute octree levels
    max_level = int(math.log2(args.res))
    min_level = min(4, max(1, max_level - 1))

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("             FaithContour Demo Pipeline")
    print("=" * 60)
    print(f"\n📋 Configuration:")
    print(f"   • Resolution: {args.res} (Octree Level: {min_level} → {max_level})")
    print(f"   • Device: {device}")
    print(f"   • Margin: {args.margin} (mesh range: [-{1.0-args.margin:.2f}, {1.0-args.margin:.2f}]^3)")
    print(f"   • Clamp Anchors: {args.clamp_anchors}")
    print(f"   • Compute Flux: {args.compute_flux}")

    # --- Load and normalize mesh ---
    print(f"\n📦 Loading mesh...")
    
    if args.mesh_path == "":
        print("   No input mesh provided. Using default icosphere.")
        mesh = trimesh.creation.icosphere(subdivisions=0, radius=0.95)
    else:
        if not os.path.exists(args.mesh_path):
            print(f"❌ Error: Mesh file not found: {args.mesh_path}")
            sys.exit(1)
        try:
            mesh = trimesh.load(args.mesh_path, force='mesh')
            print(f"   Loaded: {args.mesh_path}")
        except Exception as e:
            print(f"❌ Error loading mesh: {e}")
            sys.exit(1)

    # Normalize mesh
    mesh = normalize_mesh(mesh, margin=args.margin)
    
    # Preprocess: remove degenerate faces
    orig_faces = len(mesh.faces)
    valid_mask = mesh.nondegenerate_faces()
    if valid_mask.sum() < orig_faces:
        mesh.update_faces(valid_mask)
        mesh.remove_unreferenced_vertices()
        print(f"   ⚠️  Removed {orig_faces - len(mesh.faces)} degenerate faces")
    
    print(f"   • Vertices: {len(mesh.vertices)}")
    print(f"   • Faces: {len(mesh.faces)}")
    print(f"   • Bounds: [{mesh.vertices.min():.3f}, {mesh.vertices.max():.3f}]")

    # --- Convert to torch tensors ---
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    # --- Build BVH and Octree ---
    print(f"\n🔧 Building spatial structures...")
    
    time_start = time.time()
    bvh = MeshBVH(vertices, faces)
    time_bvh = time.time() - time_start
    print(f"   • MeshBVH built in {time_bvh:.3f}s")
    
    # Create octree with standard bounds [-1, 1]^3
    # This is the default for OctreeIndexer and matches normalize_mesh output range
    grid_bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
    octree = OctreeIndexer(max_level=max_level, bounds=grid_bounds, device=device)
    print(f"   • OctreeIndexer initialized (res={octree.res}, bounds=[-1, 1]^3)")

    # --- FCT Encoding ---
    print(f"\n🔄 FCT Encoding...")
    
    encoder = FCTEncoder(bvh, octree, device=device)
    
    time_start = time.time()
    with SuppressPrint(turn_stdout=not args.verbose):
        solver_weights = {
            'lambda_n': 1.0,
            'lambda_d': 1e-3,
            'weight_power': 1
        }
        fct_result = encoder.encode(
            min_level=min_level,
            solver_weights=solver_weights,
            compute_flux=args.compute_flux,
            clamp_anchors=args.clamp_anchors
        )
    time_encode = time.time() - time_start
    
    print(f"   ✅ Encoding completed in {time_encode:.3f}s")
    print(f"   • Active voxels: {fct_result.active_voxel_indices.shape[0]}")
    print(f"   • Anchor shape: {fct_result.anchor.shape}")
    print(f"   • Normal shape: {fct_result.normal.shape}")
    print(f"   • Edge flux shape: {fct_result.edge_flux_sign.shape}")

    # --- FCT Decoding ---
    print(f"\n🔄 FCT Decoding (tri_mode={args.tri_mode})...")
    
    decoder = FCTDecoder(resolution=args.res, bounds=grid_bounds, device=device)
    
    time_start = time.time()
    decoded_mesh = decoder.decode(
        active_voxel_indices=fct_result.active_voxel_indices,
        anchors=fct_result.anchor,
        edge_flux_sign=fct_result.edge_flux_sign,
        normals=fct_result.normal,
        triangulation_mode=args.tri_mode
    )
    time_decode = time.time() - time_start
    
    print(f"   ✅ Decoding completed in {time_decode:.3f}s")
    print(f"   • Generated vertices: {decoded_mesh.vertices.shape[0]}")
    print(f"   • Generated faces: {decoded_mesh.faces.shape[0]}")




    # --- Summary ---
    print(f"\n" + "=" * 60)
    print("                      Summary")
    print("=" * 60)
    print(f"   📊 Input:  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"   📊 Output: {decoded_mesh.vertices.shape[0]:,} vertices, {decoded_mesh.faces.shape[0]:,} faces")
    print(f"   ⏱️  Encode: {time_encode:.3f}s")
    print(f"   ⏱️  Decode: {time_decode:.3f}s")
    print(f"   ⏱️  Total:  {time_encode + time_decode:.3f}s")
    print("=" * 60)
    print(f"\n✅ Demo completed successfully!")
    

    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create trimesh for export
    recon_mesh = trimesh.Trimesh(
        vertices=decoded_mesh.vertices.cpu().numpy(),
        faces=decoded_mesh.faces.cpu().numpy(),
        process=False
    )
    
    try:
        recon_mesh.export(args.output)
        print(f"   ✅ Exported to: {args.output}")
    except Exception as e:
        print(f"❌ Error exporting mesh: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
