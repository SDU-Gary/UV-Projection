# UV-Projection Demos

This directory contains demonstration scripts for the UV-Projection project.

## Available Demos

### 1. Quick Start Demo
**File**: `quick_start.py`

A simple introduction to UV projection functionality. Demonstrates basic Method2 usage with example meshes.

```bash
python quick_start.py
```

**What it does**:
- Loads an example mesh
- Runs Method2 Gradient-Poisson UV projection
- Displays quality metrics
- Saves output mesh with UV coordinates

**Output**: `outputs/quick_start/`

### 2. UV Comparison Demo
**File**: `uv_comparison_demo.py`

Comprehensive comparison of different UV mapping methods.

```bash
# Compare all methods on default mesh
python uv_comparison_demo.py

# Compare on specific mesh
python uv_comparison_demo.py --mesh assets/examples/pirateship.glb

# Test only specific method
python uv_comparison_demo.py --method-only method4_jacobian_injective
```

**What it does**:
- Tests multiple UV projection methods
- Generates quality comparison report
- Saves individual results for each method
- Creates summary statistics

**Output**: `outputs/uv_comparison/`

### 3. FCT Demo
**File**: `fct_demo.py`

Demonstration of FaithContour Technology (FCT) encoding and decoding.

```bash
python fct_demo.py -p assets/examples/pirateship.glb -r 128
```

**What it does**:
- Loads and normalizes input mesh
- Builds BVH and Octree spatial structures
- Performs FCT encoding (mesh → FCT tokens)
- Performs FCT decoding (FCT tokens → mesh)
- Exports reconstructed mesh

## Prerequisites

All demos require:
- Python 3.9+
- PyTorch with CUDA support
- Atom3d package
- Example mesh files in `assets/examples/`

## Tips

1. **First time?** Start with `quick_start.py`
2. **Want to compare methods?** Use `uv_comparison_demo.py`
3. **Interested in mesh reconstruction?** Try `fct_demo.py`
4. **GPU required** for most UV methods (falls back to CPU if unavailable)

## Troubleshooting

**"CUDA not available"**:
- Check PyTorch installation
- Verify GPU drivers
- Falls back to CPU methods (slower)

**"Mesh not found"**:
- Ensure you're running from repository root
- Check `assets/examples/` contains mesh files

**"Out of memory"**:
- Try smaller mesh
- Reduce sampling density in config

## Next Steps

After running demos:
1. Examine output meshes in a 3D viewer
2. Review quality metrics and reports
3. Experiment with different meshes
4. Try the experiment framework: `faithc-exp run`
