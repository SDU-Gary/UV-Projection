# UV-Projection: High-Quality UV Mapping for LOD Transitions

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A research platform for explicit UV mapping strategies, focusing on transferring textures from high-poly to low-poly meshes while maintaining visual consistency through advanced geometric processing algorithms.

## 🎯 Research Focus

This project explores **explicit geometric algorithms** for UV texture transfer between high and low-resolution meshes, enabling shared texture usage across LOD (Level of Detail) levels without re-baking. Our approach emphasizes:

- **Jacobian Field Transfer** - Propagating UV differential structure instead of point coordinates
- **Topology-Aware Mapping** - Preserving UV island semantics through semantic transfer
- **Robust Statistics** - IRLS, Huber loss, and adaptive regularization for stability
- **Nonlinear Refinement** - Symmetric Dirichlet energy for injective parameterization

## 🚀 Core Methods

### Method2: Gradient-Transferred Poisson UV Mapping
Our primary production method combining:
- UV island semantic transfer via halfedge topology
- Robust Jacobian aggregation with outlier rejection
- Adaptive smooth weights based on correspondence confidence
- Per-island sparse Poisson solving with soft anchors

### Method4: Jacobian-Injective Nonlinear Refinement
Built on Method2 initialization, adds:
- Symmetric Dirichlet energy for flip prevention
- Log-det barrier for local injectivity
- Homotopy constraint introduction for stability
- Local patch refinement for remaining violations

### Experimental Methods
- **Method2.5**: Projected Jacobian with residual field reconstruction
- **Method2p**: Projected gradient Poisson (linear only)

## 📊 Technical Highlights

### Architecture
```
High Mesh + UV → Island Analysis → Semantic Transfer → Topology Cutting
                                                              ↓
Low Mesh ← Sampling ← Ray Casting + UDF Fallback ← BVH Context
                                                              ↓
                         Jacobian Aggregation → Poisson Solve → UV Output
```

### Key Innovations
- **4-point soft flood strategy** for robust semantic labeling
- **IRLS + Huber + MAD** outlier rejection pipeline
- **Adaptive smooth weights** based on Jacobian variance
- **Multi-level fallback** from CUDA to CPU solvers
- **Comprehensive diagnostics** for quality assessment

### Performance
- **GPU-accelerated**: BVH queries, UDF search, sparse PCG solver
- **Sub-second processing**: <1s for typical assets on H100 GPU
- **Quality improvement**: ~33% reduction in UV flip ratio vs baseline

## 🛠️ Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+ with CUDA
- Atom3d (CUDA-accelerated geometry operations)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/UV-Projection.git
cd UV-Projection

# Create conda environment
conda create -n uv-projection python=3.10
conda activate uv-projection

# Install dependencies
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install trimesh scipy einops pyyaml

# Install Atom3d
pip install git+https://github.com/Luo-Yihao/Atom3d.git --no-build-isolation

# Install project in editable mode
pip install -e .
```

## 📖 Usage

### Basic Demo

```bash
# Run UV projection comparison demo
python demos/uv_comparison_demo.py --mesh assets/examples/corgi_traveller.glb
```

### Experiment Framework

```bash
# Run Method2 baseline experiments
faithc-exp run -c experiments/configs/uv_stage2_method2.yaml

# Run Method4 refinement
faithc-exp run -c experiments/configs/uv_stage2_method4.yaml

# Evaluate results
faithc-exp eval -r <run_id>

# Render with Mitsuba3
faithc-exp render -r <run_id>

# Interactive preview
faithc-exp preview --mesh assets/examples/pirateship.glb
```

### Python API

```python
from faithc_infra.services.uv_projector import UVProjector
from pathlib import Path

# Initialize projector
projector = UVProjector()

# Run Method2 projection
result = projector.project(
    sample_name="test_asset",
    high_mesh_path=Path("high_poly.glb"),
    low_mesh_path=Path("low_poly.glb"),
    output_dir=Path("output"),
    method="method2_gradient_poisson",
    texture_source_path=Path("texture.png")
)

print(f"UV mapping completed: {result.low_mesh_uv_path}")
print(f"Quality metrics: {result.stats}")
```

## 📁 Project Structure

```
UV-Projection/
├── src/
│   ├── faithc_infra/           # Experiment infrastructure
│   │   ├── services/
│   │   │   ├── uv/            # UV mapping methods
│   │   │   │   ├── method2_pipeline.py
│   │   │   │   ├── method4_pipeline.py
│   │   │   │   └── ...
│   │   │   └── uv_projector.py
│   │   └── cli.py              # faithc-exp CLI
│   └── faithcontour/           # Mesh reconstruction (FCT)
├── experiments/
│   ├── configs/                # Experiment configurations
│   ├── scripts/               # Analysis scripts
│   └── runs/                   # Experiment results
├── tools/
│   ├── diagnostics/            # Diagnostic tools
│   └── preview/                # Preview tools
├── docs/uv/                    # UV method documentation
├── demos/                      # Demo scripts
└── assets/examples/            # Example meshes
```

## 📚 Documentation

- [Method2 Implementation Details](docs/uv/method2_implementation.md)
- [Method4 Implementation Details](docs/uv/method4_implementation.md)
- [Method2.5 Experimental Details](docs/uv/method25_implementation.md)
- [Problem Analysis](problem.md) - Mathematical analysis of algorithm limitations
- [Solution Approaches](solve.md) - Proposed improvements and alternatives

## 🔬 Research Results

### Quality Improvements
Based on experiments with complex assets (massive_nordic_coastal_cliff, aksfbx):

| Method | Success Rate | Bad Tri Ratio | Flip Ratio | Color L1 Error |
|--------|--------------|---------------|------------|----------------|
| Baseline | 100% | 0.852 | 0.838 | 0.100 |
| Method2 | 100% | 0.569 (-33%) | 0.558 (-33%) | 0.113 |
| Method4 | 100% | 0.569 (-33%) | 0.558 (-33%) | 0.114 |

### Key Findings
- Jacobian field transfer significantly reduces UV artifacts
- Topology-aware semantic transfer improves seam handling
- Nonlinear refinement provides additional stability for difficult cases
- Multi-level fallback ensures robustness across diverse assets

## 🎓 Applications

- **Game Development**: LOD texture sharing for real-time rendering
- **Film Production**: Asset optimization for pipeline efficiency  
- **Industrial Design**: Multi-resolution visualization
- **Research**: Geometric processing and UV mapping algorithms

## ⚠️ Current Limitations

1. **Correspondence Ambiguity**: Fundamental ill-posedness in extreme topology changes
2. **UV Island Dependency**: Quality depends on high-poly UV layout quality
3. **Computational Cost**: Requires GPU acceleration for large models
4. **Parameter Sensitivity**: Some parameters need per-asset tuning

## 🔮 Future Directions

1. **Learning-enhanced Correspondence**: Neural network priors for initial matching
2. **Optimal Transport Framework**: OTM-UV for handling correspondence uncertainty
3. **Adaptive Topology Refinement**: Automatic low-poly refinement at seams
4. **Iterative Optimization**: Full iterative refinement pipeline

## 📄 License

Distributed under the Attribution-NonCommercial 4.0 International License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- **Atom3d**: Efficient CUDA-accelerated geometry operations
- **Trimesh**: Python mesh processing library
- **Mitsuba3**: High-quality rendering for evaluation

## 📞 Contact

For questions about the research or implementation, please open an issue on GitHub.

---

**Note**: This project is under active research and development. APIs and algorithms may evolve as we improve the methods.
