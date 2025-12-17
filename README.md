
# Faithful Contouring
🤬 ***Enough with SDF + Marching Cubes? 📐 Time to Bring Geometry back — Faithfully.***

![Teaser](imgs/Cover_FCT.png)
**Faithful Contouring**: A high-fidelity, near-lossless 3D mesh representation method that eliminates the need for distance-field conversion and iso-surface extraction.  
This official library provides a GPU-accelerated **Encoder/Decoder pipeline** to transform arbitrary meshes into compact **Faithful Contour Tokens (FCTs)**, together with an efficient remeshing algorithm for precise reconstruction.


[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/) [![arXiv](https://img.shields.io/badge/arXiv-2511.04029-b31b1b.svg)](https://arxiv.org/abs/2511.04029)  [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


## 📢 News

- **[2025-12-17]** 🎉 **Code fully open-sourced!** Complete encoder/decoder implementation now available. No more waiting for source access!
- **[2025-12-17]** 🚀 **v1.5 released** — Pure Python implementation with Atom3d, no C++ compilation required.


## What's New in v1.5

> **🚀 Major Update: Pure Python Implementation with Atom3d**

### Architecture Changes

| Aspect | v0.1 (Legacy) | v1.5 (Current) |
|--------|---------------|----------------|
| **Backend** | Custom CUDA C++ kernels | Pure Python + Atom3d CUDA operators |
| **Dependencies** | External `cubvh` library | Internal Atom3d primitives |
| **Build** | Requires C++ compilation | **No compilation needed** |
| **Installation** | Complex, CUDA version matching | Simple `pip install` |

### FCT Format Changes

| Field | v0.1 (69 dims) | v1.5 (18 dims) |
|-------|----------------|----------------|
| **Primal anchor** | position (3) + normal (3) | position (3) + normal (3) |
| **Dual anchors** | 8 × (position + normal + mask) = 56 dims | ❌ Removed |
| **Semi-axis directions** | 6 dims | ❌ Removed |
| **Edge flux signs** | ❌ Not included | ✅ **12 dims** (new) |

### Decoder Changes

| Aspect | v0.1 | v1.5 |
|--------|------|------|
| **Reconstruction** | Primal-dual connectivity | **Edge-based quad extraction** |
| **Face generation** | Connect primal to dual points | Form quads from 4 incident voxel anchors |
| **Triangulation** | Fixed pattern | **Adaptive** (normal/length-based) |

### Key Benefits
- ✅ **No C++ compilation** — Pure Python, easy to install and modify
- ✅ **Simpler FCT format** — 18 dims vs 69 dims, more efficient storage
- ✅ **Edge flux decoding** — More robust mesh reconstruction
- ✅ **Atom3d integration** — Shared CUDA operators with other geometry projects

### Performance (v1.5)

Benchmark on icosphere mesh (NVIDIA H100 GPU):

| Resolution | Active Voxels | Encode | Decode | Total |
|------------|---------------|--------|--------|-------|
| 128 | 71K | 0.27s | 0.02s | 0.29s |
| 256 | 287K | 0.45s | 0.06s | 0.51s |
| 512 | 1.1M | 0.52s | 0.17s | 0.70s |
| 1024 | 4.6M | 0.82s | 0.61s | 1.42s |
| 2048 | 18.4M | 2.16s | 2.51s | 4.68s |


## Overview

Conventional voxel-based mesh representations typically rely on distance fields (SDF/UDF) and iso-surface extraction through Marching Cubes and its variations. These pipelines require watertight preprocessing and global sign computation, which often introduce artifacts including surface thickening, jagged iso-surfaces, and loss of internal structures.

**Faithful Contouring** avoids these issues by directly operating on the raw mesh. It identifies all surface-intersecting voxels and solves for a compact set of local anchor features.

This design ensures:
- **High fidelity** – sharp edges and internal structures are preserved, even for open or non-manifold meshes.  
- **Scalability** – efficient GPU kernels enable resolutions up to 2048+.  
- **Flexibility** – token-based format supports filtering, texturing, manipulation, and assembly for downstream applications.  

![Compare](imgs/WUKONGCOMPARE.png) 


## How It Works

The pipeline consists of two main components: an encoder and a decoder.

1.  **Encoder (`FCTEncoder`)**:
    - Takes a standard triangle mesh (vertices, faces) as input.
    - Builds a BVH (Bounding Volume Hierarchy) for fast intersection queries.
    - Performs hierarchical octree traversal from coarse to fine levels.
    - At each level, uses BVH-accelerated AABB intersection to prune empty regions.
    - At the finest level, performs SAT polygon clipping to compute precise anchors and normals.
    - Computes **edge flux signs** via segment-triangle intersection for surface crossing directions.

2.  **Decoder (`FCTDecoder`)**:
    - Takes the FCT tokens (anchor, normal, edge_flux_sign) as input.
    - Finds edges with non-zero flux (indicating surface crossing).
    - Forms **quads** from the 4 voxels incident to each active edge.
    - Triangulates quads based on normal consistency.
    - Outputs a reconstructed triangle mesh.


## FCT Format (v1.5)

Encoding produces an `FCTResult` dataclass with:

| Field | Shape | Description |
|-------|-------|-------------|
| `active_voxel_indices` | `[K]` | Linear indices of active voxels |
| `anchor` | `[K, 3]` | Per-voxel surface anchor point |
| `normal` | `[K, 3]` | Per-voxel surface normal direction |
| `edge_flux_sign` | `[K, 12]` | Edge crossing signs {-1, 0, +1} for 12 edges |

**Total: 18 dimensions per voxel** (vs 69 dims in v0.1)

> The simplified format focuses on **edge flux** for reconstruction, which is more robust than the dual-anchor approach.


## Installation
<a id="installation"></a>

This project requires a system with an NVIDIA GPU and PyTorch.

### Step 1: Install Atom3d (Required Dependency)

FaithContour v1.5 is built on [Atom3d](https://github.com/Luo-Yihao/Atom3d), which provides efficient BVH-accelerated geometry operations.

```bash
pip install git+https://github.com/Luo-Yihao/Atom3d.git --no-build-isolation
```

### Step 2: Install FaithContour

```bash
# Create conda environment (recommended)
conda create -n faithc python=3.10
conda activate faithc

# Install PyTorch (match your CUDA version)
# Example for CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install trimesh scipy einops

# Clone and install FaithContour
git clone https://github.com/Luo-Yihao/FC_dev.git
cd FC_dev
pip install -e .
```

**Requirements**:
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable NVIDIA GPU
- Atom3d (installed in Step 1)


## Usage

### Demo Script

The provided `demo.py` is the easiest way to test the pipeline:

```bash
# Default icosphere
python demo.py -r 128

# Custom mesh
python demo.py -p assets/examples/pirateship.glb -r 512 -o output/pirateship.glb
```

**Arguments:**
- `-p, --mesh_path`: Path to input mesh file
- `-r, --res`: Grid resolution (power of 2). Default: `128`
- `-o, --output`: Output mesh path. Default: `output/reconstructed_mesh.glb`
- `--margin`: Grid boundary margin. Default: `0.05`
- `--tri_mode`: Triangulation mode (`auto`, `length`, `angle`, `normal_abs`). Default: `auto`

### Library API

```python
import torch
import trimesh
from faithcontour import FCTEncoder, FCTDecoder, normalize_mesh
from atom3d import MeshBVH
from atom3d.grid import OctreeIndexer

# --- Load and Normalize Mesh ---
mesh = trimesh.load("my_model.obj", force='mesh')
mesh = normalize_mesh(mesh, rescalar=0.95)

V = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
F = torch.tensor(mesh.faces, dtype=torch.long, device='cuda')

# --- Build Spatial Structures ---
bvh = MeshBVH(V, F, device='cuda')
octree = OctreeIndexer(max_level=9, bounds=bvh.get_bounds(), device='cuda')  # 512^3

# --- Encoding ---
encoder = FCTEncoder(bvh, octree, device='cuda')
fct_result = encoder.encode(
    min_level=4,
    compute_flux=True,
    clamp_anchors=True
)

print(f"Active voxels: {fct_result.active_voxel_indices.shape[0]}")
print(f"Anchor shape: {fct_result.anchor.shape}")
print(f"Edge flux shape: {fct_result.edge_flux_sign.shape}")

# --- Decoding ---
decoder = FCTDecoder(resolution=512, bounds=bvh.get_bounds(), device='cuda')
mesh_result = decoder.decode_from_result(fct_result)

# Export
final_mesh = trimesh.Trimesh(
    mesh_result.vertices.cpu().numpy(), 
    mesh_result.faces.cpu().numpy()
)
final_mesh.export("reconstructed_mesh.glb")
```


## Roadmap

- [x] Wheel Package for Linux (v0.1)
- [x] **Pure Python + Atom3d Implementation (v1.5)** ✨
- [ ] Faithful Contour Tokens based VAE Release
- [ ] Diffusion Model Release


## License

Distributed under the Attribution-NonCommercial 4.0 International License. See `LICENSE` for more information.


## Citation

If you find this project useful in your research, please consider citing:
```bibtex
@misc{luo2025faithfulcontouringnearlossless3d,
      title={Faithful Contouring: Near-Lossless 3D Voxel Representation Free from Iso-surface}, 
      author={Yihao Luo and Xianglong He and Chuanyu Pan and Yiwen Chen and Jiaqi Wu and Yangguang Li and Wanli Ouyang and Yuanming Hu and Guang Yang and ChoonHwai Yap},
      year={2025},
      eprint={2511.04029},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.04029}, 
}
```


## Contact

Yihao Luo - y.luo23@imperial.ac.uk

Project Link: [https://github.com/Luo-Yihao/FC_dev](https://github.com/Luo-Yihao/FC_dev)
