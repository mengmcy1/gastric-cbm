## 📦 Environment & Checkpoints

### 1) Create environment ([miniforge](https://github.com/conda-forge/miniforge) is recommended)

If using `conda`, replace `mamba` with `conda` in the following commands and use the `conda-forge` channel when installing `gxx`:

```bash
mamba create -n infinisplat python=3.10
mamba activate infinisplat

# Optional: When gsplat compilation fails due to g++ version or CUDA toolkit issues.
# mamba install gxx=10
# mamba install nvidia/label/cuda-12.8.0::cuda-toolkit -c nvidia/label/cuda-12.8.0
# export CUDA_HOME=$CONDA_PREFIX
```

### 2) Install dependencies

```bash
# Install uv
pip install uv

# Install PyTorch with CUDA 12.8
uv pip install torch==2.9.0 torchvision==0.24.0 xformers==0.0.33.post1 --index-url https://download.pytorch.org/whl/cu128

# Install package dependencies
uv pip install -r requirements.txt
```

### 3) Optional output dependencies

PLY export works without the dependencies in this section.

Install `gsplat` only when novel-view video rendering is needed:

```bash
uv pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
```

Interactive HTML export requires Node.js and the PlayCanvas `splat-transform` CLI, but does not require `gsplat`:

```bash
npm install -g @playcanvas/splat-transform
splat-transform -v
```

Missing optional dependencies are handled independently: without `gsplat`, MP4 rendering is skipped; without `splat-transform`, HTML conversion is skipped. In both cases, inference still exports the Gaussian PLY.

### 4) Download checkpoints

Download both released checkpoints with:

```bash
bash scripts/download_checkpoints.sh
```

This downloads the model files from [`PLUS-WAVE/InfiniSplat`](https://huggingface.co/PLUS-WAVE/InfiniSplat) into the local `checkpoints/` directory.

#### Model Zoo

| Category | Model | Use Case | Download |
|---|---|---|---|
| 3DGS | `InfiniSplat RGB` | RGB-Only Gaussian Inference | [infinisplat_rgb.ckpt](https://huggingface.co/PLUS-WAVE/InfiniSplat/blob/main/checkpoints/infinisplat_rgb.ckpt) |
| 3DGS | `InfiniSplat Depth Sensor` | Gaussian Inference with RGB + Depth | [infinisplat_lidar.ckpt](https://huggingface.co/PLUS-WAVE/InfiniSplat/blob/main/checkpoints/infinisplat_lidar.ckpt) |

The expected layout is:

```text
checkpoints/
├── infinisplat_rgb.ckpt
└── infinisplat_lidar.ckpt
```
