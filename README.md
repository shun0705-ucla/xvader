## ___***XVADER: eXternal-View Assisted Depth Estimation from RGB cameras***___

> ⚠️ *This paper is currently under double-blind review. The paper will be made publicly available upon acceptance.*



## Overview:
**XVADER** predicts dense, metrically scaled depth maps and camera poses from an estimation frame and its assisting frames using their camera intrinsics.

<img src="assets/xvader_concept.jpg" width="600">

## Architecture

Each input image is patchified and processed into feature tokens by ViT encoder.
Camera intrinsics are embedded alongside image features to enable metric-scale estimation.
As proposed in VGGT, alternating attention allows information sharing across input images.
Finally, the depth and camera heads predict dense depth maps and camera poses in metric scale.

<img src="assets/xvader_model.jpg">


## Result

<img src="assets/XVADER_qualitative_comparison.jpg">

## Installation

Tested on:

- **Ubuntu 22.04**
- CUDA-compatible GPU required for inference

### 1. Clone the repository
```bash
git clone https://github.com/shun0705-ucla/xvader.git
```
### 2. Create the Conda environment and install dependencies
```bash
cd xvader
conda env create -f environment.yml

# External repos
pip install "vggt@git+https://github.com/facebookresearch/vggt.git" --no-deps
pip install "unidepth@git+https://github.com/lpiccinelli-eth/UniDepth.git" --no-deps

# Install XVADER package
pip install -e .
```

### 3. Download the pretrained checkpoint
```bash
bash download_checkpoint.sh
```

## Demo
Run the full XVADER demo pipeline:
```bash
bash demo.sh
```