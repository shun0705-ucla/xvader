#!/usr/bin/env python3
"""
infer_depthanythingv2_kitti.py

Run DepthAnythingV2 depth estimation model on preprocessed KITTI dataset (single-frame inference).

This script:
  1. Loads a pretrained DepthAnythingV2 model ("Intel/DepthAnything-Depth-M")
  2. Iterates over KITTI clips (from preprocessing script)
  3. Runs depth prediction on each image in the clip/rgb/ folder
  4. Saves output depth maps as .npy files in clip/pred_depth_depthanythingv2/

Usage:
  python infer_depthanythingv2_kitti.py /path/to/kitti_preprocessed --device cuda

Folder structure (input):
  /path/to/kitti_preprocessed/
  ├── scene_0000/
  │   ├── clip_000/
  │   │   ├── rgb/
  │   │   │   ├── 0000000005.png
  │   │   │   ├── ...
  │   │   ├── depth/            # optional
  │   │   ├── intrinsics/       # optional

Folder structure (output):
  /path/to/kitti_preprocessed/
  ├── scene_0000/
  │   ├── clip_000/
  │   │   ├── pred_depth_depthanythingv2/
  │   │   │   ├── 0000000005.npy   # float32 depth (arbitrary scale)
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import cv2

from third_party.UniDepth.unidepth.models.unidepthv2 import UniDepthV2
from third_party.UniDepth.unidepth.utils.camera import Pinhole

def main():
    kitti_root = Path("./kitti_clipsize_10")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    scene_dirs = sorted(kitti_root.glob("scene_*"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        clip_dirs = sorted(scene_dir.glob("clip_*"))
        for clip_dir in tqdm(clip_dirs, desc=f"  {scene_dir.name}", leave=False):
            rgb_dir = clip_dir / "rgb_resized"
            k_dir = clip_dir / "K_resized"
            out_dir = clip_dir / "pred_unidepthv2"
            out_dir.mkdir(parents=True, exist_ok=True)

            for rgb_file in sorted(rgb_dir.glob("*.png")):
                frame_id = rgb_file.stem
                out_path = out_dir / f"{frame_id}.npy"

                rgb = np.array(Image.open(str(rgb_file)))
                rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
                k_file = k_dir / f"{frame_id}.npy"
                intrinsics_torch = torch.from_numpy(np.load(str(k_file)))
                camera = Pinhole(K=intrinsics_torch.unsqueeze(0))

                rgb_torch = rgb_torch.to(device)
                camera = camera.to(device)

                predictions = model.infer(rgb_torch, camera)
                pred_depth = predictions["depth"].squeeze().cpu().numpy()

                np.save(out_path, pred_depth)


if __name__ == "__main__":
    main()