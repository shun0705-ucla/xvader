#!/usr/bin/env python3
"""
infer_ue5_single.py

Run UniDepthV2 depth estimation model on preprocessed UE5 dataset (single-frame inference).

This script:
  1. Loads a pretrained UniDepthV2 model ("lpiccinelli/unidepth-v2-vitl14")
  2. Iterates over UE5 clips (from preprocessing script)
  3. Runs depth prediction on each image in the scene_name+"_ego" and scene_name+"_exo" folders
  4. Saves output depth maps as .npy files in clip/pred_unidepthv2/

Usage:
  python infer_ue5_single.py /path/to/ue5_root --device cuda
Folder structure (input):
  /path/to/ue5_root/
  ├── LowerSector_test_ego/
  │   ├── rgb_resized/
  │   │   ├── 000005.png
  │   ├── depth_resized/
  │   │   ├── 000005.npy
  │   ├── K_resized/
  │   │   ├── 000005.npy
  │   ├── pose_ego.txt
  ├── LowerSector_test_exo/ 
  │   ├── rgb_resized/
  │   │   ├── 000000.png
  │   ├── depth_resized/
  │   │   ├── 000000.npy
  │   ├── K_resized/
  │   │   ├── 000000.npy
  │   ├── pose_exo.txt
  ├── WildWest_test_ego/
  │   ├── ...
  ├── WildWest_test_exo/
  │   ├── ...
  

Folder structure (output):
  /path/to/ue5_root/
  ├── LowerSector_test_ego/
  │   ├── pred_unidepthv2/
  │   │   ├── 0000000005.npy   # float32 depth (metric scale)
"""

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
    ue5_root = Path("./ue5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    scene_dirs = sorted(ue5_root.glob("*_ego")) + sorted(ue5_root.glob("*_exo"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
      rgb_dir = scene_dir / "rgb_resized"
      k_dir = scene_dir / "K_resized"
      out_dir = scene_dir / "pred_unidepthv2"
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