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
from PIL import Image
import cv2

def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Scale pinhole camera intrinsics for a resize.
    (sx = W'/W, sy = H'/H)

    Args:
        K: (N, 3, 3) numpy array of intrinsic matrices
        sx: scale factor for width
        sy: scale factor for height

    Returns:
        Scaled intrinsics as a new numpy array
    """
    K = np.asarray(K)
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 2] *= sy  # cy
    return K_scaled

def main():
    kitti_root = Path("./kitti_clipsize_10")
    image_size = 518

    scene_dirs = sorted(kitti_root.glob("scene_*"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        clip_dirs = sorted(scene_dir.glob("clip_*"))
        for clip_dir in tqdm(clip_dirs, desc=f"  {scene_dir.name}", leave=False):
            # input dirs
            rgb_dir = clip_dir / "rgb"
            k_dir = clip_dir / "K"
            depth_dir = clip_dir / "depth"
            # output dirs
            out_rgb_dir = clip_dir / "rgb_resized"
            out_k_dir = clip_dir / "K_resized"
            out_depth_dir = clip_dir / "depth_resized"
            # create output dirs
            out_rgb_dir.mkdir(parents=True, exist_ok=True)
            out_k_dir.mkdir(parents=True, exist_ok=True)
            out_depth_dir.mkdir(parents=True, exist_ok=True)

            for rgb_file in sorted(rgb_dir.glob("*.png")):
                frame_id = rgb_file.stem
                out_rgb_path = out_rgb_dir / f"{frame_id}.png"
                out_k_dir_path = out_k_dir / f"{frame_id}.npy"
                out_depth_path = out_depth_dir / f"{frame_id}.npy"

                image = cv2.imread(str(rgb_file))
                H, W = image.shape[:2]
                sx, sy = image_size / W, image_size / H
                # process image
                #interp_rgb = cv2.INTER_AREA if (image_size < W or image_size < H) else cv2.INTER_LINEAR
                resized_image = cv2.resize(image, (image_size, image_size))#, interpolation=interp_rgb)
                cv2.imwrite(out_rgb_path, resized_image)
                
                # process intrinsic
                k_file = k_dir / f"{frame_id}.npy"
                k = np.load(k_file)  # (3, 3)
                resized_k = scale_intrinsics(k, sx, sy)
                np.save(out_k_dir_path, resized_k)

                # process depth
                depth_file = depth_dir / f"{frame_id}.npy"
                depth = np.load(depth_file)
                resized_depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                np.save(out_depth_path, resized_depth)


if __name__ == "__main__":
    main()