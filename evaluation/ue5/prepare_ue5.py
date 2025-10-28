#!/usr/bin/env python3
"""
prepare_ue5.py

resize & rescale rgb.png & depth.npy, create K for each frame.

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

def compute_intrinsics(width, height, fov_deg):
    """
    Compute intrinsic camera matrix K given image size and horizontal FoV.
    
    Args:
        width (int): image width in pixels
        height (int): image height in pixels
        fov_deg (float): horizontal field of view in degrees
    
    Returns:
        np.ndarray: 3x3 intrinsic matrix
    """
    fov_rad = np.deg2rad(fov_deg)
    
    # focal lengths
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx * (height / width)  # assuming square pixels
    
    # principal point (center)
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ])
    
    return K

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
    ue5_root = Path("./ue5")
    image_size = 518

    scene_dirs = sorted(ue5_root.glob("*_ego")) + sorted(ue5_root.glob("*_exo"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        # output dirs
        out_rgb_dir = scene_dir / "rgb_resized"
        out_k_dir = scene_dir / "K_resized"
        out_depth_dir = scene_dir / "depth_resized"
        # create output dirs
        out_rgb_dir.mkdir(parents=True, exist_ok=True)
        out_k_dir.mkdir(parents=True, exist_ok=True)
        out_depth_dir.mkdir(parents=True, exist_ok=True)
        for rgb_file in sorted(scene_dir.glob("*_rgb.png")):
          frame_id = rgb_file.stem.replace("_rgb", "")
          out_rgb_path = out_rgb_dir / f"{frame_id}.png"
          out_k_dir_path = out_k_dir / f"{frame_id}.npy"
          out_depth_path = out_depth_dir / f"{frame_id}.npy"
          
          image = cv2.imread(str(rgb_file))
          H, W = image.shape[:2]
          sx, sy = image_size / W, image_size / H
          # process image
          resized_image = cv2.resize(image, (image_size, image_size))#, interpolation=interp_rgb)
          cv2.imwrite(out_rgb_path, resized_image)
                
          # process intrinsic
          k = compute_intrinsics(W, H, 90.0)  # assume 90° horizontal FoV
          resized_k = scale_intrinsics(k, sx, sy)
          np.save(out_k_dir_path, resized_k)

          # process depth
          depth_file = scene_dir / f"{frame_id}_depth.npy"
          depth = np.load(depth_file)/100.0  # convert cm to m
          resized_depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
          np.save(out_depth_path, resized_depth)


if __name__ == "__main__":
    main()