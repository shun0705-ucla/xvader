import numpy as np
import torch
import cv2
from typing import Tuple, Optional
from vggt.utils.geometry import unproject_depth_map_to_point_map

def preprocess_image_for_vit(
    rgb: np.ndarray,
    K: np.ndarray = None,
    target_size: int = 518,
    patch_size: int = 14,
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int,int], Tuple[int,int]]:
    """
    Preprocess an RGB image for ViT-based models (Xvader, UniDepth, VGGT).

    Steps:
      1. Convert to float32 in [0,1]
      2. Resize to (target_size, target_size)
      3. Make sure size is divisible by patch size (14)
      4. Adjust intrinsics K accordingly

    Args:
      rgb: HxWx3 uint8 numpy array (RGB)
      K:   3x3 intrinsics matrix (optional). If None → no scaling.
      target_size: base resolution for model (518 is VGGT default)
      patch_size: patch size of ViT (14 for ViT-L/14)

    Returns:
      rgb_torch: (3, H_new, W_new) float tensor
      K_new:     3x3 intrinsics (np.ndarray)
      orig_size: (H, W)
      new_size:  (H_new, W_new)
    """
    orig_h, orig_w = rgb.shape[:2]

    # Make square target for simplicity
    H_new = target_size
    W_new = target_size

    # Ensure new size is divisible by patch size
    H_new = (H_new // patch_size) * patch_size
    W_new = (W_new // patch_size) * patch_size

    # Resize image
    rgb_resized = cv2.resize(
        rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32) / 255.0

    # Convert to tensor (3, H, W)
    rgb_torch = torch.from_numpy(rgb_resized).permute(2, 0, 1)

    # Scale intrinsics if provided
    if K is not None:
        # Ensure numpy
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()

        K_new = K.copy().astype(np.float32)

        sx = W_new / orig_w
        sy = H_new / orig_h
        K_new[0, 0] *= sx  # fx
        K_new[1, 1] *= sy  # fy
        K_new[0, 2] *= sx  # cx
        K_new[1, 2] *= sy  # cy
    else:
        K_new = None

    return rgb_torch, K_new, (orig_h, orig_w), (H_new, W_new)

def depth_rgb_to_colored_point_cloud(
    depth: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a single depth map + intrinsics (+ optional extrinsic) + RGB into
    a colored point cloud.

    Args:
      depth: (H, W) depth map (float32)
      K:     (3, 3) intrinsics
      rgb:   (H, W, 3) uint8 colors in [0,255]
      extrinsic: (3, 4) or (4, 4) camera-to-world. If None -> identity
                (i.e., world == camera coords).

    Returns:
      points_flat: (N, 3) float32
      colors_flat: (N, 3) uint8
    """
    H, W = depth.shape

    # VGGT unprojection expects shapes:
    #  depth: (S, H, W, 1)
    #  extrinsic: (S, 3, 4)
    #  intrinsic: (S, 3, 3)
    depth_4d = depth.astype(np.float32)[..., None][None, ...]   # (1, H, W, 1)

    if extrinsic is None:
        ext = np.eye(4, dtype=np.float32)[:3, :]                 # (3, 4)
    else:
        ext = extrinsic.astype(np.float32)
        if ext.shape == (4, 4):
            ext = ext[:3, :]                                    # (3, 4)

    extrinsics = ext[None, ...]                                 # (1, 3, 4)
    intrinsics = K.astype(np.float32)[None, ...]                # (1, 3, 3)

    # Unproject to 3D
    point_map = unproject_depth_map_to_point_map(
        depth_4d,           # (1, H, W, 1)
        extrinsics,         # (1, 3, 4)
        intrinsics,         # (1, 3, 3)
    )                        # (1, H, W, 3)

    points = point_map[0]    # (H, W, 3)
    points_flat = points.reshape(-1, 3).astype(np.float32)

    # Match colors: flatten to (N, 3)
    colors_flat = rgb.reshape(-1, 3).astype(np.uint8)

    return points_flat, colors_flat

def save_pointcloud_pcd_xyzrgb(points: np.ndarray,
                               colors: np.ndarray,
                               out_path_pcd: str):
    """
    Save colored point cloud in ASCII PCD format using packed float 'rgb'.
      points: (N, 3) float32 (x,y,z)
      colors: (N, 3) uint8  (r,g,b)
    """
    import os
    os.makedirs(os.path.dirname(out_path_pcd), exist_ok=True)

    assert points.shape[0] == colors.shape[0], \
        "points and colors must have same length"
    N = points.shape[0]

    pts = points.astype(np.float32)
    cols = colors.astype(np.uint8)

    # Pack RGB into one float32 (PCL convention)
    rgb_uint32 = (
        (cols[:, 0].astype(np.uint32) << 16) |  # R
        (cols[:, 1].astype(np.uint32) << 8)  |  # G
        (cols[:, 2].astype(np.uint32))          # B
    )
    rgb_packed = rgb_uint32.view(np.float32)

    header = [
        "VERSION .7",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {N}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {N}",
        "DATA ascii",
    ]

    with open(out_path_pcd, "w") as f:
        for line in header:
            f.write(line + "\n")
        for p, rgb in zip(pts, rgb_packed):
            f.write(
                f"{float(p[0])} {float(p[1])} {float(p[2])} {rgb}\n"
            )

def flat_to_map(x: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """
    Convert flattened patch tokens to a spatial map.

    Args:
        x: Tensor of shape (B, H*W, C)
        patch_h: Number of patches along height
        patch_w: Number of patches along width

    Returns:
        Tensor of shape (B, C, patch_h, patch_w)
    """
    B, HW, C = x.shape
    assert HW == patch_h * patch_w, f"num patch mismatch {patch_h * patch_w}, {HW}"
    return x.view(B, patch_h, patch_w, C).permute(0, 3, 1, 2).contiguous()


def map_to_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert spatial map back to flattened patch tokens.

    Args:
        x: Tensor of shape (B, C, H, W)

    Returns:
        Tensor of shape (B, H*W, C)
    """
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()