import os
import argparse
from typing import List

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib

from xvader.xvader import Xvader
from xvader.xvader_utils.load_weight import load_checkpoint_into_model
from xvader.xvader_utils.xvader_utils import preprocess_image_for_vit, depth_rgb_to_colored_point_cloud, save_pointcloud_pcd_xyzrgb
from xvader.xvader_utils.skyseg import get_sky_keep_mask, load_skyseg_session
from unidepth.models import UniDepthV2


def colorize_and_save_depth(depth: np.ndarray, out_path_png: str, out_path_npy: str):
    """
    Save raw depth as .npy and a colormapped visualization as .png.
    """
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)

    # Save raw depth
    np.save(out_path_npy, depth)

    # Normalize & colorize
    dmin, dmax = depth.min(), depth.max()
    depth_norm = (depth - dmin) / (dmax - dmin + 1e-6)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth_color = (cmap(depth_norm)[..., :3] * 255).astype(np.uint8)  # RGB

    # OpenCV expects BGR
    cv2.imwrite(out_path_png, depth_color[..., ::-1])


def run_demo(args):
    # Device + dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA is required for this demo.")
    major_cc = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
    print(f"[INFO] Using device={device}, dtype={dtype}")

    # Optional sky segmentation
    sky_session = load_skyseg_session() if args.mask_sky else None

    ############ Load model ###############
    model = Xvader(encoder="vitl")
    print(f"[INFO] Loading model from: {args.checkpoint}")
    compat, _sd = load_checkpoint_into_model(
        model,
        args.checkpoint,
        device=device,
        strict=True,        # keep strict=True to catch real mismatches
        prefer_ema=False,   # set True if you want EMA weights when present
        verbose=True,
    )
    model = model.to(device)
    model.eval()

    # Optional UniDepthV2 for intrinsics estimation
    if args.estimate_intrinsics:
        unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
        unidepth_model.eval()
    else:
        unidepth_model = None

    ########### Load inputs #############
    images = []
    Ks = []

    for idx, img_path in enumerate(args.image):
        # Load image (RGB)
        rgb = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3)
        # Load intrinsics
        if args.estimate_intrinsics:
            print("[INFO] Estimating intrinsics with UniDepth...")
            # 1) Use UniDepth to estimate intrinsics
            with torch.no_grad():
                rgb_for_unidepth = torch.from_numpy(rgb).permute(2, 0, 1).to(device)  # unnormalized
                preds = unidepth_model.infer(rgb_for_unidepth)
                K_t = preds["intrinsics"]
                if K_t.dim() == 3 and K_t.shape[0] == 1:
                    K_t = K_t[0]
                K_t = K_t.cpu().float()
        else:
            # 2) Ground-truth / provided intrinsics
            k_path = args.intrinsics[idx]
            K_np = np.load(k_path).astype(np.float32)
            if K_np.shape != (3, 3):
                raise ValueError(f"Invalid intrinsics shape {K_np.shape} for file {k_path}")
            K_t = torch.from_numpy(K_np)

        rgb_t, K_new, orig_sz, new_sz = preprocess_image_for_vit(
            rgb, K_t, target_size=args.resolution
        )

        images.append(rgb_t)
        Ks.append(torch.from_numpy(K_new))

    # Stack into tensors
    rgb_torch = torch.stack(images, dim=0).to(device)      # (S, 3, H, W)
    intrinsics_torch = torch.stack(Ks, dim=0).to(device)   # (S, 3, 3)

    # For coloring later: bring RGB back to uint8 and match depth resolution
    rgb_for_colors = (rgb_torch.detach().cpu().numpy() * 255.0)
    rgb_for_colors = np.clip(rgb_for_colors, 0, 255).astype(np.uint8)  # (S, 3, H, W)
    rgb_for_colors = np.transpose(rgb_for_colors, (0, 2, 3, 1))        # (S, H, W, 3)

    ######### Inference ##############
    print("[INFO] Running depth inference...")
    with torch.amp.autocast("cuda", dtype=dtype):
        depths, depth_confs = model.infer_depth(
            rgb_torch,
            intrinsics_torch,
            resolution=args.resolution,
        )

    # Move to CPU numpy
    depths_np = depths.squeeze().detach().cpu().numpy()         # (S,H,W) or (H,W)
    depth_confs_np = depth_confs.squeeze().detach().cpu().numpy()  # (S,H,W) or (H,W)
    # Ensure shape is (S, H, W) even if only 1 frame
    if depths_np.ndim == 2:
        depths_np = depths_np[None, ...]        # (1, H, W)
    if depth_confs_np.ndim == 2:
        depth_confs_np = depth_confs_np[None, ...]  # (1, H, W)

    S, H, W = depths_np.shape
    print(f"[INFO] Got {S} depth maps of size {H}x{W}")

    os.makedirs(args.outdir, exist_ok=True)

    # Intrinsics as numpy (S, 3, 3)
    Ks_np = intrinsics_torch.detach().cpu().numpy().astype(np.float32)

    ################ Save outputs ##############
    basename_list = [
        os.path.splitext(os.path.basename(p))[0] for p in args.image
    ]
    for i in range(S):
        depth = depths_np[i]             # (H, W)
        conf  = depth_confs_np[i]        # (H, W)
        rgb_i = rgb_for_colors[i]        # (H, W, 3), uint8
        K_i   = Ks_np[i]                 # (3, 3)

        base = basename_list[i]
        out_depth_npy = os.path.join(args.outdir, f"{base}_depth.npy")
        out_depth_png = os.path.join(args.outdir, f"{base}_depth.png")
        out_points_npy = os.path.join(args.outdir, f"{base}_points.npy")
        out_points_pcd = os.path.join(args.outdir, f"{base}_points.pcd")

        # 1) depth + colored depth (unmasked, for visualization)
        colorize_and_save_depth(depth, out_depth_png, out_depth_npy)

        # 2) create colored point cloud using utils (all pixels initially)
        points_flat, colors_flat = depth_rgb_to_colored_point_cloud(
            depth, K_i, rgb_i, extrinsic=None
        )   # (N=H*W, 3)

        # 3) build a single boolean mask of length H*W
        N = depth.size  # H*W
        valid = np.ones(N, dtype=bool)

        # 3-1) sky mask (keep non-sky)
        if args.mask_sky and sky_session is not None:
            sky_keep = get_sky_keep_mask(rgb_i, sky_session)   # (H, W) bool
            sky_keep_flat = sky_keep.reshape(-1)               # (N,)
            valid &= sky_keep_flat

        # 3-2) confidence threshold
        if args.conf_threshold > 0.0:
            conf_flat = conf.reshape(-1)                       # (N,)
            valid &= (conf_flat >= args.conf_threshold)

        # 4) apply the combined mask
        points_flat = points_flat[valid]
        colors_flat = colors_flat[valid]

        # Save raw points (after masking)
        np.save(out_points_npy, points_flat)

        # Save colored PCD
        save_pointcloud_pcd_xyzrgb(points_flat, colors_flat, out_points_pcd)

        print(f"[INFO] Frame {i}:")
        print(f"       depth  -> {out_depth_npy}, {out_depth_png}")
        print(f"       points -> {out_points_npy}, {out_points_pcd}")

    print(f"[INFO] Saved {S} depth maps and point clouds to: {args.outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xvader depth demo")

    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="List of input image paths (one per frame).",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        nargs="+",
        default=[],
        help="List of 3x3 intrinsics .npy paths (aligned with --image).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained Xvader checkpoint (.pt).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Directory to save depth and point cloud results.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=518,
        help="Inference resolution for the model.",
    )
    parser.add_argument(
        "--estimate_intrinsics",
        action="store_true",
        help="Use UniDepth to predict intrinsics when .npy intrinsics are not provided."
    )

    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.0,
        help="If > 0, mask out points whose depth_conf is below this value when saving point clouds.",
    )

    parser.add_argument(
        "--mask_sky",
        action="store_true",
        help="Use sky segmentation to remove sky points from point cloud.",
    )

    args = parser.parse_args()
    run_demo(args)
