#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from tqdm import tqdm

def calc_gt_scale(pred_depth_map: np.ndarray,
                  gt_depth_map: np.ndarray,
                  method: str = "median",
                  eps: float = 1e-8) -> float:
    """Compute a scalar 'gt_scale' so that (gt_scale * pred_depth_map) matches the sparse gt_depth_map."""
    if pred_depth_map.shape != gt_depth_map.shape:
        raise ValueError(f"Shape mismatch: pred {pred_depth_map.shape} vs gt {gt_depth_map.shape}")

    pred = pred_depth_map.astype(np.float64).ravel()
    gt   = gt_depth_map.astype(np.float64).ravel()

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > eps)
    if valid.sum() == 0:
        raise ValueError("No valid overlapping pixels between prediction and sparse GT.")

    p = pred[valid]
    g = gt[valid]

    if method == "median":
        ratios = g / np.maximum(p, eps)
        scale = float(np.median(ratios))
    elif method == "l2":
        denom = float(np.dot(p, p)) + eps
        num   = float(np.dot(p, g))
        scale = num / denom
    else:
        raise ValueError("method must be 'median' or 'l2'")
    return scale

def calc_sequence_scale(pred_dir: Path,
                        gt_dir: Path,
                        pattern: str = "*.npy",
                        method: str = "median",
                        eps: float = 1e-8,
                        verbose: bool = False) -> float:
    """
    Compute a single scale for an entire sequence by aggregating all valid pixels
    across files matched by stem between pred_dir and gt_dir.
    """
    pred_files = {p.stem: p for p in sorted(pred_dir.glob(pattern))}
    gt_files   = {g.stem: g for g in sorted(gt_dir.glob(pattern))}
    common_stems = sorted(set(pred_files.keys()) & set(gt_files.keys()))
    if not common_stems:
        raise ValueError("No matching files (by stem) between the two folders.")

    if verbose:
        print(f"Found {len(common_stems)} matching frames.")

    # Accumulate all valid pixels across frames
    all_p, all_g = [], []

    for s in common_stems:
        pred = np.load(pred_files[s])
        gt   = np.load(gt_files[s])

        if pred.shape != gt.shape:
            if verbose:
                print(f"[WARN] Shape mismatch for {s}: pred{pred.shape} vs gt{gt.shape}. Skipping.")
            continue

        pred = pred.astype(np.float64).ravel()
        gt   = gt.astype(np.float64).ravel()
        valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > eps)
        if valid.any():
            all_p.append(pred[valid])
            all_g.append(gt[valid])
        elif verbose:
            print(f"[WARN] No valid pixels for {s}. Skipping.")

    if not all_p:
        raise ValueError("After filtering, no valid pixels across the sequence.")

    p = np.concatenate(all_p, axis=0)
    g = np.concatenate(all_g, axis=0)

    if method == "median":
        ratios = g / np.maximum(p, eps)
        scale = float(np.median(ratios))
    elif method == "l2":
        denom = float(np.dot(p, p)) + eps
        num   = float(np.dot(p, g))
        scale = num / denom
    else:
        raise ValueError("method must be 'median' or 'l2'")

    if verbose:
        print(f"Computed {method} sequence scale over {p.size} valid pixels: {scale:.8f}")
    return scale

def main():
    kitti_root = Path("./kitti_clipsize_10")
    scene_dirs = sorted(kitti_root.glob("scene_*"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        clip_dirs = sorted(scene_dir.glob("clip_*"))
        for clip_dir in tqdm(clip_dirs, desc=f"  {scene_dir.name}", leave=False):
            gt_depth_dir = clip_dir / "depth_resized"
            pred_depth_vggt_dir = clip_dir / "pred_depth_vggt"
            out_dir = clip_dir / "pred_depth_vggt_scaled"
            out_dir.mkdir(parents=True, exist_ok=True)

            scale = calc_sequence_scale(pred_depth_vggt_dir, gt_depth_dir, pattern="*.npy",
                                        method="median")

            for pred_depth_file in sorted(pred_depth_vggt_dir.glob("*.npy")):
                frame_id = pred_depth_file.stem
                out_path = out_dir / f"{frame_id}.npy"

                scaled_depth = np.load(pred_depth_file) * scale
                np.save(out_path, scaled_depth)

if __name__ == "__main__":
    main()