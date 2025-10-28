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

def scale_translation(txt_path, scale):
    # load all numbers from txt
    values = np.loadtxt(txt_path)

    # scale first 3
    values[:3] *= scale

    return values


def main():
    ue5_root = Path("./ue5")
    ego_scene_dirs = sorted(ue5_root.glob("*_ego"))
    for ego_scene_dir in tqdm(ego_scene_dirs, desc="Scenes"):
        ego_gt_depth_dir = ego_scene_dir / "depth_resized"
        ego_pred_depth_vggt_dir = ego_scene_dir / "pred_depth_vggt"
        ego_pred_camera_dir = ego_scene_dir / "pred_camera_vggt"
        ego_depth_out_dir = ego_scene_dir / "pred_depth_vggt_scaled"
        ego_camera_out_dir = ego_scene_dir / "pred_camera_vggt_scaled"
        ego_depth_out_dir.mkdir(parents=True, exist_ok=True)
        ego_camera_out_dir.mkdir(parents=True, exist_ok=True)

        exo_scene_dir = Path(str(ego_scene_dir).replace("_ego", "_exo"))
        exo_gt_depth_dir = exo_scene_dir / "depth_resized"
        exo_pred_depth_vggt_dir = exo_scene_dir / "pred_depth_vggt"
        exo_pred_camera_dir = exo_scene_dir / "pred_camera_vggt"
        exo_depth_out_dir = exo_scene_dir / "pred_depth_vggt_scaled"
        exo_camera_out_dir = exo_scene_dir / "pred_camera_vggt_scaled"
        exo_depth_out_dir.mkdir(parents=True, exist_ok=True)
        exo_camera_out_dir.mkdir(parents=True, exist_ok=True)

        for ego_pred_depth_file in sorted(ego_pred_depth_vggt_dir.glob("*.npy")):
            frame_id = ego_pred_depth_file.stem
            ego_gt_depth_file = ego_gt_depth_dir / f"{frame_id}.npy"
            exo_pred_depth_file = exo_pred_depth_vggt_dir / f"{frame_id}.npy"
            exo_gt_depth_file = exo_gt_depth_dir / f"{frame_id}.npy"

            # load files
            ego_pred_depth = np.load(ego_pred_depth_file)
            exo_pred_depth = np.load(exo_pred_depth_file)
            ego_gt_depth   = np.load(ego_gt_depth_file)
            exo_gt_depth   = np.load(exo_gt_depth_file)

            # concat properly
            pred_depth_concat = np.concatenate([ego_pred_depth, exo_pred_depth], axis=1)
            gt_depth_concat   = np.concatenate([ego_gt_depth, exo_gt_depth], axis=1)

            # compute scale
            scale = calc_gt_scale(pred_depth_concat, gt_depth_concat, method="median")

            # apply scale
            ego_depth_scaled = ego_pred_depth * scale
            exo_depth_scaled = exo_pred_depth * scale

            # scale translations
            ego_camera_scaled = scale_translation(ego_pred_camera_dir / f"{frame_id}.txt", scale)
            exo_camera_scaled = scale_translation(exo_pred_camera_dir / f"{frame_id}.txt", scale)

            # save
            np.save(ego_depth_out_dir / f"{frame_id}.npy", ego_depth_scaled)
            np.savetxt(ego_camera_out_dir / f"{frame_id}.txt", ego_camera_scaled)
            np.save(exo_depth_out_dir / f"{frame_id}.npy", exo_depth_scaled)
            np.savetxt(exo_camera_out_dir / f"{frame_id}.txt", exo_camera_scaled)
            '''
            pred_depth_concat = np.stack([np.load(ego_pred_depth_file), np.load(exo_pred_depth_file)], axis=0)
            gt_depth_concat = np.stack([np.load(ego_gt_depth_file), np.load(exo_gt_depth_file)], axis=0)

            scale = calc_gt_scale(pred_depth_concat,
                                  gt_depth_concat,
                                  method = "median")
            
            ego_depth_scaled = np.load(ego_pred_depth_file) * scale
            exo_depth_scaled = np.load(exo_pred_depth_file) * scale
            ego_camera_scaled = scale_translation(ego_pred_camera_dir / f"{frame_id}.txt", scale)
            exo_camera_scaled = scale_translation(exo_pred_camera_dir / f"{frame_id}.txt", scale)
            
            

            
            ego_depth_out_path = ego_depth_out_dir / f"{frame_id}.npy"
            ego_camera_out_path = ego_camera_out_dir / f"{frame_id}.txt"
            exo_depth_out_path = exo_depth_out_dir / f"{frame_id}.npy"
            exo_camera_out_path = exo_camera_out_dir / f"{frame_id}.txt"

            
            np.save(ego_depth_out_path, ego_depth_scaled)
            np.savetxt(ego_camera_out_path, ego_camera_scaled)
            np.save(exo_depth_out_path, exo_depth_scaled)
            np.savetxt(exo_camera_out_path, exo_camera_scaled)
            '''


            

if __name__ == "__main__":
    main()