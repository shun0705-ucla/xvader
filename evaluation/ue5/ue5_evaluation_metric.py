import math
import numpy as np
from pathlib import Path

def calc_depth_eval_metrics(path_to_pred_depthmap_folder, path_to_gt_depthmap_folder,
                            max_depth=None, min_depth=0.0):
    """
    Depth metrics over matching .npy files (meters):
      - AbsRel_mean
      - MAE_mean
      - RMSE
      - logMAE_mean
      - logRMSE

    Notes:
      * log-metrics are computed only at pixels where pred>0 (and valid GT).
      * RMSE/logRMSE are sqrt of the mean of per-image (log-)MSE.
    """
    pred_dir = Path(path_to_pred_depthmap_folder)
    gt_dir   = Path(path_to_gt_depthmap_folder)

    pred_files = {p.stem: p for p in pred_dir.glob("*.npy")}
    pairs = [(pred_files[g.stem], g) for g in gt_dir.glob("*.npy") if g.stem in pred_files]
    if not pairs:
        raise ValueError("No matching .npy depth files by stem between the two folders.")

    absrel_vals, mae_vals, mse_vals = [], [], []
    logmae_vals, logmse_vals = [], []
    n_imgs = 0
    n_log_imgs = 0

    for p_pred, p_gt in sorted(pairs, key=lambda x: x[0].stem):
        d_pred = np.asarray(np.load(p_pred), dtype=np.float64)
        d_gt   = np.asarray(np.load(p_gt),   dtype=np.float64)
        if d_pred.shape != d_gt.shape:
            raise ValueError(f"Shape mismatch for {p_pred.name} vs {p_gt.name}: {d_pred.shape} vs {d_gt.shape}")

        # --- clamp to evaluation range (applied to BOTH pred and GT) ---
        if min_depth is not None:
            d_pred = np.maximum(d_pred, min_depth)
            d_gt   = np.maximum(d_gt,   min_depth)
        if max_depth is not None:
            d_pred = np.minimum(d_pred, max_depth)
            d_gt   = np.minimum(d_gt,   max_depth)

        # valid mask (based on GT>0 & finite; pred finite)
        mask = np.isfinite(d_gt) & (d_gt > 0) & np.isfinite(d_pred)
        if not np.any(mask):
            continue

        dp = d_pred[mask]
        dg = d_gt[mask]
        err = dp - dg

        # Abs Rel
        absrel_vals.append(np.mean(np.abs(err) / np.clip(dg, 1e-12, None)))

        # MAE / MSE (for RMSE)
        mae_vals.append(np.mean(np.abs(err)))
        mse_vals.append(np.mean(err**2))
        n_imgs += 1
        print("n,mae",n_imgs,mae_vals[-1])

        # log metrics (only where pred > 0)
        pos = dp > 0
        if np.any(pos):
            lerr = np.log(np.clip(dp[pos], 1e-12, None)) - np.log(np.clip(dg[pos], 1e-12, None))
            logmae_vals.append(np.mean(np.abs(lerr)))
            logmse_vals.append(np.mean(lerr**2))
            n_log_imgs += 1

    if n_imgs == 0:
        raise ValueError("No valid pixels found across matched depth pairs.")

    rmse    = float(np.sqrt(np.mean(mse_vals))) if mse_vals else float("nan")
    logrmse = float(np.sqrt(np.mean(logmse_vals))) if logmse_vals else float("nan")
    logmae  = float(np.mean(logmae_vals)) if logmae_vals else float("nan")

    return {
        "count_images": int(n_imgs),
        "count_log_images": int(n_log_imgs),
        "AbsRel_mean": float(np.mean(absrel_vals)),
        "MAE_mean":   float(np.mean(mae_vals)),
        "RMSE":       rmse,
        "logMAE_mean": logmae,
        "logRMSE":     logrmse,
    }


def calc_camera_eval_metrics(path_to_pred_campose_folder, path_to_gt_campose_folder):
    """
    Pose metrics over matching files:
      - mean absolute geodesic rotation error (deg)
      - mean translation L2 (m)

    Assumptions:
      * Each file has [tx,ty,tz,qx,qy,qz,qw]
      * Filenames (stems) match between pred/ and gt/
      * Accepts .txt or .npy
    """
    def _quat_geodesic_deg(q_pred, q_gt):
        """
        q = [qx,qy,qz,qw], returns absolute geodesic angle in degrees.
        """
        qp = np.asarray(q_pred, dtype=np.float64)
        qg = np.asarray(q_gt,   dtype=np.float64)
        # normalize
        qp /= (np.linalg.norm(qp) + 1e-15)
        qg /= (np.linalg.norm(qg) + 1e-15)
        # same hemisphere
        if np.dot(qp, qg) < 0:
            qp = -qp
        # relative quaternion wrt gt: q_rel = qp * conj(qg)
        x1,y1,z1,w1 = qp
        x2,y2,z2,w2 = qg
        xr =  w1*(-x2) + x1*w2 + y1*(-z2) - z1*(-y2)
        yr =  w1*(-y2) - x1*(-z2) + y1*w2 + z1*(-x2)
        zr =  w1*(-z2) + x1*(-y2) - y1*(-x2) + z1*w2
        wr =  w1*w2 - x1*(-x2) - y1*(-y2) - z1*(-z2)
        wr = float(np.clip(wr, -1.0, 1.0))
        ang = 2.0 * math.degrees(math.acos(wr))
        # map to [0, 180]
        if ang > 180.0:
            ang = 360.0 - ang
        return abs(ang)
    
    def _load_pose_simple(path):
        """
        Reads 7 floats [tx,ty,tz,qx,qy,qz,qw] from .txt (whitespace) or .npy.
        """
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float64).reshape(-1)
        else:
            arr = np.fromstring(path.read_text(), sep=" ", dtype=np.float64)
        if arr.size < 7:
            raise ValueError(f"{path.name}: expected 7 values, got {arr.size}")
        return arr[:7]

    pred_dir = Path(path_to_pred_campose_folder)
    gt_dir   = Path(path_to_gt_campose_folder)

    pred_files = {p.stem: p for p in list(pred_dir.glob("*.txt")) + list(pred_dir.glob("*.npy"))}
    pairs = []
    for g in list(gt_dir.glob("*.txt")) + list(gt_dir.glob("*.npy")):
        if g.stem in pred_files:
            pairs.append((pred_files[g.stem], g))
    if not pairs:
        raise ValueError("No matching pose files by stem between the two folders.")

    rot_errs = []
    trans_errs = []
    n = 0

    for p_pred, p_gt in sorted(pairs, key=lambda x: x[0].stem):
        pred = _load_pose_simple(p_pred)
        gt   = _load_pose_simple(p_gt)

        t_pred, q_pred = pred[:3], pred[3:7]
        t_gt,   q_gt   = gt[:3],   gt[3:7]

        rot_deg = _quat_geodesic_deg(q_pred, q_gt)
        trans_l2 = float(np.linalg.norm(t_pred - t_gt))

        rot_errs.append(rot_deg)
        trans_errs.append(trans_l2)
        n += 1

    return {
        "count_frames": int(n),
        "rotation_mean_abs_geodesic_deg": float(np.mean(np.abs(rot_errs))),
        "translation_mean_L2_m": float(np.mean(trans_errs)),
    }

def main():
    #pred_depth_path = Path("./ue5/LowerSector_test_ego/pred_depth_vggt_scaled")
    #gt_depth_path = Path("./ue5/LowerSector_test_ego/depth_resized")
    #pred_camera_path = Path("./ue5/LowerSector_test_ego/pred_camera_vggt_scaled")
    #gt_camera_path = Path("./ue5/LowerSector_test_ego/gt_camera")
    pred_depth_path = Path("./ue5/LowerSector_test_ego/pred_unidepthv2")
    gt_depth_path = Path("./ue5/LowerSector_test_ego/depth_resized")
    #pred_camera_path = Path("./ue5/LowerSector_test_ego/pred_camera_xvader")
    #gt_camera_path = Path("./ue5/LowerSector_test_ego/gt_camera")

    depth_metrics = calc_depth_eval_metrics(pred_depth_path, gt_depth_path, max_depth=80)
    #pose_metrics  = calc_camera_eval_metrics(pred_camera_path, gt_camera_path)
    print(depth_metrics)
    #print(pose_metrics)

if __name__ == "__main__":
    main()
