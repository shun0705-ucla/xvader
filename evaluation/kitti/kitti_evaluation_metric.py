from pathlib import Path
import numpy as np
from tqdm import tqdm
import math

# ---------- per-folder metrics (pred_depth vs gt_depth) ----------

def calc_depth_eval_metrics(pred_dir: Path, gt_dir: Path, max_depth=None, min_depth=0.0):
    """
    Depth metrics over matching .npy files (meters):
      - AbsRel (mean over images)
      - MAE    (mean over images)
      - RMSE   (sqrt(mean of per-image MSE))
      - logMAE (mean over images; natural log)
      - logRMSE(sqrt(mean of per-image squared log error))
    Assumptions: filenames (stems) match between pred_dir and gt_dir.
    """
    pred_files = {p.stem: p for p in pred_dir.glob("*.npy")}
    pairs = [(pred_files[g.stem], g) for g in gt_dir.glob("*.npy") if g.stem in pred_files]
    if not pairs:
        return None  # nothing to evaluate in this pair

    absrel_vals = []
    mae_vals    = []
    mse_vals    = []  # per-image mean squared error (for RMSE)
    logmae_vals = []
    logmse_vals = []  # per-image mean squared log error (for logRMSE)

    n_imgs, n_log_imgs = 0, 0

    for p_pred, p_gt in sorted(pairs, key=lambda x: x[0].stem):
        d_pred = np.asarray(np.load(p_pred), dtype=np.float64)
        d_gt   = np.asarray(np.load(p_gt),   dtype=np.float64)
        if d_pred.shape != d_gt.shape:
            continue  # keep it simple: skip mismatched shapes

        # clamp to eval range
        if min_depth is not None:
            d_pred = np.maximum(d_pred, min_depth)
            d_gt   = np.maximum(d_gt,   min_depth)
        if max_depth is not None:
            d_pred = np.minimum(d_pred, max_depth)
            d_gt   = np.minimum(d_gt,   max_depth)

        # valid pixels
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

        # log metrics (only where pred > 0)
        pos = dp > 0
        if np.any(pos):
            lerr = np.log(np.clip(dp[pos], 1e-12, None)) - np.log(np.clip(dg[pos], 1e-12, None))
            logmae_vals.append(np.mean(np.abs(lerr)))
            logmse_vals.append(np.mean(lerr**2))
            n_log_imgs += 1

    if n_imgs == 0:
        return None

    # aggregate per-folder
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

# ---------- root/scene/clip walker (starts from your snippet) ----------

kitti_root = Path("./kitti_clipsize_10")

scene_dirs = sorted(kitti_root.glob("scene_*"))
results_per_scene = {}

# global accumulators (AbsRel/MAE/logMAE are simple weighted means by #images (or #log_images);
# RMSE/logRMSE need squared-error sums)
total_imgs = 0
total_log_imgs = 0

sum_absrel = 0.0
sum_mae    = 0.0
sum_logmae = 0.0

sum_mse        = 0.0   # accumulate mean per-image MSE * count_images
sum_logmse     = 0.0   # accumulate mean per-image logMSE * count_log_images

for scene_dir in tqdm(scene_dirs, desc="Scenes"):
    clip_dirs = sorted(scene_dir.glob("clip_*"))
    scene_acc = {
        "count_images": 0,
        "count_log_images": 0,
        "sum_absrel": 0.0,
        "sum_mae": 0.0,
        "sum_logmae": 0.0,
        "sum_mse": 0.0,
        "sum_logmse": 0.0,
        "per_clip": {}
    }

    for clip_dir in tqdm(clip_dirs, desc=f"  {scene_dir.name}", leave=False):
        # adjust names if yours differ
        pred_depth = clip_dir / "pred_unidepthv2"
        gt_depth   = clip_dir / "depth_resized"
        if not (pred_depth.is_dir() and gt_depth.is_dir()):
            continue

        m = calc_depth_eval_metrics(pred_depth, gt_depth, max_depth=80.0, min_depth=0.0)
        if m is None:
            continue

        # store per-clip
        scene_acc["per_clip"][clip_dir.name] = m

        # accumulate to scene
        n     = m["count_images"]
        n_log = m["count_log_images"]
        scene_acc["count_images"]     += n
        scene_acc["count_log_images"] += n_log

        scene_acc["sum_absrel"]  += m["AbsRel_mean"] * n
        scene_acc["sum_mae"]     += m["MAE_mean"]    * n
        if np.isfinite(m["logMAE_mean"]) and n_log > 0:
            scene_acc["sum_logmae"] += m["logMAE_mean"] * n_log

        if np.isfinite(m["RMSE"]) and n > 0:
            # RMSE^2 = mean MSE; multiply by n to get sum of per-image MSEs
            scene_acc["sum_mse"] += (m["RMSE"] ** 2) * n
        if np.isfinite(m["logRMSE"]) and n_log > 0:
            scene_acc["sum_logmse"] += (m["logRMSE"] ** 2) * n_log

    # finalize per-scene means
    if scene_acc["count_images"] > 0:
        n     = scene_acc["count_images"]
        n_log = scene_acc["count_log_images"]

        scene_absrel  = scene_acc["sum_absrel"] / n
        scene_mae     = scene_acc["sum_mae"]    / n
        scene_rmse    = float(np.sqrt(scene_acc["sum_mse"] / n)) if scene_acc["sum_mse"] > 0 else float("nan")

        scene_logmae  = float(scene_acc["sum_logmae"] / n_log) if n_log > 0 else float("nan")
        scene_logrmse = float(np.sqrt(scene_acc["sum_logmse"] / n_log)) if n_log > 0 else float("nan")

        results_per_scene[scene_dir.name] = {
            "count_images": n,
            "AbsRel_mean": float(scene_absrel),
            "MAE_mean":    float(scene_mae),
            "RMSE":        scene_rmse,
            "logMAE_mean": scene_logmae,
            "logRMSE":     scene_logrmse,
            "per_clip":    scene_acc["per_clip"],
        }

        # roll into global
        total_imgs     += n
        total_log_imgs += n_log
        sum_absrel     += scene_acc["sum_absrel"]
        sum_mae        += scene_acc["sum_mae"]
        sum_logmae     += scene_acc["sum_logmae"]
        sum_mse        += scene_acc["sum_mse"]
        sum_logmse     += scene_acc["sum_logmse"]

# ---------- global summary ----------

global_summary = None
if total_imgs > 0:
    global_absrel  = sum_absrel / total_imgs
    global_mae     = sum_mae    / total_imgs
    global_rmse    = float(np.sqrt(sum_mse / total_imgs)) if sum_mse > 0 else float("nan")

    global_logmae  = float(sum_logmae / total_log_imgs) if total_log_imgs > 0 else float("nan")
    global_logrmse = float(np.sqrt(sum_logmse / total_log_imgs)) if total_log_imgs > 0 else float("nan")

    global_summary = {
        "count_images": int(total_imgs),
        "AbsRel_mean":  float(global_absrel),
        "MAE_mean":     float(global_mae),
        "RMSE":         float(global_rmse),
        "logMAE_mean":  float(global_logmae),
        "logRMSE":      float(global_logrmse),
    }

# quick printout
print("GLOBAL:", global_summary)
for s, m in results_per_scene.items():
    print(s, {k: m[k] for k in ("count_images", "AbsRel_mean", "MAE_mean", "RMSE", "logMAE_mean", "logRMSE")})
