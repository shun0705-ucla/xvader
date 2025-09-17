#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms as TF
from PIL import Image

from xvader.xvader import Xvader
from scripts.load_weight import load_checkpoint_into_model
from third_party.vggt.vggt.models.vggt import VGGT

def _load_model(model_name: str, device: torch.device):
    if model_name == "xvader":
        model = Xvader()
        compat, _sd = load_checkpoint_into_model(
            model,
            "./logs/exp001/ckpts/checkpoint_20.pt",
            device=device,
            strict=True,        # keep strict=True to catch real mismatches
            prefer_ema=False,   # set True if you want EMA weights when present
            verbose=True,
        )

    elif model_name == "vggt":
        model = VGGT()
        model.load_state_dict(torch.load("./checkpoints/vggt_vitl.pt", map_location=device), strict=True)

    else:
        raise ValueError(f"Unknown model name '{model_name}'")
    
    model = model.to(device)
    model.eval()
    return model

def main():
    to_tensor = TF.ToTensor()

    ue5_root = Path("./ue5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    model_name = "xvader" # xvader or vggt

    model = _load_model(model_name, device)

    ego_scene_dirs = sorted(ue5_root.glob("*_ego"))
    for ego_scene_dir in tqdm(ego_scene_dirs, desc="Scenes"):
        exo_scene_dir = Path(str(ego_scene_dir).replace("_ego", "_exo"))
        ego_rgb_dir = ego_scene_dir / "rgb_resized"
        exo_rgb_dir = exo_scene_dir / "rgb_resized"
        ego_k_dir = ego_scene_dir / "K_resized"
        exo_k_dir = exo_scene_dir / "K_resized"
        ego_depth_out_dir = ego_scene_dir / f"pred_depth_{model_name}"
        exo_depth_out_dir = exo_scene_dir / f"pred_depth_{model_name}"
        ego_camera_out_dir = ego_scene_dir / f"pred_camera_{model_name}"
        exo_camera_out_dir = exo_scene_dir / f"pred_camera_{model_name}"
        ego_depth_out_dir.mkdir(parents=True, exist_ok=True)
        exo_depth_out_dir.mkdir(parents=True, exist_ok=True)
        ego_camera_out_dir.mkdir(parents=True, exist_ok=True)
        exo_camera_out_dir.mkdir(parents=True, exist_ok=True)

        for ego_rgb_file in sorted(ego_rgb_dir.glob("*.png")):
            frame_id = ego_rgb_file.stem
            exo_rgb_file = exo_rgb_dir / f"{frame_id}.png"
            ego_k_file = ego_k_dir / f"{frame_id}.npy"
            exo_k_file = exo_k_dir / f"{frame_id}.npy"

            # load ego image
            ego_rgb = Image.open(ego_rgb_file)
            ego_rgb = ego_rgb.convert("RGB")
            ego_rgb = to_tensor(ego_rgb)
            # load exo image
            exo_rgb = Image.open(exo_rgb_file)
            exo_rgb = exo_rgb.convert("RGB")
            exo_rgb = to_tensor(exo_rgb)
            # load intrinsics
            ego_intrinsics = torch.from_numpy(np.load(ego_k_file)).float()
            exo_intrinsics = torch.from_numpy(np.load(exo_k_file)).float()
            # batchify exo:reference frame, ego:source frame
            images = torch.stack([exo_rgb, ego_rgb], dim=0).to(device)  # [B, C, H, W]
            intrinsics = torch.stack([exo_intrinsics, ego_intrinsics], dim=0).to(device)  # [B, 3, 3]

            with torch.no_grad():
                if model_name == "xvader":
                    predictions = model(images, intrinsics)
                elif model_name == "vggt":
                    predictions = model(images)
                else:
                    raise ValueError(f"Unknown model name '{model_name}'")
            depth = predictions["depth"] # (1, 2, H, W, 1)
            depth_np = depth.squeeze(0).squeeze(-1).detach().cpu().numpy()  # (2, H, W)
            ego_depth = depth_np[1]
            exo_depth = depth_np[0]
            pose = predictions["pose_enc"] # (1, 2, 7 or 9)
            pose_np = pose.squeeze(0).detach().cpu().numpy()   # (2, 9)
            ego_pose = pose_np[1]
            exo_pose = pose_np[0]
            
            # save output
            ego_depth_out_path = ego_depth_out_dir / f"{frame_id}.npy"
            exo_depth_out_path = exo_depth_out_dir / f"{frame_id}.npy"
            ego_camera_out_path = ego_camera_out_dir / f"{frame_id}.txt"
            exo_camera_out_path = exo_camera_out_dir / f"{frame_id}.txt"
            np.save(ego_depth_out_path, ego_depth)
            np.save(exo_depth_out_path, exo_depth)
            np.savetxt(ego_camera_out_path, ego_pose)
            np.savetxt(exo_camera_out_path, exo_pose)

if __name__ == "__main__":
    main()