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

    kitti_root = Path("./kitti_clipsize_10")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model_name = "xvader" # xvader or vggt

    model = _load_model(model_name, device)

    scene_dirs = sorted(kitti_root.glob("scene_*"))
    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        clip_dirs = sorted(scene_dir.glob("clip_*"))
        for clip_dir in tqdm(clip_dirs, desc=f"  {scene_dir.name}", leave=False):
            rgb_dir = clip_dir / "rgb_resized"
            k_dir = clip_dir / "K_resized"
            out_depth_dir = clip_dir / f"pred_depth_{model_name}"
            out_camera_dir = clip_dir / f"pred_camera_{model_name}"
            out_depth_dir.mkdir(parents=True, exist_ok=True)
            out_camera_dir.mkdir(parents=True, exist_ok=True)

            # preprocess
            list_images = []
            list_intrinsics = []
            rgb_files = sorted(rgb_dir.glob("*.png"))
            for rgb_file in rgb_files:
                # load image
                img = Image.open(rgb_file)
                img = img.convert("RGB")
                img = to_tensor(img)
                list_images.append(img)
                # load intrinsics
                frame_id = rgb_file.stem
                k_file = k_dir / f"{frame_id}.npy"
                intri = torch.from_numpy(np.load(k_file)).float()
                list_intrinsics.append(intri)
            # to Tensor
            images = torch.stack(list_images).to(device)
            intrinsics = torch.stack(list_intrinsics, dim=0).to(device)  # (N,3,3)

            # run model
            with torch.no_grad():
                if model_name == "xvader":
                    predictions = model(images, intrinsics)
                elif model_name == "vggt":
                    predictions = model(images)
                else:
                    raise ValueError(f"Unknown model name '{model_name}'")
                
            depth = predictions["depth"] # (B, S, H, W, 1)
            depth_np = depth.squeeze(0).squeeze(-1).detach().cpu().numpy()  # (S, H, W)
            pose = predictions["pose_enc"] # (B, S, 7 or 9)
            pose_np = pose.squeeze(0).detach().cpu().numpy()   # (S, 9)

            
            # save output
            for i, rgb_file in enumerate(rgb_files):
                frame_id = rgb_file.stem
                out_depth_path = out_depth_dir / f"{frame_id}.npy"
                out_camera_path = out_camera_dir / f"{frame_id}.txt"

                pred_depth = depth_np[i]
                pred_pose = pose_np[i]

                np.save(out_depth_path, pred_depth)
                np.savetxt(out_camera_path, pred_pose)

if __name__ == "__main__":
    main()