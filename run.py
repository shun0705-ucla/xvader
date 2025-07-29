import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from xvader.xvader import Xvader
from third_party.vggt.vggt.utils.load_fn import load_and_preprocess_images
from third_party.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from third_party.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

def run_model(file_names, model) -> dict:
    """
    file_names: Sorted list of image file paths.
    model: Pretrained Xvader model.
    """
    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")
    
    # load pretrained weights
    encoder_sd = torch.load("./checkpoints/depth_anything_v2_vitl_pretrained.pt", map_location=device)
    aa_frame_sd = torch.load("./checkpoints/vggt_frame_blocks.pt", map_location=device)
    aa_global_sd = torch.load("./checkpoints/vggt_global_blocks.pt", map_location=device)
    camera_sd = torch.load("./checkpoints/vggt_camera_head.pt", map_location=device)
    dpt_sd = torch.load("./checkpoints/depth_anything_v2_vitl_depth_head.pt", map_location=device)
    xvader.pretrained.load_state_dict(encoder_sd, strict=False)
    xvader.alternating_attention.frame_blocks.load_state_dict(aa_frame_sd, strict=False)
    xvader.alternating_attention.global_blocks.load_state_dict(aa_global_sd, strict=False)
    xvader.camera_head.load_state_dict(camera_sd, strict=False)
    xvader.depth_head.load_state_dict(dpt_sd, strict=False)

    # Move model to device
    model = model.to(device)
    model.eval()

    images = load_and_preprocess_images(file_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()#.squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    #world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    #predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--ref_path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    xvader = Xvader(**model_configs[args.encoder])


    # get input file names
    ref_names = glob.glob(os.path.join(args.ref_path, "*.jpg")) + \
            glob.glob(os.path.join(args.ref_path, "*.jpeg")) + \
            glob.glob(os.path.join(args.ref_path, "*.png"))
    image_names = glob.glob(os.path.join(args.img_path, "*.jpg")) + \
            glob.glob(os.path.join(args.img_path, "*.jpeg")) + \
            glob.glob(os.path.join(args.img_path, "*.png"))
    image_names = sorted(image_names)
    file_names = ref_names + image_names
    print(f"Found {len(image_names)} images and {len(ref_names)} reference images.")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")
    if len(ref_names) == 0:
        raise ValueError("No reference images found. Check your upload.")

    # main
    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    predictions = run_model(file_names, xvader)
    depth_maps = predictions["depth"]
    images = predictions["images"]

    for i, file_name in enumerate(file_names):
        image = images[i]
        image = np.transpose(image, (1, 2, 0)) # Convert to (H, W, 3)
        depth = depth_maps[i]
        # Normalize depth to 0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # Save results
        split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
        image = np.ascontiguousarray(image, dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.uint8)
        split_region = np.ascontiguousarray(split_region, dtype=np.uint8)
        print(image.shape, split_region.shape, depth.shape)
        combined_result = cv2.hconcat([image, split_region, depth])
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(file_name))[0] + '.png'), combined_result)