"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
-> modified for processing batched camera intrinsics
"""

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from third_party.UniDepth.unidepth.utils.coordinate import coords_grid
from third_party.UniDepth.unidepth.utils.camera import BatchCamera


def invert_pinhole(K):
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    K_inv = torch.zeros_like(K)
    K_inv[..., 0, 0] = 1.0 / fx
    K_inv[..., 1, 1] = 1.0 / fy
    K_inv[..., 0, 2] = -cx / fx
    K_inv[..., 1, 2] = -cy / fy
    K_inv[..., 2, 2] = 1.0
    return K_inv

class Camera:
    """
    This is meant to be an abstract parent class, please use the others as actual cameras.
    Pinhole, FIsheye624, MEI, OPENCV, EUCM, Spherical (Equirectangular).

    """

    def __init__(self, params=None, K=None):
        if params.ndim == 1:
            params = params.unsqueeze(0)

        if K is None:
            K = (
                torch.eye(3, device=params.device, dtype=params.dtype)
                .unsqueeze(0)
                .repeat(params.shape[0], 1, 1)
            )
            K[..., 0, 0] = params[..., 0]
            K[..., 1, 1] = params[..., 1]
            K[..., 0, 2] = params[..., 2]
            K[..., 1, 2] = params[..., 3]

        self.params = params
        self.K = K
        self.overlap_mask = None
        self.projection_mask = None

    def project(self, xyz):
        raise NotImplementedError

    def unproject(self, uv):
        raise NotImplementedError

    def get_projection_mask(self):
        return self.projection_mask

    def get_overlap_mask(self):
        return self.overlap_mask

    def reconstruct(self, depth):
        id_coords = coords_grid(
            1, depth.shape[-2], depth.shape[-1], device=depth.device
        )
        rays = self.unproject(id_coords)
        return (
            rays / rays[:, -1:].clamp(min=1e-4) * depth.clamp(min=1e-4)
        )  # assumption z>0!!!

    def resize(self, factor):
        self.K[..., :2, :] *= factor
        self.params[..., :4] *= factor
        return self

    def to(self, device, non_blocking=False):
        self.params = self.params.to(device, non_blocking=non_blocking)
        self.K = self.K.to(device, non_blocking=non_blocking)
        return self

    def get_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(b, h, w, device=self.K.device, noisy=noisy)
        rays = self.unproject(uv)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def get_pinhole_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(b, h, w, device=self.K.device, homogeneous=True, noisy=noisy)
        rays = (invert_pinhole(self.K) @ uv.reshape(b, 3, -1)).reshape(b, 3, h, w)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

    def flip(self, H, W, direction="horizontal"):
        new_cx = (
            W - self.params[:, 2] if direction == "horizontal" else self.params[:, 2]
        )
        new_cy = H - self.params[:, 3] if direction == "vertical" else self.params[:, 3]
        self.params = torch.stack(
            [self.params[:, 0], self.params[:, 1], new_cx, new_cy], dim=1
        )
        self.K[..., 0, 2] = new_cx
        self.K[..., 1, 2] = new_cy
        return self

    def clone(self):
        return deepcopy(self)

    def crop(self, left, top, right=None, bottom=None):
        self.K[..., 0, 2] -= left
        self.K[..., 1, 2] -= top
        self.params[..., 2] -= left
        self.params[..., 3] -= top
        return self

    # helper function to get how fov changes based on new original size and new size
    def get_new_fov(self, new_shape, original_shape):
        new_hfov = 2 * torch.atan(
            self.params[..., 2] / self.params[..., 0] * new_shape[1] / original_shape[1]
        )
        new_vfov = 2 * torch.atan(
            self.params[..., 3] / self.params[..., 1] * new_shape[0] / original_shape[0]
        )
        return new_hfov, new_vfov

    def mask_overlap_projection(self, projected):
        B, _, H, W = projected.shape
        id_coords = coords_grid(B, H, W, device=projected.device)

        # check for mask where flow would overlap with other part of the image
        # eleemtns coming from the border are then masked out
        flow = projected - id_coords
        gamma = 0.1
        sample_grid = gamma * flow + id_coords  # sample along the flow
        sample_grid[:, 0] = sample_grid[:, 0] / (W - 1) * 2 - 1
        sample_grid[:, 1] = sample_grid[:, 1] / (H - 1) * 2 - 1
        sampled_flow = F.grid_sample(
            flow,
            sample_grid.permute(0, 2, 3, 1),
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        mask = (
            (1 - gamma) * torch.norm(flow, dim=1, keepdim=True)
            < torch.norm(sampled_flow, dim=1, keepdim=True)
        ) | (torch.norm(flow, dim=1, keepdim=True) < 1)
        return mask

    def _pad_params(self):
        # Ensure params are padded to length 16
        if self.params.shape[1] < 16:
            padding = torch.zeros(
                16 - self.params.shape[1],
                device=self.params.device,
                dtype=self.params.dtype,
            )
            padding = padding.unsqueeze(0).repeat(self.params.shape[0], 1)
            return torch.cat([self.params, padding], dim=1)
        return self.params

    @staticmethod
    def flatten_cameras(cameras):  # -> list[Camera]:
        # Recursively flatten BatchCamera into primitive cameras
        flattened_cameras = []
        for camera in cameras:
            if isinstance(camera, BatchCamera):
                flattened_cameras.extend(BatchCamera.flatten_cameras(camera.cameras))
            elif isinstance(camera, list):
                flattened_cameras.extend(camera)
            else:
                flattened_cameras.append(camera)
        return flattened_cameras

    @staticmethod
    def _stack_or_cat_cameras(cameras, func, **kwargs):
        # Generalized method to handle stacking or concatenation
        flat_cameras = BatchCamera.flatten_cameras(cameras)
        K_matrices = [camera.K for camera in flat_cameras]
        padded_params = [camera._pad_params() for camera in flat_cameras]

        stacked_K = func(K_matrices, **kwargs)
        stacked_params = func(padded_params, **kwargs)

        # Keep track of the original classes
        original_class = [x.__class__.__name__ for x in flat_cameras]
        return BatchCamera(stacked_params, stacked_K, original_class, flat_cameras)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch.cat:
            return Camera._stack_or_cat_cameras(args[0], func, **kwargs)

        if func is torch.stack:
            return Camera._stack_or_cat_cameras(args[0], func, **kwargs)

        if func is torch.flatten:
            return Camera._stack_or_cat_cameras(args[0], torch.cat, **kwargs)

        return super().__torch_function__(func, types, args, kwargs)

    @property
    def device(self):
        return self.K.device

    # here we assume that cx,cy are more or less H/2 and W/2
    @property
    def hfov(self):
        return 2 * torch.atan(self.params[..., 2] / self.params[..., 0])

    @property
    def vfov(self):
        return 2 * torch.atan(self.params[..., 3] / self.params[..., 1])

    @property
    def max_fov(self):
        return 150.0 / 180.0 * np.pi, 150.0 / 180.0 * np.pi


class Pinhole(Camera):
    def __init__(self, params=None, K=None):
        assert params is not None or K is not None
        if params is None:
            params = torch.stack(
                [K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]], dim=-1
            )
        super().__init__(params=params, K=K)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def project(self, pcd):
        b, _, h, w = pcd.shape
        pcd_flat = pcd.reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = self.K @ pcd_flat
        pcd_proj = cam_coords[:, :2] / cam_coords[:, -1:].clamp(min=0.01)
        pcd_proj = pcd_proj.reshape(b, 2, h, w)
        invalid = (
            (pcd_proj[:, 0] >= 0)
            & (pcd_proj[:, 0] < w)
            & (pcd_proj[:, 1] >= 0)
            & (pcd_proj[:, 1] < h)
        )
        self.projection_mask = (~invalid).unsqueeze(1)
        return pcd_proj

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        b, _, h, w = uv.shape
        uv_flat = uv.reshape(b, 2, -1)  # [B, 2, H*W]
        uv_homogeneous = torch.cat(
            [uv_flat, torch.ones(b, 1, h * w, device=uv.device)], dim=1
        )  # [B, 3, H*W]
        K_inv = torch.inverse(self.K.float())
        xyz = K_inv @ uv_homogeneous
        xyz = xyz / xyz[:, -1:].clip(min=1e-4)
        xyz = xyz.reshape(b, 3, h, w)
        self.unprojection_mask = xyz[:, -1:] > 1e-4
        return xyz

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def reconstruct(self, depth):
        b, _, h, w = depth.shape
        uv = coords_grid(b, h, w, device=depth.device)
        xyz = self.unproject(uv) * depth.clip(min=0.0)
        return xyz