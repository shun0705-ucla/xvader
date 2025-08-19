import torch
import torch.nn as nn

from third_party.UniDepth.unidepth.layers import (MLP, AttentionBlock, AttentionLayer, PositionEmbeddingSine)
from third_party.UniDepth.unidepth.utils.camera import BatchCamera, Camera, Pinhole
from third_party.UniDepth.unidepth.utils.geometric import flat_interpolate
from third_party.UniDepth.unidepth.utils.positional_embedding import generate_fourier_features
from .xvader_utils import map_to_flat


class CameraEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim :int,
        num_layers :int
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_camera = nn.ModuleList([])

        for i in range(num_layers):
            self.prompt_camera.append(
                AttentionLayer(
                    num_blocks=1,
                    dim=hidden_dim,
                    num_heads=hidden_dim//64,
                    expansion=4.0,
                    dropout=0.0,
                    layer_scale=-1.0,
                    context_dim=hidden_dim,
                    use_bias=False,
                )
            )
        
    def forward(self, features_list: list[torch.Tensor], rays_embedding: torch.Tensor, patch_h:int, patch_w:int):
        rays_embedding = rays_embedding / torch.norm(rays_embedding, dim=-1, keepdim=True).clip(min=1e-4)
        x, y, z = rays_embedding[..., 0], rays_embedding[..., 1], rays_embedding[..., 2]
        polar = torch.acos(z)
        x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
        azimuth = torch.atan2(y, x_clipped)
        rays_embedding = torch.stack([polar, azimuth], dim=-1)
        rays_embedding = generate_fourier_features(
            rays_embedding,
            dim=self.hidden_dim,
            max_freq=max(patch_h,patch_w) // 2,
            use_log=True,
            cat_orig=False,
        )

        conditioned_features_list = [
            prompter(features, rays_embedding)
            for prompter, features in zip(self.prompt_camera, features_list)
        ]

        return conditioned_features_list