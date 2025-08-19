import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.UniDepth.unidepth.layers import (MLP, AttentionBlock, AttentionLayer, PositionEmbeddingSine, ResUpsampleBil)
from .xvader_utils import flat_to_map, map_to_flat

class MetricDepthHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion: int = 4,
        out_dim: int = 1,
        use_norm=False,
    ) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.process_features = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.depth_mlp = nn.ModuleList([])
        
        self.to_latents = nn.Linear(hidden_dim, hidden_dim)

        iter_upsample = 3
        mult = 2
        for i in range(iter_upsample):
            current_dim = min(hidden_dim, mult * hidden_dim // int(2**i))
            next_dim = mult * hidden_dim // int(2 ** (i + 1))
            output_dim = max(next_dim, out_dim)
            self.process_features.append(
                nn.ConvTranspose2d(
                    hidden_dim,
                    current_dim,
                    kernel_size=max(1, 2 * i),
                    stride=max(1, 2 * i),
                    padding=0,
                )
            )

            self.ups.append(
                ResUpsampleBil(
                    current_dim,
                    output_dim=output_dim,
                    expansion=expansion,
                    layer_scale=1.0,
                    kernel_size=3,
                    num_layers=2,
                    use_norm=use_norm,
                )
            )

        self.depth_mlp = nn.Sequential(
            nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim)
        )

        self.confidence_mlp = nn.Sequential(
            nn.LayerNorm(next_dim), nn.Linear(next_dim, output_dim)
        )

        self.to_depth_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_confidence_lr = nn.Conv2d(
            output_dim,
            output_dim // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.to_confidence_hr = nn.Sequential(
            nn.Conv2d(
                output_dim // 2, 32, kernel_size=3, padding=1, padding_mode="reflect"
            ),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def set_original_shapes(self, H:int, W:int):
        self.H = H
        self.W = W

    def set_input_feature_shapes(self, patch_h:int, patch_w):
        self.patch_h = patch_h
        self.patch_w = patch_w

    def process(self, features_list):
        init_latents = self.to_latents(features_list[0]) # (B*S, N, C)
        init_latents = flat_to_map(init_latents, self.patch_h, self.patch_w) #(B*S, C, patch_h, patch_w)
        latents = init_latents

        for i, up in enumerate(self.ups):
            latents = latents + self.process_features[i](flat_to_map(features_list[i+1], self.patch_h, self.patch_w))
            latents = up(latents)
        upsampled_features = latents

        return upsampled_features

    def depth_proj(self, features_map):
        h_out, w_out = features_map.shape[-2:]

        depth_features_map = self.depth_mlp(features_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        depth_features_map = F.interpolate(
            depth_features_map,
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )
        logdepth = self.to_depth_lr(depth_features_map)
        logdepth = F.interpolate(
            logdepth, size=(self.H, self.W), mode="bilinear", align_corners=True
        )
        logdepth = self.to_depth_hr(logdepth)
        return logdepth

    def confidence_proj(self, features_map):
        highres_features = features_map.permute(0, 2, 3, 1)
        confidence = self.confidence_mlp(highres_features).permute(0, 3, 1, 2)
        confidence = self.to_confidence_lr(confidence)
        confidence = F.interpolate(
            confidence, size=(self.H, self.W), mode="bilinear", align_corners=True
        )
        confidence = self.to_confidence_hr(confidence)
        return confidence

    def decode(self, features_map):
        logdepth = self.depth_proj(features_map)
        confidence = self.confidence_proj(features_map)
        return logdepth, confidence

    def forward(self, features_list: list[torch.Tensor]):
        #B, S, N, C = features_list.shape
        #features = features.view(B*S, N, C)# process per image

        upsampled_features_map = self.process(features_list)
        logdepth, logconf = self.decode(upsampled_features_map)
        depth_radius = torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)
        confidence = torch.exp(logconf.clip(min=-8.0, max=8.0))
        return depth_radius, confidence