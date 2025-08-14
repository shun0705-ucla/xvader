import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from third_party.Depth_Anything_V2.depth_anything_v2.dinov2 import DINOv2
from .dpt import DPTHead
from third_party.Depth_Anything_V2.metric_depth.depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch



from .camera_head import CameraHead
#from .dpt import DPTHead
from .alternating_attention import AlternatingAttention
from .feature_attention import FeatureAttention

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

@dataclass
class AttentionConfig:
    aa_depth: int
    aa_num_heads: int
    fa_depth: int
    fa_num_heads: int

@dataclass
class DPTConfig:
    features: int
    out_channels: list[int]

class Xvader(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        use_bn=False, 
        use_clstoken=False,
    ):
        super(Xvader, self).__init__()

        self.intermediate_layer_idx_list = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],          
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.intermediate_layer_idx = self.intermediate_layer_idx_list[encoder]

        if encoder not in self.intermediate_layer_idx_list:
            raise ValueError(f"Unknown encoder '{encoder}'. Supported: {list(self.intermediate_layer_idx_list.keys())}")

        self.dpt_config = {
            #['features', 'out_channels']
            'vits': DPTConfig(64, [48, 96, 192, 384]),
            'vitb': DPTConfig(128, [96, 192, 384, 768]),          
            'vitl': DPTConfig(256, [256, 512, 1024, 1024]), 
            'vitg': DPTConfig(384, [1536, 1536, 1536, 1536])
        }[encoder]

        self.attention_config = {
            # [aa_depth, aa_num_heads, fa_depth, fa_num_heads]
            'vits': AttentionConfig(8, 6, 2, 6),
            'vitb': AttentionConfig(8, 12, 2, 12),
            'vitl': AttentionConfig(24, 16, 8, 16),
            'vitg': AttentionConfig(24, 24, 8, 24)
        }[encoder]
        
        self.pretrained = DINOv2(model_name=encoder)
        self.embed_dim = self.pretrained.embed_dim
        self.patch_size = self.pretrained.patch_size

        self.alternating_attention = AlternatingAttention(embed_dim=self.embed_dim,
                                                          depth=self.attention_config.aa_depth,
                                                          num_heads=self.attention_config.aa_num_heads,
                                                           patch_size=self.patch_size)
        self.feature_attention = FeatureAttention(embed_dim=self.embed_dim,
                                                  depth=self.attention_config.fa_depth,
                                                  num_heads=self.attention_config.fa_num_heads,
                                                  patch_size=self.patch_size)
        
        self.depth_head = DPTHead(self.embed_dim, self.dpt_config.features, use_bn, out_channels=self.dpt_config.out_channels, use_clstoken=use_clstoken)
        self.camera_head = CameraHead(2*self.embed_dim)
        
        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    
    def forward(self, images: torch.Tensor):
        if len(images.shape) == 4: # (S, C, H, W)
            images = images.unsqueeze(0) # (1, S, C, H, W)
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        # 1. patch & embed by DINOv2   
        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.pretrained.get_intermediate_layers(images, self.intermediate_layer_idx, return_class_token=False)
        
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # 2. Update 4th layer by alternating attention (from VGGT)
        # Convert tuple to list to inject updated patch tokens
        patch_tokens = torch.stack(patch_tokens, dim=0)
        final_token = patch_tokens[-1].clone()
        camera_tokens, updatad_final_token = self.alternating_attention(final_token, patch_h, patch_w, images.device)
        patch_tokens[-1] = updatad_final_token

        # 3. Update all layers by feature attention
        patch_tokens = self.feature_attention(patch_tokens, patch_h, patch_w, images.device)

        # 4. prediction heads
        
        # retrieve batch size for prediction
        images = images.view(B, S, C_in, H, W)
        _, C2 = camera_tokens.shape
        camera_tokens = camera_tokens.reshape(B, S, C2)
        # DPT processes patch_tokens in (L, B, S, P, C)
        L, _, P, C =patch_tokens.shape # _ is B*S
        patch_tokens = patch_tokens.view(L, B, S, P, C)

        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(camera_tokens)
                predictions["pose_enc"] = pose_enc_list[-1] # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth = self.depth_head(patch_tokens, patch_h, patch_w)
                #depth = F.relu(depth)
                predictions["depth"] = depth
        
        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions