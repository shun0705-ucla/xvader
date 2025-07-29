import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as compose

from third_party.Depth_Anything_V2.depth_anything_v2.dinov2 import DINOv2
from third_party.Depth_Anything_V2.depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from third_party.vggt.vggt.heads.camera_head import CameraHead

from dpt import DPTHead
from alternating_attention import AlternatingAttention

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Xvader(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False, 
        use_clstoken=False
    ):
        super(Xvader, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],          
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.embed_dim = self.pretrained.embed_dim
        self.patch_size = self.pretrained.patch_size

        self.alternating_attention = AlternatingAttention(embed_dim=1024, patch_size=self.patch_size)
        
        self.camera_head = CameraHead(2*self.embed_dim)
        self.depth_head = DPTHead(self.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    
    def forward(self, images: torch.Tensor):
        if len(images.shape) == 4: # (S, C, H, W)
            images = images.unsqueeze(0) # (1, S, C, H, W)
        B, S, C_in, H, W = images.shape
        patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        # 1. patch & embed by DINOv2   
        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std
        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.pretrained.get_intermediate_layers(images, self.intermediate_layer_idx[self.encoder], return_class_token=False)        

        # 2. Update final layer by alternating attention (from VGGT)
        # Convert tuple to list to inject updated patch tokens
        patch_tokens = list(patch_tokens)
        aggregated_tokens_list, patch_start_idx = self.alternating_attention(images, patch_tokens[-1])
        updated_final_tokens = aggregated_tokens_list[-1] # [B, S, cam_tokens+reg_tokens+N_patch, 2*D]
        updated_final_tokens = updated_final_tokens.squeeze(0) # [S, cam_tokens+reg_tokens+N_patch, 2*D]
        updated_final_tokens = updated_final_tokens[:,patch_start_idx:,self.embed_dim:] # [S, N_patch, D]
        patch_tokens[-1] = updated_final_tokens

        # 3. prediction heads
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]

            if self.depth_head is not None:
                depth = self.depth_head(patch_tokens, patch_h, patch_w)
                depth = F.relu(depth).squeeze(1)
                predictions["depth"] = depth
        
        predictions["images"] = images

        return predictions