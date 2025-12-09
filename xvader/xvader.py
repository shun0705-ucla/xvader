import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.layers.vision_transformer import vit_small, vit_base, vit_large
from .metric_depth_head import MetricDepthHead
from .camera_embed import CameraEmbedding
from .camera_model import Pinhole

from .xvader_utils import flat_to_map, map_to_flat


from .camera_head import CameraHead
from .alternating_attention import AlternatingAttention

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class Xvader(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        use_bn=False, 
        use_clstoken=False,
    ):
        super(Xvader, self).__init__()

        '''
        self.intermediate_layer_idx_list = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],          
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.intermediate_layer_idx = self.intermediate_layer_idx_list[encoder]

        if encoder not in self.intermediate_layer_idx_list:
            raise ValueError(f"Unknown encoder '{encoder}'. Supported: {list(self.intermediate_layer_idx_list.keys())}")
        '''
        # patch embed
        self.encoder = vit_large(img_size=518,
                                 patch_size=14,
                                 num_register_tokens=4,
                                 interpolate_antialias=True,
                                 interpolate_offset=0.0,
                                 block_chunks=0,
                                 init_values=1.0)
        self.embed_dim = 1024
        self.patch_size = self.encoder.patch_size


        # adapter for camera embed
        self.hidden_dim = self.embed_dim // 2
        self.dimension_adapter = nn.ModuleList([])
        for i in range(4):
            self.dimension_adapter.append(nn.Linear(self.embed_dim, self.hidden_dim))
        
        self.dimension_adapter_camera = nn.Linear(self.embed_dim, self.hidden_dim)

        # camera embed
        self.camera_embed = CameraEmbedding(hidden_dim = self.embed_dim,
                                            num_layers = 4)

        # alternating attention
        self.alternating_attention = AlternatingAttention(embed_dim=self.embed_dim,
                                                          depth=24,
                                                          num_heads=self.embed_dim//64,
                                                          patch_size=self.patch_size)
        
        self.camera_head = CameraHead(dim_in=self.hidden_dim, pose_encoding_type = "absT_quaR")
        self.depth_head = MetricDepthHead(hidden_dim=self.hidden_dim)
        
        
        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

    
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor):
        
        # 0. Process images
        if len(images.shape) == 4: # (S, C, H, W)
            images = images.unsqueeze(0) # (1, S, C, H, W)
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        patch_h, patch_w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size

        # 1. patch & embed by DINOv2   
        # Normalize images and reshape for patch embed
        #images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)

        # patch_tokens_list
        patch_tokens_list = list(self.encoder.get_intermediate_layers(images, {5, 11, 17, 23}))
        #for i in range(len(patch_tokens_list)):
        #    patch_tokens_list[i] = patch_tokens_list[i].view(B, S, *patch_tokens_list[i].shape[-2:]) #(B, S, P, D)

        # dimension_adapter
        #for i, dimension_adapter in enumerate(self.dimension_adapter):
        #    patch_tokens_list[i] = dimension_adapter(patch_tokens_list[i]) # 1024 -> 512
        #    patch_tokens_list[i] = patch_tokens_list[i].view(B*S, patch_h*patch_w, self.hidden_dim)
        

        # 0. Process intrinsics
        if len(intrinsics.shape) == 3: # (S, 3, 3)
            intrinsics = intrinsics.unsqueeze(0) # (1, S, 3, 3)
        B_intri, S_intri, _, _ = intrinsics.shape
        assert (B, S)==(B_intri, S_intri), "Batch or Sequence size mismatch"
        intrinsics = intrinsics.view(B_intri*S_intri, *intrinsics.shape[-2:])

        # 2. Embed camera intrinsic
        camera = Pinhole(K=intrinsics)
        rays = camera.get_rays(shapes=(B*S, H, W)) # (B*S, 3, H, W)
        rays_embedding = F.interpolate(rays, size=(patch_h, patch_w), mode="bilinear", align_corners=False, antialias=True)
        rays_embedding = map_to_flat(rays_embedding)
        conditioned_tokens_list = self.camera_embed(patch_tokens_list, rays_embedding, patch_h, patch_w)    

        # 3. Update last layer by alternating attention (from VGGT)
        for i in range(len(conditioned_tokens_list)):
            conditioned_tokens_list[i] = conditioned_tokens_list[i].view(B, S, *conditioned_tokens_list[i].shape[-2:])
        camera_tokens, conditioned_tokens_list[-1] = self.alternating_attention(conditioned_tokens_list[-1], patch_h, patch_w, images.device)


        # 4. prediction heads
        predictions = {}

        if self.camera_head is not None:
            camera_tokens_red = self.dimension_adapter_camera(camera_tokens)
            pose_enc_list = self.camera_head(camera_tokens_red)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
            
            # pose_encoding to extrinsics
            #T = pose_encoding[..., :3]
            #quat = pose_encoding[..., 3:7]
            #R = quat_to_mat(quat)
            #extrinsics = torch.cat([R, T[..., None]], dim=-1)
            #predictions["extrinsic"] = extrinsics

        if self.depth_head is not None:
            conditioned_tokens_list_red = []
            for i, dimension_adapter in enumerate(self.dimension_adapter):
                conditioned_tokens_list_red.append(dimension_adapter(conditioned_tokens_list[i]))
            self.depth_head.set_input_feature_shapes(patch_h, patch_w)
            self.depth_head.set_original_shapes(H,W)
            depth_radius, confidence = self.depth_head(conditioned_tokens_list_red)
            # project depth_radius to depth_map
            local_points = rays * depth_radius # (B*S, 3, H, W) * (B*S, 1, H, W) -> (B*S, 3, H, W)
            local_points = local_points.view(B, S, 3, H, W)
            predictions["depth"] = local_points[:, :, -1:].permute(0, 1, 3, 4, 2) # extract z-axis (B, S, 1, H, W) -> (B, S, H, W, 1)
            predictions["local_points"] = local_points

            confidence = confidence.view(B,S,1,H,W)
            predictions["depth_conf"] = confidence.permute(0, 1, 3, 4, 2).squeeze(4) # (B, S, 1, H, W) -> (B, S, H, W)


        
        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
    
    @torch.no_grad()
    def infer_depth(self, images, intrinsics, resolution=518):
        
        def _scale_intrinsics(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
            """Scale pinhole intrinsics for a resize (sx = W'/W, sy = H'/H)."""
            K = K.clone()
            K[:, 0, 0] *= sx  # fx
            K[:, 1, 1] *= sy  # fy
            K[:, 0, 2] *= sx  # cx
            K[:, 1, 2] *= sy  # cy
            return K
        
        def _resize_back(x):
            assert x.ndim == 5 # input should be (B, S, C, Ht, Wt)
            
            B,S,Cx,Hx,Wx = x.shape
            x = x.view(B*S,Cx,Hx,Wx)
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            BS,Cx,H_resize,W_resize = x.shape
            x = x.view(B,S,Cx,H_resize,W_resize)
            return x
            
        assert len(images.shape) == 4 #(B,3,H,W)
        assert len(intrinsics.shape) == 3 #(B,3,3)
        assert (intrinsics.shape[-1] == 3 and intrinsics.shape[-2] == 3), "camera shape is not (..., 3, 3)"

        S, _, H, W = images.shape

        Ht, Wt = resolution, resolution
        sx, sy = Wt / W, Ht / H

        images_resized = F.interpolate(images, size=(resolution,resolution), mode="bilinear", align_corners=False)
        K_resized = _scale_intrinsics(intrinsics, sx=sx, sy=sy)

        outputs = self.forward(images_resized, K_resized)
        # depth: (B,S,H,W,1) -> resize -> (B,S,1,H,W)
        depth = _resize_back(outputs["depth"].permute(0,1,4,2,3))

        depth_conf = outputs["depth_conf"].unsqueeze(2)      # (B,S,1,Ht,Wt)
        depth_conf = _resize_back(depth_conf)                # (B,S,1,H,W)
        depth_conf = depth_conf.squeeze(2)                   # (B,S,H,W)

        return depth, depth_conf
    




