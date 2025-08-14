from turtle import pos
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from typing import Tuple, List

from third_party.vggt.vggt.layers.block import Block
from third_party.vggt.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from .alternating_attention import slice_expand_and_flatten



class FeatureAttention(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        patch_size=14,
    ):
        super().__init__()

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )
        
        self.depth = depth
        self.patch_size = patch_size
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = num_register_tokens
        # Initialize parameters with small values
        nn.init.normal_(self.register_token, std=1e-6)

        self.use_reentrant = False

    def forward(self, patch_tokens: torch.Tensor, patch_h: int, patch_w: int, device: torch.device) -> List[torch.Tensor]:
        """
        Args:
            patch_tokens (torch.Tensor): shape (L, S, P, C)
        Returns:
            torch.Tensor: shape (L, S, P, C) after feature attention
        """
        L, S, P, C = patch_tokens.shape

        patch_tokens = patch_tokens.permute(1, 0, 2, 3).reshape(S * L, P, C)
        
        # Expand register tokens to match batch size and sequence length
        register_token = slice_expand_and_flatten(self.register_token, S, L)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(S * L, patch_h, patch_w, device=device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(S * L, self.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # update P because we added special tokens
        _, P_R, C = tokens.shape # Patch and Register tokens

        for i in range(self.depth):
            if tokens.shape != (S, L * P_R, C):
                tokens = tokens.view(S, L, P_R, C).view(S, L * P_R, C)
            if pos is not None and pos.shape != (S, L * P_R, 2):
                pos = pos.view(S, L, P_R, 2).view(S, L * P_R, 2)
            if self.training:
                tokens = checkpoint(self.blocks[i], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.blocks[i](tokens, pos=pos)

        # Remove register tokens and reshape back to (L, S, P, C)
        patch_tokens = tokens.view(S, L, P_R, C)[:, :, self.patch_start_idx:, :].permute(1, 0, 2, 3)  # (L, S, P, C)
        return patch_tokens