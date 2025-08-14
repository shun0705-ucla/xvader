import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from typing import Tuple, List

from third_party.vggt.vggt.layers.block import Block
from third_party.vggt.vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class AlternatingAttention(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        patch_size=14,
    ):
        super().__init__()

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
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

        self.global_blocks = nn.ModuleList(
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
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.patch_size = patch_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        
        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and the other for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        self.use_reentrant = False

    def forward(self, patch_tokens: torch.Tensor, patch_h:int, patch_w:int, device: torch.device) -> List[torch.Tensor]:
        """
        Args:
            patch_tokens (torch.Tensor): shape (S, P, C) # from 4th layer
        Returns:
            final_tokens_concat (torch.Tensor): shape (S, P_A, 2*C) # for camera_head
            patch_tokens (torch.Tensor): shape (S, P, C) # for 4th layer
        """
        assert patch_tokens.dim() == 3, "Expected input shape (S, P, C)"
        S, P, C = patch_tokens.shape
        
        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, 1, S)
        register_token = slice_expand_and_flatten(self.register_token, 1, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(S, patch_h, patch_w, device=device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(S, self.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # update P because we added special tokens
        _, P_A, C = tokens.shape # Patch + Additional tokens (camera and register)

        frame_idx = 0
        global_idx = 0
        tokens_frame = []
        tokens_global = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, tokens_frame = self._process_frame_attention(
                        tokens, S, P_A, C, frame_idx, pos=pos
                    )
                elif attn_type =="global":
                    tokens, global_idx, tokens_global = self._process_global_attention(
                        tokens, S, P_A, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
        final_tokens_concat = torch.cat([tokens_frame, tokens_global], dim=-1) # (S, P_A, 2C)
        camera_tokens = final_tokens_concat[:,0,:] # (S, 2C)
        patch_tokens = tokens[:, self.patch_start_idx:, :] # (S, P, C)

        return camera_tokens, patch_tokens

    def _process_frame_attention(self, tokens, S, N, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (S, N, C):
            tokens = tokens.view(S, N, C)

        if pos is not None and pos.shape != (S, N, 2):
            pos = pos.view(S, N, 2)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1

        tokens_frame = tokens.view(S, N, C)
        return tokens, frame_idx, tokens_frame

    def _process_global_attention(self, tokens, S, N, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (1, S * N, C):
            tokens = tokens.view(S, N, C).view(1, S * N, C)

        if pos is not None and pos.shape != (1, S * N, 2):
            pos = pos.view(1, S * N, 2)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1


        tokens = tokens.view(S, N, C)
        tokens_global = tokens.view(S, N, C)
        return tokens, global_idx, tokens_global
    
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined