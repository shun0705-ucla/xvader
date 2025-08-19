import torch

def flat_to_map(x: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """
    Convert flattened patch tokens to a spatial map.

    Args:
        x: Tensor of shape (B, H*W, C)
        patch_h: Number of patches along height
        patch_w: Number of patches along width

    Returns:
        Tensor of shape (B, C, patch_h, patch_w)
    """
    B, HW, C = x.shape
    assert HW == patch_h * patch_w, f"num patch mismatch {patch_h * patch_w}, {HW}"
    return x.view(B, patch_h, patch_w, C).permute(0, 3, 1, 2).contiguous()


def map_to_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert spatial map back to flattened patch tokens.

    Args:
        x: Tensor of shape (B, C, H, W)

    Returns:
        Tensor of shape (B, H*W, C)
    """
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

