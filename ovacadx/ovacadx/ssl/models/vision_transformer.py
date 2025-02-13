import math
from typing import Sequence

import torch.nn as nn
from monai.utils import ensure_tuple_rep
from monai.networks.nets import ViT as VisionTransformerMONAI
from monai.networks.blocks import PatchEmbeddingBlock as PatchEmbeddingBlockMONAI


class VisionTransformer(VisionTransformerMONAI):
    """
    A Vision Transformer model with an identity classification head and support for variable input sizes.
    """
    def __init__(
            self, 
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int, 
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            proj_type: str = "conv",
            pos_embed_type: str = "learnable",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            post_activation="Tanh",
            qkv_bias: bool = False,
            save_attn: bool = False,
        ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            classification=classification,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            post_activation=post_activation,
            qkv_bias=qkv_bias,
            save_attn=save_attn
        )
        self.embed_dim = hidden_size
        self.classification_head = nn.Identity()
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims
        )
    
    def forward(self, x):
        x, _ = super().forward(x)
        return x


class PatchEmbeddingBlock(PatchEmbeddingBlockMONAI):
    """
    A PatchEmbeddingBlock with support for variable input sizes.
    """
    def __init__(
            self, 
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int,
            hidden_size: int,
            num_heads: int,
            proj_type: str = "conv",
            pos_embed_type: str = "learnable",
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
        ):
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        super().__init__(
            in_channels=in_channels, 
            img_size=img_size, 
            patch_size=patch_size, 
            hidden_size=hidden_size, 
            num_heads=num_heads, 
            proj_type=proj_type, 
            pos_embed_type=pos_embed_type, 
            dropout_rate=dropout_rate, 
            spatial_dims=spatial_dims
        )
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.n_patches
    
        if npatch == N and w == h:
            return self.position_embeddings
        pos_embed = self.position_embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == pos_embed.shape[-2] and int(h0) == pos_embed.shape[-1]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed
    
    def forward(self, x):
        w, h = x.shape[-2:]
        x = self.patch_embeddings(x)

        if self.proj_type == "conv":
            x = x.flatten(2).transpose(-1, -2)

        embeddings = x + self.interpolate_pos_encoding(x, w, h)
        embeddings = self.dropout(embeddings)
        return embeddings


def vit_t_16(in_channels: int, img_size: Sequence[int] | int, **kwargs):
    return VisionTransformer(in_channels, img_size, patch_size=16, hidden_size=192, num_heads=3, **kwargs)


def vit_s_16(in_channels: int, img_size: Sequence[int] | int, **kwargs):
    return VisionTransformer(in_channels, img_size, patch_size=16, hidden_size=384, num_heads=6, **kwargs)


def vit_b_16(in_channels: int, img_size: Sequence[int] | int, **kwargs):
    return VisionTransformer(in_channels, img_size, patch_size=16, hidden_size=768, num_heads=12, **kwargs)


def vit_l_16(in_channels: int, img_size: Sequence[int] | int, **kwargs):
    return VisionTransformer(in_channels, img_size, patch_size=16, hidden_size=1024, num_heads=16, **kwargs)


def vit_h_16(in_channels: int, img_size: Sequence[int] | int, **kwargs):
    return VisionTransformer(in_channels, img_size, patch_size=16, hidden_size=1280, num_heads=16, **kwargs)
