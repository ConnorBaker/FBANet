from collections.abc import Sequence

import equinox as eqx
import jax
from einops import repeat
from equinox import field, nn
from jax import numpy as jnp
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.downsample_flatten import DownsampleFlattenLayer
from fba_net.layers.upsample_flatten import UpsampleFlattenLayer

from .residual import ResBlock


class FAFBlock(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    num_feats: int = 64
    num_frames: int = 14
    center_frame_idx: int = 0

    # Computed attributes
    temporal_attn0: nn.Conv2d = field(init=False)
    temporal_attn1: nn.Conv2d = field(init=False)
    feature_fusion: nn.Sequential = field(init=False)
    downsample0: nn.Conv2d = field(init=False)
    downsample1: nn.Conv2d = field(init=False)
    upsample0: nn.ConvTranspose2d = field(init=False)
    upsample1: nn.ConvTranspose2d = field(init=False)
    res_blocks: Sequence[nn.Sequential] = field(init=False)
    fusion_tail: nn.Conv2d = field(init=False)
    lrelu: nn.PReLU = field(init=False)

    def __post_init__(self) -> None:
        # spatial attention
        self.temporal_attn0 = Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, padding=1)
        self.temporal_attn1 = Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, padding=1)

        # feature fusion
        self.feature_fusion: nn.Sequential = nn.Sequential(
            [
                Conv2dLayer(
                    in_channels=self.num_feats * self.num_frames,
                    out_channels=self.num_feats,
                    kernel_size=1,
                ),
                nn.PReLU(init_alpha=0.1),
            ]
        )

        # Top of hour glass, which shrinks
        self.downsample0 = DownsampleFlattenLayer(in_channels=self.num_feats, out_channels=self.num_feats * 2)
        self.downsample1 = DownsampleFlattenLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats * 4)

        # Bottom of hour glass, which expands
        self.upsample0 = UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats * 2)
        self.upsample1 = UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats)

        # Residual blocks used throughout the hour glass
        self.res_blocks: Sequence[nn.Sequential] = [
            nn.Sequential([ResBlock(num_feats=self.num_feats * feat_multiplier) for _ in range(2)])
            for feat_multiplier in (1, 2, 4, 4, 2)
        ]

        # Tail of the hour glass
        self.fusion_tail = Conv2dLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats)

    def compute_guided_aligned_features(self, aligned_frames: Float[Array, "F H W C"]) -> Float[Array, "F H W C"]:
        """
        Compute the attention map, highlight distinctions while keep similarities, using them to
        align the features in the aligned frames.

        Input: Aligned frames, [F, H, W, C]
        Output: Aligned features, [F, H, W, C]
        """
        frames, height, width, channels = aligned_frames.shape
        assert frames == self.num_frames, f"Expected {self.num_frames} frames, got {frames}"

        # attention map, highlight distinctions while keep similarities
        embedding_ref: Float[Array, "H W C"] = self.temporal_attn0(aligned_frames[self.center_frame_idx])
        embeddings: Float[Array, "F H W C"] = self.temporal_attn1(aligned_frames)

        affinity_map: list[Float[Array, "H W 1"]] = [
            (embedding - embedding_ref).sum(axis=-1, keepdims=True) for embedding in embeddings
        ]
        affinity_map_diffs: list[Float[Array, "H W 1"]] = [jnp.abs(corr - affinity_map[0]) for corr in affinity_map[1:]]

        # compute the attention map
        guide_weights: Float[Array, "F-1 H W C"] = repeat(
            jax.nn.sigmoid(jnp.concatenate(affinity_map_diffs)),
            "f h w -> f h w c",
            h=height,
            w=width,
            f=frames - 1,
            c=channels,
        )
        guided_aligned_features: Float[Array, "F H W C"] = jnp.concatenate(
            [aligned_frames[0:1], aligned_frames[1:] * guide_weights]
        )

        return guided_aligned_features

    def fuse_features(self, guided_aligned_features: Float[Array, "F H W C"]) -> Float[Array, "H W C"]:
        """
        Fuse the aligned features into a single frame.

        Input: Aligned features, [F, H, W, C]
        Output: Fused frame, [H, W, C]
        """
        # fuse the feat under the guidance of computed attention map
        feat = self.feature_fusion(guided_aligned_features)

        # Hourglass for spatial attention
        feat_res: dict[int, Float[Array, "..."]] = {}

        # Top of hour glass, which shrinks
        feat_res[0] = self.res_blocks[0](feat)
        feat_res[1] = self.res_blocks[1](self.downsample0(feat_res[0]))
        feat_res[2] = self.res_blocks[2](self.downsample1(feat_res[1]))

        # Bottom of hour glass, which expands
        feat_res[3] = self.res_blocks[3](jnp.concatenate([self.upsample0(feat_res[2]), feat_res[1]], axis=1))
        feat_res[4] = self.res_blocks[4](jnp.concatenate([self.upsample1(feat_res[3]), feat_res[0]], axis=1))

        # Skip connection
        feat_out = self.fusion_tail(feat_res[4]) + feat

        return feat_out

    def __call__(self, aligned_frames: Float[Array, "F H W C"]) -> Float[Array, "H W C"]:
        """
        Compute the attention map, highlight distinctions while keep similarities

        Input: Aligned frames, [F, H, W, C]
        Output: Fused frame, [H, W, C]
        """
        guided_aligned_features: Float[Array, "F H W C"] = self.compute_guided_aligned_features(aligned_frames)
        frame: Float[Array, "H W C"] = self.fuse_features(guided_aligned_features)
        return frame
