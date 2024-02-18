from collections.abc import Sequence

import equinox as eqx
import jax
from einops import rearrange, reduce, repeat
from equinox import field, nn
from jax import numpy as jnp
from jaxtyping import Array, Float

from fba_net.assert_shape import assert_shape
from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.downsample_flatten import DownsampleFlattenLayer
from fba_net.layers.upsample_flatten import UpsampleFlattenLayer

from .residual import ResBlock


class FAFBlock(eqx.Module, strict=True):
    # Input attributes
    num_feats: int = 64
    num_frames: int = 14

    # Computed attributes
    temporal_attn0: nn.Sequential = field(init=False)
    temporal_attn1: nn.Sequential = field(init=False)
    feature_fusion: nn.Sequential = field(init=False)
    downsample0: nn.Sequential = field(init=False)
    downsample1: nn.Sequential = field(init=False)
    upsample0: nn.Sequential = field(init=False)
    upsample1: nn.Sequential = field(init=False)
    res_blocks: Sequence[nn.Sequential] = field(init=False)
    fusion_tail: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        # spatial attention
        self.temporal_attn0 = Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats)
        self.temporal_attn1 = Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats)

        # feature fusion, applied to features from each frame
        self.feature_fusion: nn.Sequential = nn.Sequential([
            Conv2dLayer(
                in_channels=self.num_feats * self.num_frames,
                out_channels=self.num_feats,
                padding=0,
                kernel_size=1,
            ),
            nn.Lambda(nn.PReLU(init_alpha=0.1)),
        ])

        # Top of hour glass, which shrinks
        self.downsample0 = DownsampleFlattenLayer(in_channels=self.num_feats, out_channels=self.num_feats * 2)
        self.downsample1 = DownsampleFlattenLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats * 4)

        # Bottom of hour glass, which expands
        self.upsample0 = UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats * 2)
        self.upsample1 = UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats)

        # Residual blocks used throughout the hour glass
        self.res_blocks: Sequence[nn.Sequential] = [
            nn.Sequential([nn.Lambda(ResBlock(num_feats=self.num_feats * feat_multiplier)) for _ in range(2)])
            for feat_multiplier in (1, 2, 4, 4, 2)
        ]

        # Tail of the hour glass
        self.fusion_tail = Conv2dLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats)

    def compute_guided_aligned_features(self, aligned_features: Float[Array, "F H W NF"]) -> Float[Array, "F H W NF"]:
        """
        Compute the attention map, highlight distinctions while keep similarities, using them to
        align the features in the aligned frames.

        Input: Aligned features, [F, H, W, NF], where NF is `num_feats`.
        Output: Guided and aligned features, [F, H, W, NF], where NF is `num_feats`.
        """
        _, height, width, _ = aligned_features.shape
        assert_shape((self.num_frames, height, width, self.num_feats), aligned_features)

        # attention map, highlight distinctions while keep similarities
        embedding_ref: Float[Array, "H W NF"] = self.temporal_attn0(aligned_features[0])
        assert_shape((height, width, self.num_feats), embedding_ref)
        embeddings: Float[Array, "F H W NF"] = jax.vmap(self.temporal_attn1)(aligned_features)
        assert_shape((self.num_frames, height, width, self.num_feats), embeddings)

        affinity_map: Float[Array, "F H W"] = reduce(
            embeddings - embedding_ref,
            "f h w nf -> f h w",
            reduction="sum",
        )
        assert_shape((self.num_frames, height, width), affinity_map)

        affinity_map_diffs: Float[Array, "F-1 H W"] = jnp.abs(affinity_map[1:] - affinity_map[0])
        assert_shape((self.num_frames - 1, height, width), affinity_map_diffs)

        # compute the attention map
        guide_weights: Float[Array, "F-1 H W NF"] = repeat(
            jax.nn.sigmoid(affinity_map_diffs),
            "f h w -> f h w nf",
            nf=self.num_feats,
        )
        assert_shape((self.num_frames - 1, height, width, self.num_feats), guide_weights)

        guided_aligned_features: Float[Array, "F H W NF"] = jnp.concatenate([
            aligned_features[0:1],
            aligned_features[1:] * guide_weights,
        ])
        assert_shape((self.num_frames, height, width, self.num_feats), guided_aligned_features)

        return guided_aligned_features

    def fuse_features(self, guided_aligned_features: Float[Array, "F H W C"]) -> Float[Array, "H W C"]:
        """
        Fuse the aligned features into a single frame.

        Input: Aligned features, [F, H, W, C]
        Output: Fused frame, [H, W, C]
        """
        _, height, width, _ = guided_aligned_features.shape
        assert_shape((self.num_frames, height, width, self.num_feats), guided_aligned_features)

        # fuse the feat under the guidance of computed attention map
        feat = self.feature_fusion(
            # Make sure to gropu the frames and features together in the last dimesnion
            # to be able to apply the convolutional layer defined in `feature_fusion`.
            rearrange(
                guided_aligned_features,
                "f h w nf -> h w (f nf)",
            )
        )
        assert_shape((height, width, self.num_feats), feat)

        # Hourglass for spatial attention
        feat_res: dict[int, Float[Array, "..."]] = {}

        # Top of hour glass, which shrinks
        feat_res[0] = self.res_blocks[0](feat)
        assert_shape((height, width, self.num_feats), feat_res[0])
        feat_res[1] = self.res_blocks[1](self.downsample0(feat_res[0]))
        assert_shape((height // 2, width // 2, self.num_feats * 2), feat_res[1])
        feat_res[2] = self.res_blocks[2](self.downsample1(feat_res[1]))
        assert_shape((height // 4, width // 4, self.num_feats * 4), feat_res[2])

        # Bottom of hour glass, which expands
        feat_res[3] = self.res_blocks[3](
            rearrange(
                # Performs concatenation along the last dimension.
                [self.upsample0(feat_res[2]), feat_res[1]],
                "two h w nf -> h w (two nf)",
            )
        )
        assert_shape((height // 2, width // 2, self.num_feats * 4), feat_res[3])
        feat_res[4] = self.res_blocks[4](
            rearrange(
                # Performs concatenation along the last dimension.
                [self.upsample1(feat_res[3]), feat_res[0]],
                "two h w nf -> h w (two nf)",
            )
        )
        assert_shape((height, width, self.num_feats * 2), feat_res[4])

        # Skip connection
        feat_out = self.fusion_tail(feat_res[4]) + feat
        assert_shape((height, width, self.num_feats), feat_out)

        return feat_out

    def __call__(self, aligned_features: Float[Array, "F H W NF"]) -> Float[Array, "H W NF"]:
        """
        Compute the attention map, highlight distinctions while keep similarities

        Input: Aligned features, [F, H, W, NF], where NF is `num_feats`.
        Output: Fused frame, [H, W, NF], where NF is `num_feats`.
        """
        _, height, width, _ = aligned_features.shape
        assert_shape((self.num_frames, height, width, self.num_feats), aligned_features)

        guided_aligned_features: Float[Array, "F H W NF"] = self.compute_guided_aligned_features(aligned_features)
        assert_shape((self.num_frames, height, width, self.num_feats), guided_aligned_features)

        frame: Float[Array, "H W NF"] = self.fuse_features(guided_aligned_features)
        assert_shape((height, width, self.num_feats), frame)

        return frame
