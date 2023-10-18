from dataclasses import InitVar

import equinox as eqx
import jax
from einops import repeat
from equinox import field, nn
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.downsample_flatten import DownsampleFlattenLayer
from fba_net.layers.upsample_flatten import UpsampleFlattenLayer

from .residual import ResBlock


class FAFBlock(eqx.Module, strict=True, frozen=True, kw_only=True):
    # Input attributes
    key: InitVar[jrandom.KeyArray]
    num_feats: int = 64
    num_frames: int = 14
    center_frame_idx: int = 0

    # Computed attributes
    temporal_attn1: nn.Conv2d = field(init=False)
    temporal_attn2: nn.Conv2d = field(init=False)
    feat_fusion: nn.Conv2d = field(init=False)
    downsample1: nn.Conv2d = field(init=False)
    downsample2: nn.Conv2d = field(init=False)
    upsample1: nn.ConvTranspose2d = field(init=False)
    upsample2: nn.ConvTranspose2d = field(init=False)
    res_block1: nn.Sequential = field(init=False)
    res_block2: nn.Sequential = field(init=False)
    res_block3: nn.Sequential = field(init=False)
    res_block4: nn.Sequential = field(init=False)
    res_block5: nn.Sequential = field(init=False)
    fusion_tail: nn.Conv2d = field(init=False)
    lrelu: nn.PReLU = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        """
        # Compuate the attention map, highlight distinctions while keep similarities

        Input: Aligned frames, [T, C, H, W]
        Output: Fused frame, [C, H, W]
        """
        keys = list(jrandom.split(key, 18))

        object.__setattr__(
            self,
            "temporal_attn1",
            Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, padding=1, key=keys.pop()),
        )
        object.__setattr__(
            self,
            "temporal_attn2",
            Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, padding=1, key=keys.pop()),
        )
        object.__setattr__(
            self,
            "feat_fusion",
            Conv2dLayer(
                in_channels=self.num_feats * self.num_frames, out_channels=self.num_feats, kernel_size=1, key=keys.pop()
            ),
        )

        # spatial attention
        object.__setattr__(
            self,
            "downsample1",
            DownsampleFlattenLayer(in_channels=self.num_feats, out_channels=self.num_feats * 2, key=keys.pop()),
        )
        object.__setattr__(
            self,
            "downsample2",
            DownsampleFlattenLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats * 4, key=keys.pop()),
        )

        object.__setattr__(
            self,
            "upsample1",
            UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats * 2, key=keys.pop()),
        )
        object.__setattr__(
            self,
            "upsample2",
            UpsampleFlattenLayer(in_channels=self.num_feats * 4, out_channels=self.num_feats, key=keys.pop()),
        )

        # Residual blocks
        for res_block_idx, feat_multiplier in enumerate([1, 2, 4, 4, 2], start=1):
            object.__setattr__(
                self,
                f"res_block{res_block_idx}",
                nn.Sequential([ResBlock(num_feats=self.num_feats * feat_multiplier, key=keys.pop()) for _ in range(2)]),
            )

        object.__setattr__(
            self,
            "fusion_tail",
            Conv2dLayer(in_channels=self.num_feats * 2, out_channels=self.num_feats, key=keys.pop()),
        )
        object.__setattr__(self, "lrelu", nn.PReLU(init_alpha=0.1))

        assert len(keys) == 0, "All keys should be used"

    def forward(
        self, aligned_feat: Float[Array, "num_frames height width channels"]
    ) -> Float[Array, "height width channels"]:
        num_frames, height, width, channels = aligned_feat.shape

        # attention map, highlight distinctions while keep similarities
        type Embedding = Float[Array, f"{height} {width} {channels}"]
        type Embeddings = Float[Array, f"{num_frames} {height} {width} {channels}"]
        embedding_ref: Embedding = self.temporal_attn1(aligned_feat[self.center_frame_idx])
        embeddings: Embeddings = self.temporal_attn2(aligned_feat)

        type Correlation = Float[Array, f"{height} {width} 1"]
        corr_l: list[Correlation] = [
            (embedding - embedding_ref).sum(axis=-1, keepdims=True) for embedding in embeddings
        ]
        corr_diff: list[Correlation] = [jnp.abs(corr - corr_l[0]) for corr in corr_l[1:]]

        # compute the attention map
        type AlignedFeats = Embeddings
        type AlignedOtherFeats = Float[Array, f"{num_frames - 1} {height} {width} {channels}"]
        corr_prob_repeat: AlignedOtherFeats = repeat(
            jax.nn.sigmoid(jnp.concatenate(corr_diff)),
            "num_frames height width -> num_frames height width channels",
            height=height,
            width=width,
            num_frames=num_frames - 1,
            channels=channels,
        )

        aligned_feat_guided: AlignedFeats = jnp.concatenate([aligned_feat[0:1], aligned_feat[1:] * corr_prob_repeat])

        # fuse the feat under the guidance of computed attention map
        type Feat = Float[Array, f"{height} {width} {channels}"]
        feat: Feat = self.lrelu(self.feat_fusion(aligned_feat_guided))

        # Hourglass for spatial attention
        feat_res1 = self.res_block1(feat)
        down_feat1 = self.downsample1(feat_res1)
        feat_res2 = self.res_block2(down_feat1)
        down_feat2 = self.downsample2(feat_res2)

        feat3 = self.res_block3(down_feat2)

        up_feat3 = self.upsample1(feat3)
        concat_2_1 = jnp.concatenate([up_feat3, feat_res2], axis=1)
        feat_res4 = self.res_block4(concat_2_1)
        up_feat4 = self.upsample2(feat_res4)
        concat_1_0 = jnp.concatenate([up_feat4, feat_res1], axis=1)
        feat_res5 = self.res_block5(concat_1_0)

        feat_out = self.fusion_tail(feat_res5) + feat

        return feat_out
