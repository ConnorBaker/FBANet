"""
## Towards Real-World Burst Image Super-Resolution: Benchmark and Method
## Code by YujingSun
"""


from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

import equinox as eqx
from equinox import field, nn
from jax import image as jim
from jax import numpy as jnp
from jaxtyping import Array, Float

from fba_net.blocks.fba_net import FBANetBlock
from fba_net.blocks.federated_affinity_fusion import FAFBlock
from fba_net.blocks.residual import ResBlock
from fba_net.blocks.upsampler import UpsamplerBlock
from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.downsample import DownsampleLayer
from fba_net.layers.input_projection import InputProjLayer
from fba_net.layers.output_projection import OutputProjLayer
from fba_net.layers.output_projection_hwc import OutputProjHWCLayer
from fba_net.layers.upsample import UpsampleLayer


class FBANetModel(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    num_frames: int = 14
    img_size: int = 128
    in_channels: int = 3
    embed_dim: int = 32
    depths: Sequence[int] = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    heads: Sequence[int] = [1, 2, 4, 8, 16, 16, 8, 4, 2]
    window_length: int = 8
    mlp_ratio: float = 4.0
    use_qkv_bias: bool = True
    qk_scale: None | float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    normalization: None | Callable[[Float[Array, "..."]], Float[Array, "..."]] = None
    token_projection: Literal["linear", "linear_concat", "conv"] = "linear"
    token_mlp: Literal["ffn", "leff"] = "ffn"
    use_se_layer: bool = False

    # Computed attributes
    pos_drop: nn.Dropout = field(init=False)
    input_proj: InputProjLayer = field(init=False)
    output_proj: OutputProjHWCLayer = field(init=False)
    output_proj_2: OutputProjLayer = field(init=False)
    output_proj_HG2_0: OutputProjHWCLayer = field(init=False)
    output_proj_HG2_1: OutputProjHWCLayer = field(init=False)
    fusion: FAFBlock = field(init=False)
    head: nn.Conv2d = field(init=False)
    body: nn.Sequential = field(init=False)
    tail: nn.Sequential = field(init=False)
    HG1_encoderlayer_0: FBANetBlock = field(init=False)
    HG1_downsample_0: DownsampleLayer = field(init=False)
    HG1_encoderlayer_1: FBANetBlock = field(init=False)
    HG1_downsample_1: DownsampleLayer = field(init=False)
    conv_HG1: FBANetBlock = field(init=False)
    HG1_upsample_0: UpsampleLayer = field(init=False)
    HG1_decoderlayer_0: FBANetBlock = field(init=False)
    HG1_upsample_1: UpsampleLayer = field(init=False)
    HG1_decoderlayer_1: FBANetBlock = field(init=False)
    HG2_encoderlayer_0: FBANetBlock = field(init=False)
    HG2_downsample_0: DownsampleLayer = field(init=False)
    HG2_encoderlayer_1: FBANetBlock = field(init=False)
    HG2_downsample_1: DownsampleLayer = field(init=False)
    conv_HG2: FBANetBlock = field(init=False)
    HG2_upsample_0: UpsampleLayer = field(init=False)
    HG2_decoderlayer_0: FBANetBlock = field(init=False)
    HG2_upsample_1: UpsampleLayer = field(init=False)
    HG2_decoderlayer_1: FBANetBlock = field(init=False)

    def __post_init__(self) -> None:
        self.pos_drop = nn.Dropout(self.drop_rate)
        # Input/Output
        self.input_proj = InputProjLayer(in_channels=self.in_channels, out_channels=self.embed_dim)
        self.output_proj = OutputProjHWCLayer(in_channels=2 * self.embed_dim, out_channels=self.embed_dim)
        self.output_proj_2 = OutputProjLayer(in_channels=2 * self.embed_dim, out_channels=self.embed_dim)
        self.output_proj_HG2_0 = OutputProjHWCLayer(in_channels=8 * self.embed_dim, out_channels=4 * self.embed_dim)
        self.output_proj_HG2_1 = OutputProjHWCLayer(in_channels=4 * self.embed_dim, out_channels=2 * self.embed_dim)
        self.fusion = FAFBlock(
            num_feats=self.embed_dim,
            num_frames=self.num_frames,
            center_frame_idx=0,
        )
        self.head = Conv2dLayer(in_channels=self.in_channels, out_channels=self.embed_dim)
        self.body = nn.Sequential([ResBlock(num_feats=self.embed_dim) for _ in range(2)])
        self.tail = nn.Sequential(
            [
                UpsamplerBlock(scale_pow_two=1, num_feats=self.embed_dim),
                Conv2dLayer(in_channels=self.embed_dim, out_channels=self.in_channels),
            ]
        )

        # stochastic depth for encoding, convolutional, and decoding layers
        enc_dpr: list[float] = jnp.linspace(
            start=0, stop=self.drop_path_rate, num=sum(self.depths[: len(self.depths) // 2])
        ).tolist()
        conv_dpr = [self.drop_path_rate] * self.depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers
        class FBANetKwargs(TypedDict):
            window_length: int
            mlp_ratio: float
            use_qkv_bias: bool
            qk_scale: None | float
            drop_rate: float
            attn_drop_rate: float
            normalization: None | Callable[[Float[Array, "..."]], Float[Array, "..."]]
            token_projection: Literal["linear", "linear_concat", "conv"]
            token_mlp: Literal["ffn", "leff"]
            use_se_layer: bool

        fba_net_kwargs: FBANetKwargs = FBANetKwargs(
            window_length=self.window_length,
            mlp_ratio=self.mlp_ratio,
            use_qkv_bias=self.use_qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            normalization=self.normalization,
            token_projection=self.token_projection,
            token_mlp=self.token_mlp,
            use_se_layer=self.use_se_layer,
        )

        # HG Block1
        # Encoder
        self.HG1_encoderlayer_0 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim,
            input_resolution=(self.img_size, self.img_size),
            depth=self.depths[0],
            heads=self.heads[0],
            drop_path_rate=enc_dpr[sum(self.depths[:0]) : sum(self.depths[:1])],
        )
        self.HG1_downsample_0 = DownsampleLayer(in_channels=self.embed_dim, out_channels=self.embed_dim * 2)
        self.HG1_encoderlayer_1 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 2,
            input_resolution=(self.img_size // 2, self.img_size // 2),
            depth=self.depths[1],
            heads=self.heads[1],
            drop_path_rate=enc_dpr[sum(self.depths[:1]) : sum(self.depths[:2])],
        )

        self.HG1_downsample_1 = DownsampleLayer(in_channels=self.embed_dim * 2, out_channels=self.embed_dim * 4)

        # Bottleneck
        self.conv_HG1 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 4,
            input_resolution=(self.img_size // (2**2), self.img_size // (2**2)),
            depth=self.depths[4],
            heads=self.heads[4],
            drop_path_rate=conv_dpr,
        )

        # Decoder
        self.HG1_upsample_0 = UpsampleLayer(in_channels=self.embed_dim * 4, out_channels=self.embed_dim * 2)
        self.HG1_decoderlayer_0 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 4,
            input_resolution=(self.img_size // 2, self.img_size // 2),
            depth=self.depths[5],
            heads=self.heads[5],
            drop_path_rate=dec_dpr[: self.depths[5]],
        )
        self.HG1_upsample_1 = UpsampleLayer(in_channels=self.embed_dim * 4, out_channels=self.embed_dim)
        self.HG1_decoderlayer_1 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 2,
            input_resolution=(self.img_size, self.img_size),
            depth=self.depths[6],
            heads=self.heads[6],
            drop_path_rate=dec_dpr[sum(self.depths[5:6]) : sum(self.depths[5:7])],
        )

        # HG Block2
        # Encoder
        self.HG2_encoderlayer_0 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim,
            input_resolution=(self.img_size, self.img_size),
            depth=self.depths[0],
            heads=self.heads[0],
            drop_path_rate=enc_dpr[sum(self.depths[:0]) : sum(self.depths[:1])],
        )
        self.HG2_downsample_0 = DownsampleLayer(in_channels=self.embed_dim, out_channels=self.embed_dim * 2)
        self.HG2_encoderlayer_1 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 2,
            input_resolution=(self.img_size // 2, self.img_size // 2),
            depth=self.depths[1],
            heads=self.heads[1],
            drop_path_rate=enc_dpr[sum(self.depths[:1]) : sum(self.depths[:2])],
        )
        self.HG2_downsample_1 = DownsampleLayer(in_channels=self.embed_dim * 2, out_channels=self.embed_dim * 4)

        # Bottleneck
        self.conv_HG2 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 4,
            input_resolution=(self.img_size // (2**2), self.img_size // (2**2)),
            depth=self.depths[4],
            heads=self.heads[4],
            drop_path_rate=conv_dpr,
        )

        # Decoder
        self.HG2_upsample_0 = UpsampleLayer(in_channels=self.embed_dim * 4, out_channels=self.embed_dim * 2)
        self.HG2_decoderlayer_0 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 4,
            input_resolution=(self.img_size // 2, self.img_size // 2),
            depth=self.depths[5],
            heads=self.heads[5],
            drop_path_rate=dec_dpr[: self.depths[5]],
        )
        self.HG2_upsample_1 = UpsampleLayer(in_channels=self.embed_dim * 4, out_channels=self.embed_dim)
        self.HG2_decoderlayer_1 = FBANetBlock(
            **fba_net_kwargs,
            dim=self.embed_dim * 2,
            input_resolution=(self.img_size, self.img_size),
            depth=self.depths[6],
            heads=self.heads[6],
            drop_path_rate=dec_dpr[sum(self.depths[5:6]) : sum(self.depths[5:7])],
        )

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x: Float[Array, "F H W C"]) -> Float[Array, "H W C"]:
        # Input Multi-Frame Conv
        b, t, c, h, w = x.shape
        assert c == 3, "In channels should be 3!"

        x_base = x[:, 0, :, :, :]  # [b, c, h, w]

        # feature extraction of aligned frames
        x_feat_head = self.head(x.view(-1, c, h, w))  # [b*t, embed_dim, h, w]
        x_feat_body = self.body(x_feat_head)  # [b*t, embed_dim, h, w]

        feat = x_feat_body.view(b, t, -1, h, w)  # [b, t, embed_dim, h, w]

        # fusion of aligned features
        fusion_feat = self.fusion(feat)  # fusion feat [b, embed_dim, h, w]

        assert fusion_feat.ndim == 4, "Fusion Feat should be [B,C,H,W]!"

        # Input Projection
        y = self.input_proj(fusion_feat)  # B, H*W, C
        y = self.pos_drop(y)

        # HG1
        # Encoder
        conv0 = self.HG1_encoderlayer_0(y)
        pool0 = self.HG1_downsample_0(conv0)

        conv1 = self.HG1_encoderlayer_1(pool0)
        pool1 = self.HG1_downsample_1(conv1)

        # Bottleneck
        conv2 = self.conv_HG1(pool1)

        # Decoder
        up0 = self.HG1_upsample_0(conv2)
        deconv0 = jnp.concatenate([up0, conv1], axis=-1)
        deconv0 = self.HG1_decoderlayer_0(deconv0)

        up1 = self.HG1_upsample_1(deconv0)
        deconv1 = jnp.concatenate([up1, conv0], axis=-1)
        deconv1 = self.HG1_decoderlayer_1(deconv1)

        # Output Projection
        y_1 = self.output_proj(deconv1)

        # HG2
        # Encoder
        conv0_2 = self.HG2_encoderlayer_0(y_1)
        pool0_2 = self.HG2_downsample_0(conv0_2)

        conv1_2 = self.HG2_encoderlayer_1(pool0_2)
        pool1_2 = self.HG2_downsample_1(conv1_2)

        # Bottleneck
        conv2_2 = self.conv_HG2(pool1_2)

        # Decoder
        up0_2 = self.HG2_upsample_0(conv2_2)
        deconv0_2 = self.output_proj_HG2_0(jnp.concatenate([up0, conv1, up0_2, conv1_2], axis=-1))  # B, H/2*W/2, C*8
        deconv0_2 = self.HG2_decoderlayer_0(deconv0_2)

        up1_2 = self.HG2_upsample_1(deconv0_2)
        deconv1_2 = self.output_proj_HG2_1(jnp.concatenate([up1, conv0, up1_2, conv0_2], axis=-1))
        deconv1_2 = self.HG2_decoderlayer_1(deconv1_2)

        # Output Projection
        y_2 = self.output_proj_2(deconv1_2)

        output_2 = self.tail(y_2)

        base = jim.resize(x_base, shape=[dim * 4 for dim in x_base.shape], method="bilinear")
        # base = F.interpolate(x_base, scale_factor=4, mode="bilinear", align_corners=False)

        out = output_2 + base

        return out
