from collections.abc import Callable
from functools import partial
from typing import Literal

import equinox as eqx
import jax
from einops import rearrange
from equinox import field, nn
from jax import nn as jnn
from jax import numpy as jnp
from jaxtyping import Array, Float

from fba_net.assert_shape import assert_shape
from fba_net.layers.drop_path import DropPath
from fba_net.layers.locally_enhanced_feed_forward import LeFFLayer
from fba_net.layers.multi_layer_perceptron import MLPLayer
from fba_net.layers.window_attention import WindowAttentionLayer


class FBANetLayer(eqx.Module, strict=True):
    # Input attributes
    dim: int
    input_resolution: tuple[int, int]
    heads: int
    window_height: int = 8
    window_width: int = 8
    shift_size_height: int = 0
    shift_size_width: int = 0
    mlp_ratio: float = 4.0
    use_qkv_bias: bool = True
    qk_scale: None | float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu
    normalization: None | Callable[[Float[Array, "..."]], Float[Array, "..."]] = None
    token_projection: Literal["linear", "linear_concat", "conv"] = "linear"
    token_mlp: Literal["ffn", "leff"] = "leff"
    use_se_layer: bool = False

    # Computed attributes
    num_height_windows: int = field(init=False)
    num_width_windows: int = field(init=False)
    norm1: nn.LayerNorm = field(init=False)
    attn: WindowAttentionLayer = field(init=False)
    drop_path: nn.Identity | nn.Dropout = field(init=False)
    norm2: nn.LayerNorm = field(init=False)
    mlp: nn.MLP | LeFFLayer = field(init=False)

    def __post_init__(self) -> None:
        assert self.dim % self.heads == 0, "dim must be divisible by number of heads"
        assert self.window_height == self.window_width, "window height and width must be equal (for now)"
        assert self.shift_size_height == self.shift_size_width, "shift size height and width must be equal (for now)"

        # Safeguard against images which are smaller in some dimension than the window size.
        H, W = self.input_resolution
        if H <= self.window_height:
            self.shift_size_height = 0
            self.window_height = H
        assert 0 <= self.shift_size_height < self.window_height, "shift_size_height must in [0,window_height)"

        if W <= self.window_width:
            self.shift_size_width = 0
            self.window_width = W
        assert 0 <= self.shift_size_width < self.window_width, "shift_size_width must in [0,window_width)"

        assert (
            H % self.window_height == 0
        ), f"input resolution height ({H}) is not divisible by window length ({self.window_height})"
        assert (
            W % self.window_width == 0
        ), f"input resolution width ({W}) is not divisible by window length ({self.window_width})"

        self.num_height_windows = H // self.window_height
        self.num_width_windows = W // self.window_width

        if self.normalization is None:
            self.norm1 = nn.LayerNorm(self.dim)
            self.norm2 = nn.LayerNorm(self.dim)
        else:
            self.norm1 = self.normalization  # type: ignore
            self.norm2 = self.normalization  # type: ignore

        # TODO: Add support for different window lengths in height and width.
        self.attn = WindowAttentionLayer(
            dim=self.dim,
            window_length=self.window_height,
            heads=self.heads,
            use_qkv_bias=self.use_qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
            token_projection=self.token_projection,
            use_se_layer=self.use_se_layer,
        )

        if self.drop_path_rate <= 0.0:
            self.drop_path = nn.Identity()
        else:
            self.drop_path = DropPath(p=self.drop_path_rate)  # type: ignore

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        if self.token_mlp == "ffn":
            self.mlp = MLPLayer(
                in_size=self.dim,
                width_size=mlp_hidden_dim,
                drop_rate=self.drop_rate,
                activation=self.activation,
            )
        else:
            self.mlp = LeFFLayer(dim=self.dim, hidden_dim=mlp_hidden_dim, activation=self.activation)

    def window_partition(
        self,
        x: Float[Array, "num_height_windows*window_height num_width_windows*window_width dim"],
    ) -> Float[Array, "num_height_windows*num_width_windows window_height*window_width dim"]:
        return rearrange(
            x,
            "(nhw wh) (nww ww) d -> (nhw nww) wh ww d",
            nhw=self.num_height_windows,
            nww=self.num_width_windows,
            wh=self.window_height,
            ww=self.window_width,
        )

    def window_reverse(
        self,
        windows: Float[Array, "num_height_windows*num_width_windows window_height*window_width dim"],
    ) -> Float[Array, "num_height_windows*window_height num_width_windows*window_width dim"]:
        return rearrange(
            windows,
            "(nhw nww) wh ww d -> (nhw wh) (nww ww) d",
            nhw=self.num_height_windows,
            nww=self.num_width_windows,
            wh=self.window_height,
            ww=self.window_width,
        )

    def __call__(self, x: Float[Array, "height*width dim"]) -> Float[Array, "height*width dim"]:
        height, width = self.input_resolution
        length_sq, dim = x.shape
        assert length_sq == height * width, "input feature has wrong size"
        assert dim == self.dim, "input feature has wrong dimension"

        # attn_mask is not None when self.shift_size > 0.
        attn_mask: None | Float[Array, "..."] = None

        # TODO: Add support for different window lengths in height and width.
        if self.shift_size_height > 0 or self.shift_size_width > 0:
            # calculate attention mask for SW-MSA
            shift_mask = jnp.zeros((height, width))
            h_slices = (
                slice(0, -self.window_height),
                slice(-self.window_height, -self.shift_size_height),
                slice(-self.shift_size_height, None),
            )
            w_slices = (
                slice(0, -self.window_width),
                slice(-self.window_width, -self.shift_size_width),
                slice(-self.shift_size_width, None),
            )
            cnt: int = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask = shift_mask.at[h, w].set(cnt)
                    cnt += 1

            # Add a dimension to the mask so that it can be broadcasted to the correct shape.
            shift_mask = rearrange(shift_mask, "h w -> h w ()")
            shift_attn_mask = rearrange(
                self.window_partition(shift_mask),
                "nw wh ww () -> nw (wh ww)",
            )

            # Add a dimension so we can subtract the mask from itself.
            shift_attn_mask = rearrange(
                shift_attn_mask,
                "nw wh_ww -> nw () wh_ww",
            ) - rearrange(
                shift_attn_mask,
                "nw wh_ww -> nw wh_ww ()",
            )
            shift_attn_mask = jnp.where(shift_attn_mask != 0, -100.0, 0.0)
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
            # TODO: Error here because I don't know what the shape should be.
            assert_shape(
                (
                    self.num_height_windows * self.num_width_windows,
                    self.window_height * self.window_width,
                    self.window_height * self.window_width,
                ),
                attn_mask,
            )

        skip_connection = x
        x = jax.vmap(self.norm1)(x)
        x = rearrange(
            x,
            "(h w) d -> h w d",
            h=height,
            w=width,
            d=dim,
        )

        # cyclic shift
        if self.shift_size_height > 0 or self.shift_size_width > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size_height, -self.shift_size_width), axis=(0, 1))
        else:
            shifted_x = x

        # partition windows
        x_windows = rearrange(
            self.window_partition(shifted_x),  # nW, window_height, window_width, dim
            "num_windows wh ww d -> num_windows (wh ww) d",
            num_windows=self.num_height_windows * self.num_width_windows,
            wh=self.window_height,
            ww=self.window_width,
            d=dim,
        )

        # W-MSA/SW-MSA
        attn_windows = jax.vmap(partial(self.attn, mask=attn_mask))(x_windows)  # nW, window_height*window_width, dim
        assert_shape((None, self.window_height * self.window_width, dim), attn_windows)

        # merge windows
        attn_windows = rearrange(
            attn_windows,
            "num_windows (wh ww) d -> num_windows wh ww d",
            wh=self.window_height,
            ww=self.window_width,
            d=dim,
        )

        shifted_x = self.window_reverse(attn_windows)  # height, width, dim

        # reverse cyclic shift
        if self.shift_size_height > 0 or self.shift_size_width > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size_height, self.shift_size_width), axis=(0, 1))
        else:
            x = shifted_x

        x = rearrange(x, "height width dim -> (height width) dim")

        # FFN
        x = skip_connection + self.drop_path(x)
        x = jax.vmap(self.norm2)(x)
        x = jax.vmap(self.mlp)(x)
        x = x + self.drop_path(x)
        assert_shape((length_sq, dim), x)
        return x
