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


def window_partition(
    x: Float[Array, "height*window_length width*window_length dim"],
    window_length: int,
) -> Float[Array, "height*width window_length*window_length dim"]:
    return rearrange(
        x,
        "(height wl1) (width wl2) dim -> (height width) wl1 wl2 dim",
        wl1=window_length,
        wl2=window_length,
    )


def window_reverse(
    windows: Float[Array, "height*width window_length*window_length dim"],
    height: int,
    width: int,
    window_length: int,
) -> Float[Array, "height*window_length width*window_length dim"]:
    return rearrange(
        windows,
        "(height width) window_length window_length dim -> (height window_length) (width window_length) dim",
        height=height,
        width=width,
        window_length=window_length,
    )


class FBANetLayer(eqx.Module, strict=True):
    # Input attributes
    dim: int
    input_resolution: tuple[int, int]
    heads: int
    window_length: int = 8
    shift_size: int = 0
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
    norm1: nn.LayerNorm = field(init=False)
    attn: WindowAttentionLayer = field(init=False)
    drop_path: nn.Identity | nn.Dropout = field(init=False)
    norm2: nn.LayerNorm = field(init=False)
    mlp: nn.MLP | LeFFLayer = field(init=False)

    def __post_init__(self) -> None:
        if min(self.input_resolution) <= self.window_length:
            self.shift_size = 0
            self.window_length = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_length, "shift_size must in 0-window_length"

        if self.normalization is None:
            self.norm1 = nn.LayerNorm(self.dim)
            self.norm2 = nn.LayerNorm(self.dim)
        else:
            self.norm1 = self.normalization  # type: ignore
            self.norm2 = self.normalization  # type: ignore

        self.attn = WindowAttentionLayer(
            dim=self.dim,
            window_length=self.window_length,
            heads=self.heads,
            use_qkv_bias=self.use_qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop_rate=self.attn_drop_rate,
            proj_drop_rate=self.drop_rate,
            token_projection=self.token_projection,
            use_se_layer=self.use_se_layer,
        )

        if self.drop_path_rate <= 0.0:  # noqa: PLR2004
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

    def __call__(self, x: Float[Array, "length*length dim"]) -> Float[Array, "length*length dim"]:
        height, width = self.input_resolution
        length_sq, dim = x.shape
        assert length_sq == height * width, "input feature has wrong size"
        assert dim == self.dim, "input feature has wrong dimension"

        # attn_mask is not None when self.shift_size > 0.
        attn_mask: None | Float[Array, "..."] = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = jnp.zeros((height, width, 1))
            h_slices = (
                slice(0, -self.window_length),
                slice(-self.window_length, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_length),
                slice(-self.window_length, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    indices_h, indices_w = jnp.ogrid[h, w]
                    cnt_mask = jnp.full((len(indices_h), len(indices_w), 1), cnt)
                    mask_update = jnp.where(jnp.zeros_like(shift_mask[h, w, :]) == 0, cnt_mask, 0)
                    shift_mask += mask_update  # This is not in-place due to immutability of jnp arrays
                    cnt += 1
            shift_attn_mask = rearrange(
                window_partition(shift_mask, self.window_length),
                "num_windows window_length window_length 1 -> num_windows (window_length window_length)",
                window_length=self.window_length,
            )
            shift_attn_mask = jnp.expand_dims(shift_attn_mask, 1) - jnp.expand_dims(shift_attn_mask, 2)
            shift_attn_mask = jax.lax.select(
                shift_attn_mask != 0,
                jnp.full(shift_attn_mask.shape, float(-100.0)),
                jnp.full(shift_attn_mask.shape, float(0.0)),
            )
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
            # TODO: Error here because I don't know what the shape should be.
            assert_shape((height, width, dim), attn_mask)

        skip_connection = x
        x = jax.vmap(self.norm1)(x)
        x = rearrange(
            x,
            "(height width) dim -> height width dim",
            height=height,
            width=width,
        )
        assert_shape((height, width, dim), x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(0, 1))
        else:
            shifted_x = x

        # partition windows
        x_windows = rearrange(
            window_partition(shifted_x, window_length=self.window_length),  # nW, window_length, window_length, C
            "num_windows wl1 wl2 dim -> num_windows (wl1 wl2) dim",
            wl1=self.window_length,
            wl2=self.window_length,
            dim=dim,
        )

        # W-MSA/SW-MSA
        attn_windows = jax.vmap(partial(self.attn, mask=attn_mask))(x_windows)  # nW, window_length*window_length, C
        assert_shape((None, self.window_length * self.window_length, dim), attn_windows)

        # merge windows
        attn_windows = rearrange(
            attn_windows,
            "num_windows (wl1 wl2) dim -> num_windows wl1 wl2 dim",
            wl1=self.window_length,
            wl2=self.window_length,
            dim=dim,
        )

        shifted_x = window_reverse(
            attn_windows, height=height, width=width, window_length=self.window_length
        )  # H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(0, 1))
        else:
            x = shifted_x

        x = rearrange(x, "height width dim -> (height width) dim")

        # FFN
        x = skip_connection + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
