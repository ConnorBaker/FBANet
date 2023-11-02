from collections.abc import Callable
from typing import Literal, overload

import equinox as eqx
from einops import rearrange, repeat
from equinox import field, nn
from jax import lax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float

from fba_net.keygen import KEYS

from .conv_projection import ConvProjectionLayer
from .linear_projection import LinearProjectionLayer
from .linear_projection_concat_kv import LinearProjectionConcatKVLayer
from .squeeze_and_excitation import SELayer


class WindowAttentionLayer(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    dim: int
    window_length: int
    heads: int
    token_projection: Literal["linear", "linear_concat", "conv"] = "linear"
    use_qkv_bias: bool = True
    qk_scale: None | float = None
    attn_drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    use_se_layer: bool = False

    # Computed attributes
    dim_head: int = field(init=False)
    # qkv_scale, or if that is None/0.0, dim_head ** -0.5
    scale: float = field(init=False)
    relative_position_bias_table: Float[Array, "(2*window_height-1)*(2*window_width-1) heads"] = field(init=False)
    relative_position_index: Float[Array, "window_height*window_width window_height*window_width"] = field(init=False)
    qkv: LinearProjectionLayer | LinearProjectionConcatKVLayer | ConvProjectionLayer = field(init=False)
    attn_drop: nn.Dropout = field(init=False)
    proj: nn.Linear = field(init=False)
    proj_drop: nn.Dropout = field(init=False)
    se: nn.Identity | SELayer = field(init=False)
    softmax: Callable[[Float[Array, "..."]], Float[Array, "..."]] = field(init=False)

    @staticmethod
    def mk_relative_position_bias_table(
        window_length: int,
        heads: int,
        *,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> Float[Array, "(2*window_length-1)*(2*window_length-1) heads"]:
        type RetType = Float[Array, f"{2 * window_length - 1}*{2 * window_length - 1} heads"]
        relative_position_bias_table: RetType = mean + std * jrandom.truncated_normal(
            key=next(KEYS), lower=-2, upper=2, shape=((2 * window_length - 1) * (2 * window_length - 1), heads)
        )
        return relative_position_bias_table

    @staticmethod
    def mk_relative_position_index(
        window_length: int,
    ) -> Float[Array, "window_length*window_length window_length*window_length"]:
        # Generate 2D coordinate grid
        coords_h, coords_w = jnp.meshgrid(*map(jnp.arange, [window_length, window_length]))  # Wh, Ww

        # Compute pairwise relative coordinates and rearrange dimensions to get (Wh*Ww, Wh*Ww, 2)
        relative_coords = rearrange(coords_h[:, :, None] - coords_h[:, None, :], "h w d -> (h w) (h w) d") + rearrange(
            coords_w[:, :, None] - coords_w[:, None, :], "h w d -> (h w) (h w) d"
        ) * (2 * window_length - 1)

        # Sum over the last dimension to obtain relative position indices
        type RetType = Float[Array, f"{window_length}*{window_length} {window_length}*{window_length}"]
        relative_position_index: RetType = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        return relative_position_index

    @overload
    @staticmethod
    def mk_qkv(
        dim: int,
        heads: int,
        *,
        use_bias: bool = True,
        token_projection: Literal["linear"] = "linear",
    ) -> LinearProjectionLayer: ...

    @overload
    @staticmethod
    def mk_qkv(
        dim: int,
        heads: int,
        *,
        use_bias: bool = True,
        token_projection: Literal["linear_concat"] = "linear_concat",
    ) -> LinearProjectionConcatKVLayer: ...

    @overload
    @staticmethod
    def mk_qkv(
        dim: int,
        heads: int,
        *,
        use_bias: bool = True,
        token_projection: Literal["conv"] = "conv",
    ) -> ConvProjectionLayer: ...

    @staticmethod
    def mk_qkv(
        dim: int,
        heads: int,
        *,
        use_bias: bool = True,
        token_projection: Literal["linear", "linear_concat", "conv"] = "linear",
    ) -> LinearProjectionLayer | LinearProjectionConcatKVLayer | ConvProjectionLayer:
        match token_projection:
            case "linear":
                QkvLayer = LinearProjectionLayer
            case "linear_concat":
                QkvLayer = LinearProjectionConcatKVLayer
            case "conv":
                QkvLayer = ConvProjectionLayer

        return QkvLayer(dim=dim, heads=heads, use_bias=use_bias)

    def __post_init__(self) -> None:
        self.dim_head = self.dim // self.heads
        self.scale = self.qk_scale or self.dim_head**-0.5
        self.relative_position_bias_table = self.mk_relative_position_bias_table(
            self.window_length, self.heads, std=0.02
        )
        self.relative_position_index = self.mk_relative_position_index(self.window_length)
        self.qkv = self.mk_qkv(
            dim=self.dim,
            heads=self.heads,
            use_bias=self.use_qkv_bias,
            token_projection=self.token_projection,
        )
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(self.dim, self.dim, key=next(KEYS))
        self.proj_drop = nn.Dropout(self.proj_drop_rate)
        self.se = SELayer(channels=self.dim) if self.use_se_layer else nn.Identity()
        self.softmax = jnn.softmax

    def __call__(self, x: Float[Array, "n c"], mask: None | Float[Array, "m n"] = None) -> Float[Array, "n c"]:
        seq_length, channels = x.shape
        q, k, v = self.qkv(x)
        q = q * self.scale
        attn = lax.dot_general(q, k, (((-2,), (-1,)), ((), ())))  # equivalent to q @ k.transpose(-2, -1)

        # Use rearrange to do the reshaping and permuting
        relative_position_bias = rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            "(window_length window_length) heads -> heads window_length window_length",
            window_length=self.window_length,
            heads=self.heads,
        )

        dimension_expansion_factor: int = attn.shape[-1] // relative_position_bias.shape[-1]
        relative_position_bias = repeat(
            relative_position_bias,
            "heads window_length channels -> heads window_length (channels d)",
            heads=self.heads,
            window_length=self.window_length,
            channels=channels,
            d=dimension_expansion_factor,
        )

        # Add relative position bias, indexing with None to add an extra dimension
        attn = attn + relative_position_bias[None]

        if mask is not None:
            # Add an extra dimension to mask for compatibility with attn dimensions
            mask_expanded = repeat(mask, "num_windows seq_length -> num_windows seq_length ()")
            # Reshape attn to separate out the heads and sequence length dimensions
            attn_reshaped = rearrange(
                attn, "(heads seq_length) len -> heads seq_length len", heads=self.heads, seq_length=seq_length
            )
            # Expand dimensions of mask to align with attn dimensions
            attn_mask_aligned = attn_reshaped + rearrange(
                mask_expanded, "num_windows seq_length () -> num_windows () seq_length ()"
            )
            # Flatten the first two dimensions back to original shape for further processing
            attn = rearrange(attn_mask_aligned, "heads seq_length len -> (heads seq_length) len")

        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn)
        # Apply dropout to attention weights
        attn_dropped = self.attn_drop(attn_weights)

        # Matrix multiplication with value matrix, and re-arrange dimensions to get the output
        x = rearrange(
            attn_dropped @ v,
            "heads seq_length (heads dim_head) -> seq_length (heads dim_head)",
            heads=self.heads,
            dim_head=self.dim_head,
        )
        # Pass through the projection layer
        x_projected = self.proj(x)
        # Apply Squeeze-and-Excitation if applicable
        x_se = self.se(x_projected)
        # Apply dropout to the projected output
        x_dropped = self.proj_drop(x_se)
        return x_dropped
