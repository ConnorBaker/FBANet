from collections.abc import Callable
from typing import Literal, overload

import equinox as eqx
import jax
from einops import rearrange, repeat
from equinox import field, nn
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float

from fba_net.assert_shape import assert_shape
from fba_net.keygen import KEYS

from .conv_projection import ConvProjectionLayer
from .linear_projection import LinearProjectionLayer
from .linear_projection_concat_kv import LinearProjectionConcatKVLayer
from .squeeze_and_excitation import SELayer


class WindowAttentionLayer(eqx.Module, strict=True):
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
        relative_position_bias_table = mean + std * jrandom.truncated_normal(
            key=next(KEYS),
            lower=-2,
            upper=2,
            shape=(
                (2 * window_length - 1) * (2 * window_length - 1),
                heads,
            ),
        )
        assert_shape(((2 * window_length - 1) ** 2, heads), relative_position_bias_table)
        return relative_position_bias_table

    @staticmethod
    def mk_relative_position_index(
        window_length: int,
    ) -> Float[Array, "window_length*window_length window_length*window_length"]:
        # Generate 2D coordinate grid
        coords = jnp.stack(jnp.mgrid[:window_length, :window_length])
        assert_shape((2, window_length, window_length), coords)

        # Compute pairwise relative coordinates
        relative_coords = rearrange(coords, "d i j -> d (i j) ()") - rearrange(coords, "d i j -> d () (i j)")
        assert_shape((2, window_length * window_length, window_length * window_length), relative_coords)

        # Processing step and Summing the x and y differences after preventing overlap between x and y axes
        relative_coords = rearrange(relative_coords, "d h w -> h w d")
        assert_shape((window_length * window_length, window_length * window_length, 2), relative_coords)
        relative_coords = relative_coords + (window_length - 1)
        assert_shape((window_length * window_length, window_length * window_length, 2), relative_coords)
        relative_coords = relative_coords * (2 * window_length - 1)
        assert_shape((window_length * window_length, window_length * window_length, 2), relative_coords)

        relative_position_index = jnp.sum(relative_coords, -1)
        assert_shape((window_length * window_length, window_length * window_length), relative_position_index)

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

    def __call__(self, x: Float[Array, "N D"], mask: None | Float[Array, "NW WH*WW N"] = None) -> Float[Array, "N D"]:
        """
        N: sequence length
        NW: number of windows
        WH: window height
        WW: window width
        D: dimension (`self.dim`)
        """
        H, (N, D) = self.heads, x.shape
        assert D == self.dim, f"Expected {self.dim} channels, got {D} channels"
        # TODO: This seems like a pre-condition -- it's the purpose of calculating dimension_expansion_factor.
        assert (
            N % self.window_length == 0
        ), f"Sequence length {N} is not divisible by window length {self.window_length}"
        q, k, v = self.qkv(x)
        q = q * self.scale
        assert_shape((H, N, D), q)
        assert_shape((H, N, D), k)
        assert_shape((H, N, D), v)

        # Multiply by the transpose of k
        # Matrix multiplication rules: (h, n, d) @ (h, d, n) -> (h, n, n)
        attn: Float[Array, "H N N"] = q @ rearrange(k, "h n d -> h d n", h=H, n=N, d=D)
        assert_shape((H, N, N), attn)

        # Use rearrange to do the reshaping and permuting
        # TODO: In this code, the product of the window height and width is the sequence length.
        # Is it possible that one of these dimensions is actually the sequence length, and
        # that we don't have two dimensions that are both the product of the window height and width?
        relative_position_bias = rearrange(
            self.relative_position_bias_table[self.relative_position_index],
            "Wh_Ww1 Wh_Ww2 h -> h Wh_Ww1 Wh_Ww2",
            Wh_Ww1=self.window_length * self.window_length,
            Wh_Ww2=self.window_length * self.window_length,
            h=H,
        )

        dimension_expansion_factor: int = attn.shape[-1] // relative_position_bias.shape[-1]
        relative_position_bias = repeat(
            relative_position_bias,
            "h Wh_Ww1 Wh_Ww2 -> h Wh_Ww1 (Wh_Ww2 exp_factor)",
            h=H,
            Wh_Ww1=self.window_length * self.window_length,
            Wh_Ww2=self.window_length * self.window_length,
            exp_factor=dimension_expansion_factor,
        )
        assert_shape((H, N, N), relative_position_bias)

        # Add relative position bias
        # TODO: Look up shape broadcasting rules.
        # Do we need the product of window width and height to equal the sequence length?
        assert N == self.window_length * self.window_length, f"Expected N to be {self.window_length**2}, got {N}"
        attn = attn + relative_position_bias
        assert_shape((self.heads, N, N), attn)

        if mask is not None:
            assert False, "there is where you left off"
            # Add an extra dimension to mask for compatibility with attn dimensions
            mask_expanded = repeat(mask, "nw wh_ww n -> nw wh_ww (n exp_factor)", exp_factor=dimension_expansion_factor)
            # Reshape attn to separate out the heads and sequence length dimensions
            attn_reshaped = rearrange(attn, "(h n) len -> h n len", h=H, n=N)
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

        # Matrix multiplication with value matrix, and re-arrange dimensions to get the output.
        # TODO: I guess self.heads must always be 1 so we can squeeze that dimension away?
        x = rearrange(
            attn_dropped @ v,
            "h n dh -> n (h dh)",
            h=H,
            n=N,
            dh=self.dim_head,
        )
        assert_shape((N, D), x)
        # Pass through the projection layer
        x_projected = jax.vmap(self.proj)(x)
        # Apply Squeeze-and-Excitation if applicable
        x_se = self.se(x_projected)
        # Apply dropout to the projected output
        x_dropped = self.proj_drop(x_se)
        return x_dropped
