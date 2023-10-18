import operator
from dataclasses import InitVar
from functools import reduce

import equinox as eqx
from einops import rearrange
from equinox import field
from jax import random as jrandom
from jaxtyping import Array, Float

from .separable_conv2d import SepConv2dLayer


class ConvProjectionLayer(eqx.Module, strict=True, frozen=True, kw_only=True):
    # Input attributes
    dim: int
    key: InitVar[jrandom.KeyArray]
    heads: int = 8
    kernel_size: int = 3
    q_stride: int = 1
    k_stride: int = 1
    v_stride: int = 1
    dropout: float = 0.0
    last_stage: bool = False
    use_bias: bool = True

    # Computed attributes
    dim_head: int = field(init=False)
    inner_dim: int = field(init=False)
    padding: int = field(init=False)
    to_q: SepConv2dLayer = field(init=False)
    to_k: SepConv2dLayer = field(init=False)
    to_v: SepConv2dLayer = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray):
        object.__setattr__(self, "dim_head", self.dim // self.heads)
        object.__setattr__(self, "inner_dim", self.dim_head * self.heads)
        object.__setattr__(self, "padding", (self.kernel_size - self.q_stride) // 2)

        def mk_sep_conv2d_layer(key: jrandom.KeyArray, stride: int) -> SepConv2dLayer:
            return SepConv2dLayer(
                in_channels=self.dim,
                out_channels=self.inner_dim,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
                use_bias=self.use_bias,
                key=key,
            )

        for name, layer in zip(
            ["to_q", "to_k", "to_v"],
            map(mk_sep_conv2d_layer, jrandom.split(key, 3), [self.q_stride, self.k_stride, self.v_stride]),
        ):
            object.__setattr__(self, name, layer)

    # Originally took `attn_kv` as an argument, but it's not used anywhere in the codebase
    # so for the sake of simplicity, I removed it.
    def __call__(
        self, x: Float[Array, "n*n c"]
    ) -> tuple[Float[Array, "h n*n d"], Float[Array, "h n*n d"], Float[Array, "h n*n d"]]:
        x = rearrange(x, "(n n) c -> c n n")

        type RetType = Float[Array, f"{self.heads} n*n d"]
        q: RetType = rearrange(self.to_q(x), "(h d) n n -> h (n n) d", h=self.heads)
        k: RetType = rearrange(self.to_k(x), "(h d) n n -> h (n n) d", h=self.heads)
        v: RetType = rearrange(self.to_v(x), "(h d) n n -> h (n n) d", h=self.heads)

        return q, k, v

    def flops(self, H: int, W: int) -> float:
        return reduce(operator.add, (layer.flops(H, W) for layer in (self.to_q, self.to_k, self.to_v)), 0.0)
