from dataclasses import InitVar
from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jax import random as jrandom
from jaxtyping import Array, Float

from .conv2d import Conv2dLayer


class DownsampleLayer(eqx.Module, strict=True, frozen=True, kw_only=True):
    # Input attributes
    in_channels: int
    out_channels: int
    key: InitVar[jrandom.KeyArray]

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        object.__setattr__(
            self,
            "body",
            nn.Sequential(
                [
                    # Reshape
                    nn.Lambda(partial(rearrange, pattern="(length length) channels -> length length channels")),
                    # Convolution
                    Conv2dLayer(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        key=key,
                    ),
                    # Reshape
                    nn.Lambda(partial(rearrange, pattern="length length channels -> (length length) channels")),
                ]
            ),
        )

    def __call__(self, x: Float[Array, "4*length*length channels"]) -> Float[Array, "length*length channels"]:
        return self.body(x)

    def flops(self, H: int, W: int) -> float:
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channels * self.out_channels * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops
