from collections.abc import Callable
from dataclasses import InitVar
from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from .conv2d import Conv2dLayer


# TODO: Is this literally the same as OutputProjLayer, just with the height and width dimensions combined?
class OutputProjHWCLayer(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    in_channels: int = 64
    out_channels: int = 3
    kernel_size: int = 3
    stride: int = 1
    normalization: InitVar[None | Callable[[Float[Array, "..."]], Float[Array, "..."]]] = None
    activation: InitVar[None | Callable[[Float[Array, "..."]], Float[Array, "..."]]] = None

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(
        self,
        normalization: None | Callable[[Float[Array, "..."]], Float[Array, "..."]],
        activation: None | Callable[[Float[Array, "..."]], Float[Array, "..."]],
    ) -> None:
        self.body = nn.Sequential(
            [
                # Rearrange
                nn.Lambda(partial(rearrange, pattern="(length length) channels -> length length channels")),
                # Projection
                Conv2dLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.kernel_size // 2,
                ),
                nn.PReLU() if activation is None else activation,
                # Normalization
                nn.Identity() if normalization is None else normalization,
                # Rearrange
                nn.Lambda(partial(rearrange, pattern="length length channels -> (length length) channels")),
            ]
        )

    # TODO: Need to change callsites to make sure channels are last
    def __call__(self, x: Float[Array, "length*length channels"]) -> Float[Array, "length*length channels"]:
        return self.body(x)
