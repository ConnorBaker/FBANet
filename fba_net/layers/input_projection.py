from collections.abc import Callable
from dataclasses import InitVar
from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from .conv2d import Conv2dLayer


class InputProjLayer(eqx.Module, strict=True):
    # Input attributes
    in_channels: int
    out_channels: int
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
                # Projection
                Conv2dLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),
                nn.Lambda(nn.PReLU() if activation is None else activation),
                # Rearrange
                nn.Lambda(partial(rearrange, pattern="height width channels-> (height width) channels")),
                # Normalization
                nn.Identity() if normalization is None else nn.Lambda(normalization),
            ]
        )

    def __call__(self, x: Float[Array, "height width channels"]) -> Float[Array, "height*width channels"]:
        return self.body(x)
