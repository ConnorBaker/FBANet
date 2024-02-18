from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from .upsample_flatten import UpsampleFlattenLayer


class UpsampleLayer(eqx.Module, strict=True):
    # Input attributes
    in_channels: int
    out_channels: int

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.body = nn.Sequential([
            # Reshape
            nn.Lambda(partial(rearrange, pattern="(length length) channels -> length length channels")),
            # Convolution
            UpsampleFlattenLayer(in_channels=self.in_channels, out_channels=self.out_channels),
            # Reshape
            nn.Lambda(partial(rearrange, pattern="length length channels -> (length length) channels")),
        ])

    def __call__(self, x: Float[Array, "length*length channels"]) -> Float[Array, "4*length*length channels"]:
        return self.body(x)
