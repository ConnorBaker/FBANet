from typing import final

import equinox as eqx
from equinox import field, nn
from jax import nn as jnn
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer


@final
class ResBlock(eqx.Module, strict=True):
    """
    Residual block with two convolutional layers sandwiching a ReLU, and a skip connection.
    """

    num_feats: int
    kernel_size: int = 3
    body: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.body = nn.Sequential(
            [
                Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats),
                nn.Lambda(jnn.relu),
                Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats),
            ]
        )

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.body(x) + x
