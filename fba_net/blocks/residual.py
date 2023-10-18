from dataclasses import InitVar

import equinox as eqx
from equinox import field, nn
from jax import nn as jnn
from jax import random as jrandom
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer


class ResBlock(eqx.Module, strict=True, frozen=True, kw_only=True):
    """
    Residual block with two convolutional layers sandwiching a ReLU, and a skip connection.
    """

    num_feats: int
    key: InitVar[jrandom.KeyArray]
    kernel_size: int = 3
    body: nn.Sequential = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        key1, key2 = jrandom.split(key)
        object.__setattr__(
            self,
            "body",
            nn.Sequential(
                [
                    Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, key=key1),
                    nn.Lambda(jnn.relu),
                    Conv2dLayer(in_channels=self.num_feats, out_channels=self.num_feats, key=key2),
                ]
            ),
        )

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.body(x) + x
