from dataclasses import InitVar
from itertools import chain
from typing import Literal

import equinox as eqx
from equinox import field, nn
from jax import random as jrandom
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.pixel_shuffle import PixelShuffleLayer


class UpsamplerBlock(eqx.Module, strict=True, frozen=True, kw_only=True):
    scale_pow_two: Literal[1, 2, 3, 4]
    num_feats: int
    key: InitVar[jrandom.KeyArray]
    body: nn.Sequential = field(init=False)

    # Only support powers of two for now.
    def __post_init__(self, key: jrandom.KeyArray) -> None:
        # Since we're upsampling by a factor of 2^n, we need to do n PixelShuffles,
        # as each PixelShuffle increases the resolution by 2.
        keys = list(jrandom.split(key, self.scale_pow_two))
        object.__setattr__(
            self,
            "body",
            nn.Sequential(
                list(
                    chain.from_iterable(
                        (
                            Conv2dLayer(
                                in_channels=self.num_feats,
                                out_channels=4 * self.num_feats,
                                key=keys.pop()
                            ),
                            nn.Lambda(PixelShuffleLayer(2)),
                        )
                        for _ in range(self.scale_pow_two)
                    )
                )
            ),
        )

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.body(x)
