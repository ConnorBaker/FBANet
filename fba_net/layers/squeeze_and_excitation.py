from dataclasses import InitVar
from functools import partial

import equinox as eqx
from einops import rearrange, reduce
from equinox import field, nn
from jax import nn as jnn
from jax import random as jrandom
from jaxtyping import Array, Float


class SELayer(eqx.Module, strict=True, frozen=True, kw_only=True):
    channels: int
    key: InitVar[jrandom.KeyArray]
    reduction: int = 16
    body: nn.Sequential = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        key1, key2 = jrandom.split(key)
        object.__setattr__(
            self,
            "body",
            nn.Sequential(
                [
                    # Squeeze step using average pooling
                    nn.Lambda(partial(reduce, pattern="h w c -> c", reduction="mean")),
                    # Excitation step
                    nn.Linear(self.channels, self.channels // self.reduction, use_bias=False, key=key1),
                    nn.Lambda(jnn.relu),
                    nn.Linear(self.channels // self.reduction, self.channels, use_bias=False, key=key2),
                    nn.Lambda(jnn.sigmoid),
                    nn.Lambda(partial(rearrange, pattern="c -> c () ()")),
                ]
            ),
        )

    def __call__(self, x: Float[Array, "h w c"]) -> Float[Array, "h w c"]:
        return self.body(x) * x
