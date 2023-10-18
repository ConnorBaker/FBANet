from collections.abc import Callable
from dataclasses import InitVar
from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jax import nn as jnn
from jax import random as jrandom
from jaxtyping import Array, Float

from .conv2d import Conv2dLayer


class LeFFLayer(eqx.Module, strict=True, frozen=True, kw_only=True):
    # Input attributes
    key: InitVar[jrandom.KeyArray]
    dim: int = 32
    hidden_dim: int = 128
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        key1, key2, key3 = jrandom.split(key, 3)
        object.__setattr__(
            self,
            "body",
            nn.Sequential(
                [
                    # Linear layer 1
                    nn.Linear(self.dim, self.hidden_dim, key=key1),
                    nn.Lambda(self.activation),
                    # Reshape
                    nn.Lambda(
                        partial(
                            rearrange,
                            pattern="(dim dim) hidden_dim -> dim dim hidden_dim",
                            dim=self.dim,
                            hidden_dim=self.hidden_dim,
                        )
                    ),
                    # Depthwise convolution
                    Conv2dLayer(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        groups=self.hidden_dim,
                        padding=(1, 1),
                        key=key2,
                    ),
                    nn.Lambda(self.activation),
                    # Reshape
                    nn.Lambda(
                        partial(
                            rearrange,
                            pattern="dim dim hidden_dim -> (dim dim) hidden_dim",
                            dim=self.dim,
                            hidden_dim=self.hidden_dim,
                        )
                    ),
                    # Linear layer 2
                    nn.Linear(self.hidden_dim, self.dim, key=key3),
                ]
            ),
        )

    def __call__(self, x: Float[Array, "n*n c"]) -> Float[Array, "n*n c"]:
        return self.body(x)

    def flops(self, H: int, W: int) -> float:
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops
