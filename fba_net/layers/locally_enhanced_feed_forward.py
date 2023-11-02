from collections.abc import Callable
from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jax import nn as jnn
from jaxtyping import Array, Float

from fba_net.keygen import KEYS

from .conv2d import Conv2dLayer


class LeFFLayer(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    dim: int = 32
    hidden_dim: int = 128
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.body = nn.Sequential(
            [
                # Linear layer 1
                nn.Linear(self.dim, self.hidden_dim, key=next(KEYS)),
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
                nn.Linear(self.hidden_dim, self.dim, key=next(KEYS)),
            ]
        )

    def __call__(self, x: Float[Array, "n*n c"]) -> Float[Array, "n*n c"]:
        return self.body(x)
