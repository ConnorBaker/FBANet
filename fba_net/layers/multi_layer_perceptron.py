from collections.abc import Callable

from equinox import nn
from jax import nn as jnn
from jax import random as jrandom
from jaxtyping import Array, Float


def MLPLayer(
    *,
    in_size: int,
    out_size: None | int = None,
    width_size: int,
    depth: int = 2,
    drop_rate: None | float = None,
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu,
    key: jrandom.KeyArray,
) -> nn.MLP:
    if out_size is None:
        out_size = in_size
    activation = nn.Sequential([nn.Lambda(activation)] + ([] if drop_rate is None else [nn.Dropout(drop_rate)]))
    return nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation=activation,
        key=key,
    )
