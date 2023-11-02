from collections.abc import Callable

import equinox as eqx
from equinox import nn
from jax import nn as jnn
from jaxtyping import Array, Float

from fba_net.keygen import KEYS


def MLPLayer(
    *,
    in_size: int,
    out_size: None | int = None,
    width_size: int,
    depth: int = 2,
    drop_rate: None | float = None,
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu,
) -> nn.MLP:
    if out_size is None:
        out_size = in_size

    activation_layers: list[eqx.Module] = [nn.Lambda(activation)]
    if drop_rate is not None:
        activation_layers.append(nn.Dropout(drop_rate))

    return nn.MLP(
        in_size=in_size,
        out_size=in_size,
        width_size=width_size,
        depth=depth,
        activation=nn.Sequential(activation_layers),  # type: ignore
        key=next(KEYS),
    )
