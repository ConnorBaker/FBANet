from collections.abc import Sequence

from equinox import nn
from jax import random as jrandom


def Conv2dLayer(
    *,
    in_channels: int,
    out_channels: int,
    key: jrandom.KeyArray,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = (1, 1),
    padding: int | Sequence[int] | Sequence[tuple[int, int]] = (0, 0),
    dilation: int | Sequence[int] = (1, 1),
    groups: int = 1,
    use_bias: bool = True,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        key=key,
    )
