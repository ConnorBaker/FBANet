from equinox import nn
from jax import random as jrandom

from .conv2d import Conv2dLayer


def DownsampleFlattenLayer(in_channels: int, out_channels: int, key: jrandom.KeyArray) -> nn.Conv2d:
    return Conv2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        key=key,
    )
