from equinox import nn
from jax import random as jrandom


def UpsampleFlattenLayer(in_channels: int, out_channels: int, key: jrandom.KeyArray) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2,
        key=key,
    )
