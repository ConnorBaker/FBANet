from equinox import nn

from fba_net.keygen import KEYS


def ConvTranspose2dLayer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        key=next(KEYS),
    )
