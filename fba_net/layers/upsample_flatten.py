from equinox import nn

from .conv2d_transpose import ConvTranspose2dLayer


def UpsampleFlattenLayer(in_channels: int, out_channels: int) -> nn.Sequential:
    return ConvTranspose2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2,
    )
