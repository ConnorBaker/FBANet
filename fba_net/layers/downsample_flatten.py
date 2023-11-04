from equinox import nn

from .conv2d import Conv2dLayer


def DownsampleFlattenLayer(in_channels: int, out_channels: int) -> nn.Sequential:
    return Conv2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )
