from collections.abc import Sequence

from equinox import nn

from fba_net.keygen import KEYS
from fba_net.swap_channels_last_to_third_from_last import swap_channels_last_to_third_from_last

# Equinox doesn't support channels-last: https://github.com/patrick-kidger/equinox/issues/432
# As such we need to manually transpose the input and output of the convolutional layer.


def Conv2dLayer(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int | Sequence[int] = 3,
    stride: int | Sequence[int] = 1,
    # Padding defaults to half the kernel size, rounded down.
    padding: None | int | Sequence[int] = None,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    use_bias: bool = True,
) -> nn.Sequential:
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [k // 2 for k in kernel_size]
    return nn.Sequential(
        [
            # Transpose so that channels are third from the end.
            nn.Lambda(swap_channels_last_to_third_from_last),
            # Convolution
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                key=next(KEYS),
            ),
            # Swap back
            nn.Lambda(swap_channels_last_to_third_from_last),
        ]
    )
