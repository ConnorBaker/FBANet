from equinox import nn

from fba_net.keygen import KEYS
from fba_net.swap_channels_last_to_third_from_last import swap_channels_last_to_third_from_last

# Equinox doesn't support channels-last: https://github.com/patrick-kidger/equinox/issues/432
# As such we need to manually transpose the input and output of the convolutional layer.


def ConvTranspose2dLayer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 2,
    stride: int = 2,
) -> nn.Sequential:
    return nn.Sequential([
        # Transpose so that channels are third from the end.
        nn.Lambda(swap_channels_last_to_third_from_last),
        # Convolution
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            key=next(KEYS),
        ),
        # Swap back
        nn.Lambda(swap_channels_last_to_third_from_last),
    ])
