from collections.abc import Callable

import equinox as eqx
from equinox import field, nn
from jax import nn as jnn
from jaxtyping import Array, Float

from .conv2d import Conv2dLayer


class SepConv2dLayer(eqx.Module, strict=True):
    # Input attributes
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    use_bias: bool = True
    act_layer: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.relu

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.body = nn.Sequential(
            [
                # Depthwise
                Conv2dLayer(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.in_channels,
                    use_bias=self.use_bias,
                ),
                # Activation
                nn.Lambda(self.act_layer),
                # Pointwise
                Conv2dLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    stride=1,
                    kernel_size=1,
                    padding=0,
                    use_bias=self.use_bias,
                ),
            ]
        )

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.body(x)
