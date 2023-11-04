from itertools import chain
from typing import Literal, final

import equinox as eqx
from equinox import field, nn
from jaxtyping import Array, Float

from fba_net.layers.conv2d import Conv2dLayer
from fba_net.layers.pixel_shuffle import PixelShuffleLayer


@final
class UpsamplerBlock(eqx.Module, strict=True):
    scale_pow_two: Literal[1, 2, 3, 4]
    num_feats: int
    body: nn.Sequential = field(init=False)

    # Only support powers of two for now.
    def __post_init__(self) -> None:
        # Since we're upsampling by a factor of 2^n, we need to do n PixelShuffles,
        # as each PixelShuffle increases the resolution by 2.
        self.body = nn.Sequential(
            list(
                chain.from_iterable(
                    (
                        Conv2dLayer(in_channels=self.num_feats, out_channels=4 * self.num_feats),
                        nn.Lambda(PixelShuffleLayer(2)),
                    )
                    for _ in range(self.scale_pow_two)
                )
            )
        )

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.body(x)
