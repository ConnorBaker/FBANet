from collections.abc import Callable
from functools import partial

from einops import rearrange
from jaxtyping import Array, Float


# NOTE: PixelShuffle and UnPixelShuffle are taken from https://github.com/google/flax/discussions/2899#discussioncomment-5088214.
def PixelShuffleLayer(scale: int) -> Callable[[Float[Array, "h w h2*w2"]], Float[Array, "h*h2 w*w2"]]:
    return partial(rearrange, pattern="h w (h2 w2) -> (h h2) (w w2)", h2=scale, w2=scale)


def UnPixelShuffleLayer(scale: int) -> Callable[[Float[Array, "h*h2 w*w2"]], Float[Array, "h w h2*w2"]]:
    return partial(rearrange, pattern="(h h2) (w w2) -> h w (h2 w2)", h2=scale, w2=scale)
