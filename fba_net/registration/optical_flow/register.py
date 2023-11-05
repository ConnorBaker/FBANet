from functools import partial

import jax
from einops import rearrange
from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Array, Float, Int32, Shaped


# A specialized version of scipy.ndimage.map_coordinates which is more amenable to currying
def warp_frame(
    coodinates: Float[Array, "2 height width"],
    frame: Shaped[Array, "height width"],
) -> Shaped[Array, "height width"]:
    return jsp.ndimage.map_coordinates(
        input=frame,
        coordinates=coodinates,  # type: ignore -- Array also satisfies Sequence
        order=1,
        mode="nearest",
    )


def register_frame(
    frame: Shaped[Array, "height width channels"],
    flow: Shaped[Array, "height width 2"],
) -> Shaped[Array, "height width channels"]:
    """
    Warps the frame according to the optical flow
    """
    # Move channels to the front
    frame = rearrange(frame, "height width channels -> channels height width")
    flow = rearrange(flow, "height width xy -> xy height width")

    # Compute the grid
    _channels, height, width = frame.shape
    grid: Int32[Array, "2 height width"] = jnp.mgrid[:height, :width]

    # Compute the warped grid
    warp_grid: Float[Array, "2 height width"] = grid - flow

    # Interpolate the frame by warping each channel
    warped_frame: Shaped[Array, "channels height width"] = jax.vmap(partial(warp_frame, warp_grid))(frame)

    # Move channels back to the end
    warped_frame = rearrange(warped_frame, "channels height width -> height width channels")

    return warped_frame
