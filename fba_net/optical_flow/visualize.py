from itertools import starmap
from operator import ge
from typing import Annotated

from beartype.vale import Is
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int32, Scalar, UInt8

NonNegativeInt = Annotated[int, Is[lambda n: ge(n, 0)]]  # pyright: ignore[reportUnknownLambdaType]
UInt8RGB = tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]


def make_optical_flow_color_wheel_segment(
    start_color: UInt8RGB, end_color: UInt8RGB, length: NonNegativeInt
) -> Float[Array, "length 3"]:
    """
    Generates a smooth color gradient for a given color segment
    """
    return jnp.stack([jnp.linspace(start, stop, length) for start, stop in zip(start_color, end_color)], axis=-1)


def make_optical_flow_color_wheel() -> Float[Array, "55 3"]:
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    # Make the color wheel segments: red to yellow, yellow to green, etc.
    segments = list(
        starmap(
            make_optical_flow_color_wheel_segment,
            (
                ((255, 0, 0), (255, 255, 0), 15),  # red to yellow
                ((255, 255, 0), (0, 255, 0), 6),  # yellow to green
                ((0, 255, 0), (0, 255, 255), 4),  # green to cyan
                ((0, 255, 255), (0, 0, 255), 11),  # cyan to blue
                ((0, 0, 255), (255, 0, 255), 13),  # blue to magenta
                ((255, 0, 255), (255, 0, 0), 6),  # magenta to red
            ),
        )
    )

    return jnp.concatenate(segments, axis=0)


ColorWheel: Float[Array, "55 3"] = make_optical_flow_color_wheel()


def compute_optical_flow_colors(
    u: Float[Array, "height width"],
    v: Float[Array, "height width"],
) -> Float[Array, "height width 3"]:
    num_colors, _num_channels = ColorWheel.shape
    flow_angle: Float[Array, "height width"] = jnp.arctan2(-v, -u) / jnp.pi

    # Map flow angle to hue color space
    color_index_float: Float[Array, "height width"] = (flow_angle + 1) / 2 * (num_colors - 1)
    color_index_floor: Int32[Array, "height width"] = jnp.floor(color_index_float).astype(jnp.int32)
    color_index_ceil: Int32[Array, "height width"] = jnp.where(
        color_index_floor + 1 == num_colors, 0, color_index_floor + 1
    )
    color_interpolation_factor: Float[Array, "height width"] = color_index_float - color_index_floor

    # Compute colors
    base_color: Float[Array, "height width 3"] = ColorWheel[color_index_floor] / 255.0
    next_color: Float[Array, "height width 3"] = ColorWheel[color_index_ceil] / 255.0

    # Broadcast color_interpolation_factor along the channel dimension
    interpolated_color: Float[Array, "height width 3"] = (
        1 - color_interpolation_factor[..., None]
    ) * base_color + color_interpolation_factor[..., None] * next_color

    return interpolated_color


def normalize_flow(
    flow_uv: Float[Array, "height width 2"], max: float = float("inf"), epsilon: float = 1e-5
) -> Float[Array, "height width 2"]:
    """
    Clip flow by the provided max value, then normalize by the maximum magnitude, plus a small epsilon
    to avoid division by zero
    """
    clipped: Float[Array, "height width 2"] = jnp.clip(flow_uv, 0, max)
    magnitude: Float[Scalar, ""] = jnp.max(jnp.linalg.norm(clipped, axis=2))
    return flow_uv / (magnitude + epsilon)


def compute_optical_flow_image(
    flow_uv: Float[Array, "height width 2"],
    max: float = float("inf"),
) -> UInt8[Array, "height width 3"]:
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: array of shape [H,W,2]
    :param max: float, maximum clipping value for flow, defaults to inf
    :return:
    """
    # Normalize the flow
    normed_flow_uv: Float[Array, "height width 2"] = normalize_flow(flow_uv, max)

    # Find out which pixels are in range
    magnitude: Float[Array, "height width"] = jnp.linalg.norm(normed_flow_uv, axis=2)
    within_range: Bool[Array, "height width"] = magnitude <= 1

    # Get our colors
    interpolated_color: Float[Array, "height width 3"] = compute_optical_flow_colors(
        normed_flow_uv[..., 0], normed_flow_uv[..., 1]
    )

    # Broadcast within_range and flow_magnitude along the channel dimension
    adjusted_color: Float[Array, "height width 3"] = jnp.where(
        within_range[..., None], 1 - magnitude[..., None] * (1 - interpolated_color), interpolated_color * 0.75
    )

    # Convert to uint8
    flow_image: UInt8[Array, "height width 3"] = jnp.floor(255 * adjusted_color).astype(jnp.uint8)

    return flow_image
