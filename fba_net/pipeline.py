import time
from collections.abc import Iterator
from functools import cache, partial
from pathlib import Path

import dm_pix
import jax
import numpy as np
from einops import rearrange
from jax import numpy as jnp
from jax import scipy as jsp
from jaxtyping import Array, Bool, Float, Scalar, UInt8
from nvidia.dali import fn, pipeline_def  # type: ignore
from nvidia.dali.plugin.jax import DALIGenericIterator  # type: ignore
from PIL import Image

UInt8RGB = tuple[int, int, int]


def make_optical_flow_colorwheel_segment(
    start_color: UInt8RGB, end_color: UInt8RGB, length: int
) -> Float[Array, "length 3"]:
    """
    Generates a smooth color gradient for a given color segment
    """

    return jnp.stack(list(map(partial(jnp.linspace, num=length), start_color, end_color)), axis=-1)


@cache
def make_optical_flow_colorwheel() -> Float[Array, "55 3"]:
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    """

    # Make the color wheel segments: red to yellow, yellow to green, etc.
    RY = make_optical_flow_colorwheel_segment((255, 0, 0), (255, 255, 0), 15)
    YG = make_optical_flow_colorwheel_segment((255, 255, 0), (0, 255, 0), 6)
    GC = make_optical_flow_colorwheel_segment((0, 255, 0), (0, 255, 255), 4)
    CB = make_optical_flow_colorwheel_segment((0, 255, 255), (0, 0, 255), 11)
    BM = make_optical_flow_colorwheel_segment((0, 0, 255), (255, 0, 255), 13)
    MR = make_optical_flow_colorwheel_segment((255, 0, 255), (255, 0, 0), 6)

    return jnp.concatenate([RY, YG, GC, CB, BM, MR], axis=0)


def compute_optical_flow_colors(
    u: Float[Array, "height width"],
    v: Float[Array, "height width"],
) -> Float[Array, "num_colors 3"]:
    colorwheel: Float[Array, "55 3"] = make_optical_flow_colorwheel()
    num_colors, _num_channels = colorwheel.shape
    flow_angle: Float[Array, "height width"] = jnp.arctan2(-v, -u) / jnp.pi

    # Map flow angle to hue color space
    color_index_float: Float[Array, "height width"] = (flow_angle + 1) / 2 * (num_colors - 1)
    color_index_floor: Float[Array, "height width"] = jnp.floor(color_index_float).astype(jnp.int32)
    color_index_ceil: Float[Array, "height width"] = jnp.where(
        color_index_floor + 1 == num_colors, 0, color_index_floor + 1
    )
    color_interpolation_factor: Float[Array, "height width"] = color_index_float - color_index_floor

    # Compute colors
    base_color: Float[Scalar, ""] = colorwheel[color_index_floor] / 255.0
    next_color: Float[Scalar, ""] = colorwheel[color_index_ceil] / 255.0

    # Broadcast color_interpolation_factor along the channel dimension
    interpolated_color: Float[Array, "height width 3"] = (
        1 - color_interpolation_factor[..., None]
    ) * base_color + color_interpolation_factor[..., None] * next_color

    return interpolated_color


def compute_optical_flow_image(
    flow_uv: Float[Array, "height width 2"],
    clip_flow: None | float = None,
    convert_to_bgr: bool = False,
) -> UInt8[Array, "height width 3"]:
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    :param flow_uv: array of shape [H,W,2]
    :param clip_flow: float, maximum clipping value for flow
    :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
    :return:
    """
    if clip_flow is not None:
        flow_uv = jnp.clip(flow_uv, 0, clip_flow)

    def normalize_flow(
        flow_uv: Float[Array, "height width 2"], epsilon: float = 1e-5
    ) -> Float[Array, "height width 2"]:
        """
        Normalize flow by the maximum magnitude, plus a small epsilon
        """
        return flow_uv / (jnp.max(jnp.linalg.norm(flow_uv, axis=2)) + epsilon)

    # Normalize the flow
    flow_uv = normalize_flow(flow_uv)

    # Find out which pixels are in range
    flow_magnitude: Float[Array, "height width"] = jnp.linalg.norm(flow_uv, axis=2)
    within_range: Bool[Array, "height width"] = flow_magnitude <= 1

    # Get our colors
    interpolated_color: Float[Array, "height width 3"] = compute_optical_flow_colors(flow_uv[..., 0], flow_uv[..., 1])

    # Broadcast within_range and flow_magnitude along the channel dimension
    adjusted_color: Float[Array, "height width 3"] = jnp.where(
        within_range[..., None], 1 - flow_magnitude[..., None] * (1 - interpolated_color), interpolated_color * 0.75
    )

    # If convert_to_bgr, reverse the channel order
    if convert_to_bgr:
        adjusted_color = adjusted_color[..., ::-1]

    flow_image: UInt8[Array, "height width 3"] = jnp.floor(255 * adjusted_color).astype(jnp.uint8)
    return flow_image


@pipeline_def(num_threads=1, device_id=0)
def sintel_pipeline(
    file_root: str,
    num_frames: int = 5,
    seed: int = 0,
) -> tuple[UInt8[Array, "batches frames height width channels"], Float[Array, "batches frames-1 height width 2"],]:
    frames = fn.readers.sequence(file_root=file_root, sequence_length=num_frames).gpu()  # type: ignore
    # Move to the GPU for optical flow
    flows = fn.optical_flow(  # type: ignore
        frames,
        output_grid=1,  # Highest resolution
        hint_grid=8,  # For temporal hints
        enable_temporal_hints=True,  # Our frames have high temporal coherence
        seed=seed,
    )
    return frames, flows


def register_frame(
    frame: UInt8[Array, "height width channels"],
    flow: Float[Array, "height width 2"],
) -> UInt8[Array, "height width channels"]:
    """
    Warps the frame according to the optical flow
    """
    # Move channels to the front
    frame = rearrange(frame, "height width channels -> channels height width")
    flow = rearrange(flow, "height width xy -> xy height width", xy=2)

    # Compute the grid
    _channels, height, width = frame.shape
    grid: Float[Array, "2 height width"] = jnp.mgrid[0:height, 0:width]

    # Compute the warped grid
    warped_grid: Float[Array, "2 height width"] = grid - flow

    # Interpolate the frame by warping each channel
    warped_frame: Float[Array, "channels height width"] = jax.vmap(
        partial(
            jsp.ndimage.map_coordinates,
            coordinates=warped_grid,  # type: ignore
            order=1,
            mode="nearest",
        )
    )(frame)

    # Move channels back to the end
    warped_frame = rearrange(warped_frame, "channels height width -> height width channels")
    return warped_frame


def run_pipeline() -> None:
    num_batches: int = 1
    data_path: Path = Path("/home/connorbaker/FBANet/data")
    optical_flow_path: Path = Path("/home/connorbaker/FBANet/optical_flow")

    # jax_iter will have length num_batches
    jax_iter: Iterator[UInt8[Array, "1 frame height width channel"]] = DALIGenericIterator(
        pipelines=[
            sintel_pipeline(  # type: ignore
                batch_size=1,
                file_root=data_path.as_posix(),
            )
        ],
        output_map=["frames", "flows"],
        size=num_batches,
    )

    for blob in jax_iter:
        frames: UInt8[Array, "frames height width channels"] = rearrange(
            blob["frames"], "1 frames height width channels -> frames height width channels"
        )
        flows: Float[Array, "frames-1 height width 2"] = rearrange(
            blob["flows"],
            "1 frames height width channels -> frames height width channels",
            frames=frames.shape[0] - 1,
            height=frames.shape[1],
            width=frames.shape[2],
            channels=2,
        )

        # Store the reference frame
        optical_flow_path.mkdir(exist_ok=True, parents=True)
        reference_frame: UInt8[Array, "height width channels"] = frames[0]
        float_reference_frame: Float[Array, "height width channels"] = reference_frame.astype(jnp.float32) / 255.0
        Image.fromarray(np.asarray(reference_frame)).save(optical_flow_path / "reference.jpg")

        # Store the other frames and their optical flow
        for i, (frame, flow) in enumerate(zip(frames[1:], flows, strict=True)):
            Image.fromarray(np.asarray(frame)).save(optical_flow_path / f"frame_{i}.jpg")
            float_frame: Float[Array, "height width channels"] = frame.astype(jnp.float32) / 255.0
            Image.fromarray(np.asarray(compute_optical_flow_image(flow))).save(optical_flow_path / f"flow_{i}.jpg")
            registered_frame: UInt8[Array, "height width channels"] = register_frame(frame, flow)
            Image.fromarray(np.asarray(registered_frame)).save(optical_flow_path / f"registered_frame_{i}.jpg")
            float_registered_frame: Float[Array, "height width channels"] = registered_frame.astype(jnp.float32) / 255.0
            print(f"frame={i}")
            for metric in [dm_pix.psnr, dm_pix.ssim]:
                print(f"{metric.__name__} unreg. = {metric(float_reference_frame, float_frame)}")
                print(f"{metric.__name__} reg.   = {metric(float_reference_frame, float_registered_frame)}")


if __name__ == "__main__":
    start_time: float = time.time()
    run_pipeline()
    end_time: float = time.time()
    print(f"Pipeline took {end_time - start_time} seconds to run")

# Getting the batch as a list of numpy arrays, for displaying
# for blob in jax_iter:
#     image: Int8[Array, "batch height width channel"] = blob["image"]
#     label: Int8[Array, "batch label"] = blob["label"]

#     def write_img(
#         image: Int8[Array, "height width channel"],
#         label: Int8[Array, "label"],
#     ) -> None:
#         Image.fromarray(np.asarray(image)).save(f"test_{int(label)}.jpg")

#     for img, lbl in zip(image, label):
#         write_img(img, lbl)
