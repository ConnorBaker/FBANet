from collections.abc import Iterator
from pathlib import Path

import dm_pix
import numpy as np
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, Float, UInt8
from nvidia.dali import fn, pipeline_def  # type: ignore
from nvidia.dali.plugin.jax import DALIGenericIterator  # type: ignore
from PIL import Image

from fba_net.optical_flow import compute_optical_flow_image, register_frame
from fba_net.homography_alignment import register_frame as homography_register_frame
from fba_net.registration.pyramid import register as pyramid_register_frame


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


def save_image(
    image: UInt8[Array, "height width 3"] | Float[Array, "height width 3"],
    path: Path,
) -> None:
    if isinstance(image, Float[Array, "height width 3"]):  # type: ignore
        image = (image * 255.0).astype(jnp.uint8)
    Image.fromarray(np.asarray(image)).save(path)


def compare_registrations(
    registration_path: Path,
    reference_frame: Float[Array, "height width 3"],
    frame: Float[Array, "height width 3"],
    flow: Float[Array, "height width 2"],
    i: int,
) -> None:
    # Save the optical flow visualization
    optical_flow_frame: UInt8[Array, "height width 3"] = compute_optical_flow_image(flow)
    save_image(optical_flow_frame, registration_path / f"flow_{i}.jpg")

    # Save the optical flow-registrered frame
    optical_flow_registered_frame: Float[Array, "height width channels"] = register_frame(frame, flow)
    save_image(optical_flow_registered_frame, registration_path / f"optical_flow_registered_frame_{i}.jpg")

    # Compute the metrics
    for metric in [dm_pix.psnr, dm_pix.ssim]:
        print(f"{metric.__name__} optical flow unreg. = {metric(reference_frame, frame)}")
        print(f"{metric.__name__} optical flow reg. = {metric(reference_frame, optical_flow_registered_frame)}")

    # Save the homography-registered frame and run the metrics (they are in the callable)
    homography_registered_frame: Float[Array, "height width channels"] = homography_register_frame(
        reference_frame, frame
    )
    save_image(homography_registered_frame, registration_path / f"homography_registered_frame_{i}.jpg")

    # Save the pyramid-registered frame and run the metrics (they are in the callable)
    pyramid_registered_frame: Float[Array, "height width channels"] = pyramid_register_frame(
        "MapperGradProj", reference_frame, frame
    )
    save_image(pyramid_registered_frame, registration_path / f"pyramid_registered_frame_{i}.jpg")


def run_pipeline() -> None:
    num_batches: int = 1
    data_path: Path = Path("/home/connorbaker/FBANet/data")
    registration_path = Path("/home/connorbaker/FBANet/registration")
    registration_path.mkdir(exist_ok=True, parents=True)

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
        Image.fromarray(np.asarray(frames[0])).save(registration_path / "reference.jpg")
        reference_frame: Float[Array, "height width channels"] = frames[0].astype(jnp.float32) / 255.0
        print(f"reference frame shape = {reference_frame.shape}")
        print(f"reference frame dtype = {reference_frame.dtype}")

        # Store the other frames and their optical flow
        for i, (frame, flow) in enumerate(zip(frames[1:], flows, strict=True)):
            save_image(frame, registration_path / f"frame_{i}.jpg")
            frame = frame.astype(jnp.float32) / 255.0
            compare_registrations(registration_path, reference_frame, frame, flow, i)


if __name__ == "__main__":
    run_pipeline()
