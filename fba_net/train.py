from collections.abc import Iterator
from pathlib import Path

import dm_pix as pix
import equinox as eqx
import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
from optax import OptState  # pyright: ignore[reportMissingTypeStubs]

from fba_net.models.fba_net import FBANetModel
from fba_net.pipeline.real_bsr_dataset import RealBSRData, RealBSRDatasetKwargs

from .assert_shape import assert_shape
from .pipeline.real_bsr_iterator import DaliKwargs, real_bsr_iterator

LRFramesBatch = Float[Array, "batch frame 160 160 3"]
LRFrames = Float[Array, "frame 160 160 3"]
LRFrame = Float[Array, "160 160 3"]
HRFrameBatch = Int[Array, "batch 640 640 3"]
HRFrame = Int[Array, "640 640 3"]


def compute_loss(
    model: FBANetModel,
    lr_frames_batch: LRFramesBatch,
    hr_frame_batch: HRFrameBatch,
) -> Float[Array, " batch"]:
    """Compute the loss for a batch of frames."""
    batch_size = lr_frames_batch.shape[0]
    assert_shape((batch_size, None, 160, 160, 3), lr_frames_batch)
    assert_shape((batch_size, 640, 640, 3), hr_frame_batch)

    hr_frame_predicted_batch: HRFrameBatch = jax.vmap(model)(lr_frames_batch)
    assert_shape((batch_size, 640, 640, 3), hr_frame_predicted_batch)

    # TODO: Make sure values are normalized prior to feeding to pix.psnr, which assumes the
    # the difference between the maximum and minimum values are 1.0.
    loss_batch = pix.psnr(hr_frame_predicted_batch, hr_frame_batch)
    assert_shape((batch_size,), loss_batch)

    return loss_batch


@eqx.filter_jit
def make_step(
    model: FBANetModel,
    optimizer_update_fn: optax.TransformUpdateFn,
    opt_state: PyTree,
    lr_frames_batch: Float[Array, "batch frame 160 160 3"],
    hr_frame_batch: Int[Array, "batch 640 640 3"],
) -> tuple[FBANetModel, OptState, Float[Array, " batch"]]:
    """Make a step of training."""
    batch_size = lr_frames_batch.shape[0]
    assert_shape((batch_size, None, 160, 160, 3), lr_frames_batch)
    assert_shape((batch_size, 640, 640, 3), hr_frame_batch)

    losses, grads = eqx.filter_value_and_grad(compute_loss)(model, lr_frames_batch, hr_frame_batch)
    assert_shape((batch_size,), losses)

    updates, opt_state = optimizer_update_fn(
        grads,
        opt_state,
        model,  # type: ignore
    )
    model = eqx.apply_updates(model, updates)

    return model, opt_state, losses


def train(
    model: FBANetModel,
    training_data: Iterator[RealBSRData],
    optimizer: optax.GradientTransformation,
    steps: int,
) -> FBANetModel:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for step, data in zip(range(steps), training_data):
        # Normalize the data to be between 0 and 1.
        lr_frames_batch = data["lr_frames"].astype(jnp.float32) / 255.0
        hr_frame_batch = data["hr_frame"].astype(jnp.float32) / 255.0
        model, opt_state, loss = make_step(model, optimizer.update, opt_state, lr_frames_batch, hr_frame_batch)
        print(f"Step {step}: loss = {loss.mean()}")

    return model


if __name__ == "__main__":
    fba_net_model = FBANetModel(img_size=160)
    training_data = real_bsr_iterator(
        dataset_kwargs=RealBSRDatasetKwargs(
            data_dir=Path("/home/connorbaker/FBANet/data/RealBSR_RGB_trainpatch"),
            num_frames=14,
            seed=0,
        ),
        dali_kwargs=DaliKwargs(
            seed=0,
            num_threads=1,
            device_id=0,
            batch_size=1,
        ),
    )

    trained_model = train(
        model=fba_net_model,
        training_data=training_data,
        optimizer=optax.adam(1e-4),
        steps=100,
    )
