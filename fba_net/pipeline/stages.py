from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
from jaxtyping import Array, Float, UInt8
from numpy import typing as npt
from nvidia.dali import fn, newaxis, types  # type: ignore
from nvidia.dali.fn import experimental as fnx  # type: ignore
from nvidia.dali.pipeline import DataNode  # type: ignore
from nvidia.dali.pipeline import _pipeline_def_experimental as pipeline_defx  # type: ignore
from nvidia.dali.plugin.jax import DALIGenericIterator  # type: ignore

from .real_bsr_dataset import RealBSRDataset


class RealBSRData(TypedDict):
    hr_frame: UInt8[Array, "batch height width channel"]
    lr_frames: UInt8[Array, "batch frame height width channel"]
    flows: Float[Array, "batch frame-1 height width 2"]
    lr_frames_unregistered: UInt8[Array, "batch frame height width channel"]


def _01_load_frames(
    source: Callable[[types.SampleInfo], Sequence[npt.NDArray[np.uint8]]],
    seed: int,
    num_frames: int,
) -> Sequence[DataNode]:
    """
    Returns a list of `DataNode` representing deserialized frames. No frame has been decoded yet. Each frame is a
    `UInt8[Array, "length"]`. The first frame is the HR frame, the rest are LR frames.
    """
    num_outputs: int = num_frames + 1
    return fn.external_source(
        source=source,
        num_outputs=num_outputs,
        layout="N",  # It's all just bytes read directly from disk
        dtype=[types.DALIDataType.UINT8] * num_outputs,
        ndim=[1] * num_outputs,
        batch=False,  # Automatically set to False when parallel=True
        parallel=False,
        seed=seed,
    )


def _02_decode_frames(encoded_frames: Sequence[DataNode], seed: int) -> Sequence[DataNode]:
    """
    Returns a list of `DataNode` representing decoded frames. Each frame is a `UInt8[Array, "height width channel"]`.
    The first frame is the HR frame, the rest are LR frames.
    """
    return fnx.decoders.image(
        encoded_frames,
        device="mixed",
        dtype=types.DALIDataType.UINT8,
        output_type=types.DALIImageType.RGB,
        seed=seed,
    )


def _03_compute_flows(lr_frames: DataNode, seed: int) -> DataNode:
    """
    Returns a `DataNode` representing the optical flow between the LR frames. The shape is
    `Float[Array, "frame-1 height width 2"]`. The last dimension is (y, x), we don't have channels.
    """
    return fn.optical_flow(  # type: ignore
        lr_frames,
        image_type=types.DALIImageType.RGB,
        preset=0.0,  # Slowest, but highest quality, preset
        output_grid=1,  # Highest resolution
        hint_grid=8,  # For temporal hints
        enable_temporal_hints=True,  # Our frames have high temporal coherence
        seed=seed,
    )


def _04_register_lr_frames(lr_frames: DataNode, flows: DataNode, seed: int) -> DataNode:
    """
    Registers the LR frames according to the optical flow. Returns a `DataNode` representing the registered LR frames.
    NOTE: The number of frames should match the number of flows -- that is, the reference frame should not be included.
    """
    # grid: 2 x RealBSRDataset.lr_frame_shape[0, 1]
    # This isn't a scalar type, but there's no way to type ConstantNode :l
    grid = types.Constant(
        value=np.mgrid[0: RealBSRDataset.lr_frame_shape[0], 0: RealBSRDataset.lr_frame_shape[1]],
        dtype=types.DALIDataType.INT64,
        shape=(2, RealBSRDataset.lr_frame_shape[0], RealBSRDataset.lr_frame_shape[1]),
        layout="NHW",
        device="gpu",
    )

    # map_xs: num_frames x RealBSRDataset.lr_frame_shape[0, 1]
    map_xs: DataNode = grid[1] - flows[:, :, :, 1]  # type: ignore
    # map_ys: num_frames x RealBSRDataset.lr_frame_shape[0, 1]
    map_ys: DataNode = grid[0] - flows[:, :, :, 0]  # type: ignore

    # num_frames x RealBSRDataset.lr_frame_shape
    return fnx.remap(  # type: ignore
        lr_frames,
        map_xs,
        map_ys,
        # INTERP_NN, INTERP_LINEAR, INTERP_CUBIC all work.
        # INTERP_LANCZOS3 fails with:
        #   Error when executing GPU operator experimental__Remap encountered:
        #   npp error (-22): NPP_INTERPOLATION_ERROR
        # INTERP_TRIANGULAR fails with:
        #   Error when executing GPU operator experimental__Remap encountered:
        #   [/opt/dali/dali/npp/npp.h:45] Unsupported interpolation type. Interpolation type (INTERP_TRIANGULAR)
        #   is not supported by NPP.
        # INTERP_GAUSSIAN fails with:
        #   Error when executing GPU operator experimental__Remap encountered:
        #   [/opt/dali/dali/npp/npp.h:45] Unsupported interpolation type. Interpolation type (INTERP_GAUSSIAN)
        #   is not supported by NPP.
        interp=types.DALIInterpType.INTERP_CUBIC,
        pixel_origin="center",  # TODO: Compare corner and center
        seed=seed,
    )


# max_batch_size (which is batch_size) must be greater than zero
@pipeline_defx(num_threads=1, device_id=0, batch_size=1, debug=False)
def _real_bsr_pipeline(dataset: RealBSRDataset):
    encoded_frames: Sequence[DataNode] = _01_load_frames(
        source=dataset.__call__,
        seed=dataset.seed,
        num_frames=dataset.num_frames,
    )
    frames: Sequence[DataNode] = _02_decode_frames(encoded_frames, seed=dataset.seed)

    # Unpack the list of frames and stack the LR frames
    # hr_frame: RealBSRDataset.hr_frame_shape
    # lr_frames: num_frames x RealBSRDataset.lr_frame_shape
    hr_frame: DataNode = frames[0]
    lr_frames_unregistered: DataNode = fn.stack(  # type: ignore
        *frames[1:],
        axis=0,
        axis_name="F",
        seed=dataset.seed,
    )

    flows: DataNode = _03_compute_flows(lr_frames_unregistered, seed=dataset.seed)

    # lr_frames_stacked: num_frames x RealBSRDataset.lr_frame_shape
    lr_frames: DataNode = fn.cat(  # type: ignore
        # 1 x RealBSRDataset.lr_frame_shape
        lr_frames_unregistered[0][newaxis("F")],
        # num_frames-1 x RealBSRDataset.lr_frame_shape
        _04_register_lr_frames(lr_frames_unregistered[1:], flows, seed=dataset.seed),
        axis_name="F",
    )

    ret = {
        "hr_frame": hr_frame,
        "lr_frames": lr_frames,
        "flows": flows,
        "lr_frames_unregistered": lr_frames_unregistered,
    }

    # Ensure consistent ordering of the outputs
    # NOTE: Can use a `tuple`, but cannot use a `list`.
    return tuple(ret[key] for key in RealBSRData.__annotations__.keys())


def real_bsr_pipeline(data_dir: Path, num_frames: int, seed: int) -> Iterator[RealBSRData]:
    dataset = RealBSRDataset(data_dir=data_dir, seed=seed, num_frames=num_frames)
    return DALIGenericIterator(
        pipelines=[_real_bsr_pipeline(dataset)],  # type: ignore
        output_map=RealBSRData.__annotations__.keys(),
        size=1,
    )
