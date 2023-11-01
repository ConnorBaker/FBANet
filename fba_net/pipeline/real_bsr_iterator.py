from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, final

import numpy as np
from numpy import typing as npt
from nvidia.dali import fn, newaxis, types  # pyright: ignore[reportMissingTypeStubs]
from nvidia.dali.fn import experimental as fnx  # type: ignore
from nvidia.dali.pipeline import DataNode  # pyright: ignore[reportMissingTypeStubs]
from nvidia.dali.plugin.jax import data_iterator  # pyright: ignore[reportMissingTypeStubs]

from .real_bsr_dataset import RealBSRData, RealBSRDataset, RealBSRDatasetKwargs


@final
@dataclass(frozen=True, slots=True, eq=True, kw_only=True)
class DaliKwargs:
    # Seed is required for the sake of reproducibility
    seed: int

    enable_conditionals: bool = False
    batch_size: int = -1
    num_threads: int = -1
    device_id: int = -1
    exec_pipelined: bool = True
    prefetch_queue_depth: int = 2
    exec_async: bool = True
    bytes_per_sample: int = 0
    set_affinity: bool = False
    max_streams: int = -1
    default_cuda_stream_priority: int = 0
    enable_memory_stats: bool = False
    enable_checkpointing: bool = False
    checkpoint: None | str = None
    py_num_workers: int = 1
    py_start_method: Literal["fork", "spawn"] = "fork"
    py_callback_pickler: None | tuple[Any, ...] | ModuleType = None
    output_dtype: None | types.DALIDataType | Sequence[types.DALIDataType] = None
    output_ndim: None | int | Sequence[int] = None

    def create_iterator(self, dataset: RealBSRDataset) -> Iterator[RealBSRData]:
        return _mk_real_bsr_dali_iterator(  # type: ignore
        dataset, **{key: self.__getattribute__(key) for key in self.__slots__}
    )


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

    # We have `num_outputs`, each of type `types.DALIDataType.UINT8`.
    dtypes: Sequence[types.DALIDataType] = [types.DALIDataType.UINT8] * num_outputs

    # Each output has a dimension of 1 because it's an encoded JPEG -- it's just a byte array.
    dimensions: Sequence[int] = [1] * num_outputs

    frames: Sequence[DataNode] = fn.external_source(
        source=source,
        num_outputs=num_outputs,
        layout="N",  # It's all just bytes read directly from disk
        dtype=dtypes,
        ndim=dimensions,
        batch=False,  # Automatically set to False when parallel=True
        parallel=False,
        seed=seed,
    )
    assert isinstance(frames, Sequence)

    return frames


def _02_decode_frames(encoded_frames: Sequence[DataNode], seed: int) -> Sequence[DataNode]:
    """
    Returns a list of `DataNode` representing decoded frames. Each frame is a `UInt8[Array, "height width channel"]`.
    The first frame is the HR frame, the rest are LR frames.
    """
    output_dtype: types.DALIDataType = types.DALIDataType.UINT8
    output_color_space: types.DALIImageType = types.DALIImageType.RGB
    decoded_frames: Sequence[DataNode] = fnx.decoders.image(
        encoded_frames,
        device="mixed",
        dtype=output_dtype,
        output_type=output_color_space,
        seed=seed,
    )
    assert isinstance(decoded_frames, Sequence)
    return decoded_frames


def _03_compute_flows(lr_frames: DataNode, seed: int) -> DataNode:
    """
    Returns a `DataNode` representing the optical flow between the LR frames. The shape is
    `Float[Array, "frame-1 height width 2"]`. The last dimension is (y, x), we don't have channels.
    """
    input_color_space: types.DALIImageType = types.DALIImageType.RGB
    preset: float = 0.0  # Slowest, but highest quality, preset
    output_grid: int = 1  # Highest resolution
    hint_grid: int = 8  # For temporal hints
    enable_temporal_hints: bool = True  # Our frames have high temporal coherence
    frames_and_flows: DataNode = fn.optical_flow(  # type: ignore
        lr_frames,
        image_type=input_color_space,
        preset=preset,
        output_grid=output_grid,
        hint_grid=hint_grid,
        enable_temporal_hints=enable_temporal_hints,
        seed=seed,
    )
    assert isinstance(frames_and_flows, DataNode)
    return frames_and_flows


def _04_register_lr_frames(lr_frames: DataNode, flows: DataNode, seed: int) -> DataNode:
    """
    Registers the LR frames according to the optical flow. Returns a `DataNode` representing the registered LR frames.
    NOTE: The number of frames should match the number of flows -- that is, the reference frame should not be included.
    """
    # grid: 2 x RealBSRDataset.lr_frame_shape[0, 1]
    # This isn't a scalar type, but there's no way to type ConstantNode :l
    grid = types.Constant(
        value=np.mgrid[0 : RealBSRDataset.lr_frame_shape[0], 0 : RealBSRDataset.lr_frame_shape[1]],
        dtype=types.DALIDataType.INT64,
        shape=(2, RealBSRDataset.lr_frame_shape[0], RealBSRDataset.lr_frame_shape[1]),
        layout="NHW",
        device="gpu",
    )

    # map_xs: num_frames x RealBSRDataset.lr_frame_shape[0, 1]
    map_xs: DataNode = grid[1] - flows[:, :, :, 1]  # type: ignore
    # map_ys: num_frames x RealBSRDataset.lr_frame_shape[0, 1]
    map_ys: DataNode = grid[0] - flows[:, :, :, 0]  # type: ignore

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
    interpolation_type: types.DALIInterpType = types.DALIInterpType.INTERP_CUBIC

    # num_frames x RealBSRDataset.lr_frame_shape
    registered_frames: DataNode = fnx.remap(
        lr_frames,
        map_xs,
        map_ys,
        interp=interpolation_type,
        pixel_origin="center",  # TODO: Compare corner and center
        seed=seed,
    )
    assert isinstance(registered_frames, DataNode)

    return registered_frames


@data_iterator(  # pyright: ignore[reportUntypedFunctionDecorator]
    output_map=RealBSRData.__annotations__.keys(),
    size=1,
)
def _mk_real_bsr_dali_iterator(dataset: RealBSRDataset) -> tuple[DataNode, DataNode, DataNode, DataNode]:
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
    return tuple(ret[key] for key in RealBSRData.__annotations__.keys())  # type: ignore


def real_bsr_iterator(
    # Dataset arguments
    dataset_kwargs: RealBSRDatasetKwargs,
    # DALI arguments
    dali_kwargs: DaliKwargs,
) -> Iterator[RealBSRData]:
    dataset: RealBSRDataset = dataset_kwargs.create_dataset()
    iterator: Iterator[RealBSRData] = dali_kwargs.create_iterator(dataset)
    return iterator
