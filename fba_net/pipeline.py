import re
from collections.abc import Iterator, Sequence, Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import ClassVar, TypedDict

import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, UInt8
from numpy import typing as npt
from nvidia.dali import fn, newaxis, types  # type: ignore
from nvidia.dali.fn import experimental as fnx  # type: ignore
from nvidia.dali.pipeline import DataNode  # type: ignore
from nvidia.dali.pipeline import _pipeline_def_experimental as pipeline_defx  # type: ignore

# from nvidia.dali.pipeline.experimental import pipeline_def  # type: ignore
from nvidia.dali.plugin.jax import DALIGenericIterator  # type: ignore
from PIL import Image


@dataclass(slots=True, frozen=True, eq=True, kw_only=True)
class RealBSRDataset:
    # Variables set by the user
    data_dir: Path
    seed: int
    num_frames: int = 14
    """The length of the sequence returned is num_frames + 1, where the first frame is the HR frame"""
    batch_size: int = 1
    shard_id: int = 0
    num_shards: int = 1

    # Variables set during initialization
    shard_size: int = field(init=False)
    shard_offset: int = field(init=False)
    full_iterations: int = field(init=False)
    subdirs: Sequence[Path] = field(init=False)

    # Class variables
    max_num_frames: ClassVar[int] = 14
    file_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"""
        ^(?P<filename>                   # Begin filename
            (?P<base_image_number>\d{3})    # 3-digit base frame number patches are from
            _MFSR_Sony_                     # Dataset name
            (?P<patch_number>\d{4})         # 4-digit patch number
            _x(?P<scale_factor>\d{1})       # 1-digit scale factor
            (?P<frame_kind>                 # Begin frame kind
                _(?P<frame_number>\d{2})        # 2-digit frame number
                |warp                           # Or the HR frame
            )                               # End frame kind
            .png                            # File extension
        )$                               # End filename
        """,
        re.VERBOSE,
    )
    hr_frame_shape: ClassVar[tuple[int, int, int]] = (640, 640, 3)
    lr_frame_shape: ClassVar[tuple[int, int, int]] = (160, 160, 3)

    @cache
    @staticmethod
    def get_indices_for_epoch(seed: int, length: int, epoch_idx: int) -> npt.NDArray[np.uint16]:
        # There are very few samples, so we can use a uint16.
        # Also makes it easier to cache.
        return np.random.default_rng(seed=seed + epoch_idx).permutation(np.arange(length, dtype=np.uint16))

    def get_index(self, epoch_idx: int, idx_in_epoch: int) -> int:
        """A function that takes an epoch index and an index in the epoch and returns the index of the sample to use"""
        return RealBSRDataset.get_indices_for_epoch(self.seed, len(self.subdirs), epoch_idx)[
            idx_in_epoch + self.shard_offset
        ]

    def __post_init__(self):
        # Assert that the number of frames desired is less than the number of frames in the dataset
        assert self.num_frames <= self.max_num_frames, "The dataset only contains 14 LR frames"

        # Get a list of the directories in dir, sorted for consistency
        object.__setattr__(self, "subdirs", sorted(self.data_dir.iterdir()))
        assert all(map(Path.is_dir, self.subdirs)), "All paths must be directories"

        # If the dataset size is not divisibvle by number of shards, the trailing samples will
        # be omitted.
        object.__setattr__(self, "shard_size", len(self.subdirs) // self.num_shards)
        object.__setattr__(self, "shard_offset", self.shard_size * self.shard_id)

        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        object.__setattr__(self, "full_iterations", self.shard_size // self.batch_size)

    def __len__(self) -> int:
        return self.full_iterations

    # NOTE: No shape checking done because the DALI pipeline handles decoding for us
    def __getitem__(self, idx: int) -> Sequence[npt.NDArray[np.uint8]]:
        # Lexicographically sort the files in the directory to ensure that the frames are in the correct order
        image_files = sorted(self.subdirs[idx].iterdir())

        lr_frames: list[npt.NDArray[np.uint8]] = list()
        hr_frame: npt.NDArray[np.uint8] | None = None
        for image_file in image_files:
            match = self.file_pattern.match(image_file.name)
            assert match is not None, f"File {image_file} does not match the expected pattern"

            # Frame is a single-dimensional array of bytes
            frame: npt.NDArray[np.uint8] = np.fromfile(image_file, dtype=np.uint8)

            if match["frame_kind"] == "warp":
                hr_frame = frame
            elif len(lr_frames) < self.num_frames:
                lr_frames.append(frame)

            # Early exit condition
            if len(lr_frames) >= self.num_frames and hr_frame is not None:
                break

        assert hr_frame is not None, f"HR frame not found in {image_files}"
        return hr_frame, *lr_frames

    # NOTE: No shape checking done because the DALI pipeline handles decoding for us
    def __call__(self, sample_info: types.SampleInfo) -> Sequence[npt.NDArray[np.uint8]]:
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        sample_idx: int = self.get_index(sample_info.epoch_idx, sample_info.idx_in_epoch)
        return self[sample_idx]


class RealBSRData(TypedDict):
    hr_frame: UInt8[Array, "batch height width channel"]
    lr_frames: UInt8[Array, "batch frame height width channel"]
    flows: Float[Array, "batch frame-1 height width 2"]
    lr_frames_unregistered: UInt8[Array, "batch frame height width channel"]


def _get_frames(
    source: Callable[[types.SampleInfo], Sequence[npt.NDArray[np.uint8]]],
    seed: int,
    num_frames: int,
) -> Sequence[DataNode]:
    num_outputs: int = num_frames + 1

    # Get the frames from disk
    encoded_frames: Sequence[DataNode] = fn.external_source(
        source=source,
        num_outputs=num_outputs,
        layout="N",  # It's all just bytes read directly from disk
        dtype=[types.DALIDataType.UINT8] * num_outputs,
        ndim=[1] * num_outputs,
        batch=False,  # Automatically set to False when parallel=True
        parallel=False,
        seed=seed,
    )

    # Decode the frames
    decoded_frames: Sequence[DataNode] = fnx.decoders.image(
        encoded_frames,
        device="mixed",
        dtype=types.DALIDataType.UINT8,
        output_type=types.DALIImageType.RGB,
        seed=seed,
    )

    return decoded_frames


def _register_lr_frames(lr_frames: DataNode, seed: int) -> tuple[DataNode, DataNode]:
    # Take the optical flow between the LR frames
    # flows: num_frames-1 x RealBSRDataset.lr_frame_shape[0, 1] x 2
    # NOTE: The last dimension is (y, x), we don't have channels
    flows: DataNode = fn.optical_flow(  # type: ignore
        lr_frames,
        image_type=types.DALIImageType.RGB,
        preset=0.0,  # Slowest, but highest quality, preset
        output_grid=1,  # Highest resolution
        hint_grid=8,  # For temporal hints
        enable_temporal_hints=True,  # Our frames have high temporal coherence
        seed=seed,
    )

    # Align the LR frames to the first LR frame
    # lr_reference_frame: RealBSRDataset.lr_frame_shape
    # lr_frames_to_register: num_frames-1 x RealBSRDataset.lr_frame_shape
    # NOTE: lr_frames_to_register_stacked must match the dimensions of flows (excluding the last dimension)
    lr_reference_frame: DataNode = lr_frames[0]
    lr_frames_to_register: DataNode = lr_frames[1:]

    # grid: 2 x RealBSRDataset.lr_frame_shape[0, 1]
    # This isn't a scalar type, but there's no way to type ConstantNode :l
    grid = types.Constant(
        value=np.mgrid[0 : RealBSRDataset.lr_frame_shape[0], 0 : RealBSRDataset.lr_frame_shape[1]],
        dtype=types.DALIDataType.INT64,
        shape=(2, RealBSRDataset.lr_frame_shape[0], RealBSRDataset.lr_frame_shape[1]),
        layout="NHW",
        device="gpu",
    )

    # map_xs: num_frames-1 x RealBSRDataset.lr_frame_shape[0, 1]
    map_xs: DataNode = grid[1] - flows[:, :, :, 1]  # type: ignore
    # map_ys: num_frames-1 x RealBSRDataset.lr_frame_shape[0, 1]
    map_ys: DataNode = grid[0] - flows[:, :, :, 0]  # type: ignore

    # lr_frames_stacked: num_frames x RealBSRDataset.lr_frame_shape
    lr_frames_registered: DataNode = fn.cat(  # type: ignore
        # 1 x RealBSRDataset.lr_frame_shape
        lr_reference_frame[newaxis("F")],
        # num_frames-1 x RealBSRDataset.lr_frame_shape
        fnx.remap(  # type: ignore
            lr_frames_to_register,
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
        ),
        axis_name="F",
    )

    return flows, lr_frames_registered


# max_batch_size (which is batch_size) must be greater than zero
@pipeline_defx(num_threads=1, device_id=0, batch_size=1, debug=False)
def _real_bsr_pipeline(dataset: RealBSRDataset):
    # Get the frames
    frames = _get_frames(
        source=dataset.__call__,
        seed=dataset.seed,
        num_frames=dataset.num_frames,
    )

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

    # Register the LR frames
    flows, lr_frames = _register_lr_frames(lr_frames_unregistered, seed=dataset.seed)

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


def save_image(
    image: UInt8[Array, "height width 3"] | Float[Array, "height width 3"],
    path: Path,
) -> None:
    if isinstance(image, Float[Array, "height width 3"]):  # type: ignore
        image = (image * 255.0).astype(jnp.uint8)
    Image.fromarray(np.asarray(image)).save(path)


def run_pipeline() -> None:
    # pipeline = _real_bsr_pipeline(Path("/home/connorbaker/FBANet/data/RealBSR_RGB_trainpatch"), num_frames=5, seed=0)
    # pipeline.build()
    # pipeline.run()
    for blob in real_bsr_pipeline(Path("/home/connorbaker/FBANet/data/RealBSR_RGB_trainpatch"), num_frames=5, seed=0):
        print(blob)
        break
        # frames: UInt8[Array, "frames height width channels"] = rearrange(
        #     blob["frames"], "1 frames height width channels -> frames height width channels"
        # )
        # flows: Float[Array, "frames-1 height width 2"] = rearrange(
        #     blob["flows"],
        #     "1 frames height width channels -> frames height width channels",
        #     frames=frames.shape[0] - 1,
        #     height=frames.shape[1],
        #     width=frames.shape[2],
        #     channels=2,
        # )

        # # Store the reference frame
        # Image.fromarray(np.asarray(frames[0])).save(registration_path / "reference.jpg")
        # reference_frame: Float[Array, "height width channels"] = frames[0].astype(jnp.float32) / 255.0
        # print(f"reference frame shape = {reference_frame.shape}")
        # print(f"reference frame dtype = {reference_frame.dtype}")

        # # Store the other frames and their optical flow
        # for i, (frame, flow) in enumerate(zip(frames[1:], flows, strict=True)):
        #     save_image(frame, registration_path / f"frame_{i}.jpg")
        #     frame = frame.astype(jnp.float32) / 255.0
        #     compare_registrations(registration_path, reference_frame, frame, flow, i)


if __name__ == "__main__":
    run_pipeline()
