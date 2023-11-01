import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import ClassVar, TypedDict, final

import numpy as np
from jaxtyping import Array, Float, UInt8
from numpy import typing as npt
from nvidia.dali.types import SampleInfo  # type: ignore


class RealBSRData(TypedDict):
    hr_frame: UInt8[Array, "batch height width channel"]
    lr_frames: UInt8[Array, "batch frame height width channel"]
    flows: Float[Array, "batch frame-1 height width 2"]
    lr_frames_unregistered: UInt8[Array, "batch frame height width channel"]


@final
@dataclass(slots=True, frozen=True, eq=True, kw_only=True)
class RealBSRDataset:
    # Variables set by the user
    data_dir: Path
    seed: int
    num_frames: int
    batch_size: int
    shard_id: int
    num_shards: int

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
    def __call__(self, sample_info: SampleInfo) -> Sequence[npt.NDArray[np.uint8]]:
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        sample_idx: int = self.get_index(sample_info.epoch_idx, sample_info.idx_in_epoch)
        return self[sample_idx]


@final
@dataclass(frozen=True, slots=True, eq=True)
class RealBSRDatasetKwargs:
    data_dir: Path
    seed: int
    num_frames: int = 14
    """The length of the sequence returned is num_frames + 1, where the first frame is the HR frame"""
    batch_size: int = 1
    shard_id: int = 0
    num_shards: int = 1

    def create_dataset(self) -> RealBSRDataset:
        return RealBSRDataset(**{key: self.__getattribute__(key) for key in self.__slots__})
