import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

import cv2  # pyright: ignore[reportMissingTypeStubs]
import dm_pix
import numpy as np
import numpy.typing as npt
from jax import numpy as jnp
from jaxtyping import Array, Float32, Shaped

# TODO:
# 1. Add an option for batch processing
# 2. Use the inverse map of the previously computed frame as the initial guess for the current frame

MapperName = Literal["MapperGradShift", "MapperGradEuclid", "MapperGradSimilar", "MapperGradAffine", "MapperGradProj"]
NPImage = Shaped[np.ndarray, "height width channels"]  # pyright: ignore[reportMissingTypeArgument]
JaxImage = Shaped[Array, "height width channels"]


class Map(Protocol):
    def inverseMap(self) -> Self: ...

    def inverseWarp(self, img1: NPImage) -> NPImage: ...

    def warp(self, img1: NPImage) -> NPImage: ...


class Mapper(Protocol):
    def calculate(
        self,
        img1: NPImage,
        img2: NPImage,
    ) -> Map: ...

    def getMap(self) -> Map: ...


@dataclass(frozen=True, slots=True)
class MapperPyramid(Mapper, cv2.reg.MapperPyramid):  # type: ignore
    numIterPerScale_: int
    numLev_: int

    def __new__(cls, *, mapper: Mapper, numIterPerScale_: int, numLev_: int) -> Callable[[Mapper], Self]:
        pyramid_mapper = cv2.reg.MapperPyramid(mapper)  # type: ignore
        pyramid_mapper.numIterPerScale_ = numIterPerScale_
        pyramid_mapper.numLev_ = numLev_
        return pyramid_mapper


def register(
    mapper_name: MapperName,
    reference: JaxImage,
    frame: JaxImage,
) -> JaxImage:
    start = time.time()

    mapper: Mapper = getattr(cv2.reg, mapper_name)()  # type: ignore
    mapper_pyramid: MapperPyramid = MapperPyramid(
        mapper=mapper,
        numIterPerScale_=3,
        numLev_=3,
    )
    np_reference: npt.NDArray[Any] = np.asarray(reference)
    np_frame: npt.NDArray[Any] = np.asarray(frame)
    map_ptr: Map = mapper_pyramid.calculate(
        img1=np_frame,
        img2=np_reference,
    )
    np_registered_frame: npt.NDArray[Any] = map_ptr.warp(np_frame)
    registered_frame: JaxImage = jnp.asarray(np_registered_frame)

    end = time.time()
    print(f"pyramid {mapper_name} time cost: {end - start}")

    # Metrics require they be floating point arrays
    float_reference: Float32[Array, "height width channels"] = reference.astype(jnp.float32) / 255.0
    float_frame: Float32[Array, "height width channels"] = frame.astype(jnp.float32) / 255.0
    float_registered_frame: Float32[Array, "height width channels"] = registered_frame.astype(jnp.float32) / 255.0
    for metric in [dm_pix.psnr, dm_pix.ssim]:
        print(f"{metric.__name__} pyramid {mapper_name} unreg. = {metric(float_reference, float_frame)}")
        print(f"{metric.__name__} pyramid {mapper_name} reg. = {metric(float_reference, float_registered_frame)}")

    return registered_frame
