from collections.abc import Iterable

from jaxtyping import Array, Shaped


def assert_shape(shape: Iterable[None | int], array: Shaped[Array, "..."]) -> None:
    """Assert that the shape of an array matches the expected shape."""
    try:
        for expected, actual in zip(shape, array.shape, strict=True):
            if expected is not None:
                assert expected == actual, f"Expected shape {shape}, got {array.shape}"
    except ValueError:
        raise AssertionError(f"Expected shape {shape}, got {array.shape}")
