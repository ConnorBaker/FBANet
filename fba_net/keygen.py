from collections.abc import Iterator

from jax import random as jrandom
from jaxtyping import Array, Key


def keygen(seed: int) -> Iterator[Key[Array, ""]]:
    key: Key[Array, ""] = jrandom.key(seed)
    key, key_to_yield = jrandom.split(key)
    while True:
        yield key_to_yield
        key, key_to_yield = jrandom.split(key)


KEYS = keygen(0)
