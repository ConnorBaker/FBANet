from dataclasses import InitVar

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jax import random as jrandom
from jaxtyping import Array, Float


class LinearProjectionLayer(eqx.Module, strict=True, frozen=True, kw_only=True):
    # Input attributes
    dim: int
    key: InitVar[jrandom.KeyArray]
    heads: int = 8
    dropout: float = 0.0
    use_bias: bool = True

    # Computed attributes
    dim_head: int = field(init=False)
    inner_dim: int = field(init=False)
    to_q: nn.Linear = field(init=False)
    to_kv: nn.Linear = field(init=False)

    def __post_init__(self, key: jrandom.KeyArray) -> None:
        object.__setattr__(self, "dim_head", self.dim // self.heads)
        object.__setattr__(self, "inner_dim", self.dim_head * self.heads)
        key1, key2 = jrandom.split(key)
        object.__setattr__(self, "to_q", nn.Linear(self.dim, self.inner_dim, self.use_bias, key=key1))
        object.__setattr__(self, "to_kv", nn.Linear(self.dim, self.inner_dim * 2, self.use_bias, key=key2))

    def __call__(
        self, x: Float[Array, "n c"]
    ) -> tuple[Float[Array, "n d"], Float[Array, "h n c"], Float[Array, "h n c"]]:
        q = rearrange(self.to_q(x), "n (h d) -> () h n d", h=self.heads)
        kv = rearrange(self.to_kv(x), "n (hd c) -> hd h n c", hd=2, h=self.heads)
        q = q[0]
        k, v = kv
        return q, k, v

    def flops(self, H: int, W: int) -> int:
        return H * W * self.dim * self.inner_dim * 3
