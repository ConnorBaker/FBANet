import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from fba_net.keygen import KEYS


class LinearProjectionLayer(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    dim: int
    heads: int = 8
    dropout: float = 0.0
    use_bias: bool = True

    # Computed attributes
    dim_head: int = field(init=False)
    inner_dim: int = field(init=False)
    to_q: nn.Linear = field(init=False)
    to_kv: nn.Linear = field(init=False)

    def __post_init__(self) -> None:
        self.dim_head = self.dim // self.heads
        self.inner_dim = self.dim_head * self.heads
        self.to_q = nn.Linear(self.dim, self.inner_dim, self.use_bias, key=next(KEYS))
        self.to_kv = nn.Linear(self.dim, self.inner_dim * 2, self.use_bias, key=next(KEYS))

    def __call__(
        self, x: Float[Array, "n c"]
    ) -> tuple[Float[Array, "n d"], Float[Array, "h n c"], Float[Array, "h n c"]]:
        q = rearrange(self.to_q(x), "n (h d) -> h n d", h=self.heads)
        kv = rearrange(self.to_kv(x), "n (hd c) -> hd h n c", hd=2, h=self.heads)
        k, v = kv
        return q, k, v
