import equinox as eqx
import jax
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from fba_net.assert_shape import assert_shape
from fba_net.keygen import KEYS


class LinearProjectionLayer(eqx.Module, strict=True):
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
        self, x: Float[Array, "n d"]
    ) -> tuple[Float[Array, "h n d"], Float[Array, "h n d"], Float[Array, "h n d"]]:
        """
        n: number of elements
        h: number of heads (`self.heads`)
        d: dimension (`self.dim`)
        """
        assert_shape((None, self.dim), x)
        q = rearrange(jax.vmap(self.to_q)(x), "n (h d) -> h n d", h=self.heads)
        assert_shape((self.heads, None, self.dim_head), q)
        k, v = rearrange(jax.vmap(self.to_kv)(x), "n (hd h c) -> hd h n c", hd=2, h=self.heads)
        assert_shape((self.heads, None, self.dim_head), k)
        assert_shape((self.heads, None, self.dim_head), v)
        return q, k, v
