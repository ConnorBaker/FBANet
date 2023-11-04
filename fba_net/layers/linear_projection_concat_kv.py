import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from fba_net.keygen import KEYS


class LinearProjectionConcatKVLayer(eqx.Module, strict=True):
    # Input attributes
    dim: int
    heads: int = 8
    dropout: float = 0.0
    use_bias: bool = True

    # Computed attributes
    dim_head: int = field(init=False)
    inner_dim: int = field(init=False)
    to_qkv: nn.Linear = field(init=False)
    to_kv: nn.Linear = field(init=False)

    def __post_init__(self) -> None:
        self.dim_head = self.dim // self.heads
        self.inner_dim = self.dim_head * self.heads
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, self.use_bias, key=next(KEYS))
        self.to_kv = nn.Linear(self.dim, self.inner_dim * 2, self.use_bias, key=next(KEYS))

    def __call__(
        self, x: Float[Array, "n c"]
    ) -> tuple[Float[Array, "h n d"], Float[Array, "h n 2*d"], Float[Array, "h n 2*d"]]:
        # hdc: Query, Key, Value for decoder; h: Attention heads; d: Feature dimensions per head
        qkv_dec = rearrange(self.to_qkv(x), "n (hdc d) -> hdc h n d", hdc=3, h=self.heads)
        # hd: Key, Value for encoder; h: Attention heads; d: Feature dimensions per head
        kv_enc = rearrange(self.to_kv(x), "n (hd d) -> hd h n d", hd=2, h=self.heads)
        # Splitting query, key, value for decoder
        q, k_d, v_d = qkv_dec
        # Splitting key, value for encoder
        k_e, v_e = kv_enc
        # Concatenating keys from decoder and encoder along the sequence length dimension
        k = rearrange([k_d, k_e], "(kv h n d) -> h n (kv d)", kv=2)
        v = rearrange([v_d, v_e], "(kv h n d) -> h n (kv d)", kv=2)
        return q, k, v
