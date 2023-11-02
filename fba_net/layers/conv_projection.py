from functools import partial

import equinox as eqx
from einops import rearrange
from equinox import field, nn
from jaxtyping import Array, Float

from .separable_conv2d import SepConv2dLayer


class ConvProjectionLayer(eqx.Module, strict=True, kw_only=True):
    # Input attributes
    dim: int
    heads: int = 8
    kernel_size: int = 3
    q_stride: int = 1
    k_stride: int = 1
    v_stride: int = 1
    dropout: float = 0.0
    last_stage: bool = False
    use_bias: bool = True

    # Computed attributes
    dim_head: int = field(init=False)
    inner_dim: int = field(init=False)
    padding: int = field(init=False)
    to_q: nn.Sequential = field(init=False)
    to_k: nn.Sequential = field(init=False)
    to_v: nn.Sequential = field(init=False)

    def __post_init__(self):
        self.dim_head = self.dim // self.heads
        self.inner_dim = self.dim_head * self.heads
        self.padding = (self.kernel_size - self.q_stride) // 2

        def mk_sep_conv2d_layer(stride: int) -> SepConv2dLayer:
            return SepConv2dLayer(
                in_channels=self.dim,
                out_channels=self.inner_dim,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=self.padding,
                use_bias=self.use_bias,
            )

        rearrange_fn = partial(rearrange, pattern="(h n n) d -> (h d) n n", h=self.heads)
        for letter in "qkv":
            self.__setattr__(
                f"to_{letter}",
                nn.Sequential(
                    [
                        mk_sep_conv2d_layer(self.__getattribute__(f"{letter}_stride")),
                        nn.Lambda(rearrange_fn),
                    ]
                ),
            )

    # Originally took `attn_kv` as an argument, but it's not used anywhere in the codebase
    # so for the sake of simplicity, I removed it.
    def __call__(
        self, x: Float[Array, "n*n c"]
    ) -> tuple[Float[Array, "h n*n d"], Float[Array, "h n*n d"], Float[Array, "h n*n d"]]:
        x_rearranged = rearrange(x, "(n n) c -> c n n")

        q: Float[Array, "h n*n d"] = self.to_q(x_rearranged)
        k: Float[Array, "h n*n d"] = self.to_k(x_rearranged)
        v: Float[Array, "h n*n d"] = self.to_v(x_rearranged)

        return q, k, v
