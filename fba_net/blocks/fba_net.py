from collections.abc import Callable, Sequence
from typing import Literal, final

import equinox as eqx
from equinox import field, nn
from jax import nn as jnn
from jaxtyping import Array, Float

from fba_net.layers.fba_net import FBANetLayer


@final
class FBANetBlock(eqx.Module, strict=True):
    # Input attributes
    dim: int
    input_resolution: tuple[int, int]
    depth: int
    heads: int
    window_length: int = 8
    mlp_ratio: float = 4.0
    use_qkv_bias: bool = True
    qk_scale: None | float = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float | Sequence[float] = 0.0
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jnn.gelu
    normalization: None | Callable[[Float[Array, "..."]], Float[Array, "..."]] = None
    token_projection: Literal["linear", "linear_concat", "conv"] = "linear"
    token_mlp: Literal["ffn", "leff"] = "ffn"
    use_se_layer: bool = False

    # Computed attributes
    body: nn.Sequential = field(init=False)

    def __post_init__(self) -> None:
        self.body = nn.Sequential(
            [
                nn.Lambda(
                    FBANetLayer(
                        dim=self.dim,
                        input_resolution=self.input_resolution,
                        heads=self.heads,
                        window_height=self.window_length,
                        window_width=self.window_length,
                        shift_size_height=0 if (i % 2 == 0) else self.window_length // 2,
                        shift_size_width=0 if (i % 2 == 0) else self.window_length // 2,
                        mlp_ratio=self.mlp_ratio,
                        use_qkv_bias=self.use_qkv_bias,
                        qk_scale=self.qk_scale,
                        drop_rate=self.drop_rate,
                        attn_drop_rate=self.attn_drop_rate,
                        drop_path_rate=self.drop_path_rate[i]
                        if isinstance(self.drop_path_rate, list)
                        else self.drop_path_rate,  # type: ignore
                        activation=self.activation,
                        normalization=self.normalization,
                        token_projection=self.token_projection,
                        token_mlp=self.token_mlp,
                        use_se_layer=self.use_se_layer,
                    )
                )
                for i in range(self.depth)
            ]
        )

    def __call__(self, x: Float[Array, "height width channels"]) -> Float[Array, "height width channels"]:
        return self.body(x)
