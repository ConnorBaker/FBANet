# https://github.com/paganpasta/eqxvision/blob/7da790c7bfae76a3631a269b67c4846c329f1fa5/eqxvision/layers/drop_path.py
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array


class DropPath(eqx.Module, strict=True, frozen=True, kw_only=True):
    """Effectively dropping a sample from the call.
    Often used inside a network along side a residual connection.
    Equivalent to `torchvision.stochastic_depth`."""

    p: float = 0.0
    """
    The probability to drop a sample entirely during forward pass.
    Defaults to `0.0`.
    """

    inference: bool = False
    """
    Defaults to `False`. If `True`, then the input is returned unchanged.
    This may be toggled with `equinox.tree_inference`.
    """

    mode: str = "global"
    """
    Can be set to `global` or `local`. If `global`, the whole input is dropped or retained.
    If `local`, then the decision on each input unit is computed independently. Defaults to `global`.

    !!! note

            For `mode = local`, an input `(channels, dim_0, dim_1, ...)` is reshaped and transposed to
            `(channels, dims).transpose()`. For each `dim x channels` element,
            the decision to drop/keep is made independently.
    """

    def __call__(self, x, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to drop
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        """
        if self.inference or self.p == 0.0:
            return x
        if key is None:
            raise RuntimeError(
                "DropPath requires a key when running in non-deterministic mode. Did you mean to enable inference?"
            )

        keep_prob = 1 - self.p
        if self.mode == "global":
            noise = jrandom.bernoulli(key, p=keep_prob)
        else:
            noise = jnp.expand_dims(
                jrandom.bernoulli(key, p=keep_prob, shape=[x.shape[0]]).reshape(-1),
                axis=[i for i in range(1, len(x.shape))],
            )
        if keep_prob > 0.0:
            noise /= keep_prob
        return x * noise
