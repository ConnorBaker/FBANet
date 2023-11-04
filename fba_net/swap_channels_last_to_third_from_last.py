from jaxtyping import Array, Shaped

from .assert_shape import assert_shape


def swap_channels_last_to_third_from_last(x: Shaped[Array, "... h w c"]) -> Shaped[Array, "... h w c"]:
    assert x.ndim >= 3

    # assuming input n-dimensional array a with n > 2
    new_shape = list(x.shape)
    perm = list(range(len(x.shape)))

    # switch last and third to last dimensions
    new_shape[-1], new_shape[-3] = new_shape[-3], new_shape[-1]
    perm[-1], perm[-3] = perm[-3], perm[-1]

    # reorder dimensions based on perm
    x_reordered = x.transpose(perm)
    assert_shape(new_shape, x_reordered)

    return x_reordered
