import jax.numpy as jnp


def mode_flatten(tensor, mode):
    """
    Flatten a tensor along its k-th mode.
    """
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def product(tensorX, matrixY, mode):
    """
    Compute the product of two tensors along their k-th mode.
    """
    return (matrixY @ mode_flatten(tensorX, mode)).reshape((matrixY.shape[0], *tensorX.shape[1:]))
