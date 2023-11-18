import jax.numpy as jnp


def mode_flatten(tensor, k):
    """
    Flatten a tensor along its k-th mode.
    """
    return jnp.moveaxis(tensor, k, 0).reshape(tensor.shape[k], -1)

