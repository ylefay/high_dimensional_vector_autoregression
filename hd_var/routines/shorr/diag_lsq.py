import jax.numpy as jnp


def diag_lsq(y, X):
    """
    Solve the diagonal least square problem
    minimize over diagonal D,
        ||y - X D||_F^2
    The solution is
        D_ii = (y^T X)_ii / (X^T X)_ii
    """
    return jnp.diag(jnp.einsum('li,li->i', y, X) / jnp.einsum('li,li->i', X, X))
