import jax.numpy as jnp


def diag_lsq(y, X):
    """
    Solve the diagonal least square problem
    minimize over diagonal D,
        ||y - X D||_F^2
    The solution is
        D = diag(<x_i, y_i> / ||x_i||_2^2)
    """
    return jnp.diag(jnp.sum(X * y, axis=-1) / jnp.linalg.norm(X, axis=-1, ord=2))
