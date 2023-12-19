import jax.numpy as jnp
import jax
from hd_var.utils import minimize_matrix_input


def soc(J, r=1.0, max_iter=10):
    """
    Reference: A Splitting Method for Orthogonality Constrained Problems
    An implementation of the Algorithm described in 2.3 (Bregman Iteration) for the minimization problem:
        min J(X) s.t. X^T X = I
    """

    def criterion(inps):
        iter, X, prev_X, _ = inps
        return jnp.linalg.norm(X - prev_X, norm='fro') / jnp.linalg.norm(prev_X, norm='fro') > 1e-3 & iter <= max_iter

    def bregman_iter(inps):
        iter, X, prev_X, prev_B = inps
        new_X = minimize_matrix_input(lambda _X: J(_X) + r / 2 * jnp.linalg.norm(_X - prev_X + prev_B, norm='fro'), X)
        new_X2 = orthogonal_QP(new_X + prev_B)
        new_B = prev_B + new_X - new_X2
        return (iter + 1, new_X2, new_X, new_B)

    X, _, _ = jax.lax.while_loop(criterion, bregman_iter, (0, J, J, jnp.zeros_like(J)))
    return X


def orthogonal_QP(Y):
    """
    See Eq. 19.
    P* = argmin_{P in R^{n x m}} {||Y - P||_F^2} s.t. P^T P = I
    """
    n, m = Y.shape
    if jnp.linalg.rank(Y) == m:
        D, V = jnp.linalg.eigh(Y.T @ Y)  # ici
        Pstar = Y @ V @ jnp.diag(1 / jnp.sqrt(D)) @ V.T
    else:
        U, D, V = jnp.linalg.svd(Y, compute_uv=True)
        Pstar = U @ D @ V.T
    return Pstar
