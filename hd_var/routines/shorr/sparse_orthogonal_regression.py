import jax.numpy as jnp
import jax
from hd_var.routines.shorr.splitting_orthogonal_constraint import soc
from hd_var.operations import vec
from hd_var.utils import minimize_matrix_input
from functools import partial


def subroutine(y, X, B, pen_l, pen_k, T, max_iter=5):
    """
    ADMM subroutine for sparse and orthogonal regression as described
    in Algorithm 3.
    min_{B} {1 / T ||y - X vec(B)||_2^2 + lambda ||B||_1}, s.t. B^T B = I,
    where y = [y_1^T, ..., y_n^T] in R^(n x m)
    In our case, we have n = T * N
    """

    soc_p = partial(soc, T=T, y=y, X=X, pen_k=pen_k)

    def criterion(inps):
        n_iter, B, old_B, _, _ = inps
        return (n_iter < max_iter) & (jnp.linalg.norm(B - old_B, ord='fro') / jnp.linalg.norm(old_B, ord='fro') > 1e-3)

    def iter_fun(inps):
        n_iter, B, _, W, M = inps
        new_B = soc_p(B=B, W=W, M=M)
        new_W, _ = minimize_matrix_input(
            lambda _W: pen_l * jnp.linalg.norm(_W, ord=1) + 2 * pen_k * jnp.trace((-_W) @ (new_B + M).T),
            W)  # according to the paper, we should use explicit soft-tresholding
        new_M = M + new_B - new_W
        return n_iter + 1, new_B, B, new_W, new_M

    """inps = (0, B, jnp.zeros_like(B), B, jnp.zeros_like(B))
    while (criterion(inps)):
        inps = iter_fun(inps)
    _, B, *_ = inps"""
    _, B, *_ = jax.lax.while_loop(criterion, iter_fun, (0, B, jnp.zeros_like(B), B, jnp.zeros_like(B)))
    return B
