import jax.numpy as jnp
import jax
from hd_var.routines.shorr.splitting_orthogonal_constraint import soc
from hd_var.operations import vec
from hd_var.utils import minimize_matrix_input


def subroutine(y, X, B, pen_l, pen_k, max_iter=10):
    """
    ADMM subroutine for sparse and orthogonal regression as described
    in Algorithm 3.
    min_{B} {1 / n ||y - X vec(B)||_2^2 + lambda ||B||_1}, s.t. B^T B = I,
    where y = [y_1^T, ..., y_n^T] in R^(n x m)
    """

    T = y.shape[0]

    def criterion(inps):
        iter, B, old_B, _, _ = inps
        return (iter <= max_iter) & (jnp.linalg.norm(B - old_B, ord='fro') / jnp.linalg.norm(old_B, ord='fro') > 1e-3)

    def J_base(B):
        return 1 / T * jnp.linalg.norm(y - X @ vec(B), ord=2) ** 2

    def iter_fun(inps):
        iter, B, _, W, M = inps
        new_B = soc(lambda _B: J_base(_B) + pen_k * jnp.linalg.norm(_B - W + M, ord='fro') ** 2, B)
        new_W, _ = minimize_matrix_input(
            lambda _W: pen_l * jnp.linalg.norm(_W, ord=1) + pen_k * jnp.linalg.norm(-_W - new_B + M, ord='fro') ** 2,
            W)  # according to the paper, we should use explicit soft-tresholding
        new_M = M + new_B - new_W
        return iter + 1, new_B, B, new_W, new_M

    _, B, _, _, _ = jax.lax.while_loop(criterion, iter_fun, (0, B, B, B, jnp.zeros_like(B)))
    return B
