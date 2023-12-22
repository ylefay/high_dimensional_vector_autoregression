import jax.numpy as jnp
import jax
from hd_var.utils import minimize_matrix_input


def soc(J, X, r=2.0, max_iter=1):
    """
    Reference: A Splitting Method for Orthogonality Constrained Problems
    An implementation of the Algorithm described in 2.3 (Bregman Iteration) for the minimization problem:
        X* = argmin J(X) s.t. X^T X = I
    """

    def criterion(inps):
        n_iter, X, prev_X, _ = inps
        return (n_iter < max_iter) & (
                jnp.linalg.norm(X - prev_X, ord='fro') / jnp.linalg.norm(prev_X, ord='fro') > 1e-2)

    def first_convex_subproblem(X):
        return J(X) + r / 2 * jnp.linalg.norm(X, ord='fro') ** 2

    def bregman_iter(inps):
        n_iter, X, prev_X, prev_B = inps
        new_X, _ = minimize_matrix_input(first_convex_subproblem, X)
        new_X2 = orthogonal_QP(new_X + prev_B)
        new_B = prev_B + new_X - new_X2
        return n_iter + 1, new_X2, new_X, new_B

    """inps = (0, X, jnp.zeros_like(X), jnp.zeros_like(X))
    while (criterion(inps)):
        inps = bregman_iter(inps)
    _, X, *_ = inps"""
    _, X, *_ = jax.lax.while_loop(criterion, bregman_iter, (0, X, jnp.zeros_like(X), jnp.zeros_like(X)))
    return X


def orthogonal_QP(Y):
    """
    Ref: A Splitting Method for Orthogonality Constrained Problems.
    See Eq. 19.
        P* = argmin_{P in R^{n x m}} {||Y - P||_F^2} s.t. P^T P = I
    """
    n, m = Y.shape

    def full_rank(Y):
        D, V = jnp.linalg.eigh(Y.T @ Y)  # ici
        Pstar = Y @ V @ jnp.diag(1 / jnp.sqrt(D)) @ V.T
        return Pstar

    def otherwise(Y):
        U, D, V = jnp.linalg.svd(Y)
        Pstar = U @ jnp.eye(U.shape[1], V.shape[0]) @ V.T
        return Pstar

    Pstar = jax.lax.cond(jnp.linalg.matrix_rank(Y) == m,
                         full_rank,
                         otherwise,
                         Y
                         )

    return Pstar


def unbalanced_procruste(X, L):
    """
    Solution of:
        argmin_{P} ||X-PL||^2_F s.t. P^TP = I,
    with P in R^{n x m} and L in R^{m x d}
    X in R^{n x d}
    """
    return orthogonal_QP(X @ L.T)
