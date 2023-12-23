import jax.numpy as jnp
import jax
from hd_var.utils import minimize_matrix_input
from hd_var.operations import vec


def soc(B, W, M, T, y, X, pen_k, r=2.0, max_iter=1):
    """
    Reference: A Splitting Method for Orthogonality Constrained Problems
    An implementation of the Algorithm described in 2.3 (Bregman Iteration) for the minimization problem:
        B* = argmin J(B) s.t. B^T B= I
    In our case, the code is explicitly written for
    J(B) = 1 / T * jnp.linalg.norm(y - X @ vec(B), ord=2) ** 2 + pen_k * jnp.linalg.norm(B - W + M,
                                                                                                 ord='fro') ** 2
    """

    def criterion(inps):
        n_iter, B, prev_B, _ = inps
        return (n_iter < max_iter) & (
                jnp.linalg.norm(B - prev_B, ord='fro') / jnp.linalg.norm(prev_B, ord='fro') > 1e-2)

    def subproblem(B, B_ortho, A):
        """
        See A Splitting Method for Orthogonality Constrained Problems, SOC algorithm 1.
        The first unconstrained convex subproblem for the SOC method in our case is
        argmin_B J(B) + r/2 ||B-prev_B+prev_A||_F^2
        We only keep the terms that depend on B by using Trace.
        """
        return 1 / T * jnp.linalg.norm(y - X @ vec(B), ord=2) ** 2 + 2 * pen_k * jnp.trace(
            B @ (- W + M).T) + r * jnp.trace(B @ (-B_ortho + A).T)

    def bregman_iter(inps):
        n_iter, prev_B, prev_B_ortho, prev_A = inps
        B, _ = minimize_matrix_input(lambda _B: subproblem(_B, prev_B_ortho, prev_A), prev_B)
        B_ortho = orthogonal_QP(B + prev_A)
        A = prev_A + B - B_ortho
        return n_iter + 1, B, B_ortho, A

    """inps = (0, B, B, jnp.zeros_like(B))
    while (criterion(inps)):
        inps = bregman_iter(inps)
    _, _, B, _ = inps"""
    _, _, B, _ = jax.lax.while_loop(criterion, bregman_iter, (0, B, B, jnp.zeros_like(B)))
    return B


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
