import jax.numpy as jnp
import jax
from hd_var.routines.shorr.splitting_orthogonal_constraint import orthogonal_QP
from hd_var.operations import unvec, vec


def subroutine(y, X, B, pen_l, pen_k=1, max_iter=100):
    """
    ADMM subroutine for sparse and orthogonal regression as described
    in Algorithm 3.
    min_{B} {1 / n ||y - X vec(B)||_2^2 + lambda ||B||_1}, s.t. B^T B = I,
    where y = [y_1^T, ..., y_n^T] in R^n
    """
    n = y.shape[0]
    dim = B.shape[0] * B.shape[1]
    reg = jnp.linalg.pinv(X.T @ X) @ (X.T @ y)
    init_W = unvec(reg, B.shape)
    init_M = jnp.zeros_like(B)
    gamma = 1
    Gram = jnp.linalg.pinv(X.T @ X / n + pen_k * jnp.eye(dim) + gamma * jnp.eye(dim))
    Xyprod = X.T @ y / n

    def criterion(inps):
        n_iter, reg, W, _ = inps
        return ((n_iter < max_iter) & (
                jnp.linalg.norm(unvec(reg, B.shape) - W) / jnp.linalg.norm(W) > 1e-2)) + (
                       n_iter == 0)

    def criterion_for_orthogonal_iter(inps):
        n_iter, reg, Q, _ = inps
        return ((n_iter < max_iter) & (
                jnp.linalg.norm(unvec(reg, B.shape) - Q) / jnp.linalg.norm(Q) > 1e-2)) + (
                       n_iter == 0)

    def iter_fun(inps):
        """
        Ref: A Splitting Method for Orthogonality Constrained Problems
        SOC Algorithm 2. for convex optimization under orthogonal constraint.
        """
        n_iter, reg, W, M = inps
        gamma = 1
        Q = unvec(reg, B.shape)
        Z = jnp.zeros(B.shape)
        WmM = pen_k * vec(W - M)

        def orthogonal_iter_fun(inps):
            n_iter, reg, Q, Z = inps
            reg = Gram @ (Xyprod + WmM + gamma * vec(Q - Z))
            _B = unvec(reg, B.shape)
            Q = orthogonal_QP(_B + Z)
            Z = Z + _B - Q
            return n_iter + 1, reg, Q, Z

        inps = (0, reg, Q, Z)
        _, _, Q, _ = jax.lax.while_loop(criterion_for_orthogonal_iter, orthogonal_iter_fun, inps)
        # explicit soft thresholding
        new_W = (Q + M - (2 * pen_l) / pen_k > 0) * (Q + M - (2 * pen_l) / pen_k) - (
                -Q - M - (2 * pen_l) / pen_k > 0) * (-Q - M - (2 * pen_l) / pen_k)

        new_M = M + Q - new_W
        return n_iter + 1, vec(Q), new_W, new_M

    inps = (0, reg, init_W, init_M)
    _, _, W, _ = jax.lax.while_loop(criterion, iter_fun, inps)
    return W
