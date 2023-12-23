import jax.numpy as jnp
import jax
from hd_var.routines.shorr.splitting_orthogonal_constraint import orthogonal_QP
from hd_var.operations import unvec


def subroutine(y, X, B, pen_l, pen_k=1, max_iter=5):
    """
    ADMM subroutine for sparse and orthogonal regression as described
    in Algorithm 3.
    min_{B} {1 / n ||y - X vec(B)||_2^2 + lambda ||B||_1}, s.t. B^T B = I,
    where y = [y_1^T, ..., y_n^T] in R^n
    """
    n = y.shape[0]
    dim = B.shape[0] * B.shape[1]
    vecB = (jnp.linalg.inv(X.T @ X) @ (X.T @ y))
    init_W = unvec(vecB, B.shape)
    init_M = jnp.zeros_like(B)
    gamma = 1
    Gram = jnp.linalg.inv(X.T @ X / n + pen_k * jnp.eye(dim) + gamma * jnp.eye(dim))
    Xyprod = X.T @ y / n

    def criterion(inps):
        n_iter, reg, W, _ = inps
        return ((n_iter < max_iter) & (
                jnp.linalg.norm(reg.reshape(B.shape) - W) / jnp.linalg.norm(W) > 1e-3)) | n_iter == 0

    def criterion_for_orthogonal_iter(inps):
        n_iter, B, Q, _ = inps
        return ((n_iter < max_iter) & (
                jnp.linalg.norm(B - Q) / jnp.linalg.norm(Q) > 1e-3)) | n_iter == 0

    def iter_fun(inps):
        """
        Ref: A Splitting Method for Orthogonality Constrained Problems
        SOC Algorithm 2. for convex optimization under orthogonal constraint.
        """
        n_iter, reg, W, M = inps
        gamma = 1
        Q = reg.reshape(B.shape)
        Z = jnp.zeros(B.shape)
        WmM = pen_k * (W - M).reshape(dim, )

        def orthogonal_iter_fun(inps):
            n_iter, _, Q, Z = inps
            _B = (Gram @ (Xyprod + WmM + gamma * (Q - Z).reshape(dim, ))).reshape(B.shape)
            Q = orthogonal_QP(_B + Z)
            Z = Z + _B - Q
            return n_iter + 1, _B, Q, Z

        inps = (0, Q, Q, Z)
        _, new_B, *_ = jax.lax.while_loop(criterion_for_orthogonal_iter, orthogonal_iter_fun, inps)

        # explicit soft thresholding
        new_W = ((new_B + M - 2 * pen_l / pen_k) > 0) * (W + M - 2 * pen_l / pen_k) - (
                (-new_B - M - 2 * pen_l / pen_k) > 0) * (-new_B - M - (2 * pen_l) / pen_k)

        new_M = M + new_B - new_W
        return n_iter + 1, reg, new_W, new_M

    inps = (0, vecB, init_W, init_M)
    _, _, W, _ = jax.lax.while_loop(criterion, iter_fun, inps)
    return W
