import jax.numpy as jnp
import jax


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
