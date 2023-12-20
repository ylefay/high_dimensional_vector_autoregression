import jax.lax
from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, fast_ttm
from hd_var.operations import vec
from hd_var.routines.als.utils import constructX
import jax.numpy as jnp
from functools import partial
from hd_var.routines.shorr.sparse_orthogonal_regression import subroutine


@jax.jit
def criterion(inps):
    prev_A, A, iter, _, _, _, _ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) / jnp.linalg.norm(prev_A) > 1e-2)


def admm_compute(A_init, ranks, y_ts, pen_l, pen_k=1, criterion=criterion, iter_sor=10):
    # Computing the initial HOSVD decomposition
    A = A_init
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[0]
    N = U1.shape[0]
    # Creating the lagged tensors.
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)
    # Computing useful quantities
    G_shape = G.shape
    G_flattened_mode1 = mode_fold(G, 0)
    id_N = jnp.eye(N)
    id_r2 = jnp.eye(ranks[1])
    id_r3 = jnp.eye(ranks[2])
    T = y_ts.shape[1]
    jitted_mode_unfold = jax.jit(partial(mode_unfold, mode=0, shape=G_shape))
    jitted_subroutine = jax.jit(partial(subroutine, y=y_ts.T, pen_k=pen_k, max_iter=iter_sor))

    def iter_fun(inps):
        A, prev_A, iter, U1, U2, U3, G_flattened_mode1 = inps
        iter += 1
        # Implementing the same losses as for ALS but without vmap for the timedimension
        _ = jnp.kron(x_ts_bis @ jnp.kron(U3, U2) @ G_flattened_mode1.T, id_N)
        factor_U1 = _.reshape((T, N, _.shape[1]))
        U1 = jitted_subroutine(B=U1, X=factor_U1, pen_l=pen_l * jnp.linalg.norm(U2, ord=1) * jnp.linalg.norm(U3, ord=1))
        _ = jnp.kron(jnp.moveaxis(X_ts @ U3, -1, 1), id_r2)
        factor_U2 = jnp.einsum('ij,tjl->til', U1 @ G_flattened_mode1, _)
        U2 = jitted_subroutine(B=U2.T, X=factor_U2,
                               pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U3, ord=1)).T
        _ = jnp.kron(id_r3, U2.T @ X_ts)
        factor_U3 = jnp.einsum('ij,tjl->til', U1 @ G_flattened_mode1, _)
        U3 = jitted_subroutine(B=U3, X=factor_U3, pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U2, ord=1))
        _ = jnp.kron((jnp.kron(U3, U2).T @ x_ts_bis.T).T, U1)
        factor_G_mode1 = _.reshape(T, N, _.shape[1])
        G = jitted_mode_unfold(G_flattened_mode1)
        A = fast_ttm(G, (U1, U2, U3))
        return A, prev_A, iter, U1, U2, U3, G_flattened_mode1

    A, _, _, _, _, _, _ = jax.lax.while_loop(criterion, iter_fun, (A, A, 0, U1, U2, U3, G_flattened_mode1))
    Us, G = hosvd(A, ranks)
    return G, A, Us
