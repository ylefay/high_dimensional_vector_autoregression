import jax.lax
from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, fast_ttm
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4
from hd_var.routines.als.utils import constructX
from jax.scipy.optimize import minimize
import jax.numpy as jnp
from functools import partial


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method='BFGS', options={'maxiter': 5})

    return minimization.x.reshape(shape), minimization.fun


def als_compute(A_init, ranks, y_ts, criterion):
    A = A_init
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[0]
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    G_shape = G.shape
    iter = 0
    G_flattened_mode1 = mode_fold(G, 0)

    jitted_mode_unfold = jax.jit(partial(mode_unfold, mode=0, shape=G_shape))
    jitted_loss1 = jax.jit(partial(lossU1, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts))
    jitted_loss2 = jax.jit(partial(lossU2, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts))
    jitted_loss3 = jax.jit(partial(lossU3, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts))
    jitted_loss4 = jax.jit(partial(lossU4, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts))

    def iter_fun(inps):
        A, prev_A, iter, U1, U2, U3, G_flattened_mode1 = inps
        prev_A = A
        iter += 1
        U1, l1 = minimize_matrix_input(
            lambda _U1: jitted_loss1(U1=_U1, U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1), U1)
        U2, l2 = minimize_matrix_input(
            lambda _U2: jitted_loss2(U1=U1, U2=_U2, U3=U3, G_flattened_mode1=G_flattened_mode1), U2)
        U3, l3 = minimize_matrix_input(
            lambda _U3: jitted_loss3(U1=U1, U2=U2, U3=_U3, G_flattened_mode1=G_flattened_mode1), U3)
        G_flattened_mode1, l4 = minimize_matrix_input(
            lambda _G_flattened_mode1: jitted_loss4(U1=U1, U2=U2, U3=U3, G_flattened_mode1=_G_flattened_mode1),
            G_flattened_mode1)
        G = jitted_mode_unfold(G_flattened_mode1)
        A = fast_ttm(G, (U1, U2, U3))
        return A, prev_A, iter, U1, U2, U3, G_flattened_mode1

    A, _, _, _, _, _, _ = jax.lax.while_loop(criterion, iter_fun,
                                             (A, jnp.zeros_like(A), iter, U1, U2, U3, G_flattened_mode1))
    Us, G = hosvd(A, ranks)
    return G, A, Us
