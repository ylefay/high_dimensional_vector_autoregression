import jax.lax
import jax.numpy as jnp
from functools import partial
from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, fast_ttm
from hd_var.routines.mlr.utils import constructX
from hd_var.utils import minimize_matrix_input
import hd_var.routines.mlr.losses as losses


def criterion(inps):
    A, prev_A, iter, *_ = inps
    return (iter < 1000) & (jnp.linalg.norm(A - prev_A) / jnp.linalg.norm(prev_A) > 1e-2)


def als_compute(A_init, ranks, y_ts, criterion=criterion):
    """
    Alternative Least Square algorithm for VAR model, using HOSVD decomposition.
    Algorithm 1.
    Author: Yvann Le Fay
    """
    A = A_init
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[0]
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    G_shape = G.shape
    n_iter = 0
    G_flattened_mode1 = mode_fold(G, 0)

    mode_unfold_p = partial(mode_unfold, mode=0, shape=G_shape)
    lossU1 = partial(losses.lossU1, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts)
    lossU2 = partial(losses.lossU2, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts)
    lossU3 = partial(losses.lossU3, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts)
    lossU4 = partial(losses.lossU4, y_ts=y_ts, x_ts=x_ts, X_ts=X_ts)

    def iter_fun(inps):
        A, prev_A, n_iter, U1, U2, U3, G_flattened_mode1 = inps
        prev_A = A

        U1, l1 = minimize_matrix_input(
            lambda _U1: lossU1(U1=_U1, U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1), U1)
        U2, l2 = minimize_matrix_input(
            lambda _U2: lossU2(U1=U1, U2=_U2, U3=U3, G_flattened_mode1=G_flattened_mode1), U2)
        U3, l3 = minimize_matrix_input(
            lambda _U3: lossU3(U1=U1, U2=U2, U3=_U3, G_flattened_mode1=G_flattened_mode1), U3)
        G_flattened_mode1, l4 = minimize_matrix_input(
            lambda _G_flattened_mode1: lossU4(U1=U1, U2=U2, U3=U3, G_flattened_mode1=_G_flattened_mode1),
            G_flattened_mode1)

        G = mode_unfold_p(G_flattened_mode1)

        A = fast_ttm(G, (U1, U2, U3))
        return A, prev_A, n_iter + 1, U1, U2, U3, G_flattened_mode1

    A, *_ = jax.lax.while_loop(criterion, iter_fun,
                               (A, jnp.zeros_like(A), n_iter, U1, U2, U3, G_flattened_mode1))
    Us, G = hosvd(A, ranks)
    return G, A, Us


def als_compute_closed_form_optimization():
    raise NotImplementedError
