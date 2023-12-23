import jax.lax
import jax.numpy as jnp
from functools import partial
from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, fast_ttm, unvec
from hd_var.routines.mlr.utils import constructX
from hd_var.utils import minimize_matrix_input
import hd_var.routines.mlr.losses as losses
import hd_var.routines.shorr.losses as shorr_losses


def criterion(inps):
    A, prev_A, iter, *_ = inps
    return (iter < 1000) & (jnp.linalg.norm(A - prev_A) / jnp.linalg.norm(prev_A) > 1e-3)


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

    inps = (A, jnp.zeros_like(A), n_iter, U1, U2, U3, G_flattened_mode1)
    A, *_ = jax.lax.while_loop(criterion, iter_fun,
                               inps)
    Us, G = hosvd(A, ranks)
    return G, A, Us


def als_compute_closed_form(A_init, ranks, y_ts, criterion=criterion):
    """
    Alternative Least Square algorithm for VAR model, using HOSVD decomposition.
    Algorithm 1.
    Author: Yvann Le Fay
    """
    A = A_init
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[0]
    N, T = y_ts.shape
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)
    y_ts_reshaped = y_ts.T.reshape((-1))
    G_shape = G.shape
    n_iter = 0
    G_flattened_mode1 = mode_fold(G, 0)

    mode_unfold_p = partial(mode_unfold, mode=0, shape=G_shape)
    fun_factor_U1 = partial(shorr_losses.factor_U1, T=T, N=N, x_ts_bis=x_ts_bis)
    fun_factor_U2 = partial(shorr_losses.factor_U2, r2=ranks[1], X_ts=X_ts)
    fun_factor_U3 = partial(shorr_losses.factor_U3, r3=ranks[2], X_ts=X_ts)
    fun_factor_G = partial(shorr_losses.factor_G_mode1, T=T, N=N, x_ts_bis=x_ts_bis)

    def iter_fun(inps):
        A, prev_A, n_iter, U1, U2, U3, G_flattened_mode1 = inps
        prev_A = A

        factor_U1 = fun_factor_U1(U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U1 = factor_U1.reshape((-1, factor_U1.shape[-1]))
        vU1 = (jnp.linalg.pinv(factor_U1.T @ factor_U1) @ factor_U1.T @ y_ts_reshaped)
        U1 = unvec(vU1, U1.shape)

        factor_U2 = fun_factor_U2(U1=U1, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U2 = factor_U2.reshape((-1, factor_U2.shape[-1]))
        vU2T = (jnp.linalg.pinv(factor_U2.T @ factor_U2) @ factor_U2.T @ y_ts_reshaped)
        U2 = unvec(vU2T, (U2.shape[1], U2.shape[0])).T

        factor_U3 = fun_factor_U3(U1=U1, U2=U2, G_flattened_mode1=G_flattened_mode1)
        factor_U3 = factor_U3.reshape((-1, factor_U3.shape[-1]))
        vU3 = (jnp.linalg.pinv(factor_U3.T @ factor_U3) @ factor_U3.T @ y_ts_reshaped)
        U3 = unvec(vU3, U3.shape)

        factor_G_mode1 = fun_factor_G(U1=U1, U2=U2, U3=U3)
        factor_G_mode1 = factor_G_mode1.reshape((-1, factor_G_mode1.shape[-1]))
        vG_flattened_mode1 = (
                jnp.linalg.pinv(factor_G_mode1.T @ factor_G_mode1) @ factor_G_mode1.T @ y_ts_reshaped)
        G_flattened_mode1 = unvec(vG_flattened_mode1, G_flattened_mode1.shape)

        G = mode_unfold_p(G_flattened_mode1)

        A = fast_ttm(G, (U1, U2, U3))
        return A, prev_A, n_iter + 1, U1, U2, U3, G_flattened_mode1

    inps = (A, jnp.zeros_like(A), n_iter, U1, U2, U3, G_flattened_mode1)
    while criterion(inps):
        inps = iter_fun(inps)
    A, *_ = inps
    # A, *_ = jax.lax.while_loop(criterion, iter_fun,inps)
    Us, G = hosvd(A, ranks)
    return G, A, Us
