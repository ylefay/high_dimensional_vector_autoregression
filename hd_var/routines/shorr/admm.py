from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, fast_ttm, unvec, mode_unfold
from hd_var.routines.mlr.utils import constructX
from hd_var.routines.shorr.penalization import lambda_optimal
import hd_var.routines.shorr.sparse_orthogonal_regression as sor
from hd_var.routines.shorr.diag_lsq import diag_lsq
import hd_var.routines.shorr.losses as losses
from functools import partial
import jax.numpy as jnp
import jax.lax


def criterion(inps):
    A, prev_A, n_iter, *_ = inps
    return (n_iter < 1000) & (jnp.linalg.norm(A - prev_A) / jnp.linalg.norm(prev_A) > 1e-9)


def _admm_compute(A_init, ranks, y_ts, pen_l=None, pen_k=1.0, criterion=criterion, iter_sor=100):
    """
    See Algorithm 2. in the paper.
    Compute the SHORR estimate.
    """
    # Computing the initial HOSVD decomposition
    A = A_init
    Us, G = hosvd(A, ranks)
    G_shape = G.shape
    U1, U2, U3 = Us
    Us = (U1, U2, U3)
    P = U3.shape[0]
    N = U1.shape[0]
    T = y_ts.shape[1]
    # Creating the lagged tensors.
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)
    y_ts_reshaped = y_ts.T.reshape((-1,))
    if pen_l is None:
        pen_l = lambda_optimal(N, P, T, jnp.eye(N))  # assuming unit covariance

    pen_l *= 1 / N  # We reshape y_ts_reshaped to be of shape (T*N,1) instead of (T,N), thus the factor in the OLS loss is 1/(T * N) instead of 1/T

    subroutine = partial(sor.subroutine, y=y_ts_reshaped, max_iter=iter_sor)
    fun_factor_U1 = partial(losses.factor_U1, T=T, N=N, x_ts_bis=x_ts_bis)
    fun_factor_U2 = partial(losses.factor_U2, r2=ranks[1], X_ts=X_ts)
    fun_factor_U3 = partial(losses.factor_U3, r3=ranks[2], X_ts=X_ts)
    fun_factor_G_mode1 = partial(losses.factor_G_mode1, T=T, N=N, x_ts_bis=x_ts_bis)

    unvec_p = partial(unvec, shape=(G_shape[0], G_shape[1] * G_shape[2]))
    mode_unfold_p = partial(mode_unfold, mode=0, shape=G_shape)

    def iter_fun(inps):
        A, prev_A, n_iter, Us, G, Ds, Vs, Cs_flattened, pen_k = inps
        prev_A = A
        U1, U2, U3 = Us
        G_flattened_mode1 = mode_fold(G, 0)

        factor_U1 = fun_factor_U1(U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U1 = factor_U1.reshape((-1, factor_U1.shape[-1]))
        U1 = subroutine(B=U1, X=factor_U1,
                        pen_l=pen_l * jnp.linalg.norm(U2, ord=1) * jnp.linalg.norm(U3, ord=1),
                        pen_k=pen_k)
        factor_U2 = fun_factor_U2(U1=U1, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U2 = factor_U2.reshape((-1, factor_U2.shape[-1]))
        U2 = subroutine(B=U2.T, X=factor_U2,
                        pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U3, ord=1), pen_k=pen_k).T
        factor_U3 = fun_factor_U3(U1=U1, U2=U2, G_flattened_mode1=G_flattened_mode1)
        factor_U3 = factor_U3.reshape((-1, factor_U3.shape[-1]))
        U3 = subroutine(B=U3, X=factor_U3,
                        pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U2, ord=1),
                        pen_k=pen_k)

        factor_G_mode1 = fun_factor_G_mode1(U1=U1, U2=U2, U3=U3)
        factor_G_mode1 = factor_G_mode1.reshape((-1, factor_G_mode1.shape[-1]))
        vecG_mode1 = jnp.linalg.pinv(factor_G_mode1.T @ factor_G_mode1) @ factor_G_mode1.T @ y_ts_reshaped
        G_mode1 = unvec_p(vecG_mode1)
        G = mode_unfold_p(G_mode1)
        G_flattened_mode2 = mode_fold(G, 1)
        G_flattened_mode3 = mode_fold(G, 2)
        V1, V2, V3 = Vs
        D1, D2, D3 = Ds
        C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3 = Cs_flattened

        D1 = diag_lsq((G_flattened_mode1 + C1_flattened_mode1).T, V1)
        V1 = sor.unbalanced_procruste((G_flattened_mode1 + C1_flattened_mode1).T, D1)
        C1_flattened_mode1 = C1_flattened_mode1 + G_flattened_mode1 - D1 @ V1.T
        D2 = diag_lsq((G_flattened_mode2 + C2_flattened_mode2).T, V2)
        V2 = sor.unbalanced_procruste((G_flattened_mode2 + C2_flattened_mode2).T, D2)
        C2_flattened_mode2 = C2_flattened_mode2 + G_flattened_mode2 - D2 @ V2.T
        V3 = sor.unbalanced_procruste((G_flattened_mode3 + C3_flattened_mode3).T, D3)
        C3_flattened_mode3 = C3_flattened_mode3 + G_flattened_mode3 - D3 @ V3.T
        D3 = diag_lsq((G_flattened_mode3 + C3_flattened_mode3).T, V3)

        Us = (U1, U2, U3)
        Vs = (V1, V2, V3)
        Ds = (D1, D2, D3)
        Cs_flattened = (C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3)

        A = fast_ttm(G, Us)
        pen_k *= jax.lax.cond(n_iter < 64, lambda _: 1.0, lambda x: 1.0, None)
        return A, prev_A, n_iter + 1, Us, G, Ds, Vs, Cs_flattened, pen_k

    Ds = (jnp.eye(ranks[0]), jnp.eye(ranks[1]), jnp.eye(ranks[2]))
    Vs = (jnp.eye(ranks[1] * ranks[2], ranks[0]), jnp.eye(ranks[0] * ranks[2], ranks[1]),
          jnp.eye(ranks[0] * ranks[1], ranks[2]))
    Cs_flattened = (jnp.ones((ranks[0], ranks[1] * ranks[2])), jnp.ones((ranks[1], ranks[0] * ranks[2])),
                    jnp.ones((ranks[2], ranks[0] * ranks[1])))
    inps = (A, jnp.zeros_like(A), 0, Us, G, Ds, Vs, Cs_flattened, pen_k)
    A, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    Us, G = hosvd(A, ranks)
    return G, A, Us


def admm_compute(A_init, ranks, y_ts, pen_l=None, pen_k=1.0, criterion=criterion, iter_sor=100):
    """
    See Algorithm 2. in the paper.
    Compute the SHORR estimate.
    """
    # Computing the initial HOSVD decomposition
    A = A_init
    Us, G = hosvd(A, ranks)
    G_shape = G.shape
    U1, U2, U3 = Us
    Us = (U1, U2, U3)
    P = U3.shape[0]
    N = U1.shape[0]
    T = y_ts.shape[1]
    # Creating the lagged tensors.
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)
    y_ts_reshaped = y_ts.T.reshape((-1,))
    if pen_l is None:
        pen_l = lambda_optimal(N, P, T, jnp.eye(N))  # assuming unit covariance

    pen_l *= 1 / N  # We reshape y_ts_reshaped to be of shape (T*N,1) instead of (T,N), thus the factor in the OLS loss is 1/(T * N) instead of 1/T

    subroutine = partial(sor.subroutine, y=y_ts_reshaped, max_iter=iter_sor)
    fun_factor_U1 = partial(losses.factor_U1, T=T, N=N, x_ts_bis=x_ts_bis)
    fun_factor_U2 = partial(losses.factor_U2, r2=ranks[1], X_ts=X_ts)
    fun_factor_U3 = partial(losses.factor_U3, r3=ranks[2], X_ts=X_ts)
    fun_factor_G_mode1 = partial(losses.factor_G_mode1, T=T, N=N, x_ts_bis=x_ts_bis)

    unvec_p = partial(unvec, shape=(G_shape[0], G_shape[1] * G_shape[2]))
    mode_unfold_p = partial(mode_unfold, mode=0, shape=G_shape)

    def iter_fun(inps):
        A, prev_A, n_iter, Us, G, pen_k = inps
        prev_A = A
        U1, U2, U3 = Us
        G_flattened_mode1 = mode_fold(G, 0)

        factor_U1 = fun_factor_U1(U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U1 = factor_U1.reshape((-1, factor_U1.shape[-1]))
        U1 = subroutine(B=U1, X=factor_U1,
                        pen_l=pen_l * jnp.linalg.norm(U2, ord=1) * jnp.linalg.norm(U3, ord=1),
                        pen_k=pen_k)
        factor_U2 = fun_factor_U2(U1=U1, U3=U3, G_flattened_mode1=G_flattened_mode1)
        factor_U2 = factor_U2.reshape((-1, factor_U2.shape[-1]))
        U2 = subroutine(B=U2.T, X=factor_U2,
                        pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U3, ord=1), pen_k=pen_k).T
        factor_U3 = fun_factor_U3(U1=U1, U2=U2, G_flattened_mode1=G_flattened_mode1)
        factor_U3 = factor_U3.reshape((-1, factor_U3.shape[-1]))
        U3 = subroutine(B=U3, X=factor_U3,
                        pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U2, ord=1),
                        pen_k=pen_k)

        factor_G_mode1 = fun_factor_G_mode1(U1=U1, U2=U2, U3=U3)
        factor_G_mode1 = factor_G_mode1.reshape((-1, factor_G_mode1.shape[-1]))
        vecG_mode1 = jnp.linalg.pinv(factor_G_mode1.T @ factor_G_mode1) @ factor_G_mode1.T @ y_ts_reshaped
        G_mode1 = unvec_p(vecG_mode1)
        G = mode_unfold_p(G_mode1)
        Us = (U1, U2, U3)
        A = fast_ttm(G, Us)
        pen_k *= jax.lax.cond(n_iter < 64, lambda _: 1.0, lambda x: 1.0, None)
        return A, prev_A, n_iter + 1, Us, G, pen_k

    inps = (A, jnp.zeros_like(A), 0, Us, G, pen_k)
    A, *_ = jax.lax.while_loop(criterion, iter_fun, inps)
    Us, G = hosvd(A, ranks)
    return G, A, Us
