import jax.lax
from hd_var.hosvd import hosvd
from hd_var.utils import minimize_matrix_input
from hd_var.operations import mode_unfold, mode_fold, fast_ttm
from hd_var.routines.als.utils import constructX
import jax.numpy as jnp
from functools import partial
import hd_var.routines.shorr.sparse_orthogonal_regression as sor
import hd_var.routines.shorr.losses as losses
from hd_var.routines.shorr.diagonal_least_square import diag_lsq
from hd_var.routines.shorr.splitting_orthogonal_constraint import orthogonal_QP


def criterion(inps):
    prev_A, A, iter, _, _, _, _ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) / jnp.linalg.norm(prev_A) > 1e-2)


def admm_compute(A_init, ranks, y_ts, pen_l, pen_k=1, rhos=(1.0, 1.0, 1.0), criterion=criterion, iter_sor=10):
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
    T = y_ts.shape[1]

    mode_unfold_p = partial(mode_unfold, mode=0, shape=G_shape)
    subroutine = partial(sor.subroutine, y=y_ts.T, pen_k=pen_k, max_iter=iter_sor)
    fun_factor_U1 = partial(losses.factor_U1, T=T, N=N, x_ts_bis=x_ts_bis)
    fun_factor_U2 = partial(losses.factor_U2, r2=ranks[1], X_ts=X_ts)
    fun_factor_U3 = partial(losses.factor_U3, r3=ranks[2], X_ts=X_ts)
    loss_for_G = partial(losses.penalized_loss_G_mode1, y=y_ts.T, T=T, N=N, x_ts_bis=x_ts_bis, rhos=rhos)

    def iter_fun(inps):
        A, prev_A, iter, Us, G, Ds, Vs, Cs_flattened = inps
        U1, U2, U3 = Us
        iter += 1
        G_flattened_mode1 = mode_fold(G, 0)
        factor_U1 = fun_factor_U1(U2=U2, U3=U3, G_flattened_mode1=G_flattened_mode1)
        U1 = subroutine(B=U1, X=factor_U1, pen_l=pen_l * jnp.linalg.norm(U2, ord=1) * jnp.linalg.norm(U3, ord=1))
        factor_U2 = fun_factor_U2(U1=U1, U3=U3, G_flattened_mode1=G_flattened_mode1)
        U2 = subroutine(B=U2.T, X=factor_U2,
                        pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U3, ord=1)).T
        factor_U3 = fun_factor_U3(U1=U1, U2=U2, G_flattened_mode1=G_flattened_mode1)
        U3 = subroutine(B=U3, X=factor_U3, pen_l=pen_l * jnp.linalg.norm(U1, ord=1) * jnp.linalg.norm(U2, ord=1))
        G, _ = minimize_matrix_input(
            lambda G_tensor: loss_for_G(G_tensor=G_tensor, rhos=rhos, Ds=Ds, Us=Us, Vs=Vs, Cs_flattened=Cs_flattened),
            G)
        G_flattened_mode1 = mode_fold(G, 0, G_shape)
        G_flattened_mode2 = mode_fold(G, 1, G_shape)
        G_flattened_mode3 = mode_fold(G, 2, G_shape)
        V1, V2, V3 = Vs
        D1, D2, D3 = Ds
        C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3 = Cs_flattened
        D1 = diag_lsq((G_flattened_mode1 + C1_flattened_mode1).T, V1)
        V1 = ((1 / D1) @ orthogonal_QP(G_flattened_mode1 + C1_flattened_mode1)).T
        C1_flattened_mode1 = C1_flattened_mode1 + G_flattened_mode1 - D1 @ V1.T
        D2 = diag_lsq((G_flattened_mode2 + C2_flattened_mode2).T, V2)
        V2 = ((1 / D2) @ orthogonal_QP(G_flattened_mode2 + C2_flattened_mode2)).T
        C2_flattened_mode2 = C2_flattened_mode2 + G_flattened_mode2 - D2 @ V2.T
        V3 = ((1 / D3) @ orthogonal_QP(G_flattened_mode3 + C3_flattened_mode3)).T
        C3_flattened_mode3 = C3_flattened_mode3 + G_flattened_mode3 - D3 @ V3.T
        D3 = diag_lsq((G_flattened_mode3 + C3_flattened_mode3).T, V3)
        Us = (U1, U2, U3)
        Vs = (V1, V2, V3)
        Ds = (D1, D2, D3)
        Cs_flattened = (C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3)
        A = fast_ttm(G, Us)
        return A, prev_A, iter, Us, G, Ds, Vs, Cs_flattened

    Ds = (jnp.eye(ranks[0]), jnp.eye(ranks[1]), jnp.eye(ranks[2]))
    Vs = (None, None, None)
    Cs_flattened = (jnp.zeros((ranks[0], ranks[1] * ranks[2])), jnp.zeros((ranks[1], ranks[0] * ranks[2])),
                    jnp.zeros((ranks[2], ranks[0] * ranks[1])))
    A, _, _, _, _, _, _ = jax.lax.while_loop(criterion, iter_fun, (A, A, 0, Us, G, Ds, Vs, Cs_flattened))
    Us, G = hosvd(A, ranks)
    return G, A, Us
