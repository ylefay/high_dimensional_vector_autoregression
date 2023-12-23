import jax.numpy as jnp

from hd_var.operations import vec, mode_fold, mode_unfold
from hd_var.routines.mlr.losses import lossU4


def loss(y, X, B):
    """
    General loss function L for U1, U2, U3, G
    """
    return jnp.mean(jnp.linalg.norm(y - X @ vec(B), ord=2, axis=-1) ** 2)


def factor_U1(T, N, x_ts_bis, U2, U3, G_flattened_mode1):
    id_N = jnp.eye(N)
    _ = jnp.kron(x_ts_bis @ jnp.kron(U3, U2) @ G_flattened_mode1.T, id_N)
    factor_U1 = _.reshape((T, N, _.shape[1]))
    return factor_U1


def factor_U2(r2, X_ts, U1, U3, G_flattened_mode1):
    id_r2 = jnp.eye(r2)
    _ = jnp.kron(jnp.moveaxis(X_ts @ U3, -1, 1), id_r2)
    factor_U2 = jnp.einsum('ij,tjl->til', U1 @ G_flattened_mode1, _)
    return factor_U2


def factor_U3(r3, X_ts, U1, U2, G_flattened_mode1):
    id_r3 = jnp.eye(r3)
    _ = jnp.kron(id_r3, U2.T @ X_ts)
    factor_U3 = jnp.einsum('ij,tjl->til', U1 @ G_flattened_mode1, _)
    return factor_U3


def factor_G_mode1(T, N, x_ts_bis, U1, U2, U3):
    _ = jnp.kron((jnp.kron(U3, U2).T @ x_ts_bis.T).T, U1)
    factor_G_mode1 = _.reshape(T, N, _.shape[1])
    return factor_G_mode1


# Implementing the same losses as for ALS but without vmap for the time dimension

def loss_U1(U1, y, T, N, x_ts_bis, U2, U3, G_flattened_mode1):
    return loss(y, factor_U1(T, N, x_ts_bis, U2, U3, G_flattened_mode1), U1)


def loss_U2(U2, y, r2, X_ts, U1, U3, G_flattened_mode1):
    return loss(y, factor_U2(r2, X_ts, U1, U3, G_flattened_mode1), U2.T)


def loss_U3(U3, y, r3, U1, U2, X_ts, G_flattened_mode1):
    return loss(y, factor_U3(r3, X_ts, U1, U2, G_flattened_mode1), U3)


def loss_G_mode1(G_flattened_mode1, y, T, N, x_ts_bis, U1, U2, U3):
    return loss(y, factor_G_mode1(T, N, x_ts_bis, U1, U2, U3), G_flattened_mode1)


def penalized_loss_G(G_flattened_mode1, y_ts, x_ts, Us, Ds, Vs, Cs_flattened, rhos, ranks):
    """
    ||G_(i)||_F^2 does not depend on i, hence:
    pen = rho_1||G_(1)-D_1 V_1^T||_F^2 + rho_2||G_(2)-D_2 V_2^T||_F^2 + rho_3||G_(3)-D_3 V_3^T||_F^2
    = (rho_1+rho_2+rho_3) ||G_(1)||_F^2 + (rho_1 Tr(G_(2) A_1) + rho_2 Tr(G_(2)A_2) + rho_3 Tr(G_(3)A_3))
    + cst
    with A_1 = -2(D_1V_1^T)^T = -2V_1D_1, A_2 = -2V_2D_2, A_3 = -2V_3D_3
    """
    U1, U2, U3 = Us
    rho1, rho2, rho3 = rhos
    C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3 = Cs_flattened
    D1, D2, D3 = Ds
    V1, V2, V3 = Vs
    G = mode_unfold(G_flattened_mode1, 0, ranks)
    G_flattened_mode2 = mode_fold(G, 1)
    G_flattened_mode3 = mode_fold(G, 2)

    non_penalized_loss = lossU4(y_ts, x_ts, None, U1, U2, U3,
                                G_flattened_mode1)  # for some reason I had to use lossU4 instead of loss_G_mode1 for optimizing (otherwise nan...)
    penalization = (rho1 + rho2 + rho3) * jnp.linalg.norm(G_flattened_mode1, ord='fro') ** 2 \
                   + 2 * (rho1 * jnp.trace(G_flattened_mode1 @ (C1_flattened_mode1.T - V1 @ D1)) + \
                          rho2 * jnp.trace(G_flattened_mode2 @ (C2_flattened_mode2.T - V2 @ D2)) + \
                          rho3 * jnp.trace(G_flattened_mode3 @ (C3_flattened_mode3.T - V3 @ D3))
                          )
    return non_penalized_loss + 0 * penalization
