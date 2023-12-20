import jax.numpy as jnp
from hd_var.operations import vec, mode_fold


def loss(y, X, B):
    """
    General loss function for U1, U2, U3, G
    minimize over orthogonal B,
    1 / n ||y - X vec(B)||_2^2
    """
    T = y.shape[0]
    return 1 / T * jnp.linalg.norm(y - X @ vec(B), ord=2) ** 2


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
    return loss(y, factor_U2(r2, X_ts, U1, U3, G_flattened_mode1), U2)


def loss_U3(U3, y, r3, U1, U2, X_ts, G_flattened_mode1):
    return loss(y, factor_U3(r3, U1, U2, X_ts, G_flattened_mode1), U3)


def loss_G_mode1(G_flattened_mode1, y, T, N, x_ts_bis, U1, U2, U3):
    return loss(y, factor_G_mode1(T, N, x_ts_bis, U1, U2, U3), G_flattened_mode1)


def penalized_loss_G_mode1(G_tensor, y, T, N, x_ts_bis, Us, Ds, Vs, Cs_flattened, rhos):
    G_flattened_mode1 = mode_fold(G_tensor, 0)
    G_flattened_mode2 = mode_fold(G_tensor, 1)
    G_flattened_mode3 = mode_fold(G_tensor, 2)
    non_penalized_loss = loss_G_mode1(G_flattened_mode1, y, T, N, x_ts_bis, *Us)
    rhos1, rho2, rho3 = rhos
    C1_flattened_mode1, C2_flattened_mode2, C3_flattened_mode3 = Cs_flattened
    D1, D2, D3 = Ds
    V1, V2, V3 = Vs
    penalization = rhos1 * jnp.linalg.norm(G_flattened_mode1 - D1 @ V1.T + C1_flattened_mode1, ord='fro') ** 2 + \
                   rho2 * jnp.linalg.norm(G_flattened_mode2 - D2 @ V2.T + C2_flattened_mode2, ord='fro') ** 2 + \
                   rho3 * jnp.linalg.norm(G_flattened_mode3 - D3 @ V3.T + C3_flattened_mode3, ord='fro') ** 2
    return non_penalized_loss + penalization
