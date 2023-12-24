import jax.numpy as jnp
from hd_var.operations import vec


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
