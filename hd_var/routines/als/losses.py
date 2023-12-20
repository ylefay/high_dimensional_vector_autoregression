import jax.numpy as np
import jax
from hd_var.operations import vec

"""
Defining the losses involved in the ALS algorithm, Alg. 1.
Closed form solution exists for the unconsrained optimization problems.
"""


def lossU1(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    kronU3U2atG1T = np.kron(U3, U2) @ G_flattened_mode1.T
    id = np.eye(U1.shape[0])
    vecU1 = vec(U1)

    def _lossU1(y_t, x_t):
        _ = y_t - np.kron((x_t.reshape(-1) @ kronU3U2atG1T),
                          id) @ vecU1
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(jax.vmap(_lossU1, in_axes=(0, 0))(y_ts.T, x_ts))


def lossU2(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    U1atG1 = U1 @ G_flattened_mode1
    id = np.eye(U2.shape[1])
    vecU2T = vec(U2.T)

    def _lossU2(y_t, X_t):
        _ = y_t - U1atG1 @ np.kron((X_t @ U3).T, id) @ vecU2T
        norms = np.mean(_ ** 2, axis=-1)
        return norms

    return np.mean(jax.vmap(_lossU2, in_axes=(0, 0))(y_ts.T, X_ts))


def lossU3(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    U1atG1 = U1 @ G_flattened_mode1
    id = np.eye(U3.shape[1])
    vecU3 = vec(U3)

    def _lossU3(y_t, X_t):
        _ = y_t - U1atG1 @ np.kron(id, (U2.T @ X_t)) @ vecU3
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(jax.vmap(_lossU3, in_axes=(0, 0))(y_ts.T, X_ts))


def lossU4(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    kronU3U2T = np.kron(U3, U2).T
    vecG_flattened_mode1 = vec(G_flattened_mode1)

    def _lossU4(y_t, x_t):
        _ = y_t - np.kron((kronU3U2T @ x_t.reshape(-1, )).T, U1) @ vecG_flattened_mode1
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(jax.vmap(_lossU4, in_axes=(0, 0))(y_ts.T, x_ts))
