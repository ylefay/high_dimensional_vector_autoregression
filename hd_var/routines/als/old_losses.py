import numpy as np


def lossU1(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    kronU3U2atG1T = np.kron(U3, U2) @ G_flattened_mode1.T
    id = np.eye(U1.shape[0])
    vecU1 = U1.reshape(-1, )

    def _lossU1(y_t, x_t):
        _ = y_t - np.kron((x_t.T.reshape(-1) @ kronU3U2atG1T),
                          id) @ vecU1
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(np.vectorize(_lossU1, signature="(m),(n,o)->()")(y_ts.T, x_ts))


def lossU2(y_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    U1atG1 = U1 @ G_flattened_mode1
    id = np.eye(U2.shape[1])
    vecU2T = (U2.T).reshape(-1, )

    def _lossU2(y_t, X_t):
        _ = y_t - U1atG1 @ np.kron((X_t @ U3).T, id) @ vecU2T
        norms = np.mean(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU2, signature="(m),(n,o)->()")(y_ts.T, X_ts))


def lossU3(y_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    U1atG1 = U1 @ G_flattened_mode1
    id = np.eye(U3.shape[1])
    vecU3 = U3.reshape(-1, )

    def _lossU3(y_t, X_t):
        _ = y_t - U1atG1 @ np.kron(id, (U2.T @ X_t)) @ vecU3
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(np.vectorize(_lossU3, signature="(m),(n,o)->()")(y_ts.T, X_ts))


def lossU4(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    kronU3U2T = np.kron(U3, U2).T
    vecG_flattened_mode1 = G_flattened_mode1.reshape(-1, )

    def _lossU4(y_t, x_t):
        _ = y_t - np.kron((kronU3U2T @ x_t.reshape(-1, )).T, U1) @ vecG_flattened_mode1
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.mean(np.vectorize(_lossU4, signature="(m),(n,o)->()")(y_ts.T, x_ts))
