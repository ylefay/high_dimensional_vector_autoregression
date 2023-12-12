import numpy as np


def lossU1(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU1(y_t, x_t):
        _ = y_t - np.kron((x_t.T @ np.kron(U3, U2) @ G_flattened_mode1.T), np.eyes(U1.shape[0])) @ U1.reshape(-1, 1)
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU1)(y_ts, x_ts))


def constructX(y_ts, order):
    T = y_ts.shape[1]

    def lag(y, i, order):
        return np.pad(y[:, 1 + i:order + 1 + i], ((0, 0), (0, i)), 'constant')

    X_ts = np.vstack([lag(y_ts, i, order) for i in range(T)])
    return X_ts


def lossU2(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    P = U3.shape[1]

    def _lossU2(y_t, X_t):
        _ = y_t - U1 @ G_flattened_mode1 @ np.kron((X_t @ U3).T, np.eye(U2.shape[1])) @ (U2.T).reshape(-1, 1)
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU2)(y_ts, constructX(y_ts, P)))


def lossU3(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    P = U3.shape[0]

    def _lossU3(y_t, X_t):
        _ = y_t - U1 @ G_flattened_mode1 @ np.kron(np.eye(U3.shape[1]), (U2.T @ X_t)) @ U3.reshape(-1, 1)
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU3)(y_ts, constructX(y_ts, P)))


def lossU4(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU4(y_t, x_t):
        _ = y_t - np.kron((np.kron(U3, U2).T @ x_t).T, U1) @ G_flattened_mode1.reshape(-1, 1)
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU4)(y_ts, x_ts))