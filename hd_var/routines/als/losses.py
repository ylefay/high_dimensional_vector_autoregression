import numpy as np


def lossU1(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU1(y_t, x_t):
        y_t = y_t.T
        _ = y_t - np.kron((x_t.reshape(-1) @ np.kron(U3, U2) @ G_flattened_mode1.T),
                          np.eye(U1.shape[0])) @ U1.reshape(-1, )
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU1, signature="(m),(n,o)->()")(y_ts.T, x_ts))


def constructX(y_ts, order):
    T = y_ts.shape[1]

    def lag(y, i, order):
        y = y[:, 1 + i:order + 1 + i]
        diff = order - y.shape[1]
        left_to_pad = diff if diff > 0 else 0
        return np.pad(y, ((0, 0), (0, left_to_pad)), 'constant')

    X_ts = np.vectorize(lag, signature="(m,n),(),()->(o,p)")(y_ts, np.arange(T), order)
    return X_ts


def constructx(y_ts, order):
    return np.moveaxis(constructX(y_ts, order).T, -1, 0)


def lossU2(y_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU2(y_t, X_t):
        _ = y_t - U1 @ G_flattened_mode1 @ np.kron((X_t @ U3).T, np.eye(U2.shape[1])) @ (U2.T).reshape(-1, )
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU2, signature="(m),(n,o)->()")(y_ts.T, X_ts))


def lossU3(y_ts, X_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU3(y_t, X_t):
        _ = y_t - U1 @ G_flattened_mode1 @ np.kron(np.eye(U3.shape[1]), (U2.T @ X_t)) @ U3.reshape(-1, )
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU3, signature="(m),(n,o)->()")(y_ts.T, X_ts))


def lossU4(y_ts, x_ts, U1, U2, U3, G_flattened_mode1):
    def _lossU4(y_t, x_t):
        _ = y_t - np.kron((np.kron(U3, U2).T @ x_t.reshape(-1, )).T, U1) @ G_flattened_mode1.reshape(-1, )
        norms = np.sum(_ ** 2, axis=-1)
        return norms

    return np.sum(np.vectorize(_lossU4, signature="(m),(n,o)->()")(y_ts.T, x_ts))
