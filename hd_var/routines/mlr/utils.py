import numpy as np


def constructX(y_ts, order):
    """
    Construct the tensor X = [X_0, ..., X_T],
    where X_t =  [y_{t-1}, ..., y_{t-p}]
    """
    T = y_ts.shape[1]

    def lag(y, i, order):
        y = y[:, i - order:i][:, ::-1]
        diff = order - y.shape[1]
        if diff > 0:
            return np.pad(y, ((0, 0), (0, diff)), 'constant')
        return y

    X_ts = np.vectorize(lag, signature="(m,n),(),()->(o,p)")(y_ts, np.arange(T), order)
    return X_ts


def constructx(y_ts, order):
    """
    Construct the tensor x = [x_0, ..., x_T],
    where x_t = [y_{t-1}^T, ..., y_{t-p}^T]^T.
    """
    return np.moveaxis(constructX(y_ts, order).T, -1, 0)
