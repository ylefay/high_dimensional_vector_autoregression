from jax.scipy.optimize import minimize
import numpy as np


def minimize_matrix_input(f, init_matrix, args=()):
    """
    Wrapper around the scipy minimize function to handle matrix input.
    """
    shape = init_matrix.shape

    if args != ():
        def _f(flatten_matrix, *args):
            return f(flatten_matrix.reshape(shape), *args)
    else:
        def _f(flatten_matrix):
            return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method='BFGS', options={'maxiter': 100}, args=args)

    return minimization.x.reshape(shape), minimization.fun


def differentiate(y):
    """
    Discrete differentiation of a time series
    """
    y_diff = np.diff(y, axis=-1)
    return y_diff


def integrate_series(y_0, y_diff):
    """
    Integrate a time series
    """
    N = y_diff.shape[0]
    return np.hstack((y_0.reshape(N, 1), y_diff)).cumsum(axis=-1)


def estimate_noise_variance(y, A):
    """
    Using an estimator A, compute the noise variance
    """
    N, P = A.shape[0], A.shape[2]
    T = y.shape[1]
    y_hat = np.zeros((N, T))

    for t in range(P, T):
        for p in range(P):
            y_hat[:, t] += A[:, :, p] @ y[:, t - p - 1]
    noise = y[:, P:] - y_hat[:, P:]
    noise_variance = np.cov(noise)
    return noise_variance


def predict(y, A, futur, cov):
    """
    Sample from the VAR model
    """
    N, P = A.shape[0], A.shape[2]
    T = y.shape[1]
    y_out = np.concatenate((y, np.zeros((N, futur))), axis=1)
    noises = np.random.multivariate_normal(np.zeros(N), cov, size=futur)
    for k in range(futur):
        y_out[:, T + k] = noises[k] + np.sum(np.array([np.dot(A[:, :, i], y_out[:, T + k - 1 - i]) for i in range(P)]),
                                             axis=0)
    return y_out


def normalise_y(y, T=-1):
    """
    Normalise the time series using min, max inside a window :T
    """
    y_w = y[:, :T]
    return (y - np.min(y_w, axis=-1, keepdims=True)) / (
            np.max(y_w, axis=-1, keepdims=True) - np.min(y_w, axis=-1, keepdims=True))
