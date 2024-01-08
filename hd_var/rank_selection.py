import numpy as np
import jax.numpy as jnp

from hd_var.operations import mode_fold, unvec, vec, mode_unfold
from hd_var.routines.mlr.utils import constructx
from hd_var.utils import minimize_matrix_input


def rank_selection(A, T):
    """
    Estimated ranks according to Section 5. of the paper
    Author: Schoonaert Antoine
    """
    N, P = A.shape[1:]
    p = [N, N, P]
    c = np.sqrt((N * P * np.log(T)) / (10 * T))

    rank = -1 * np.ones(3)

    for i in range(3):
        A_i = mode_fold(A, i)
        U, S, Vh = np.linalg.svd(A_i)
        ratio_list = np.array([(S[j + 1] + c) / (S[j] + c) for j in range(p[i] - 1)])
        r_i = np.argmin(ratio_list) + 1
        rank[i] = r_i

    return rank


def NN_compute(y, P, lamb, A_init=None):
    """
    Nuclear-Norm estimator using scipy.optimize.minimize
    Author: Schoonaert Antoine, Yvann Le Fay
    """
    N, T = y.shape
    x_ts = constructx(y, P)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)

    shape = (N, N, P)

    if A_init is None:
        A1 = y @ x_ts_bis @ jnp.linalg.pinv(x_ts_bis.T @ x_ts_bis)  # OLS
    elif A_init == "random":
        A1 = np.random.randn(N, N * P)
    else:
        A1 = mode_fold(A_init, 0)

    def loss(A1):
        out = jnp.mean(jnp.linalg.norm(y - A1 @ x_ts_bis.T, ord=2, axis=0) ** 2) + lamb * jnp.linalg.norm(A1, 'nuc')
        return out

    A_nn1, val = minimize_matrix_input(loss, A1)  # not efficient, see for an implementation of the NN estimator
    A_nn = mode_unfold(A_nn1, 0, shape)
    return A_nn, val
