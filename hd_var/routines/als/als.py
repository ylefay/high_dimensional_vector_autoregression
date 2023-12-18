from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, ttm, rank_tensor, compute_A
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4, constructX
from scipy.optimize import minimize
import numpy as np


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method=None, options={'maxiter': 1})

    return minimization.x.reshape(shape), minimization.fun


def als_compute(A_init, ranks, y_ts, criterion):
    A = A_init
    prev_A = np.zeros_like(A)
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[0]
    X_ts = constructX(y_ts, P)
    x_ts = np.moveaxis(X_ts.T, -1, 0)
    G_shape = G.shape
    iter = 0
    G_flattened_mode1 = mode_fold(G, 0)
    while criterion(A, prev_A, iter):
        prev_A = A
        iter += 1
        U1, l1 = minimize_matrix_input(lambda _U1: lossU1(y_ts, x_ts, _U1, U2, U3, G_flattened_mode1), U1)
        U2, l2 = minimize_matrix_input(lambda _U2: lossU2(y_ts, X_ts, U1, _U2, U3, G_flattened_mode1), U2)
        U3, l3 = minimize_matrix_input(lambda _U3: lossU3(y_ts, X_ts, U1, U2, _U3, G_flattened_mode1), U3)
        G_flattened_mode1, l4 = minimize_matrix_input(
            lambda _G_flattened_mode1: lossU4(y_ts, x_ts, U1, U2, U3, _G_flattened_mode1),
            G_flattened_mode1)
        G = mode_unfold(G_flattened_mode1, 0, G_shape)
        A = compute_A(G, (U1, U2, U3))
        print(f'iter:{iter}, l1:{l1}, l2:{l2}, l3:{l3}, l4:{l4}')
    Us = [None for _ in range(3)]
    for i in range(3):
        A_i = mode_fold(A, i)
        _, _, Vh = np.linalg.svd(A_i, compute_uv=True)
        Us[i] = Vh[:, :ranks[i]]
    return G, A, Us[0].T, Us[1].T, Us[2].T, (l1, l2, l3, l4)
