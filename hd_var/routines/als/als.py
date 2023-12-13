from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, rank_tensor, ttm
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4, constructX
from scipy.optimize import minimize
import numpy as np


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), options={'maxiter': 1})

    return minimization.x.reshape(shape), minimization.fun


def compute_A(G, Us):
    U1, U2, U3 = Us
    return ttm(ttm(ttm(G, U1, mode=0), U2, mode=1), U3, mode=2)


def als_compute(A_init, ranks, y_ts, criterion):
    A = A_init
    prev_A = np.zeros_like(A)
    Us, G = hosvd(A, ranks)
    U1, U2, U3 = Us
    P = U3.shape[1]
    X_ts = constructX(y_ts, P)
    x_ts = np.moveaxis(X_ts.T, -1, 0)
    G_shape = G.shape
    iter = 0
    G_flattened_mode1 = mode_fold(G, 0)
    while criterion(A, prev_A, iter):
        prev_A = A
        iter += 1
        U1, _ = minimize_matrix_input(lambda _U1: lossU1(y_ts, x_ts, _U1, U2, U3, G_flattened_mode1), U1)
        U2, _ = minimize_matrix_input(lambda _U2: lossU2(y_ts, X_ts, U1, _U2, U3, G_flattened_mode1), U2)
        U3, _ = minimize_matrix_input(lambda _U3: lossU3(y_ts, X_ts, U1, U2, _U3, G_flattened_mode1), U3)
        G_flattened_mode1, _ = minimize_matrix_input(
            lambda _G_flattened_mode1: lossU4(y_ts, x_ts, U1, U2, U3, _G_flattened_mode1),
            G_flattened_mode1)
        G = mode_unfold(G_flattened_mode1, 0, G_shape)
        A = compute_A(G, (U1, U2, U3))
    Us = [None for _ in range(3)]
    for i in range(3):
        A_i = mode_fold(A, i)
        _, _, Vh = np.linalg.svd(A_i, compute_uv=True)
        Us[i] = Vh[:, :ranks[i]]
    return G, A, Us[0].T, Us[1].T, Us[2].T


T = 12
P = 6
N = 4
A_init = np.random.normal(size=(N, N, P))
A_init[:, 0] = A_init[:, 1]
ranks = rank_tensor(A_init)
print(ranks)
y_ts = np.random.normal(size=(N, T))
criterion = lambda A, prev_A, iter: iter < 5 or np.linalg.norm(A - prev_A) < 1e-2
als_compute(A_init, ranks, y_ts, criterion)
