from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4, constructX
from scipy.optimize import minimize
import numpy as np


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten())

    return minimization.x.reshape(shape)


def als_compute(A_init, ranks, y_ts, criterion):
    Us, G = hosvd(A_init, ranks)
    U1, U2, U3 = Us
    P = U3.shape[1]
    X_ts = constructX(y_ts, P)
    x_ts = X_ts.T
    G_shape = G.shape
    iter = 0
    G_flattened_mode1 = mode_fold(G, 1)
    while criterion(G, U1, U2, U3, iter):
        iter += 1
        program_U1 = minimize_matrix_input(lambda _U1: lossU1(y_ts, x_ts, _U1, U2, U3, G_flattened_mode1), U1)
        U1 = program_U1.x
        program_U2 = minimize_matrix_input(lambda _U2: lossU2(y_ts, X_ts, U1, _U2, U3, G_flattened_mode1), U2)
        U2 = program_U2.x
        program_U3 = minimize_matrix_input(lambda _U3: lossU3(y_ts, X_ts, U1, U2, _U3, G_flattened_mode1), U3)
        U3 = program_U3.x
        program_G = minimize_matrix_input(lambda _G_flattened_mode1: lossU4(y_ts, x_ts, U1, U2, U3, _G_flattened_mode1),
                                          G_flattened_mode1)
        G_flattened_mode1 = program_G.x
    G = mode_unfold(G_flattened_mode1, 1, G_shape)
    Us = [None for _ in range(3)]
    for j in range(3):
        G_i = mode_fold(G, j)
        _, _, Vh = np.linalg.svd(G_i, compute_uv=True)
        Us[j] = Vh[:, :ranks[j]]
    return G,


A_init = np.random.normal(size=(2, 5, 12))
ranks = [1, 5, 10]
y_ts = np.random.normal(size=(2, 10))
x_ts = np.random.normal(size=(5, 10))
criterion = lambda G, U1, U2, U3, iter: iter < 100
als_compute(A_init, ranks, y_ts, criterion)
