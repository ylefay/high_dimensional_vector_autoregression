from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4
from scipy.optimize import minimize
import numpy as np


def als_compute(A_init, ranks, y_ts, x_ts, criterion):
    G, U1, U2, U3 = hosvd(A_init, ranks)
    G_shape = G.shape
    iter = 0
    G_flattened_mode1 = mode_fold(G, 1)
    while criterion(G, U1, U2, U3, iter):
        iter += 1
        program_U1 = minimize(lambda _U1: lossU1(y_ts, x_ts, _U1, U2, U3, G_flattened_mode1), U1)
        U1 = program_U1.x
        program_U2 = minimize(lambda _U2: lossU2(y_ts, x_ts, U1, _U2, U3, G_flattened_mode1), U2)
        U2 = program_U2.x
        program_U3 = minimize(lambda _U3: lossU3(y_ts, x_ts, U1, U2, _U3, G_flattened_mode1), U3)
        U3 = program_U3.x
        program_G = minimize(lambda _G_flattened_mode1: lossU4(y_ts, x_ts, U1, U2, U3, _G_flattened_mode1),
                             G_flattened_mode1)
        G_flattened_mode1 = program_G.x
    G = mode_unfold(G_flattened_mode1, 1, G_shape)
    Us = [None for _ in range(3)]
    for j in range(3):
        G_i = mode_fold(G, j)
        _, _, Vh = np.linalg.svd(G_i, compute_uv=True)
        Us[j] = Vh[:, :ranks[j]]
    return G,
