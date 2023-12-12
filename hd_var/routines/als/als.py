from hd_var import hosvd
from .losses import lossU1, lossU2, lossU3, lossU4
from scipy.optimize import minimize
from hd_var.operations import mode_fold, mode_unfold


def als_compute(A_init, y_ts, x_ts, criterion):
    G, U1, U2, U3 = hosvd(A_init)
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
