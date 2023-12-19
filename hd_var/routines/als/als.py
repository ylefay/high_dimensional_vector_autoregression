import jax.lax
from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold, mode_unfold, ttm, nvecs
from hd_var.routines.als.losses import lossU1, lossU2, lossU3, lossU4
from hd_var.routines.als.utils import constructX
from jax.scipy.optimize import minimize
import numpy as np
import jax.numpy as jnp


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method='BFGS', options={'maxiter': 10})

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
    U1, U2, U3 = jnp.asarray(U1), jnp.asarray(U2), jnp.asarray(U3)
    A = jnp.asarray(A)
    x_ts = jnp.asarray(x_ts)
    X_ts = jnp.asarray(X_ts)
    G_flattened_mode1 = jnp.asarray(G_flattened_mode1)

    while criterion(A, prev_A, iter):
        prev_A = A
        iter += 1
        U1, l1 = minimize_matrix_input(lambda _U1: lossU1(y_ts, x_ts, X_ts, _U1, U2, U3, G_flattened_mode1), U1)
        U2, l2 = minimize_matrix_input(lambda _U2: lossU2(y_ts, x_ts, X_ts, U1, _U2, U3, G_flattened_mode1), U2)
        U3, l3 = minimize_matrix_input(lambda _U3: lossU3(y_ts, x_ts, X_ts, U1, U2, _U3, G_flattened_mode1), U3)
        G_flattened_mode1, l4 = minimize_matrix_input(
            lambda _G_flattened_mode1: lossU4(y_ts, x_ts, X_ts, U1, U2, U3, _G_flattened_mode1),
            G_flattened_mode1)
        G = mode_unfold(G_flattened_mode1, 0, G_shape)
        A = ttm(G, (U1, U2, U3))
        print(f'iter:{iter}, l1:{l1}, l2:{l2}, l3:{l3}, l4:{l4}')
        print(f'A:{A}')

    Us = [None for _ in range(3)]
    for j in range(3):
        Us[j] = np.array(nvecs(A, j, ranks[j]))
    G = ttm(A, Us, transp=True)
    return G, A, Us
