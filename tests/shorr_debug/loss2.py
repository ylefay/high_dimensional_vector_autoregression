from hd_var.hosvd import hosvd
from hd_var.operations import mode_fold
from hd_var.routines.mlr.utils import constructX
import jax.numpy as jnp
import numpy as np
from hd_var.generate import generate_A_given_rank
from hd_var.operations import rank_tensor, vec
import hd_var.routines.shorr.losses as shorr_losses
import hd_var.routines.mlr.losses as mlr_losses


# Creating the lagged tensors.


def main():
    np.random.seed(0)
    dataset = np.load(f'../../scripts/toy_model/data/var_10000_2_3.npz')
    y_ts, A, E = dataset['y'], dataset['A'], dataset['E']
    N, T = y_ts.shape
    P = A.shape[-1]  # cheating
    ranks = rank_tensor(A)  # cheating
    A = generate_A_given_rank(N, P, ranks)
    X_ts = constructX(y_ts, P)
    x_ts = jnp.moveaxis(X_ts.T, -1, 0)
    x_ts_bis = x_ts.reshape(x_ts.shape[0], -1)
    Us, G = hosvd(A, ranks)
    G_flattened_mode1 = mode_fold(G, 0)
    U1, U2, U3 = Us
    Us = (U1, U2, U3)
    P = U3.shape[0]
    N = U1.shape[0]
    l2_mlr = mlr_losses.lossU2(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1)
    l3_mlr = mlr_losses.lossU3(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1)
    l4_mlr = mlr_losses.lossU4(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1)
    l1_mlr = mlr_losses.lossU1(y_ts, x_ts, X_ts, U1, U2, U3, G_flattened_mode1)
    l2_shorr = shorr_losses.loss_U2(U2, y_ts.T, ranks[1], X_ts, U1, U3, G_flattened_mode1)
    l3_shorr = shorr_losses.loss_U3(U3, y_ts.T, ranks[2], U1, U2, X_ts, G_flattened_mode1)
    l4_shorr = shorr_losses.loss_G_mode1(G_flattened_mode1, y_ts.T, T, N, x_ts_bis, U1, U2, U3)
    l1_shorr = shorr_losses.loss_U1(U1, y_ts.T, T, N, x_ts_bis, U2, U3, G_flattened_mode1)
    assert np.allclose(l2_mlr, l2_shorr)
    assert np.allclose(l3_mlr, l3_shorr)
    assert np.allclose(l4_mlr, l4_shorr)
    assert np.allclose(l1_mlr, l1_shorr)
    y_ts_reshaped = y_ts.T.reshape((-1))
    factor_U1 = shorr_losses.factor_U1(T=T, N=N, x_ts_bis=x_ts_bis, U2=U2, U3=U3,
                                       G_flattened_mode1=G_flattened_mode1)

    factor_U1 = factor_U1.reshape((-1, factor_U1.shape[-1]))
    assert jnp.linalg.norm(y_ts_reshaped - factor_U1 @ vec(U1), ord=2) ** 2 / T == l1_mlr
    factor_U2 = shorr_losses.factor_U2(r2=ranks[1], X_ts=X_ts, U1=U1, U3=U3, G_flattened_mode1=G_flattened_mode1)
    factor_U2 = factor_U2.reshape((-1, factor_U2.shape[-1]))
    assert np.isclose(jnp.linalg.norm(y_ts_reshaped - factor_U2 @ vec(U2.T), ord=2) ** 2 / T, l2_mlr)
    factor_U3 = shorr_losses.factor_U3(r3=ranks[2], X_ts=X_ts, U1=U1, U2=U2, G_flattened_mode1=G_flattened_mode1)
    factor_U3 = factor_U3.reshape((-1, factor_U3.shape[-1]))
    assert np.isclose(jnp.linalg.norm(y_ts_reshaped - factor_U3 @ vec(U3), ord=2) ** 2 / T, l3_mlr)

    # mlr
    def factor_mlr(k):
        U1atG1 = U1 @ G_flattened_mode1
        id = np.eye(U2.shape[1])
        vecU2T = vec(U2.T)
        factor_U2_mlr = U1atG1 @ np.kron((X_ts[k] @ U3).T, id)
        return factor_U2_mlr

    # shorr
    id_r2 = jnp.eye(ranks[1])
    _ = jnp.kron(jnp.moveaxis(X_ts @ U3, -1, 1), id_r2)
    factor_U2_shorr = jnp.einsum('ij,tjl->til', U1 @ G_flattened_mode1, _)
    pass


if __name__ == "__main__":
    main()
