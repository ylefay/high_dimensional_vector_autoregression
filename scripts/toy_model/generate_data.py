import numpy as np
from hd_var.generate import generate
from hd_var.assumptions import check_ass2, check_ass1


def main(check=False):
    T = 10000  # Length of the time series
    sigma = 0.1  # Variance of the innovations, assuming diagonal noise
    A = np.array(
        [[[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]], [[0.1, 0.05, -0.05], [0.6, -0.2, 0.1]]])  # Coefficients of the VAR model
    # A = np.array([[[0.5], [0.25]], [[0.1], [0.6]]])  # 2,1 WORKS
    # A = np.array([[[0.5, 0.0], [0.25, 0.0]], [[0.1, 0.0], [0.6, -0.2]]])  # 2, 2 WORKS
    if check:
        # check_ass1(A)
        check_ass2(A)
    N, P = A.shape[1:]
    cov = np.eye(N, ) * sigma  # Covariance matrix of the innovations
    X, A, E = generate(A, T, P, N, cov)
    np.savez(f'./data/var_{T}_{N}_{P}.npz', X=X, A=A, E=E)


if __name__ == '__main__':
    main(check=True)
