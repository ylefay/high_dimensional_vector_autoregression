import numpy as np
from hd_var.generate import generate, generate_A_given_rank
from hd_var.assumptions import check_ass2, check_ass1


def main(check=False):
    T = 260  # Length of the time series
    sigma = 1.0  # Variance of the innovations, assuming diagonal noise
    A = np.array([[[0.5, 0.1, 0.0], [0.25, 0.0, 0.2]], [[0.1, 0.1, -0.05], [0.6, -0.2, 0.1]]])  # 2, 3 WORKS
    A = np.array(
        [[[0.5, 0.1, 0.0, 0.3], [0.25, 0.0, 0.2, -0.6]],
         [[0.1, 0.1, -0.05, 0.25], [0.6, -0.2, 0.1, -0.25]]])  # 2, 4 WORKS
    # A = np.array([[[0.5], [0.25]], [[0.1], [0.6]]])  # 2,1 WORKS
    # A = np.array([[[0.5, 0.0], [0.25, 0.0]], [[0.1, 0.0], [0.6, -0.2]]])  # 2, 2 WORKS
    # A = generate_A_given_rank(9, 5, [3, 2, 1])
    A = np.array([[[-0.31569, 0.96391], [-0.54931, 1.67723], [-0.98500, -0.34061]],
                  [[0.11848, -0.36178], [0.20617, -0.62950], [0.36969, 0.12784]], [[-0.32287, -0.14310],
                                                                                   [-0.56179, -0.24900],
                                                                                   [0.04632, -0.70431]]])  # WORKS
    if check:
        # check_ass1(A)
        check_ass2(A)
    N, P = A.shape[1:]
    cov = np.eye(N, ) * sigma  # Covariance matrix of the innovations
    y, A, E = generate(A, T, P, N, cov)
    np.savez(f'./data/var_4_{T}_{N}_{P}.npz', y=y, A=A, E=E)


if __name__ == '__main__':
    main(check=False)
