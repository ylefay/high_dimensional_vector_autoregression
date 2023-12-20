import numpy as np
from hd_var.generate import generate_A_given_rank, generate
from hd_var.assumptions import check_ass2, check_ass1


def main(case=1, check=False):
    """
    Same setting as in Section 6.2.
    Sigma = 1.0, (N, P) = (10, 5),
    (r1, r2, r3) = (3, 3, {2, 3, 4})
    """
    T = 100  # Length of the time series
    sigma = 1.0  # Variance of the innovations, assuming diagonal noise
    N, P = 10, 5
    if case == 1:
        ranks = [3, 3, 2]
    elif case == 2:
        ranks = [3, 3, 3]
    elif case == 3:
        ranks = [3, 3, 4]
    cov = np.eye(N, ) * sigma  # Covariance matrix of the innovations
    A = generate_A_given_rank(N, P, ranks)
    if check:
        while not check_ass2(A):
            A = generate_A_given_rank(N, P, ranks)
    y, A, E = generate(A, T, P, N, cov)
    np.savez(f'./data/var_62_{T}_{N}_{P}.npz', y=y, A=A, E=E)


if __name__ == '__main__':
    main(case=1, check=True)
