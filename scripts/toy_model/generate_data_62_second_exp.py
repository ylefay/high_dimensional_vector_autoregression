import numpy as np
from hd_var.generate import generate_A_according_to_section62, generate
from hd_var.assumptions import check_ass2, check_ass1


def main(case=1, check=False):
    """
    Same setting as in Section 6.2.
    Generate G by scaling a standard normal tensor.
    Generate the U_is as described in section F.
    """
    T = 2000  # Length of the time series
    sigma = 1.0  # Variance of the innovations, assuming diagonal noise
    N, P = 10, 5
    cov = np.eye(N, ) * sigma  # Covariance matrix of the innovations
    A = generate_A_according_to_section62(case)

    if check: #does not work
        while not check_ass2(A):
            A = generate_A_according_to_section62(case)
    X, A, E = generate(A, T, P, N, cov)
    np.savez(f'./data/var_62_bis_{T}_{N}_{P}.npz', X=X, A=A, E=E)


if __name__ == '__main__':
    main(case=1, check=False)
