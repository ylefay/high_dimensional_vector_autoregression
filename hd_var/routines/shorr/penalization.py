import numpy as np
from scipy.linalg import eigh


def lambda_optimal(N, P, T, cov):
    """
    Compute a lower bound on lambda for the estimator to be
    consistent. See Theorem 2.
    Assuming µ_max / µ_min (A) <= 10.
    """

    M = eigh(a=cov, subset_by_index=(N - 1, N - 1), eigvals_only=True)[0]
    factor = np.sqrt(np.log(N ** 2 * P) / T)
    return factor * M * (1 + 10)
