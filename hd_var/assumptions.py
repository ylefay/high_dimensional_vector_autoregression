import numpy as np
from hd_var.operations import mode_fold, rank_tensor
from hd_var.hosvd import hosvd
import scipy


def check_ass2(A):
    """
    Check Assumption 2 on uniqueness of singular values of U_j and positiveness of the
    first element in each column of U_j, for j = 1, 2, 3
    """
    ranks = rank_tensor(A)
    Us, G = hosvd(A, ranks)
    for i in range(3):
        if not np.all(Us[i][0, :] >= 0):
            return False
        Ai = mode_fold(A, i)
        sv = np.linalg.svd(Ai, compute_uv=False)
        if not len(np.unique(sv)) == len(sv):  # ... up to precision machine
            return False
    return True


def check_ass1(A):
    """
    Check Assumption 1: Stationarity condition on A
    Failing to converge...
    """
    N, P = A.shape[-2:]
    id = np.eye(N)
    attempt_number = 100

    def characteristic_polynomial(z):
        if np.absolute(z) < 1:
            return np.absolute(np.linalg.det(id - np.sum([z ** i * A[..., i] for i in range(P)], axis=0)))
        else:
            return np.inf

    def random_pt_inside_circle():
        alpha = 2 * np.pi * np.random.random()
        r = np.random.random()
        return r * np.exp(1j * alpha)

    try:
        roots = [scipy.optimize.newton(characteristic_polynomial, random_pt_inside_circle(), maxiter=2) for i in
                 range(attempt_number)]
        roots = roots[np.absolute(roots) < 1]
        zeros = characteristic_polynomial(roots)
        return np.all(zeros > 1e-2)
    except Exception as e:
        print(e)
        return True


def check_ass1_bis(A):
    """
    Implementing the ADF test ?
    """
    raise NotImplementedError
