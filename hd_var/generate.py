import numpy as np

from hd_var.operations import mode_fold, ttm


def generate(A, T, P, N, cov):
    """
    Generate a P-VAR time series with dimension N and length T.
    """
    if N is None:
        N = A.shape[0]
    if P is None:
        P = A.shape[2]
    assert cov.shape == (N, N)
    # Generate VAR coefficients
    if A is None:
        A = np.random.randn(N, N, P)
    assert A.shape == (N, N, P)
    if cov is None:
        cov = np.eye(N)
    # Generate VAR innovations
    E = np.moveaxis(np.random.multivariate_normal(np.zeros(N), cov, size=(T,)), 0, -1)
    y = np.zeros((N, T))
    for t in range(P, T):
        y[:, t] = E[:, t]
        for p in range(P):
            y[:, t] += A[:, :, p] @ y[:, t - p - 1]
    assert np.abs(y).max() < T * P * np.linalg.norm(
        cov)  # not sure about that, the time serie should not be too big if stationary.
    return y, A, E


def generate_core_tensor(ranks):
    """
    Generate a core tensor of shape (r1, r2, r3) by scaling a random standard normal tensor
    such that the minimum of the minimal non-zero singular value accros the three slices,
    min_{1<=i<=3} sigma_{r_i}(G_{(i)}) is equal to 1.
    """
    G = np.random.normal(size=ranks)
    min_svd = [np.min(np.linalg.svd(mode_fold(G, i), compute_uv=False)) for i in range(3)]  # it is never 0
    i = np.argmin(min_svd)
    G /= min_svd[i]
    return G


def generate_orthogonal_matrix(m, n):
    assert m > n
    mat = np.random.normal(size=(m, m))
    left, singular, right = np.linalg.svd(mat)
    O = np.zeros((m, n))
    O[:n] = left[:n]
    return O


def generate_sparse_orthonormal_matrices(case=1):
    """
    See Section F.
    Used in an experiment setting in Section 6.2.
    Case 1:
        (r1, r2, r3, s1, s2, s3) = (2, 2, 2, 3, 3, 2)
    Case 2:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 3, 3, 2)
    Case 3:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 2, 2, 2)
    """

    if case == 1:
        a = np.random.normal(size=(3, 1))
        a /= np.linalg.norm(a)
        b = np.random.normal(size=(3, 1))
        b /= np.linalg.norm(b)
        U1 = np.zeros((10, 2))
        U1[:3, 0] = a.flatten()
        U1[3:6, 1] = b.flatten()
        U2 = np.zeros((10, 2))
        c = np.random.normal(size=(3, 1))
        c /= np.linalg.norm(c)
        d = np.random.normal(size=(3, 1))
        d /= np.linalg.norm(d)
        U2[:3, 0] = c.flatten()
        U2[3:6, 1] = d.flatten()
        e = np.random.normal(size=(2, 1))
        e /= np.linalg.norm(e)
        U3 = np.zeros((5, 2))
        U3[0, 0] = 1
        U3[1:3, 1] = e.flatten()
    if case == 2:
        U1 = np.zeros((10, 3))
        a = np.random.normal(size=(3, 1))
        a /= np.linalg.norm(a)
        b = np.random.normal(size=(3, 1))
        b /= np.linalg.norm(b)
        c = np.random.normal(size=(3, 1))
        c /= np.linalg.norm(c)
        U1[:3, 0] = a.flatten()
        U1[3:6, 1] = b.flatten()
        U1[6:9, 2] = c.flatten()
        U2 = np.zeros((10, 3))
        d = np.random.normal(size=(3, 1))
        d /= np.linalg.norm(d)
        e = np.random.normal(size=(3, 1))
        e /= np.linalg.norm(e)
        f = np.random.normal(size=(3, 1))
        f /= np.linalg.norm(f)
        U2[:3, 0] = d.flatten()
        U2[3:6, 1] = e.flatten()
        U2[6:9, 2] = f.flatten()
        U3 = np.zeros((5, 3))
        g = np.random.normal(size=(2, 1))
        g /= np.linalg.norm(g)
        h = np.random.normal(size=(2, 1))
        h /= np.linalg.norm(h)
        U3[0, 0] = 1
        U3[1:4, 1] = g.flatten()
        U3[3:6, 2] = h.flatten()
    if case == 3:
        U1 = np.zeros((10, 3))
        a = np.random.normal(size=(3, 1))
        a[-1] = 0
        a /= np.linalg.norm(a)
        b = np.random.normal(size=(3, 1))
        b[-1] = 0
        b /= np.linalg.norm(b)
        c = np.random.normal(size=(3, 1))
        c[-1] = 0
        c /= np.linalg.norm(c)
        U1[:3, 0] = a.flatten()
        U1[3:6, 1] = b.flatten()
        U1[6:9, 2] = c.flatten()
        U2 = np.zeros((10, 3))
        d = np.random.normal(size=(3, 1))
        d[-1] = 0
        d /= np.linalg.norm(d)
        e = np.random.normal(size=(3, 1))
        e[-1] = 0
        e /= np.linalg.norm(e)
        f = np.random.normal(size=(3, 1))
        f[-1] = 0
        f /= np.linalg.norm(f)
        U2[:3, 0] = d.flatten()
        U2[3:6, 1] = e.flatten()
        U2[6:9, 2] = f.flatten()
        U3 = np.zeros((5, 3))
        g = np.random.normal(size=(2, 1))
        g /= np.linalg.norm(g)
        h = np.random.normal(size=(2, 1))
        h /= np.linalg.norm(h)
        U3[0, 0] = 1
        U3[1:, 1] = g.flatten()
        U3[3:, 2] = h.flatten()
    return U1, U2, U3


def generate_orthonormal_matrices(N, P, ranks):
    """
    Generate orthonormal matrices U1, U2, U3
    See Section 6.1
    """
    Us = [np.empty(shape=(N, ranks[0])), np.empty(shape=(N, ranks[1])), np.empty(shape=(P, ranks[2]))]

    for i in range(3):
        random = np.random.normal(size=(Us[i].shape[0], Us[i].shape[0]))
        left, singular, right = np.linalg.svd(random, compute_uv=True)
        Us[i] = left[:, :ranks[i]]
    return Us


def generate_A_given_rank(N=10, P=5, ranks=[3, 3, 2]):
    """
    Generate A according to Section 6.2.
    In the paper, there are three cases:
        (r1, r2, r3) = (3, 3, 2)
        (r1, r2, r3) = (3, 3, 3)
        (r1, r2, r3) = (3, 3, 4)
    The tensor might not generate stationary VAR.
    """
    Us = generate_orthonormal_matrices(N, P, ranks)
    G = generate_core_tensor(ranks)
    A = ttm(G, Us)
    return A


def generate_A_according_to_section62(case=1):
    """
    Case 1:
        (r1, r2, r3, s1, s2, s3) = (2, 2, 2, 3, 3, 2)
    Case 2:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 3, 3, 2)
    Case 3:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 2, 2, 2)
    """
    if case == 1:
        ranks = [2, 2, 2]
    if case == 2 or case == 3:
        ranks = [3, 3, 3]
    G = generate_core_tensor(ranks)
    Us = generate_sparse_orthonormal_matrices(case)
    A = ttm(G, Us)
    return A


def generate_A_given_case(ranks, case):
    r1, r2, r3 = ranks
    if case == 'a' and r1 == 2 and r2 == 2 and r3 == 2:
        N, P = 10, 5
        U1, U2, U3 = generate_sparse_orthonormal_matrices(1)
    elif case == 'a' and r1 == 3 and r2 == 3 and r3 == 3:
        N, P = 10, 5
        U1, U2, U3 = generate_sparse_orthonormal_matrices(2)
    elif case == 'b' and r1 == 2 and r2 == 2 and r3 == 2:
        print('TODO')
    elif case == 'b' and r1 == 3 and r2 == 3 and r3 == 3:
        print('TODO')
    elif case == 'c' and r1 == 2 and r2 == 2 and r3 == 2:
        N, P = 20, 5
        U1, U2, U3 = generate_sparse_orthonormal_matrices(1)
        U1 = np.vstack([U1, np.zeros((10, 2))])
        U2 = np.vstack([U2, np.zeros((10, 2))])
    elif case == 'c' and r1 == 3 and r2 == 3 and r3 == 3:
        N, P = 20, 5
        U1, U2, U3 = generate_sparse_orthonormal_matrices(2)
        U1 = np.vstack([U1, np.zeros((10, 3))])
        U2 = np.vstack([U2, np.zeros((10, 3))])
    elif case == 'd' and r1 == 2 and r2 == 2 and r3 == 2:
        N, P = 10, 10
        U1, U2, U3 = generate_sparse_orthonormal_matrices(1)
        U3 = np.vstack([U3, np.zeros((5, 2))])
    elif case == 'd' and r1 == 3 and r2 == 3 and r3 == 3:
        N, P = 10, 10
        U1, U2, U3 = generate_sparse_orthonormal_matrices(2)
        U3 = np.vstack([U3, np.zeros((5, 3))])
    G = generate_core_tensor(ranks)
    Us = [U1, U2, U3]
    A = ttm(G, Us)
    return A
