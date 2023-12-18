import numpy as np
from hd_var.operations import mode_fold, compute_A


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
    X = np.zeros((N, T))
    for t in range(P, T):
        X[:, t] = E[:, t]
        for p in range(P):
            X[:, t] += A[:, :, p] @ X[:, t - p - 1]
    assert np.abs(X).max() < T * P * N  # not sure about that, the time serie should not be too big if stationary.
    return X, A, E


def generate_core_tensor(ranks):
    """
    Generate a core tensor of shape (r1, r2, r3) by scaling a random standard normal tensor
    such that the minimum of the minimal non zero singular value accros the three slices,
    min_{1<=i<=3} sigma_{r_i}(G_{(i)}) is equal to 1.
    """
    G = np.random.normal(size=ranks)
    min_svd = [np.min(np.linalg.svd(mode_fold(G, i), compute_uv=False)) for i in range(3)]  # it is never 0
    i = np.argmin(min_svd)
    G /= min_svd[i]
    return G


def generate_sparse_orthonormal_matrices(m, n, case=1):
    """
    See Section F.
    Case 1:
        (r1, r2, r3, s1, s2, s3) = (2, 2, 2, 3, 3, 2)
    Case 2:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 3, 3, 2)
    Case 3:
        (r1, r2, r3, s1, s2, s3) = (3, 3, 3, 2, 2, 2)
    """
    assert m > n
    mat = np.random.normal(size=(m, m))
    left, singular, right = np.linalg.svd(mat)
    O = np.zeros((m, n))
    O[:n] = left[:n]
    if case == 1:
        a = np.random.normal(size=(3, 1))
        a /= np.linalg.norm(a)
        b = np.random.normal(size=(3, 1))
        b /= np.linalg.norm(b)
        U1 = np.zeros((10, 2))
        U1[:3, 0] = a.flatten()
        U1[3:, 1] = b.flatten()
        U2 = np.zeros((10, 2))
        c = np.random.normal(size=(3, 1))
        c /= np.linalg.norm(c)
        d = np.random.normal(size=(3, 1))
        d /= np.linalg.norm(d)
        U2[:3, 0] = c.flatten()
        U2[3:, 0] = d.flatten()
        e = np.random.normal(size=(2, 1))
        e /= np.linalg.norm(e)
        U3 = np.zeros((5, 2))
        U3[0, 0] = 1
        U3[1:, 1] = e.flatten()
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
        U1[6:, 2] = c.flatten()
        U2 = np.zeros((10, 3))
        d = np.random.normal(size=(3, 1))
        d /= np.linalg.norm(d)
        e = np.random.normal(size=(3, 1))
        e /= np.linalg.norm(e)
        f = np.random.normal(size=(3, 1))
        f /= np.linalg.norm(f)
        U2[:3, 0] = d.flatten()
        U2[3:6, 1] = e.flatten()
        U2[6:, 2] = f.flatten()
        U3 = np.zeros((5, 3))
        g = np.random.normal(size=(2, 1))
        g /= np.linalg.norm(g)
        h = np.random.normal(size=(2, 1))
        h /= np.linalg.norm(h)
        U3[0, 0] = 1
        U3[1:, 1] = g.flatten()
        U3[3:, 2] = h.flatten()
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
        U1[6:, 2] = c.flatten()
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
        U2[6:, 2] = f.flatten()
        U3 = np.zeros((5, 3))
        g = np.random.normal(size=(2, 1))
        g /= np.linalg.norm(g)
        h = np.random.normal(size=(2, 1))
        h /= np.linalg.norm(h)
        U3[0, 0] = 1
        U3[1:, 1] = g.flatten()
        U3[3:, 2] = h.flattfen()
    return O, U1, U2, U3


def generate_orthonormal_matrices(N, P, ranks):
    """
    Generate orthonormal matrices U1, U2, U3
    See Section 6.1
    """
    Us = [np.empty(shape=(N, ranks[0])), np.empty(shape=(N, ranks[1])), np.empty(shape=(P, ranks[2]))]

    for i in range(3):
        random = np.random.normal(size=Us[i].shape)
        left, singular, right = np.linalg.svd(random, compute_uv=True)
        Us[i] = left[:, :ranks[i]]
    return Us


def generate_A_according_to_section62(ranks=[3, 3, 2]):
    """
    Generate A according to Section 6.2.
    In the paper, there are three cases:
        (r1, r2, r3) = (3, 3, 2)
        (r1, r2, r3) = (3, 3, 3)
        (r1, r2, r3) = (3, 3, 4)
    The tensor might not generate stationary VAR.
    """
    N, P = 10, 5
    Us = generate_orthonormal_matrices(N, P, ranks)
    G = generate_core_tensor(ranks)
    A = compute_A(G, Us)
    return A
