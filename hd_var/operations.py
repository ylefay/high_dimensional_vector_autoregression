import numpy as np
from collections.abc import Sequence
from scipy.linalg import eigh
from scipy.sparse import issparse as issparse_mat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def mode_flatten(tensor, mode):
    """
    Flatten a tensor along its k-th mode.
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1, order='F')


def from_to_without(frm, to, without, step=1, skip=1, reverse=False, separate=False):
    """
    Helper function to create ranges with missing entries
    """
    if reverse:
        frm, to = (to - 1), (frm - 1)
        step *= -1
        skip *= -1
    a = list(range(frm, without, step))
    b = list(range(without + skip, to, step))
    if separate:
        return a, b
    else:
        return a + b


def ttm(X, V, mode=None, transp=False, without=False):
    """
    Tensor times matrix product

    Parameters
    ----------
    V : M x N array_like or list of M_i x N_i array_likes
        Matrix or list of matrices for which the tensor times matrix
        products should be performed
    mode : int or list of int's, optional
        Modes along which the tensor times matrix products should be
        performed
    transp: boolean, optional
        If True, tensor times matrix products are computed with
        transpositions of matrices
    without: boolean, optional
        It True, tensor times matrix products are performed along all
        modes **except** the modes specified via parameter ``mode``


    Examples
    --------
    Create dense tensor

    >>> T = np.zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]

    Create matrix

    >>> V = np.array([[1, 3, 5], [2, 4, 6]])

    Multiply tensor with matrix along mode 0

    >>> Y = ttm(T, V, 0)
    >>> Y[:, :, 0]
    array([[ 22.,  49.,  76., 103.],
           [ 28.,  64., 100., 136.]])
    >>> Y[:, :, 1]
    array([[130., 157., 184., 211.],
           [172., 208., 244., 280.]])

    """
    if mode is None:
        mode = range(X.ndim)
    if isinstance(V, np.ndarray):
        Y = ttm_compute(X, V, mode, transp)
    elif isinstance(V, Sequence):
        dims, vidx = check_multiplication_dims(mode, X.ndim, len(V), vidx=True, without=without)
        Y = ttm_compute(X, V[vidx[0]], dims[0], transp)
        for i in range(1, len(dims)):
            Y = ttm_compute(Y, V[vidx[i]], dims[i], transp)
    return Y


def ttv(X, v, modes=[], without=False):
    """
    Tensor times vector product

    Parameters
    ----------
    v : 1-d array or tuple of 1-d arrays
        Vector to be multiplied with tensor.
    modes : array_like of integers, optional
        Modes in which the vectors should be multiplied.
    without : boolean, optional
        If True, vectors are multiplied in all modes **except** the
        modes specified in ``modes``.

    """
    if not isinstance(v, tuple):
        v = (v,)
    dims, vidx = check_multiplication_dims(modes, X.ndim, len(v), vidx=True, without=without)
    for i in range(len(dims)):
        if not len(v[vidx[i]]) == X.shape[dims[i]]:
            raise ValueError('Multiplicant is wrong size')
    remdims = np.setdiff1d(range(X.ndim), dims)
    return ttv_compute(X, v, dims, vidx, remdims)


def check_multiplication_dims(dims, N, M, vidx=False, without=False):
    dims = np.array(dims, ndmin=1)
    if len(dims) == 0:
        dims = np.arange(N)
    if without:
        dims = np.setdiff1d(range(N), dims)
    if not np.in1d(dims, np.arange(N)).all():
        raise ValueError('Invalid dimensions')
    P = len(dims)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
    if vidx:
        if M > N:
            raise ValueError('More multiplicants than dimensions')
        if M != N and M != P:
            raise ValueError('Invalid number of multiplicants')
        if P == M:
            vidx = sidx
        else:
            vidx = sdims
        return sdims, vidx
    else:
        return sdims


def nvecs(X, n, rank, do_flipsign=True, dtype=float):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = mode_flatten(X, n)
    if issparse_mat(Xn):
        Xn = csr_matrix(Xn, dtype=dtype)
        Y = Xn.dot(Xn.T)
        _, U = eigsh(Y, rank, which='LM')
    else:
        Y = Xn.dot(Xn.T)
        N = Y.shape[0]
        # _, U = eigh(Y, eigvals=(N - rank, N - 1))
        _, U = eigh(Y, subset_by_index=(N - rank, N - 1))  # ici
        # _, U = eigsh(Y, rank, which='LM')
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = np.array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)
    return U


def flipsign(U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = abs(U).argmax(axis=0)
    for i in range(U.shape[1]):
        if U[midx[i], i] < 0:
            U[:, i] = -U[:, i]
    return U


def teneye(dim, order):
    """
    Create tensor with superdiagonal all one, rest zeros
    """
    I = np.zeros(dim ** order)
    for f in range(dim):
        idd = f
        for i in range(1, order):
            idd = idd + dim ** (i - 1) * (f - 1)
        I[idd] = 1
    return I.reshape(np.ones(order) * dim)


# vim: set et:


## dtensor.py

# sktensor.dtensor - base class for dense tensors
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


def ttv_compute(X, v, dims, vidx, remdims):
    """
    Tensor times vector product

    Parameter
    ---------
    """
    if not isinstance(v, tuple):
        raise ValueError('v must be a tuple of vectors')
    ndim = X.ndim
    order = list(remdims) + list(dims)
    if ndim > 1:
        T = np.transpose(X, order)
    sz = np.array(X.shape)[order]
    for i in np.arange(len(dims), 0, -1):
        T = T.reshape((sz[:ndim - 1].prod(), sz[ndim - 1]))
        T = T.dot(v[vidx[i - 1]])
        ndim -= 1
    if ndim > 0:
        T = T.reshape(sz[:ndim])
    return T


def ttm_compute(X, V, mode, transp):
    sz = np.array(X.shape)
    r1, r2 = from_to_without(0, X.ndim, mode, separate=True)
    order = [mode] + r1 + r2
    newT = np.transpose(X, axes=order)
    newT = newT.reshape(sz[mode], np.prod(sz[r1 + list(range(mode + 1, len(sz)))]))
    if transp:
        newT = V.T.dot(newT)
        p = V.shape[1]
    else:
        newT = V.dot(newT)
        p = V.shape[0]
    newsz = [p] + list(sz[:mode]) + list(sz[mode + 1:])
    newT = newT.reshape(newsz)
    # transpose + argsort(order) equals ipermute
    newT = np.transpose(newT, np.argsort(order))
    return newT
