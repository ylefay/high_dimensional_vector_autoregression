from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse import issparse as issparse_mat
from scipy.sparse.linalg import eigsh


def mode_fold(tensor, mode):
    """
    Flatten a tensor along its k-th mode.
    """
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1, order='F')


def mode_unfold(tensor, mode, shape):
    return jnp.moveaxis(np.reshape(tensor, shape, order='F'), 0, mode)


def rank_tensor(X):
    """
    Compute the ranks of a tensor X as
    rank(X, j) = rank(X_(j)) for j = 1, ..., ndims(X).
    """
    return [np.linalg.matrix_rank(mode_fold(X, n)) for n in range(X.ndim)]


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


def ttm(X, V, mode=None, transp=False):
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
        Y = ttm_compute(X, V[0], 0, transp)
        for i in range(1, X.ndim):
            Y = ttm_compute(Y, V[i], i, transp)
    return Y


def nvecs(X, n, rank, do_flipsign=True, dtype=float):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = mode_fold(X, n)
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
    newT = np.transpose(newT, np.argsort(order))
    return newT


"""
Tensor multiplication along different mode.
"""


def mode1(X, V):
    return jnp.einsum('ijk,si->sjk', X, V)


def mode2(X, V):
    return jnp.einsum('ijk,sj->isk', X, V)


def mode3(X, V):
    return jnp.einsum('ijk,sk->ijs', X, V)


def fast_ttm(X, Us, transp=False):
    """
    Similar to ttm_compute for computing
        G = X x_1 U_1 x_2 U_2 x_3 U_3
    """
    if transp:
        Us = [U.T for U in Us]
    return mode3(mode2(mode1(X, Us[0]), Us[1]), Us[2])


def vec(X):
    """
    Vectorization of a matrix.
    """
    return X.reshape(-1, order='F')


def unvec(vecX, shape):
    """
    Invert operation
    """
    return vecX.reshape(shape, order='F')
