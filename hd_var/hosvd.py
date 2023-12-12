## tucker.py

# sktensor.tucker - Algorithms to compute Tucker decompositions
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

import numpy as np
from .operations import ttm, nvecs


def hosvd(X, rank, dims=None, dtype=None, compute_core=True):
    U = [None for _ in range(X.ndim)]
    if dims is None:
        dims = range(X.ndim)

    if dtype is None:
        dtype = X.dtype
    for d in dims:
        U[d] = np.array(nvecs(X, d, rank[d]), dtype=dtype)
    if compute_core:
        core = ttm(X, U, transp=True)
        return U, core
    else:
        return U


"""
tensor = np.random.normal(0, 1, (10, 22, 34))
rank = [3, 2, 4]
result = hosvd(tensor, rank, dims=None, dtype=None, compute_core=True)
print(result)
"""
