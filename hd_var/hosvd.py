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

from .operations import nvecs, ttm


def hosvd(A, rank):
    U = [None for _ in range(A.ndim)]
    dims = range(A.ndim)
    for d in dims:
        U[d] = np.array(nvecs(A, d, rank[d]))
    core = ttm(A, U, transp=True)
    return U, core
