#!/usr/bin/env python
"""
Utility methods
"""

import numpy as np

from mpi4py import MPI


def print0(*args, **kwargs):
    """Print only on MPI global communicator rank 0"""
    if MPI.COMM_WORLD.Get_rank() == 0:
        return print(f'[{rank:d}]', x, *args, **kwargs)


def printn(x, *args, **kwargs):
    """Print with MPI global communicator rank prefixed"""
    rank = MPI.COMM_WORLD.Get_rank()
    return print(f'[{rank:d}]', x, *args, **kwargs)


def searchsortedclosest(a, v, side='left'):
    """
    Thin wrapper around numpy.searchsorted(...) to get closest element index,
    rather than the insertion index to preserve sort order.
    """
    ii = np.searchsorted(a,v)
    # smaller than all elements in array
    if ii == 0:
        return 0
    # bigger than all elements in array
    elif ii == a.size:
        return a.size - 1
    # deal with all the other cases
    iLeft = ii - 1
    iRight = ii
    dLeft = v - a[ii-1]
    dRight = a[ii] - v
    if dRight > dLeft:
        return iLeft
    elif dLeft > dRight:
        return iRight
    else:
        if side == 'left':
            return iLeft
        elif side == 'right':
            return iRight
