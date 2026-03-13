#!/usr/bin/env python
"""
Utility methods
"""

from __future__ import division, print_function

import numpy as np


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
