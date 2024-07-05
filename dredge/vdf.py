#!/usr/bin/env python
"""
Velocity distribution functions useful for plasma dispersion calculations
"""

from __future__ import division, print_function

import numpy as np
import scipy as sp


def maxwell_reduced(vperp, vth):
    """
    Reduced 1D Maxwellian distribution in vperp
    defined such that F(vperp) * 2*pi*vperp * dvperp = f(v) d^3 v.
    """
    return 1/(np.pi*vth**2) * np.exp(-(vperp/vth)**2)


def gerver(vperp, vth, R):
    """
    Reduced 1D distribution in vperp
    defined such that F(vperp) * 2*pi*vperp * dvperp = f(v) d^3 v
    (following Kotelnikov, differs from Gerver),
    but the meaning of R in both Kotelnikov and Gerver is the same.
    """
    a0 = (R+1)/(R-1)/(np.pi*vth**2)
    a1 = np.exp(-(vperp/vth)**2 * (R+1)/R)
    a2 = np.exp(-(vperp/vth)**2 * (R+1)  )
    return a0 * (a1 - a2)
