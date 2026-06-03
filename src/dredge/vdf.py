"""
Analytic velocity distribution functions (no numerical grid is assumed)
"""

import numpy as np


def bimaxwellian(vperp, vprll, vthperp, vthprll):
    """
    Bi-Maxwellian distribution, non-relativistic.
    """
    norm = 1./(np.pi**1.5 * vthperp**2 * vthprll)
    return norm * np.exp( - (vperp/vthperp)**2 - (vprll/vthprll)**2 )


def maxwellian_reduced(vperp, vth):
    """
    Reduced 1D Maxwellian distribution in vperp
    defined such that F(vperp) * 2*pi*vperp * dvperp = f(v) d^3 v.
    Non-relativistic.
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
