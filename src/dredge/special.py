#!/usr/bin/env python
"""
Special functions useful for plasma dispersion calculations
"""

from __future__ import division, print_function

import numpy as np
import scipy as sp


def Zfunc(zeta):
    """
    Plasma dispersion function, Z(zeta) = i*sqrt(pi) * W(zeta),
    where W(zeta) = exp(-zeta**2) * erfc(-i*zeta) is the Faddeeva W function.
    Uses scipy.special.wofz(...) to implement W(...).

    Inputs
        zeta : argument to Z(...), scalar or array-like, may be complex
    Returns
        Z(zeta) : complex, in general, result of Z(function)
    """
    # Tests:
    # Zfunc(9.8 + 10.0*1j)  # returns (-0.04985622714609075+0.05113379742397624j)
    # Zfunc(9.8 - 10.0*1j)  # returns (-174.76146310967394+63.63268853627574j)
    #
    # See Huasheng Xie's very useful notes on plasma dispersion function,
    # http://hsxie.me/codes/gpdf/PlasmaDispersionFunction_hsixe_20111009.pdf ,
    # and his notes on page 25 -- we need to check Im(zeta) < 0 case
    # to verify that analytic continuation is done correctly.
    #
    # I'm a bit confused because expression for Zfunc is stated w/o caveat in Stix, and in Schekochihin,
    # but NRL formulary suggests (after some mild rewriting) that equation is only valid for real argument?
    return 1j*np.sqrt(np.pi) * sp.special.wofz(zeta)
