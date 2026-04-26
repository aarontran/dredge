#!/usr/bin/env python
"""
Special functions useful for plasma dispersion calculations
"""

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
    # https://github.com/hsxie/gpdf/blob/main/gpdf/PlasmaDispersionFunction_hsixe_20111009.pdf ,
    # and his notes on page 25 -- we need to check Im(zeta) < 0 case
    # to verify that analytic continuation is done correctly.
    #
    # I'm a bit confused because expression for Zfunc is stated w/o caveat in Stix, and in Schekochihin,
    # but NRL formulary suggests (after some mild rewriting) that equation is only valid for real argument?
    return 1j*np.sqrt(np.pi) * sp.special.wofz(zeta)


def Zfuncn(n, zeta):
    """
    Generalized plasma dispersion function,
    Z_n(zeta) = (1/sqrt(pi)) * int_{-inf}^{inf} dx x^n exp(-x^2) / (x-zeta).

    These appear in ITG, interchange, etc. instability calculations involving
    magnetic curvature; see, e.g., Mishchenko+ (2018 JPP Equation (5.5)) and
    Gurcan (2014 J. Comput. Phys.)
    https://www.sciencedirect.com/science/article/pii/S0021999114001983
    https://github.com/gurcani/zpdgen

    Inputs
        n : order of the generalized plasma dispersion function, integer >= 0
        zeta : argument to Z_n(...), scalar or array-like, may be complex
    Returns
        Z_n(zeta) : complex, in general, result of Z_n(function)
    """
    Z0 = Zfunc(zeta)
    if n == 0:
        return Z0
    if n == 1:
        return 1 + zeta * Z0
    if n == 2:
        return zeta + zeta**2 * Z0
    if n == 3:
        return 0.5 + zeta**2 + zeta**3 * Z0
    if n == 4:
        return 0.5*zeta + zeta**3 + zeta**4 * Z0
    if n == 5:
        return 0.75 + 0.5*zeta**2 + zeta**4 + zeta**5 * Z0
    if n == 6:
        return 0.75*zeta + 0.5*zeta**3 + zeta**5 + zeta**6 * Z0
    # For higher orders, add a function for Gaussian moment coefficients
    # \int_{-inf}^{inf} dx x^n exp(-x^2) = (n-1)!! / 2^{n/2}
    # use something like sp.special.factorial2(n-1) / 2**(n//2)
    # but with special handling for n=0 case
    # https://chatgpt.com/share/69ebe483-9bb8-83ea-8001-1c051dab9d68
    raise NotImplementedError("Higher-order generalized plasma dispersion functions are not implemented")
