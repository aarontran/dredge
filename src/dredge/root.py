#!/usr/bin/env python
"""
Trace dispersion branches by numerical root finding instead of working on a
grid
"""

from __future__ import division, print_function

import numpy as np
import scipy as sp

# https://discuss.python.org/t/which-is-the-right-way-to-use-modules-within-a-package/15297/3
from .util import searchsortedclosest


def solver(k_arr, omega_initial_guess, disprel, solver_kws=None,
           terminate_omega_re=None, terminate_omega_im=None, **kwargs):
    """
    Given wavenumbers k_arr, find complex frequencies omega_arr that satisfy

        disprel(omega_arr[i], k_arr[i], *args, **kwargs) = 0

    by starting from k_arr[0] and using SciPy's secant root-finder
    to trace the dispersion relation branch along k_arr.
    At k = k_arr[0], use omega_initial_guess to find omega_arr[0].
    At k = k_arr[1], use omega_arr[0] to find omega_arr[1].
    At k = k_arr[i], use omega_arr[i-1] to find omega_arr[i].

    The dispersion branch explored depends on the user's initial guess for omega.

    As constructed, we use Scipy's secant method to solve D(...)=0.
    If we have the first and/or second derivatives D'(...)=0, D''(...)=0
    with respect to omega at fixed k, we could use (but don't currently use)
    the faster-converging Newton-Raphson, Hailey methods.

    The k_arr does not strictly have to be a wavenumber, it can be any argument
    to disprel(...) that causes a "step" along k in the dispersion calculation
    (for example, an array of indices into a pre-cached k grid can be provided,
    so that the user can pre-cache Bessel function integrals to avoid
    recomputing on the fly during root finding)

    Inputs:
        k_arr : numpy array of k.
            User should choose k samples to suit
            the dispersion relation, such that the root finder can easily track
            one branch and will not jump between different branches.

        omega_initial_guess : complex scalar.
            Initial estimate of dispersion relation root for k_arr[0].

        disprel : function with signature disprel(omega, k, *args, **kwargs)

        solver_kws : dict of keywords passed to scipy.optimize.newton(...)
            or None to accept defaults

        terminate_omega_re : function to test omega re for terminating root
            tracing

        terminate_omega_im : function to test omega im for terminating root
            tracing

        **kwargs : additional arguments passed to disprel.

    Returns:
        omega_arr : numpy array of complex frequencies omega
    """
    def disprel_local(om, k):
        #return disprel(om, k, *args, **kwargs)
        return disprel(om, k, **kwargs)

    # when calling scipy.optimize.newton without "fprime",
    # need to give numpy.complex dtype for initial guess
    # https://stackoverflow.com/questions/24414762/finding-zeros-of-a-complex-function-in-scipy-numpy

    om_arr = np.zeros_like(k_arr, dtype=np.cdouble)  # complex double
    om_guess = omega_initial_guess  # for k_arr[0]

    if solver_kws is None:
        solver_kws = dict(
            #tol=3e-8,  # increased from default 1.48e-8
            maxiter=500,  # increased from default 50
        )

    for ii in range(len(k_arr)):
        #print("calling newton with omguess", om_guess, "k=", k_arr[ii])
        try:
            root = sp.optimize.newton(
                disprel_local,
                om_guess,
                args=(k_arr[ii],),
                **solver_kws,
            )
        except RuntimeError:
            print("got to ii=",ii,"k=",k_arr[ii],"om_guess",om_guess)
            raise
        # allow user to specify a break criterion
        if terminate_omega_re is not None and terminate_omega_re(root.real):
            print("...halt trace due to Re(omega) at ii=",ii,"k=",k_arr[ii],"om_guess",om_guess,"root",root)
            om_arr[ii:] = np.nan + 1j*np.nan
            break
        if terminate_omega_im is not None and terminate_omega_im(root.imag):
            print("...halt trace due to Im(omega) at ii=",ii,"k=",k_arr[ii],"om_guess",om_guess,"root",root)
            om_arr[ii:] = np.nan + 1j*np.nan
            break
        # accept result
        om_arr[ii] = root
        om_guess = root

    return om_arr


def tracer_grid(k0, om0, disprel, k_arr, **kwargs):
    """
    Given initial guess for (k,omega) on some dispersion branch,
    trace the branch forward and backwards on a user-specified k mesh.

    Inputs:
        k0 = wavenumber, initial guess for root on dispersion branch
        om0 = complex frequency, initial guess for root on dispersion branch
        disprel = function with arguments (omega, ik, ...)
            where ik indexes into the array k_arr
        k_arr = 1D array-like of wavenumber
        **kwargs = passed to solver(...)
            for use in either disprel(...) or scipy.optimize.newton(...)
    """

    # get the closest k to user's initial guess
    ik = searchsortedclosest(k_arr, k0)
    ik_rev = np.arange(0, ik+1)[::-1]  # reverse
    ik_fwd = np.arange(ik, k_arr.size)  # forward

    # do the dispersion rootfinding by stepping over k...
    om_rev = solver(ik_rev, om0, disprel, **kwargs)
    om_fwd = solver(ik_fwd, om0, disprel, **kwargs)
    om_refine = np.concatenate((om_rev[::-1],om_fwd[1:]))

    # value of D can be useful for troubleshooting sometimes
    dd_refine = np.nan * np.ones_like(om_refine)

    ok = np.logical_not(np.isnan(om_refine.real))
    for ii in np.where(ok)[0]:
        dd_refine[ii] = disprel(om_refine[ii], ii)

    return om_refine, dd_refine
