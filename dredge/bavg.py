"""
Code to compute bounce-average integrals
"""

from __future__ import division, print_function

import numba
import numpy as np

# TODO: implement methods from https://arxiv.org/abs/2412.01724


@numba.njit(parallel=False)
def _bounce_average_njit_kernel(
        x_grid, Bsamp_grid, s_grid, E_grid, mu_grid, vperp_vec, vprll_vec,
        mass, dB_ds2_origin, tbounce4th, tbounce4th_max, norm=True
):
    r"""
    Internal function to compute the bounce average, hottest logic
    Input:
        x_grid = quantity to be averaged; shape (vperp,vprll,s)
        Bsamp_grid = B field magnitude on grid (vperp,vprll,s)
        s_grid = s coordinates on grid (vperp,vprll,s)
                 recall that field-line integration points may differ in
                 velocity space b/c turning-point location varies
        E_grid = energy on grid (vperp,vprll)
        mu_grid = magnetic moment on grid (vperp,vprll)
        vperp_vec = v_\perp coordinates in cm/s, 1D array
        vprll_vec = v_\parallel coordinates in cm/s, 1D array
        mass = particle species mass in grams, scalar
        dB_ds2_origin = d^2(B)/ds^2 evaluated at s=0, scalar, used to help
                        evaluate limit vprll->0 (with vperp=finite)
        tbounce4th = shape (vperp,vprll) grid of bounce times
        tbounce4th_max = upper limit on quarter bounce time
        norm = normalize?
    Return:
        shape (vperp, vprll)
    """
    result = np.empty((vperp_vec.size, vprll_vec.size), dtype=x_grid.dtype)

    for ii in numba.prange(vperp_vec.size):
        for jj in numba.prange(vprll_vec.size):
            x_vec = x_grid    [ii,jj,:]  # NOTE (s,vperp,vprll) shape
            B_vec = Bsamp_grid[ii,jj,:]  # doesn't play nice with numba
            s_vec = s_grid    [ii,jj,:]  # compilation of np.trapezoid
            E     = E_grid    [ii,jj]    # b/c [:,ii,jj] data not contiguous in c ordering
            mu    = mu_grid   [ii,jj]
            # common case; breaks with divide-by-zero or huge number
            # for vprll=0 (pitch angle 90)
            # singular line requires separate handling for vperp=0 or vperp>0
            integrand = x_vec / np.sqrt( (2./mass)*(E - mu*B_vec) )
            result[ii,jj] = np.trapezoid(integrand, s_vec)

    # special case handling
    muzero = (vperp_vec == 0)
    pitch90 = (vprll_vec == 0)

    # handle vprll=0 line with special remainder loop
    if np.any(pitch90):
        assert np.where(pitch90)[0].size == 1
        jj = np.where(pitch90)[0][0]
        for ii in range(vperp_vec.size):
            # prevent exactly zero; the zero case will be dealt with later
            #mu = mu_grid[ii,jj]
            mu = max(1e-99,mu_grid[ii,jj])
            # limiting form of bounce-average integral near the singularity,
            # valid for the case x=1, but TODO MAY NOT BE CORRECT FOR x(s)
            # spatially varying........ --ATr,2025nov06
            result[ii,jj] = x_grid[ii,jj,0] * np.pi/2 * np.sqrt(mass / dB_ds2_origin / mu)

    # apply normalization BEFORE the singular point handling
    if norm:
        result /= tbounce4th

    # handle zero point specially
    if np.any(muzero) and np.any(pitch90):
        assert np.where(muzero)[0].size == 1
        assert np.where(pitch90)[0].size == 1
        ii = np.where(muzero)[0][0]
        jj = np.where(pitch90)[0][0]
        if norm:
            # force to the moment's value at s=0 at singular point??
            # TODO is this correct???
            result[ii,jj] = x_grid[ii,jj,0]
        else:
#            # this treatment is only valid when x=1
#            # as used when computing bounce orbit periods
#            assert x_grid[ii,jj,0] == 1.
#            result[ii,jj] = min(tbounce4th_max, np.nanmax(result))
            # PROBLEM, np.nanmax(...) does not play nice with complex arguments
            # and compiler will balk
            result[ii,jj] = np.nan

    return result
