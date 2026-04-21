#!/usr/bin/env python
"""
Test dipole magnetic geometry calculation
"""

import numpy as np
import pytest

from mpi4py import MPI

import dredge as dr
from dredge.const import CLIGHT, ERG_PER_EV, ESU, M_PROTON


@pytest.mark.mpi_skip
def test_dipole_bounce_average():

    # ----------------------------------------------------------------
    # Define the magnetic geometry

    req = 50
    field = dr.field.FieldLineDipoleFarField(
            I=1e17, r0=1, req=req,
            ds=0.2, n_steps=int(200*req/30),
            axis_r = 1, axis_z = 2,
    )

    # ----------------------------------------------------------------
    # Define the plasma parameters and species

    Ti = 100 * ERG_PER_EV  # arbitrary
    vthi = np.sqrt(2*Ti/M_PROTON)

    # using odd number of points to vprll_vec to get vprll=0 exactly,
    # stress test the mu=0 edge case
    vperp_vec = np.linspace(0, 4*vthi, 20)  # cm/s
    vprll_vec = np.linspace(-4*vthi, 4*vthi, 41)  # cm/s
    vperp, vprll = np.meshgrid(vperp_vec, vprll_vec, indexing='ij')
    df = dr.vdf.bimaxwellian(vperp, vprll, vthi, vthi)  # isotropic maxwellian, (cm/s)^-3

    ion = dr.species.KineticVDFGrid(
        mass=M_PROTON,
        charge=ESU,
        vperp_vec=vperp_vec,
        vprll_vec=vprll_vec,
        df=df,
    )

    # ----------------------------------------------------------------
    # Define the (k,\omega) mesh for susceptibility calculation

    # just a dummy argument, not used for bounce-average tests
    solve_grid = dr.chi.WaveGrid(
        k_vec_global = np.linspace(1e-6, 1, 10) / ion.rLs(field.Bmag[0]),
        omega_re_vec_global = np.linspace(1e-6, 1, 11) * ion.Omcs(field.Bmag[0]),
        omega_im_vec_global = np.linspace(1e-6, 1, 12) * ion.Omcs(field.Bmag[0]),
        proc_layout = (1,1,1),
    )

    # ----------------------------------------------------------------
    # Perform the bounce-average calculation

    calc = dr.chi.BounceAvgESPerp(
        grid = solve_grid,
        species = ion,
        field = field,
    )

    calc.setup_bounce_average(NS_RESOLUTION = 30)
    # additional setup for bounce average
    # that is not fully incorporated into my code
    rsamp = field.query_r_at(calc.ssamp)  # shape (vperp,vprll,s)

    # compute bounce-averaged (omega_drift / m) where m = azimuthal mode number
    # this includes k_\perp(s) structure into the bounce average
    calc._reset_timers()
    omega_m_gradB_BA = calc.bounce_average_norm_raw(calc.v_gradB[0]/rsamp, norm=True)
    omega_m_curv_BA  = calc.bounce_average_norm_raw(calc.v_curv[0]/rsamp,  norm=True)
    omega_m_drift_BA = omega_m_gradB_BA + omega_m_curv_BA

    # ----------------------------------------------------------------
    # Compare to approximation from Kesner & Hastie (2002, Phys Plasmas)
    # as written by Mishchenko+ (2018, J Plasma Physics)

    # flux label of selected field line
    psi = field.M / field.req
    # need -1 to reconcile my (x,y,z) coordinates with Mishchenko's (r,z,phi) coordinates
    # and need factor of CLIGHT to convert SI into CGS
    # Mishchenko+ (2018 JPP), Section 5, top of page 11
    omega_m_drift_BA_expected = -1 * (8/3) * calc.E * CLIGHT / (ion.charge * psi)

    # actual bounds are (0.76849,1.125) computed with NS_RESOLUTION = 300
    # actual bounds are (0.76449,1.125) computed with NS_RESOLUTION = 30
    # allow for some numerical slop, this test is meant to catch
    # things breaking horribly by factors of 2x or 10x
    assert np.nanmax(omega_m_drift_BA / omega_m_drift_BA_expected) < 1.126
    assert np.nanmin(omega_m_drift_BA / omega_m_drift_BA_expected) > 0.760


if __name__ == '__main__':

    if MPI.COMM_WORLD.Get_size() == 1:

        test_dipole_bounce_average()

    elif MPI.COMM_WORLD.Get_size() == 4:

        print("No MPI-parallel dipole tests, rerun this script with 1 MPI rank")

    else:

        print("Invalid number of MPI ranks, use 1 or 4")
