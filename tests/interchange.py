#!/usr/bin/env python
"""
Test interchange calculation for cool bi-Maxwellian plasma
Only checks fluid long-wavelength limit,
but it exercises many different pieces of the code
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from datetime import datetime

from mpi4py import MPI
import pytest

import dredge as dr
from dredge.const import (
    CLIGHT, ERG_PER_EV, ESU, GAUSS_PER_TESLA,
    M_ELECTRON, M_PROTON,
)

# ----------------------------------------------------------------
# Parallel environment setup diagnostics

#print("Numba threads", numba.get_num_threads())

def interchange(ion_method='loop6d', proc_layout=(1,1,1)):

    # ----------------------------------------------------------------
    # Define the magnetic geometry

    B0 = 0.26 * GAUSS_PER_TESLA  # WHAM Phase 1 magnetic field in Gauss
    BT = 17 * GAUSS_PER_TESLA  # WHAM-HTS throat magnetic field
    LP = 98  # centimeters

    # starting (r, z) positions (cm) for field line trace
    r0 = 0.233
    z0 = 0

    # TESTING: concentrate plasma at z=0 to recover slab limit,
    # no magnetic geometry variation along s is seen by particles
    ds = 1e-6
    n_steps = 20
    ns_resolution = 10  # number of sample points for bounce-average integral

    field = dr.field.FieldLineParabolic(B0=B0, Bt=BT, Lp=LP, r0=r0, z0=z0,
                                        ds=ds, n_steps=n_steps,
                                        axis_r=1, axis_z=2)

    # ----------------------------------------------------------------
    # Define the plasma parameters and species

    n0 = 1e13  # ion density in cm^-3
    Ti_perp = 100 * ERG_PER_EV  # ion temperature
    Te_perp = 100 * ERG_PER_EV  # electron temperature
    Ti_aniso = 2    # ratio of Tperp/Tprll for bi-Maxwellian
    Te_aniso = 2

    # plasma species parameters
    mi = M_PROTON  # ion mass (PROTON!!)
    me = M_ELECTRON  # electron mass
    qi = ESU  # ion charge  # TODO Z > 1 is not tested --ATr,2026mar12
    qe = -ESU  # electron charge

    # inverse density scale length ~ (WHAM Phase 1 plasma radius)^-1 in cm^-1
    # etaN < 0 for coordinate system with gradients along y, k along x, B along z
    # and del(n)/n = etaN * \hat{y}
    etaN = -1./15  # in dimensionful CGS units

    # numerical mesh for VDF calculation
    vperp_vth_vec = np.linspace(0, 4, 100)
    vprll_vth_vec = np.linspace(-4, 4, 201)

    def make_aniso_df(vperp_vth_vec, vprll_vth_vec, vth, aniso=1, floor=1e-99):
        vperp, vprll = np.meshgrid(vperp_vth_vec * vth,
                                   vprll_vth_vec * vth, indexing='ij')
        df = dr.vdf.bimaxwellian(vperp, vprll, vth, vth/aniso**0.5)
        df[ df<floor ] = floor
        return df

    # thermal velocities with sqrt(2) factors
    vthi_perp = np.sqrt(2*Ti_perp/mi)
    vthe_perp = np.sqrt(2*Te_perp/me)

    ion = dr.species.KineticVDFGrid(
        mass = mi,
        charge = qi,
        vperp_vec = vperp_vth_vec * vthi_perp,
        vprll_vec = vprll_vth_vec * vthi_perp,
        df = make_aniso_df(
            vperp_vth_vec,
            vprll_vth_vec,
            vthi_perp,
            aniso = Ti_aniso,
        ),
    )

    lec = dr.species.KineticVDFGrid(
        mass = me,
        charge = qe,
        vperp_vec = vperp_vth_vec * vthe_perp,
        vprll_vec = vprll_vth_vec * vthe_perp,
        df = make_aniso_df(
            vperp_vth_vec,
            vprll_vth_vec,
            vthe_perp,
            aniso = Te_aniso,
        ),
    )

    del vthi_perp, vthe_perp

    # ----------------------------------------------------------------
    # Define the (k,\omega) mesh for susceptibility calculation

    # good for baseline interchange test
    #k_vec        = np.linspace(1e-5, 2, 121, dtype=np.float64)
    #omega_re_vec = np.linspace(1e-8, 0.05, 201, dtype=np.float64)
    #omega_im_vec = np.linspace(0, 0.04, 181, dtype=np.float64)

    # for quick testing with "loop" setup
    k_vec = np.array([0.08])
    #omega_re_vec = np.linspace(1e-8, 0.008, 41, dtype=np.float64)
    #omega_im_vec = np.linspace(0, 0.016, 41, dtype=np.float64)
    omega_re_vec = np.linspace(1e-8, 0.008, 21, dtype=np.float64)  # EVEN QUICKER TEST
    omega_im_vec = np.linspace(0, 0.016, 21, dtype=np.float64)

    # Do you want to compute damped modes?
    # Comment or uncomment the relevant block of code as needed.
    # WARNING I don't treat the resonant integral rigorously,
    # so my damped mode calculation may not be correct...

    # unstable+normal modes only
    if not np.any(omega_im_vec < 0.):
        print('Get normal modes; add Im(omega)<0 point')
        omega_im_vec = np.insert(omega_im_vec, 0, -1*omega_im_vec[1])

    # all damped+unstable+normal modes
    #if not np.any(omega_im < 0.):
    #    print('Get normal+damped modes; add Im(omega)<0 points to case={}'.format(label))
    #    case['omega_im_vec'] = np.concatenate((-1*omega_im[:0:-1], omega_im))

    # ----------------------------------------------------------------
    # Prepare the calculation

    # convert user input to dimension-ful CGS units
    solve_grid = dr.chi.WaveGrid( k_vec / ion.rLs(B0),
                                  omega_re_vec * ion.Omcs(B0),
                                  omega_im_vec * ion.Omcs(B0),
                                  proc_layout = proc_layout )

    calc_i = dr.chi.BounceAvgESPerp( grid = solve_grid,
                                     species = ion,
                                     field = field, )

    calc_e = dr.chi.BounceAvgESPerp( grid = solve_grid,
                                     species = lec,
                                     field = field, )

    # takes a few seconds for 40x100x201 grid on perlmutter
    calc_i.setup_bounce_average(NS_RESOLUTION=ns_resolution)
    calc_e.setup_bounce_average(NS_RESOLUTION=ns_resolution)

    # ----------------------------------------------------------------

    chi_i = calc_i.chi_GK(
        ns = n0,  # midplane value in cm^-3
        epsilonN = etaN,  # signed midplane value in cm^-1
        Gforce = 0,  # signed midplane value, in cm/s^2
        Teff_ceiling = 1e5 * ERG_PER_EV,  # 100 keV
        method = ion_method,
        enable_Upsilon = False,
    )

    chi_e = calc_e.chi_GK(
        ns = n0,
        epsilonN = etaN,
        Gforce = 0,
        Teff_ceiling = 1e5 * ERG_PER_EV,  # 100 keV
        method = 'expand',
    )

    # dispersion relation computed on 3D grid of (k,Re(omega),Im(omega))
    dd = 1. + chi_i + chi_e
    if solve_grid.world_size > 1:
        dd_loc = dd
        dd = solve_grid.gather(dd_loc)

    # ----------------------------------------------------------------

    if solve_grid.rank != 0:
        return  # kinda hacky

    # dispersion relation roots
    k_root, omega_re_root, omega_im_root, absd_root = solve_grid.roots(np.abs(dd))
    stable = omega_im_root == 0.
    unstab = omega_im_root > 0
    damped = omega_im_root < 0
    # probably better to define weak/strong damping w.r.t. Re(omega)...
    wkdamp = np.logical_and(damped, np.abs(omega_im_root) < 0.5*ion.Omcs(B0))  # weakly damped
    strdamp = np.logical_and(damped, np.abs(omega_im_root) >= 0.5*ion.Omcs(B0))  # strongly damped

    k_root *= ion.rLs(B0)  # convert to dimensionless coordinates
    omega_re_root /= ion.Omcs(B0)  # convert to dimensionless coordinates
    omega_im_root /= ion.Omcs(B0)  # convert to dimensionless coordinates

    print("found local minima (possible roots)")
    print("k = ", k_root[0])
    print("Re(omega) =", omega_re_root[0])
    print("Im(omega) =", omega_im_root[0])
    print("expected",  omega_re_vec[2]) # 0.0008
    print("expected",  omega_im_vec[10])  # 0.00072

    print("delta", omega_re_root[0] - omega_re_vec[2] )
    print("delta", omega_im_root[0] / omega_im_vec[2] )
    print("relative delta", np.abs( omega_re_root[0] - omega_re_vec[2] ) / omega_re_vec[2] )
    print("relative delta", np.abs( omega_im_root[0] - omega_im_vec[10] ) / omega_im_vec[10] )

    # ----------------------------------------------------------------

    #plt.imshow(
    #    np.abs(dd[0,:,:]).T, origin='lower',
    #    extent=(omega_re_vec[0], omega_re_vec[-1],
    #           omega_im_vec[0], omega_im_vec[-1]),
    #    norm=mpl.colors.LogNorm(),
    #    cmap='turbo',
    #)
    #plt.title(r'$|D|$')
    #plt.xlabel(r'Re($\omega$)')
    #plt.ylabel(r'Im($\omega$)')
    #plt.colorbar()
    #plt.savefig('interchange_test.png', dpi=300, bbox_inches='tight')
    #plt.clf()
    #plt.close()

    return k_root, omega_re_root, omega_im_root


def within_rtol(test, truth, rtol=None, dtype=np.float64):
    """Test two scalar numbers for agreement to machine precision"""
    if rtol is None:
        rtol = np.finfo(dtype).resolution
    return np.abs( (test - truth) / truth ) < rtol


@pytest.mark.mpi_skip
def test_interchange_expand():

    k_root, omega_re_root, omega_im_root = interchange(ion_method='expand')

    assert k_root.size == 1
    assert omega_re_root.size == 1
    assert omega_im_root.size == 1

    assert within_rtol( omega_re_root[0], 0.000800009, rtol = 1e-9)
    assert within_rtol( omega_im_root[0], 0.0072, rtol = 1e-8)


@pytest.mark.mpi_skip
def test_interchange_loop5d():

    k_root, omega_re_root, omega_im_root = interchange(ion_method='loop5d')

    assert k_root.size == 1
    assert omega_re_root.size == 1
    assert omega_im_root.size == 1

    assert within_rtol( omega_re_root[0], 0.000800009, rtol = 1e-9)
    assert within_rtol( omega_im_root[0], 0.0072, rtol = 1e-8)


@pytest.mark.mpi_skip
def test_interchange_loop6d():

    k_root, omega_re_root, omega_im_root = interchange(ion_method='loop6d')

    assert k_root.size == 1
    assert omega_re_root.size == 1
    assert omega_im_root.size == 1

    assert within_rtol( omega_re_root[0], 0.000800009, rtol = 1e-9)
    assert within_rtol( omega_im_root[0], 0.0072, rtol = 1e-8)


@pytest.mark.mpi(min_size=4)
def test_interchange_loop6d_nproc4():

    result = interchange(ion_method='loop6d', proc_layout=(1,2,2))

    if result is None:

        #assert 1 == 0
        # WARNING with pytest-mpi plugin
        # the result MUST be tested on rank = 0
        # TODO need to move final data / checks from cartesian comm rank = 0
        # to global comm rank = 0 . . . right now it works ok still, but could
        # fail in the future --ATr,2026mar14

        return

    else:
        # only one rank gathers data,
        # it doesn't have to be MPI.COMM_WORLD.Get_rank() == 0
        # due to use of MPI_Cart_create(...)

        assert 1 == 0

        k_root, omega_re_root, omega_im_root = result

        assert k_root.size == 1
        assert omega_re_root.size == 1
        assert omega_im_root.size == 1

        assert within_rtol( omega_re_root[0], 0.000800009, rtol = 1e-9)
        assert within_rtol( omega_im_root[0], 0.0072, rtol = 1e-8)


if __name__ == '__main__':

    if MPI.COMM_WORLD.Get_size() == 1:

        test_interchange_expand()
        test_interchange_loop5d()
        test_interchange_loop6d()
        print("To test MPI functionality, rerun this script with 4 MPI ranks")

    elif MPI.COMM_WORLD.Get_size() == 4:

        test_interchange_loop6d_nproc4()
        print("To test other functionality, rerun this script with 1 MPI rank")

    else:

        print("Invalid number of MPI ranks, use 1 or 4")
