#!/usr/bin/env python
"""
Test interchange calculation for cool bi-Maxwellian plasma on a circular
(wire-like) magnetic field line with constant |B|, using the trivial
UniformOrbitAverage instead of a bounce average.

Cross-check against tests/interchange.py: choose the circle radius to match
the midplane curvature of the parabolic mirror field used there.  Both fields
are curl-free, so grad(|B|)/|B| = curvature at the midplane, and the
parabolic case concentrates plasma at z=0, so both should give the same
susceptibility and same interchange root.

THIS IS A CLAUDE-GENERATED TEST SUITE.
"""

import numpy as np

import pytest

import dredge as dr
from dredge.const import ERG_PER_EV, GAUSS_PER_TESLA, M_ELECTRON, M_PROTON, Q_ELEMENTARY


def setup(geometry='circular'):
    """
    Return ion, electron calculators and WaveGrid for interchange test.
    Input:
        geometry = 'circular' for CircularFieldLine with UniformOrbitAverage,
                   'parabolic' for ParabolicFieldLine with BounceOrbitAverage
                   (same as tests/interchange.py)
    """
    B0 = 0.26 * GAUSS_PER_TESLA  # WHAM Phase 1 magnetic field in Gauss

    # mirror field from tests/interchange.py, concentrated at z=0
    parabolic = dr.field.ParabolicFieldLine(
        B0=B0, Bt=17*GAUSS_PER_TESLA, Lp=98, r0=0.233, z0=0,
        ds=1e-6, n_steps=20, axis_r=1, axis_z=2,
    )

    if geometry == 'circular':
        # match |B| and curvature radius at midplane s=0
        s0 = np.array([0.])
        Bmid = parabolic.query_Bmag_at(s0)[0]
        Rc = 1. / np.abs(parabolic.query_dbhat_ds_at(s0)[1,0])  # ~640 cm
        field = dr.field.CircularFieldLine(B0=Bmid, R0=Rc)
        orbit = dr.bavg.UniformOrbitAverage
        orbit_kws = dict()
    elif geometry == 'parabolic':
        field = parabolic
        orbit = dr.bavg.BounceOrbitAverage
        orbit_kws = dict(NS_RESOLUTION=10)
    else:
        raise ValueError(geometry)

    n0 = 1e13  # ion density in cm^-3
    T_perp = 100 * ERG_PER_EV  # ion, electron temperature
    aniso = 2  # ratio of Tperp/Tprll for bi-Maxwellian

    vperp_vth_vec = np.linspace(0, 4, 100)
    vprll_vth_vec = np.linspace(-4, 4, 201)

    def make_species(mass, charge, floor=1e-99):
        vth = np.sqrt(2*T_perp/mass)
        vperp, vprll = np.meshgrid(vperp_vth_vec * vth,
                                   vprll_vth_vec * vth, indexing='ij')
        df = dr.vdf.bimaxwellian(vperp, vprll, vth, vth/aniso**0.5)
        df[ df<floor ] = floor
        return dr.species.KineticVDFGrid(
            mass = mass,
            charge = charge,
            vperp_vec = vperp_vth_vec * vth,
            vprll_vec = vprll_vth_vec * vth,
            df = df,
        )

    ion = make_species(M_PROTON, Q_ELEMENTARY)
    lec = make_species(M_ELECTRON, -Q_ELEMENTARY)

    # same (k,omega) mesh as tests/interchange.py
    k_vec = np.array([0.08])
    omega_re_vec = np.linspace(1e-8, 0.008, 21, dtype=np.float64)
    omega_im_vec = np.linspace(0, 0.016, 21, dtype=np.float64)
    omega_im_vec = np.insert(omega_im_vec, 0, -1*omega_im_vec[1])

    solve_grid = dr.chi.WaveGrid( k_vec / ion.rLs(B0),
                                  omega_re_vec * ion.Omcs(B0),
                                  omega_im_vec * ion.Omcs(B0) )

    orbit_i = orbit(species=ion, field=field, **orbit_kws)
    orbit_e = orbit(species=lec, field=field, **orbit_kws)

    calc_i = dr.chi.BounceAvgESPerp(grid=solve_grid, species=ion,
                                    field=field, orbit=orbit_i)
    calc_e = dr.chi.BounceAvgESPerp(grid=solve_grid, species=lec,
                                    field=field, orbit=orbit_e)

    return calc_i, calc_e, solve_grid, ion, B0, n0


def dispersion(ion_method='expand', geometry='circular'):
    """Return D(k,omega) on WaveGrid, and (k, Re(omega), Im(omega)) roots"""
    calc_i, calc_e, solve_grid, ion, B0, n0 = setup(geometry)

    etaN = -1./15  # in dimensionful CGS units
    kws = dict(ns=n0, epsilonN=etaN, Gforce=0,
               Teff_ceiling=1e5*ERG_PER_EV, expand_order=1)
    chi_i = calc_i.chi_GK(method=ion_method, **kws)
    chi_e = calc_e.chi_GK(method='expand', **kws)
    dd = 1. + chi_i + chi_e

    k_root, omega_re_root, omega_im_root, _ = solve_grid.roots(np.abs(dd))
    k_root *= ion.rLs(B0)
    omega_re_root /= ion.Omcs(B0)
    omega_im_root /= ion.Omcs(B0)

    return dd, (k_root, omega_re_root, omega_im_root)


def within_rtol(test, truth, rtol):
    return np.abs( (test - truth) / truth ) < rtol


@pytest.mark.mpi_skip
def test_uniform_orbit_average_is_s0_value():
    calc_i, _, _, _, _, _ = setup('circular')
    orb = calc_i.orbit
    assert orb.ssamp.shape[-1] == 1
    assert np.all(orb.ssamp == 0.)
    assert np.all(orb.B_B0 == 1.)
    assert np.all(orb.r_r0 == 1.)
    x = np.random.default_rng(0).normal(size=orb.ssamp.shape)
    assert np.array_equal(orb.bounce_average_norm_raw(x), x[...,0])


@pytest.mark.mpi_skip
@pytest.mark.parametrize('ion_method', ['expand', 'loop5d', 'loop6d'])
def test_circular_interchange(ion_method):

    dd, (k_root, omega_re_root, omega_im_root) = dispersion(ion_method, 'circular')

    assert k_root.size == 1
    assert omega_re_root.size == 1
    assert omega_im_root.size == 1

    # same root as tests/interchange.py
    assert within_rtol( omega_re_root[0], 0.000800009, rtol = 1e-9)
    assert within_rtol( omega_im_root[0], 0.0072, rtol = 1e-8)


@pytest.mark.mpi_skip
def test_circular_matches_parabolic():

    dd_circ, _ = dispersion('expand', 'circular')
    dd_para, _ = dispersion('expand', 'parabolic')

    # parabolic field has tiny variation along s ~ 1e-5 cm and
    # finite-difference curvature, so agreement is close but not exact;
    # measured max relative difference ~5e-11 --ATr,2026oct28
    reldiff = np.abs(dd_circ - dd_para) / np.abs(dd_para)
    print("max relative difference in D", np.amax(reldiff))
    assert np.amax(reldiff) < 1e-8


if __name__ == '__main__':
    test_uniform_orbit_average_is_s0_value()
    for m in ['expand', 'loop5d', 'loop6d']:
        test_circular_interchange(m)
    test_circular_matches_parabolic()
