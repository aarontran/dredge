#!/usr/bin/env python
"""
Test the collisionless Liouville map of a velocity distribution along a field
line (dredge.orbit.FieldLineVDF).

Key physics invariant exercised here: a full isotropic Maxwellian source maps to
a spatially UNIFORM density with NO induced anisotropy along the field line.
The step-function cutoff in magnetic moment mu at each position (trapped
particles that mirror before s are absent) is exactly compensated by the 1/vprll
dwell-time weighting of the survivors near their turning points.  This is the
regression guard for the np.sign(0)=0 turning-point bug.
"""

import numpy as np
import pytest

from mpi4py import MPI

import dredge as dr
from dredge.const import ERG_PER_EV, Q_ELEMENTARY, M_PROTON, GAUSS_PER_TESLA


def _mirror_field():
    """Parabolic mirror field line with monotone, non-trivial |B| variation."""
    return dr.field.ParabolicFieldLine(
        B0=0.26*GAUSS_PER_TESLA, Bt=0.78*GAUSS_PER_TESLA, Lp=98.,
        r0=0.233, z0=0., ds=1.0, n_steps=90, axis_r=1, axis_z=2,
    )


def _isotropic_kinetic_grid(T):
    """Isotropic-Maxwellian KineticVDFGrid ion at temperature T (erg)."""
    vth = np.sqrt(2*T/M_PROTON)
    vperp = np.linspace(0., 6., 200) * vth
    vprll = np.linspace(-6., 6., 401) * vth
    VP, VL = np.meshgrid(vperp, vprll, indexing='ij')
    df = dr.vdf.bimaxwellian(VP, VL, vth, vth)
    return dr.species.KineticVDFGrid(mass=M_PROTON, charge=Q_ELEMENTARY,
                                     vperp_vec=vperp, vprll_vec=vprll, df=df)


@pytest.mark.mpi_skip
def test_liouville_midplane_identity():
    """At s=0 the map is the identity: density=1, Pperp=Tperp, Pprll=Tprll."""
    field = _mirror_field()
    Ti = 100 * ERG_PER_EV
    ion = _isotropic_kinetic_grid(Ti)
    profile = dr.orbit.FieldLineVDF(ion, field)

    # density normalized to unit midplane density
    assert np.isclose(profile.density(0.), 1.0, rtol=1e-9)
    # quadrature uses the same grid + rule as KineticVDFGrid.Tperp/Tprll, and
    # df_at at s=0 hits exact grid nodes, so these match to ~machine precision
    assert np.isclose(profile.Pperp(0.), ion.Tperp, rtol=1e-9)
    assert np.isclose(profile.Pprll(0.), ion.Tprll, rtol=1e-9)


@pytest.mark.mpi_skip
def test_liouville_isotropic_is_flat():
    """
    Analytic isotropic Maxwellian: density must be exactly flat and Pperp/Pprll
    exactly constant along the field line (no spurious anisotropy).  Uses the
    closed-form df_at (no interpolation), so the only thing under test is the
    map + quadrature.  This fails if the turning-point sign bug returns.
    """
    field = _mirror_field()
    Ti = 100 * ERG_PER_EV
    ion = dr.species.Maxwellian(M_PROTON, Q_ELEMENTARY, Ti)
    profile = dr.orbit.FieldLineVDF(ion, field)

    s = np.linspace(0., field.s[-1]*0.8, 6)
    # significant field variation must be present for the test to be meaningful
    assert field.query_Bmag_at(s)[-1] / field.Bmag[0] > 1.5

    n = profile.density(s)
    ratio = profile.Pperp(s) / profile.Pprll(s)

    # density flat at 1 to machine precision (integrand is node-for-node f0)
    assert np.allclose(n, 1.0, atol=1e-10)
    # anisotropy ratio constant in s (its small offset from 1 is fixed grid
    # discretization, NOT a physical s-dependent anisotropy)
    assert np.allclose(ratio, ratio[0], rtol=1e-9)


@pytest.mark.mpi_skip
def test_liouville_value_conservation():
    """vdf(s, vperp, vprll) equals f0 at the hand-computed back-mapped velocity."""
    field = _mirror_field()
    Ti = 100 * ERG_PER_EV
    ion = _isotropic_kinetic_grid(Ti)
    profile = dr.orbit.FieldLineVDF(ion, field)

    vth = np.sqrt(2*Ti/M_PROTON)
    s = field.s[-1] * 0.6
    b = field.query_Bmag_at(np.array([s]))[0] / field.Bmag[0]
    assert b > 1.0

    vperp, vprll = 2.0*vth, 1.5*vth
    vperp0 = vperp * np.sqrt(1.0/b)
    vprll0 = np.sqrt(vprll**2 + vperp**2 * (1.0 - 1.0/b))  # vprll>0 -> +sign
    # vdf is normalized to unit density: f_s / I(s), I(s)=local_species(s).norm
    expected = ion.df_at(vperp0, vprll0) / profile.local_species(s).norm

    assert np.isclose(profile.vdf(s, vperp, vprll), expected, rtol=1e-12)


if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_size() == 1:
        test_liouville_midplane_identity()
        test_liouville_isotropic_is_flat()
        test_liouville_value_conservation()
        print("Liouville map tests passed")
    else:
        print("Run this test with 1 MPI rank")
