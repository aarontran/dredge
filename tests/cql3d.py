#!/usr/bin/env python
"""
Test loading a CQL3D Fokker-Planck NetCDF distribution into a KineticVDFGrid
via dredge.species.CQL3DVDFGrid.
"""

from pathlib import Path

import numpy as np
import pytest

from mpi4py import MPI

import dredge as dr
from dredge.const import CLIGHT, GAUSS_PER_TESLA

# example CQL3D files not shipped with public repo, so skip tests if absent
_DAT = Path(__file__).resolve().parent.parent / 'data'
CQL3D_PATH = _DAT / 'pizzo_cql3d_0106.nc'                            # ngen=1
CQL3D_MULTI = _DAT / 'WHAM_baseline_edge_nH_3e17m-3_20250602.nc'     # ngen=2 (D,e)

requires_single = pytest.mark.skipif(
    not CQL3D_PATH.exists(),
    reason=f'CQL3D data file not present: {CQL3D_PATH}')
requires_multi = pytest.mark.skipif(
    not CQL3D_MULTI.exists(),
    reason=f'CQL3D data file not present: {CQL3D_MULTI}')


@pytest.mark.mpi_skip
@requires_single
def test_cql3d_load():
    """CQL3DVDFGrid builds a normalized, physical (vperp, vprll) distribution."""
    electron = dr.species.CQL3DVDFGrid(CQL3D_PATH, radial_index=0)

    # builds on the expected grid and renormalizes to unit density
    assert electron.df.shape == (electron.vperp_vec.size, electron.vprll_vec.size)
    assert np.isclose(electron.moment(1.), 1.0, rtol=1e-9)

    # physical distribution and moments
    assert np.all(electron.df > 0.)               # floored, no exact zeros
    assert np.isfinite(electron.Tperp) and electron.Tperp > 0.
    assert np.isfinite(electron.Tprll) and electron.Tprll > 0.

    # general species in this file is electrons (bnumb=-1)
    assert electron.charge < 0.
    assert electron.mass > 0.

    # relativistic deproject keeps the velocity grid sub-luminal
    assert electron.vperp_vec.max() < CLIGHT
    assert np.abs(electron.vprll_vec).max() < CLIGHT

    # provenance metadata recorded
    assert electron.radial_index == 0
    assert 0. < electron.rya < 1.    # flux-surface label, dimensionless
    assert electron.Rp > 0.          # physical outerboard radius, cm
    assert electron.b_midplane > 0.  # min |B| on flux surface


@pytest.mark.mpi_skip
@requires_single
def test_cql3d_composes_with_fieldlinevdf():
    """A loaded CQL3D VDF works as the midplane source for the Liouville map."""
    ion = dr.species.CQL3DVDFGrid(CQL3D_PATH, radial_index=0)
    field = dr.field.ParabolicFieldLine(
        B0=0.26*GAUSS_PER_TESLA, Bt=17*GAUSS_PER_TESLA, Lp=98.,
        r0=0.2, z0=0., ds=1.0, n_steps=90, axis_r=1, axis_z=2,
    )
    prof = dr.orbit.FieldLineVDF(ion, field)
    # density is referenced to the midplane, so density(0)=1 by construction
    assert np.isclose(prof.density(0.), 1.0, rtol=1e-9)


@pytest.mark.mpi_skip
@requires_multi
def test_cql3d_multi_species():
    """Load each general species from a 2-species (D, e) file"""
    D = dr.species.CQL3DVDFGrid(CQL3D_MULTI, species_index=0)   # deuterium
    e = dr.species.CQL3DVDFGrid(CQL3D_MULTI, species_index=1)   # electron

    # distinct species pulled from the leading f axis: D heavy & positive,
    # e light & negative
    assert D.mass > e.mass
    assert D.charge > 0. and e.charge < 0.
    # both normalized, physical
    for sp in (D, e):
        assert np.isclose(sp.moment(1.), 1.0, rtol=1e-9)
        assert np.all(sp.df > 0.)
        assert sp.Tperp > 0. and sp.Tprll > 0.

    # out-of-range species index raises
    with pytest.raises(AssertionError):
        dr.species.CQL3DVDFGrid(CQL3D_MULTI, species_index=2)


@pytest.mark.mpi_skip
@requires_single
def test_cql3d_radial_index_bounds():
    """An out-of-range radial surface raises."""
    with pytest.raises(AssertionError):
        dr.species.CQL3DVDFGrid(CQL3D_PATH, radial_index=999)


if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_size() != 1:
        print("Run this test with 1 MPI rank")
    else:
        # skipif markers only apply under pytest; guard by hand when run direct
        if CQL3D_PATH.exists():
            test_cql3d_load()
            test_cql3d_composes_with_fieldlinevdf()
            test_cql3d_radial_index_bounds()
        else:
            print(f"SKIP single-species tests, data file not present: {CQL3D_PATH}")
        if CQL3D_MULTI.exists():
            test_cql3d_multi_species()
        else:
            print(f"SKIP multi-species test, data file not present: {CQL3D_MULTI}")
        print("CQL3D reader tests done")
