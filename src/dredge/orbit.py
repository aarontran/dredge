"""
Map a velocity distribution along a magnetic field line using Liouville's
theorem, to obtain spatial profiles of density and pressure and the local
distribution function at any point.
"""

import numpy as np

from .species import KineticVDFGrid


class FieldLineVDF:
    """
    Collisionless (Liouville) map of a species' midplane velocity
    distribution along a magnetic field line.

    Profiles are returned PER UNIT MIDPLANE DENSITY n0 (the species distribution
    carries no absolute density): density(s) is the dimensionless ratio
    n(s)/n0, and Pperp(s)/Pprll(s) are in erg (multiply by n0 [cm^-3] for
    erg/cm^3).  At s=0 the map is the identity, so density(0)=1 and
    Pperp(0)/Pprll(0) equal the species' midplane Tperp/Tprll.
    """

    def __init__(self, species, field, vperp_vec=None, vprll_vec=None):
        r"""
        Collisionless (Liouville) map of a species' midplane velocity
        distribution along a magnetic field line.

        Inputs:
            species: a dredge.species.Species with a df_at(vperp, vprll) method
                     (KineticVDFGrid, Maxwellian, BiMaxwellian)
            field:   a dredge.field.FieldLine (e.g. PleiadesFieldLine);
                     |B| must increase monotonically from B0 at s=0
            vperp_vec, vprll_vec: optional 1D velocity grids (cm/s) for the
                     moment quadrature.  Default to the species' own grids if it
                     has them (KineticVDFGrid); otherwise build symmetric grids
                     out to ~5 thermal speeds from the species' vth_perp/vth_prll.
        """
        self.species = species
        self.field = field
        self.B0 = field.query_Bmag_at([0.])[0]

        # choose the velocity-space quadrature grid for moments
        if vperp_vec is None or vprll_vec is None:
            if hasattr(species, 'vperp_vec') and hasattr(species, 'vprll_vec'):
                self.vperp_vec = species.vperp_vec
                self.vprll_vec = species.vprll_vec
            else:
                self.vperp_vec = np.linspace(0., 5.*species.vth_perp, 200)
                self.vprll_vec = np.linspace(-5.*species.vth_prll, 5.*species.vth_prll, 401)
        else:
            self.vperp_vec = np.asarray(vperp_vec)
            self.vprll_vec = np.asarray(vprll_vec)

        # midplane density integral I(0) = local_species(0).norm; density(s) is
        # referenced to this so density(0)=1 on whatever quadrature grid is used
        self._norm = self.local_species(0.).norm  # this should be ~1.

    # ------------------------------------------------------------------
    # local distribution (Liouville map); all velocity-space integration is
    # delegated to KineticVDFGrid via local_species(s)
    # ------------------------------------------------------------------

    def _liouville_map(self, s, vperp, vprll):
        r"""
        Inverse Liouville map: evaluate the midplane distribution f0 at the
        midplane velocities (vperp0, vprll0) that map to LOCAL velocities
        (vperp, vprll) at arc length s.  By Liouville's theorem this equals the
        local distribution f(s, vperp, vprll), UNnormalized.
        Inputs:
            s = arc length in cm (CGS unit), scalar
            vperp = local perpendicular velocity in cm/s, numpy.ndarray
            vprll = local parallel velocity in cm/s, numpy.ndarray;
                    vperp and vprll broadcast against each other.
        Returns:
            f in (cm/s)^(-3), shaped as the broadcast of vperp and vprll.
        """
        b = self.field.query_Bmag_at(s) / self.B0   # scalar s only
        vperp0 = vperp * np.sqrt(1.0 / b)
        # +1 at vprll=0 (turning point; orbit symmetric there).  np.sign(0)=0
        # would zero out the magnitude on that line and bias the moments.
        sgn = np.where(vprll < 0., -1.0, 1.0)
        vprll0 = sgn * np.sqrt(vprll**2 + vperp**2 * (1.0 - 1.0/b))
        return self.species.df_at(vperp0, vprll0)

    def local_species(self, s):
        r"""
        Local velocity distribution at arc length s, represented as a
        KineticVDFGrid normalized to unit density.  Composable, and exposes the
        species moment machinery (.Tperp, .Tprll, .moment, .norm, .df_at).  Use
        density(s) for the physical n(s)/n0 scale factor.  The (vperp,vprll)
        grid is interpreted as LOCAL coordinates at s.
        Inputs:
            s = arc length in cm (CGS unit), scalar
        Returns:
            dredge.species.KineticVDFGrid instance for the local distribution,
            normalized so that \int f d^3v = 1.
        """
        # local VDF on quadrature grid,
        _local_df = self._liouville_map(s, self.vperp_vec[:,np.newaxis],
                                        self.vprll_vec[np.newaxis,:])
        return KineticVDFGrid(self.species.mass, self.species.charge,
                              self.vperp_vec, self.vprll_vec, _local_df)

    # ------------------------------------------------------------------
    # profiles along the field line (per unit midplane density n0)
    # ------------------------------------------------------------------

    def _shape_out(self, s, arr):
        """Collapse to scalar if original input s was scalar"""
        return arr[0] if np.ndim(s) == 0 else arr

    def density(self, s):
        """
        Number density relative to the midplane value, n(s)/n0.
        Inputs:
            s = arc length in cm (CGS unit), scalar or 1D numpy.ndarray
        Returns:
            n(s)/n0, dimensionless; scalar if s is scalar, else 1D array of the
            same shape as s.  density(0)=1.
        """
        out = np.array([self.local_species(si).norm / self._norm
                        for si in np.atleast_1d(s)])
        return self._shape_out(s, out)

    def Tperp(self, s):
        """
        Perpendicular temperature of the local distribution.
        Inputs:
            s = arc length in cm (CGS unit), scalar or 1D numpy.ndarray
        Returns:
            Tperp(s) in erg; scalar if s is scalar, else 1D array matching s.
            Tperp(0) recovers the species midplane Tperp.
        """
        out = np.array([self.local_species(si).Tperp for si in np.atleast_1d(s)])
        return self._shape_out(s, out)

    def Tprll(self, s):
        """
        Parallel temperature of the local distribution.
        Inputs:
            s = arc length in cm (CGS unit), scalar or 1D numpy.ndarray
        Returns:
            Tprll(s) in erg; scalar if s is scalar, else 1D array matching s.
            Tprll(0) recovers the species midplane Tprll.
        """
        out = np.array([self.local_species(si).Tprll for si in np.atleast_1d(s)])
        return self._shape_out(s, out)

    def Pperp(self, s):
        """
        Perpendicular pressure per unit midplane density n0.
        Inputs:
            s = arc length in cm (CGS unit), scalar or 1D numpy.ndarray
        Returns:
            Pperp(s) in erg (multiply by n0 in cm^-3 for erg/cm^3); scalar if s
            is scalar, else 1D array matching s.  Pperp(0)=species.Tperp.
        """
        return self.density(s) * self.Tperp(s)

    def Pprll(self, s):
        """
        Parallel pressure per unit midplane density n0.
        Inputs:
            s = arc length in cm (CGS unit), scalar or 1D numpy.ndarray
        Returns:
            Pprll(s) in erg (multiply by n0 in cm^-3 for erg/cm^3); scalar if s
            is scalar, else 1D array matching s.  Pprll(0)=species.Tprll.
        """
        return self.density(s) * self.Tprll(s)

    def vdf(self, s, vperp=None, vprll=None):
        r"""
        Local distribution f(s, vperp, vprll) via the inverse Liouville map,
        normalized to unit density (\int f d^3v = 1), matching local_species(s).
        Multiply by density(s) for the physical, n0-relative distribution.

        Inputs:
            s = arc length in cm (CGS unit), scalar
            vperp = perpendicular velocity in cm/s, scalar or numpy.ndarray
            vprll = parallel velocity in cm/s, scalar or numpy.ndarray;
                    vperp and vprll are broadcast against each other.  Both
                    default to the quadrature grid as a 2D (nvperp, nvprll) mesh.
        Returns:
            f in (cm/s)^(-3), normalized to unit density; shape is the broadcast
            of vperp and vprll.
        """
        loc = self.local_species(s)  # Liouville maps the grid once; .df is f(s) / norm
        if vperp is None or vprll is None:
            # default grid: the normalized local VDF is already stored in loc
            return loc.df
        # arbitrary velocities: map the requested points, normalize by I(s)
        return self._liouville_map(s, vperp, vprll) / loc.norm
