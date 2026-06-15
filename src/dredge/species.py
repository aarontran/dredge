"""
Represent individual particle species in an ionized plasma.
Total density is not yet specified; it enters via susceptibility calculation.
"""

import numpy as np

from scipy.interpolate import RegularGridInterpolator

from .const import CLIGHT
from . import vdf


class Species(object):
    """
    Particle species in an ionized plasma
    """

    def __init__(self, mass, charge):
        """
        Particle species in an ionized plasma
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
        """
        self.mass = mass
        self.charge = charge
        self.Tperp = np.nan
        self.Tprll = np.nan

    # ----------------------------------------------------------
    # derived constants that are fully determined by (q, m, VDF)
    # ----------------------------------------------------------

    @property
    def m(self):
        return self.mass

    @property
    def q(self):
        return self.charge

    @property
    def vth_perp(self):
        """
        Perpendicular thermal velocity in cm/s, with sqrt(2) factor
        """
        return np.sqrt(2*self.Tperp/self.mass)

    @property
    def vth_prll(self):
        """
        Parallel thermal velocity in cm/s, with sqrt(2) factor
        """
        return np.sqrt(2*self.Tprll/self.mass)

    # ------------------------------------
    # derived constants that require (n,B)
    # ------------------------------------

    def Omcs(self, B):
        """
        Cyclotron frequency (signed) in rad/s
        Input: B = magnetic field in Gauss
        """
        return self.charge * B / (self.mass * CLIGHT)

    def omps(self, n):
        """
        Plasma frequency (single species) in rad/s
        Input: n = single-species number density in cm^-3
        """
        return np.sqrt( 4*np.pi * n * (self.charge)**2 / self.mass )

    def rLs(self, B):
        """
        Larmor radius in cm, strictly non-negative
        Input: B = magnetic field in Gauss
        """
        return self.vth_perp / np.abs(self.Omcs(B))

    def c_omps(self, n):
        """
        Plasma inertial length aka skin depth
        Input: n = single-species number density in cm^-3
        """
        return CLIGHT / self.omps(n)


class Maxwellian(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    isotropic Maxwellian velocity distribution.
    """

    def __init__(self, mass, charge, T):
        """
        Particle species in an ionized plasma represented by a non-relativistic
        isotropic Maxwellian velocity distribution.
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            temperature in ergs, with Boltzmann factor absorbed
        """
        super().__init__(mass, charge)
        self.Tperp = T
        self.Tprll = T

    def df_at(self, vperp, vprll):
        r"""
        Evaluate the normalized distribution f0(vperp,vprll) at arbitrary
        velocities, broadcasting over input numpy array shapes.
        Inputs:
            vperp in cm/s
            vprll in cm/s
        Returns f0 in (cm/s)^(-3), normalized so \int f0 d^3v = 1.
        """
        return vdf.bimaxwellian(vperp, vprll, self.vth_perp, self.vth_prll)


class BiMaxwellian(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    bi-Maxwellian velocity distribution.
    """

    def __init__(self, mass, charge, Tperp, Tprll):
        """
        Particle species in an ionized plasma represented by a non-relativistic
        bi-Maxwellian velocity distribution.
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            temperature in ergs, with Boltzmann factor absorbed
        """
        super().__init__(mass, charge)
        self.Tperp = Tperp
        self.Tprll = Tprll

    def df_at(self, vperp, vprll):
        r"""
        Evaluate the normalized distribution f0(vperp,vprll) at arbitrary
        velocities, broadcasting over input numpy array shapes.
        Inputs:
            vperp in cm/s
            vprll in cm/s
        Returns f0 in (cm/s)^(-3), normalized so \int f0 d^3v = 1.
        """
        return vdf.bimaxwellian(vperp, vprll, self.vth_perp, self.vth_prll)


class KineticPerpVDFGrid(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    velocity distribution on a numerical grid (v_perp,).
    """

    def __init__(self, mass, charge, vperp_vec, df_reduced):
        r"""
        Particle species in an ionized plasma represented by a non-relativistic
        velocity distribution on a numerical grid (v_perp,).
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            vperp_vec in cm/s, 1D numpy array
            df_reduced in (cm/s)^(-2), 1D numpy array, normalized so that
                \int F_{reduced} 2*pi*v_\perp d(v_\perp) = 1.
                During initialization, df values will be adjusted to enforce
                the normalization of 1.
        """
        super().__init__(mass, charge)

        assert df_reduced.ndim == 1
        assert df_reduced.shape == vperp_vec.shape

        self.vperp_vec = vperp_vec
        self.df_reduced = df_reduced
        self.df_reduced = self.df_reduced / self.moment(1.)

        self.Tperp = self.mass * 0.5 * self.moment(self.vperp_vec**2)
        self.Tprll = np.nan

    def moment(self, x):
        """
        Compute a velocity-space moment
        Input:
            x = scalar, or numpy array with shape == self.df_reduced.shape
        """
        if np.ndim(x) == 0:
            x = x * np.ones_like(self.df_reduced)
        assert x.ndim == self.df_reduced.ndim
        return np.trapezoid(x * self.df_reduced * 2*np.pi*self.vperp_vec,
                            self.vperp_vec)


class KineticVDFGrid(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    velocity distribution on a numerical grid (v_perp, v_parallel).
    """

    def __init__(self, mass, charge, vperp_vec, vprll_vec, df):
        r"""
        Particle species in an ionized plasma represented by a non-relativistic
        velocity distribution on a numerical grid (v_perp, v_parallel).
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            vperp_vec in cm/s, 1D numpy array
            vprll_vec in cm/s, 1D numpy array
            df in (cm/s)^(-3), 2D numpy array, normalized so that
                \int F_{reduced} 2*pi*v_\perp d(v_\perp) d(v_\parallel) = 1.
                During initialization, df values will be adjusted to enforce
                the normalization of 1.
        """
        super().__init__(mass, charge)

        assert vperp_vec.ndim == 1
        assert vprll_vec.ndim == 1
        assert df.shape == (vperp_vec.size, vprll_vec.size)

        self.vperp_vec = np.asarray(vperp_vec)
        self.vprll_vec = np.asarray(vprll_vec)
        self.df = df
        # zeroth velocity moment of df AS SUPPLIED (its velocity-space integral);
        # retained so callers can recover the density of an unnormalized input
        # df.  Equals 1 after the normalization on the next line.
        self.norm = self.moment(1.)
        self.df = self.df / self.norm

        self.Tperp = self.mass * self.moment( 0.5*(self.vperp_vec**2)[:,np.newaxis] )
        self.Tprll = self.mass * self.moment(     (self.vprll_vec**2)[np.newaxis,:] )

    def to_perp_grid(self):
        r"""
        Reduce to a KineticPerpVDFGrid by integrating out v_parallel.
        Returns:
            KineticPerpVDFGrid with the same mass/charge and reduced
            distribution df_reduced(vperp) = \int f dv_parallel in (cm/s)^(-2)
            (re-normalized to unit density on construction).
        """
        df_reduced = np.trapezoid(self.df, self.vprll_vec, axis=1)
        return KineticPerpVDFGrid(self.mass, self.charge, self.vperp_vec,
                                  df_reduced)

    def moment(self, *args, **kwargs):
        return self.moment_bcast_left(*args, **kwargs)

    def moment_bcast_left(self, x):
        """
        Compute a velocity-space moment, broadcasting over LEADING axes
        This is "hot" code, called many times in loop, so must be fast.

        Input:
            x = scalar, or numpy array with valid shape (ndim >=2) of:
                    (..., vperp_vec.size, 1)
                    (..., 1, vprll_vec.size)
                    (..., *df.shape)
                with zero, one, two, or any number of trailing axes
                notice that 1D arrays do not work; it's unclear whether such
                arrays should broadcast over vperp or vprll
        """
        if np.ndim(x) == 0:
            x = x * np.ones_like(self.df)
        assert x.ndim >= self.df.ndim  # prevent ambiguous 1D broadcast
        #if x.ndim == 2:
        #    mom_reduced = np.trapezoid(x * self.df, self.vprll_vec, axis=-1)
        #    mom = np.trapezoid(mom_reduced * 2*np.pi*self.vperp_vec, self.vperp_vec)
        #else:
        #    target_shape = [1] * x.ndim
        #    target_shape[-2] = self.df.shape[0]  # vperp axis
        #    target_shape[-1] = self.df.shape[1]  # vprll axis
        #    df_wide = np.reshape(self.df, tuple(target_shape))
        #    target_shape = [1] * (x.ndim - 1)
        #    target_shape[-1] = self.df.shape[0]  # vperp axis
        #    vperp_wide = np.reshape(self.vperp_vec, tuple(target_shape))
        #    mom_reduced = np.trapezoid(x * df_wide, self.vprll_vec, axis=-1)
        #    mom = np.trapezoid(mom_reduced * 2*np.pi*vperp_wide, self.vperp_vec, axis=-1)
        # Take advantage of numpy's default broadcasting semantics,
        # https://numpy.org/devdocs/user/basics.broadcasting.html#general-broadcasting-rules
        mom_reduced = np.trapezoid(x * self.df, self.vprll_vec, axis=-1)
        mom = np.trapezoid(mom_reduced * 2*np.pi*self.vperp_vec, self.vperp_vec, axis=-1)
        return mom

    def moment_bcast_right(self, x):
        """
        Compute a velocity-space moment, broadcasting over TRAILING axes
        This is "hot" code, called many times in loop, so must be fast.

        Input:
            x = scalar, or numpy array with valid shape (ndim >=2) of:
                    (vperp_vec.size, 1, ...)
                    (1, vprll_vec.size, ...)
                    (*df.shape, ...)
                with zero, one, two, or any number of trailing axes
                notice that 1D arrays do not work; it's unclear whether such
                arrays should broadcast over vperp or vprll
        """
        if np.ndim(x) == 0:
            x = x * np.ones_like(self.df)
        assert x.ndim >= self.df.ndim
        if x.ndim == 2:
            mom_reduced = np.trapezoid(x * self.df, self.vprll_vec, axis=1)
            mom = np.trapezoid(mom_reduced * 2*np.pi*self.vperp_vec, self.vperp_vec)
        else:
            target_shape = [1] * x.ndim
            target_shape[0] = self.df.shape[0]  # vperp axis
            target_shape[1] = self.df.shape[1]  # vprll axis
            df_wide = np.reshape(self.df, tuple(target_shape))

            target_shape = [1] * (x.ndim - 1)
            target_shape[0] = self.df.shape[0]  # vperp axis
            vperp_wide = np.reshape(self.vperp_vec, tuple(target_shape))

            mom_reduced = np.trapezoid(x * df_wide, self.vprll_vec, axis=1)
            mom = np.trapezoid(mom_reduced * 2*np.pi*vperp_wide, self.vperp_vec, axis=0)
        return mom

    # TODO write template methods / extensions for
    # other distribution functions to help with code testing and structure
    # --ATr,2026mar08

    def df_at(self, vperp, vprll):
        r"""
        Evaluate the normalized distribution f0(vperp,vprll) at arbitrary
        velocities, broadcasting over input numpy array shapes,
        using 2D linear interpolation of the stored grid.

        Points outside the stored (vperp_vec, vprll_vec) grid return 0; this is
        intended for tail values beyond the grid where f0 is negligible.

        Inputs:
            vperp in cm/s
            vprll in cm/s
        Returns f0 in (cm/s)^(-3), normalized so \int f0 d^3v = 1.
        """
        # build the interpolator once and cache it; self.df is fixed after init
        if not hasattr(self, '_df_interp'):
            self._df_interp = RegularGridInterpolator(
                (self.vperp_vec, self.vprll_vec), self.df,
                bounds_error=False, fill_value=0.,
            )
        # broadcast_arrays raises clearly on incompatible shapes, and prevents
        # silent mispairing when shapes have equal size but differ (e.g. (2,3)
        # vs (3,2)) since ravel/reshape alone would not catch that
        vperp, vprll = np.broadcast_arrays(np.asarray(vperp), np.asarray(vprll))
        pts = np.column_stack([vperp.ravel(), vprll.ravel()])
        return self._df_interp(pts).reshape(vperp.shape)

    def compute_dF0_dEperp(self):
        """
        Compute the background distribution function gradient with respect
        to perpendicular energy, at fixed v_parallel,
        dF0/d(0.5*m*vperp^2) = 1/(m * vperp) * dF0/d(vperp) |_{vparallel}
                             = (dF0/dE |_µ + 1/B * dF0/dµ |_E)
        Returns:
            dF0/d(0.5*m*vperp^2) |_{vparallel} in dimensionful (CGS) units of
            1/erg/(cm/s)^3, stored as 2D array of shape (vperp,vprll)
        """
        # NOTE edge_order=2 is required to get correct Tperp at vperp=0
        # line, if coordinate array includes vperp=0 exactly.
        # When using edge_order=1 for isotropic Maxwellian,
        # resulting Tperp is 2x larger than true value.
        # --ATr,2026mar04
        df0_dvperp  = np.gradient(self.df,    self.vperp_vec, axis=0, edge_order=2)
        df0_dvperp2 = np.gradient(df0_dvperp, self.vperp_vec, axis=0, edge_order=2)

        # need special handling on µ=0 (vperp=0) line
        # because vperp=0, df/dvperp -> 0 and 1/(m*vperp) -> inf gives
        # indeterminate limit 0/0; apply l'Hopital's rule to bypass
        zeromu = (self.vperp_vec == 0)
        if np.any(zeromu):
            df0_dEperp = np.empty_like(self.df)
            df0_dEperp[ zeromu,:] =   df0_dvperp2[zeromu,:] / self.m
            df0_dEperp[~zeromu,:] = ( df0_dvperp[~zeromu,:]
                                      / (self.m * self.vperp_vec[~zeromu,np.newaxis]) )
        else:
            df0_dEperp = df0_dvperp / (self.m * self.vperp_vec[:,np.newaxis])
        return df0_dEperp  # shape (vperp, vprll)

    def compute_dF0_dEprll(self):
        """
        Compute the background distribution function gradient with respect
        to parallel energy, at fixed v_perp,
        dF0/d(0.5*m*vparallel^2) = 1/(m * vprll) * dF0/d(vprll) |_{vperp}
                                 = dF0/dE |_µ
        Returns:
            dF0/d(0.5*m*vparallel^2) |_{vperp} in dimensionful (CGS) units of
            1/erg/(cm/s)^3, stored as 2D array of shape (vperp,vprll)
        """
        df0_dvprll  = np.gradient(self.df,    self.vprll_vec, axis=1, edge_order=2)
        df0_dvprll2 = np.gradient(df0_dvprll, self.vprll_vec, axis=1, edge_order=2)
        # need special handling at pitch angle = 90 deg. (vprll->0) line
        # because df/dvprll -> 0 and 1/(m*vprll) -> inf gives
        # indeterminate limit 0/0; apply l'Hopital's rule to bypass
        pitch90 = (self.vprll_vec == 0)
        if np.any(pitch90):
            df0_dEprll = np.empty_like(self.df)
            df0_dEprll[:, pitch90] =   df0_dvprll2[:,pitch90] / self.m
            df0_dEprll[:,~pitch90] = ( df0_dvprll[:,~pitch90]
                                       / (self.m * self.vprll_vec[np.newaxis,~pitch90]) )
        else:
            df0_dEprll = df0_dvprll / (self.m * self.vprll_vec[np.newaxis,:])
        return df0_dEprll  # shape (vperp, vprll)
