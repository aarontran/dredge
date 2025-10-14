"""
Represent individual particle species in an ionized plasma.
Total density is not yet specified; it enters via susceptibility calculation.
"""

import numpy as np

from .const import CLIGHT


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
        return self.vth_perp() / np.abs(self.Omcs(B))

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


class KineticPerpVDFGrid(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    velocity distribution on a numerical grid (v_perp,).
    """

    def __init__(self, mass, charge, vperp_vec, df_reduced):
        """
        Particle species in an ionized plasma represented by a non-relativistic
        velocity distribution on a numerical grid (v_perp,).
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            vperp_vec in cm/s, 1D numpy array
            df_reduced in (cm/s)^(-2), 1D numpy array, normalized so that
                \int F_{reduced} 2*pi*v_\perp d(v_\perp) = 1.
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
        assert x.shape == self.df_reduced.shape
        return np.trapz(x * self.df_reduced * 2*np.pi*self.vperp_vec,
                        self.vperp_vec)


class KineticVDFGrid(Species):
    """
    Particle species in an ionized plasma represented by a non-relativistic
    velocity distribution on a numerical grid (v_perp, v_parallel).
    """

    def __init__(self, mass, charge, vperp_vec, vprll_vec, df):
        """
        Particle species in an ionized plasma represented by a non-relativistic
        velocity distribution on a numerical grid (v_perp, v_parallel).
        Inputs:
            mass in grams
            charge in ESU (Gaussian CGS unit)
            vperp_vec in cm/s, 1D numpy array
            vprll_vec in cm/s, 1D numpy array
            df in (cm/s)^(-3), 2D numpy array, normalized so that
                \int F_{reduced} 2*pi*v_\perp d(v_\perp) d(v_\parallel) = 1.
        """
        super().__init__(mass, charge)

        assert vperp_vec.ndim == 1
        assert vprll_vec.ndim == 1
        assert df.shape == (vperp_vec.size, vprll_vec.size)

        self.vperp_vec = vperp_vec
        self.vprll_vec = vprll_vec
        self.df = df
        self.df = self.df / self.moment(1.)
        self.df_reduced = np.trapz(self.df, self.vprll_vec, axis=1)

        self.Tperp = self.mass * self.moment( 0.5*(self.vperp_vec**2)[:,np.newaxis] )
        self.Tprll = self.mass * self.moment(     (self.vprll_vec**2)[np.newaxis,:] )

    def moment(self, x):
        """
        Compute a velocity-space moment
        Input:
            x = scalar, or numpy array with shape == self.df.shape
        """
        if np.ndim(x) == 0:
            x = x * np.ones_like(self.df)
        assert x.shape == self.df.shape
        mom_reduced = np.trapz(x * self.df, self.vprll_vec, axis=1)
        return np.trapz(mom_reduced * 2*np.pi*self.vperp_vec, self.vperp_vec)
