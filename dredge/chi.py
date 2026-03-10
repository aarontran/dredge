"""
Code to calculate plasma susceptibilities for dispersion relation
on uniform grids of (k, Re(omega), Im(omega)) to help us quickly sweep
parameter space of (Tc/T0, nc/n0, epsilon).
"""

from __future__ import division, print_function

import numpy as np
#import scipy as sp
import scipy  # avoid collision with sp = alias for species

from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

from .special import Zfunc
from .species import Species, KineticVDFGrid
from .field import FieldLineVec
from .const import CLIGHT

# beware... changing temperature, mass, charge,
# ... requires re-computing bessel functions...
# changing density or epsilon does NOT require recomputing bessels
# changing B-field strength is complicated...
# ... it modifies both Omcs/omps and rho_Ls
# ... dont worry about it for now

# package chi / meshing together
# because the numerical approach to computing different chi's may differ
# considerably...

class WaveGrid(object):

    def __init__(self, k_vec, omega_re_vec, omega_im_vec):
        """
        Grid of (k, Re(ω), Im(ω)) for dispersion relation calculations
        Although code within this class appears to not care about (k, omega)
        normalization, you must diligently construct (k, ω) in CGS units
        because dependent code relies upon that normalization convention.

        Inputs:
            k_vec = 1D array of angular wavenumber grid points in cm^{-1}.
            omega_re_vec = 1D array, real angular frequency Re(omega)
                           grid points in rad/s.
            omega_im_vec = 1D array, imaginary angular frequency Im(omega)
                           grid points in rad/s.
        """
        self.k_vec = k_vec
        self.omega_re_vec = omega_re_vec
        self.omega_im_vec = omega_im_vec
        # dont allow any fancy/weird gridding
        assert self.k_vec.ndim == 1
        assert self.omega_re_vec.ndim == 1
        assert self.omega_im_vec.ndim == 1
        # grid points must ascend monotonically, with no duplicates
        # be careful because sign of charge enters into omega
        assert np.all(np.diff(self.omega_re_vec) > 0)
        assert np.all(np.diff(self.omega_im_vec) > 0)
        assert np.all(np.diff(self.k_vec) > 0)

    def mesh_extent(self):
        """Helper method for 2D plots of dispersion or susceptibility terms"""
        extent = np.array([-1, 1, -1, 1, -1, 1])
        if self.k_vec.size > 1:
            extent[0] = self.k_vec[0]  - np.diff(self.k_vec)[0]/2
            extent[1] = self.k_vec[-1] + np.diff(self.k_vec)[-1]/2
        if self.omega_re_vec.size > 1:
            extent[2] = self.omega_re_vec[0]  - np.diff(self.omega_re_vec)[0]/2
            extent[3] = self.omega_re_vec[-1] + np.diff(self.omega_re_vec)[-1]/2
        if self.omega_im_vec.size > 1:
            extent[4] = self.omega_im_vec[0]  - np.diff(self.omega_im_vec)[0]/2
            extent[5] = self.omega_im_vec[-1] + np.diff(self.omega_im_vec)[-1]/2
        return extent

    def grid_roots(self, arr: np.ndarray):
        """
        Helper method to trace dispersion relation roots in 3D
        (k, omega_re, omega_im) coordinates.
        For each k, seeks local minima in arr[ Re(ω), Im(ω) ].
        Returns indices into the species (k, Re(ω), Im(ω)) mesh.

        Requires D for all species, which hints that this method doesn't belong
        in this class... but live with the hacky code for now.

        Useful to have indices, in some cases, to index into abs(D), D,
        individual susceptibility grids, etc.

        Input:
            arr = abs(D) to minimize
        Output:
            inds = (3,) tuple of indices into k, Re(ω), Im(ω) mesh vectors
        """
        assert arr.ndim == 3
        assert arr.shape[0] == self.k_vec.size
        assert arr.shape[1] == self.omega_re_vec.size
        assert arr.shape[2] == self.omega_im_vec.size
        # this scheme to get the local minima
        # is 10x faster than explicit loop + conditional in regular Python
        # only unique values are (-2,0,2)
        # where +2 corresponds to local minima, -2 corresponds to local maxima
        sdd_re = np.diff(np.sign(np.diff(arr, axis=1)), axis=1)
        sdd_im = np.diff(np.sign(np.diff(arr, axis=2)), axis=2)
        # combine to find the local extrema,
        # unique values are (-4,-2,0,2,4) with +/-4 signifying minima/maxima
        sdd_om = sdd_re[:,:,1:-1] + sdd_im[:,1:-1,:]

        # get indices into array
        inds = np.nonzero(sdd_om == +4)
        # adjust for offset b/c we do not find local extrema at omega boundaries
        # need to convert tuple to list
        inds = [np.array(x) for x in inds]
        inds[1] += 1
        inds[2] += 1
        # revert to tuple now
        return tuple(inds)

    def roots(self, arr: np.ndarray):
        """
        Helper method to trace dispersion relation roots.
        Same as grid_roots(...) but apply indices to return
        actual values of k, omega, omega on the grid.

        Input:
            arr = abs(D) to minimize
        Output:
            k, Re(ω), Im(ω), arr vectors of equal length, encoding approximate
            dispersion relation root positions and value of abs(D) at its local
            extrema
        """
        inds = self.grid_roots(arr)
        if len(inds[0]) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        k_root        = self.k_vec[inds[0]]
        omega_re_root = self.omega_re_vec[inds[1]]
        omega_im_root = self.omega_im_vec[inds[2]]
        arr_root      = arr[ inds[0], inds[1], inds[2] ]
        return k_root, omega_re_root, omega_im_root, arr_root


class SlabESPerp(object):

    def __init__(self,
                 grid: WaveGrid,
                 species: Species,
                 B0: float):
        """
        Susceptibility for perpendicular electrostatic waves in a slab plasma,
        computed for one species on a grid of (k, Re(ω), Im(ω)).

        Coordinate scheme:
        * k points along x
        * grad(n) points along y, so epsilon = dn/dy
        * magnetic field points along z
        * Electron diamagnetic drift towards +k, ion towards -k for epsilon > 0

        Inputs:
            grid = dredge.chi.WaveGrid(...) instance
            species = dredge.species.Species(...) instance
            B0 = magnetic field in Gauss (CGS units)
        """
        self.grid = grid
        self.species = species
        self.B0 = B0

        # internal code works in dimensionless units
        # charge sign is not used for k rescaling,
        # but charge sign is included in Omega_{cs}
        self.k_vec        = grid.k_vec        * self.species.rLs (B = self.B0)
        self.omega_re_vec = grid.omega_re_vec / self.species.Omcs(B = self.B0)
        self.omega_im_vec = grid.omega_im_vec / self.species.Omcs(B = self.B0)

        # calculation breaks at resonant denominators
        # when omega exactly equal to cyclotron harmonics
        # so ensure we only sample non-integer values
        assert np.all(self.omega_re_vec.astype(np.int64) != self.omega_re_vec)

        # grids for broadcasting, faster than np.meshgrid(...)
        kk  = self.k_vec       [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        self.kk = kk
        self.oo = omr + 1j*omi

        # dimensionless reduced distribution F(v_perp)
        try:
            self.Freduced = species.df_reduced * species.vth_perp**2
            self.vperp    = species.vperp_vec  / species.vth_perp
        except:
            self.Freduced = None
            self.vperp    = None

        # Bessel functions convolved with F, Fprime computed on demand by user;
        # either Jn^2(...) or In(...) forms can be used
        self.bessel_Fprime = None
        self.bessel_F      = None
        # Bessel function sums must be computed on demand by user
        self.bsum0 = None
        self.bsum1 = None
        # Derivatives of bessel sums w.r.t. omega
        # used to estimate electron Landau damping
        self.bsum0p = None
        self.bsum1p = None

    # -------------------------------------------------------------------------
    # Bessel function integral and sum caching
    # -------------------------------------------------------------------------

    def cache_besselI_integrals(
            self,
            bessel_nmax = 20,
            verbose = True,
    ):
        """
        Compute modified Bessel I_n(...) terms for Maxwellian distribution of
        temperature Ts, for electrostatic dispersion relation of a slab plasma
        with a density gradient, for linear waves propagating exactly
        perpendicular to B.

        Package Bessel sum and chi code together because their normalization
        factors are linked; changes to one method affect the other.
        The besselI(...) and besselJ(...) sums are defined to agree exactly for
        a Maxwellian, up to numerical precision and discretization errors.

        Inputs:
            bessel_nmax = largest Bessel index (cyclotron harmonic) to include
                          indexing runs [0,1,2,...,bessel_nmax] inclusive
            verbose = talk while computing

        Output:
            None, but the following class attributes are updated.
            bessel_Fprime, bessel_F = arrays with shape (bessel_nmax+1, k)
            bessel_Fprime = integral 2*pi*vperp*dvperp * J_n^2 * dF/dvperp / vperp
                          = -2*exp(-λ) * I_n(λ)
            bessel_F      = integral 2*pi*vperp*dvperp * J_n^2 * F
                          = exp(-λ) * I_n(λ)
        """
        started = datetime.now()

        # lambda = k^2 rho^2 / 2 where rho^2 = 2 kB Ts/(ms*Omega_cs^2)
        lamb = (self.k_vec**2)/2

        bessel_In_Fprime = np.empty((bessel_nmax+1, self.k_vec.size))
        bessel_In_F      = np.empty((bessel_nmax+1, self.k_vec.size))
        bessel_In_Fprime_vpsq = np.empty((bessel_nmax+1, self.k_vec.size))  # TODO Used for grad(B) term, NOT IMPLEMENTED YET --ATr,2025may01

        for n in range(0, bessel_nmax+1):
            arg = np.exp(-lamb) * scipy.special.iv(n, lamb)
            # \int dF/dvperp * 1/vperp * J_n^2(z) * 2*pi*vperp dvperp
            # = -2 * e^(-λ) * I_n(λ)
            bessel_In_Fprime[n,:] = -2*arg
            # \int F * J_n^2(z) * 2*pi*vperp dvperp
            # = e^(-λ) * I_n(λ)
            bessel_In_F     [n,:] = arg
            if verbose:
                print(f'Bessel I_{n:d} integral done, elapsed', datetime.now()-started)

        self.bessel_Fprime = bessel_In_Fprime
        self.bessel_F      = bessel_In_F
        self.bessel_Fprime_vpsq = bessel_In_Fprime_vpsq  # TODO Used for grad(B) term, NOT IMPLEMENTED YET --ATr,2025may01

        return

    def cache_besselJ_integrals(
            self,
            bessel_nmax = 20,
            verbose = True,
            with_bsum2 = False,
    ):
        """
        Compute Bessel J_n(...) integrals over reduced distribution F(vperp) or
        distribution gradient dF/d(dvperp) * 1/vperp, for electrostatic
        dispersion relation of a slab plasma with a density gradient, for
        linear waves propagating exactly perpendicular to B.

        Package Bessel sum and chi code together because their normalization
        factors are linked; changes to one method affect the other.
        The besselI(...) and besselJ(...) sums are defined to agree exactly for
        a Maxwellian, up to numerical precision and discretization errors.

        Inputs:
            bessel_nmax = largest Bessel index (cyclotron harmonic) to include
                          indexing runs [0,1,2,...,bessel_nmax] inclusive

            verbose = talk while computing

            with_bsum2 = True/False, whether to compute the bessel sum integral
                for the gradB term in perp electrostatic susceptibility,
                see my notes from Tang derivation

        Output:
            None, but the following class attributes are updated.
            bessel_Fprime, bessel_F = arrays with shape (bessel_nmax+1, k)
            bessel_Fprime = integral 2*pi*vperp*dvperp * J_n^2 * dF/dvperp / vperp
            bessel_F      = integral 2*pi*vperp*dvperp * J_n^2 * F
        """

        started = datetime.now()

        # Setup (k, vperp) grid for Bessel Jn-weighted moments of F(vperp)
        # working in species-specific dimensionless units
        # k*(Larmor radius), v / v_{th,perp}, etc...
        # VDF normalization = 1 enforced by KineticPerpVDFGrid(...)
        kg      = self.k_vec                            [...,np.newaxis]
        vperpg  = self.vperp                            [np.newaxis,...]
        Fg      = self.Freduced                         [np.newaxis,...]
        Fprimeg = np.gradient(self.Freduced, self.vperp)[np.newaxis,...]

        bessel_Jnsq_Fprime = np.empty((bessel_nmax+1, self.k_vec.size))
        bessel_Jnsq_F      = np.empty((bessel_nmax+1, self.k_vec.size))
        bessel_Jnsq_Fprime_vpsq = np.empty((bessel_nmax+1, self.k_vec.size))

        for n in range(0, bessel_nmax+1):

            Jnsq = scipy.special.jv(n, kg*vperpg)**2

            # \int dF/dvperp * 1/vperp * J_n^2(z) * 2*pi*vperp dvperp
            bessel_Jnsq_Fprime[n,:] = np.trapz(Fprimeg * Jnsq * 2*np.pi, vperpg, axis=-1)

            # \int F * J_n^2(z) * 2*pi*vperp dvperp
            bessel_Jnsq_F[n,:] = np.trapz(Fg * Jnsq * 2*np.pi*vperpg, vperpg, axis=-1)

            # \int dF/dvperp * vperp * J_n^2(z) * 2*pi*vperp dvperp
            if with_bsum2:
                bessel_Jnsq_Fprime_vpsq[n,:] = np.trapz(Fprimeg*vperpg * Jnsq * 2*np.pi*vperpg, vperpg, axis=-1)

            if verbose:
                print(f'Bessel J_{n:d} integral done, elapsed', datetime.now()-started)

        self.bessel_Fprime = bessel_Jnsq_Fprime
        self.bessel_F      = bessel_Jnsq_F
        self.bessel_Fprime_vpsq = bessel_Jnsq_Fprime_vpsq

        return

    def cache_bessel_sums(self, verbose=True, with_prime=False,
                          with_bsum2=False, with_gradBdrift=False,
                          epsilonB=None, Gforce=0):
        """
        Compute sums of Bessel J_n(...) integrals or I_n(...) terms which have
        already been pre-cached by the user, for use in electrostatic
        dispersion relation of a slab plasma with a density gradient, for
        linear waves propagating exactly perpendicular to B.

        The Bessel sums are crafted to omit epsilon (spatial gradient) factors,
        so that the user can cache the Bessel sums, then recompute behavior
        quickly for varying epsilon, Omega_ci/omega_pi ~ vA/c ~ density, etc.

        WARNING = the d(chi)/d(ω) Bessel sums have not been extensively tested
        as of 2024 July 05, use at your own risk and be prepared to debug
        errors.

        Inputs:
            verbose = talk while computing
            with_prime = compute additional sums for d(chi)/dω calculation,
                         which we are using to estimate electron Landau damping...

            with_bsum2 = True/False, whether to compute the bessel sum integral
                for the gradB term in perp electrostatic susceptibility,
                see my notes from Tang derivation

            with_gradBdrift = True/False, whether to include grad(B)
                drift into the resonant denominator,but using sqrt(<vperp^2>)
                as a stand-in for vperp to avoid taking the full velocity-space
                integral
                if True, you must also provide epsilonB

            epsilonB = (cm^-1) used for grad(B) drift calculation in the
                resonant denominator

            Gforce = 0 or float, external force field acceleration (cm/s^2),
                contributes particle drift to Bessel sum resonant denominators.
                Gforce is used for both gravity and external electric fields
                (hence capital rather than lowercase G).
                Sign matters; positive G points along the +y axis.

        Output:
            None, but the following class attributes are updated.

            bsum0, bsum1 = arrays with shape (k, Re(ω), Im(ω)) where the Bessel
                           sums are defined as follows, and omega is normalized
                           to the species' cyclotron frequency

            bsum0 = sum_n 1/(ω/n - 1) * 1/k^2
                    * integral 2*pi*vperp*dvperp * J_n^2 * dF/dvperp / vperp
                    for n in [-nmax, +nmax]

                  = -1*exp(-λ)/λ * sum_n I_n / (ω/n - 1)      for n in [-nmax, +nmax]
                  = -2*exp(-λ)/λ * sum_n I_n / ((ω/n)**2 - 1) for n in [1, +nmax]

            bsum1 = sum_n 1/(ω - n)
                    * integral 2*pi*vperp*dvperp * J_n^2 * F
                    for n in [-nmax, +nmax]

                  = exp(-λ) * sum_n I_n / (ω - n)                           for n in [-nmax, +nmax]
                  = exp(-λ) * [ I_0 / ω + ω * sum_n 2*I_n / (ω**2 - n**2) ] for n in [1, +nmax]
        """
        started = datetime.now()

        # broadcast Bessel integrals from (n,k) to (n,k,Re(ω),Im(ω))
        bessel_Fprime = self.bessel_Fprime[..., np.newaxis, np.newaxis]
        bessel_F      = self.bessel_F     [..., np.newaxis, np.newaxis]
        bessel_Fprime_vpsq = self.bessel_Fprime_vpsq[..., np.newaxis, np.newaxis]

        # (k, ω) grids of shape (k,Re(ω),Im(ω))
        kk = self.kk
        oo = self.oo

        if with_gradBdrift:
            sp = self.species
            # use <vperp^2> to compute FLR drift velocity
            vpsq_moment = sp.moment( sp.vperp_vec**2 ) / sp.vth_perp**2
            # rescale epsilonB from cm^-1 to current species normalization
            epsB = epsilonB * self.species.rLs(B=self.B0)
            # remap omega -> omega + k*v_{del B}
            # which is safe to do throughout these bessel sums
            oo = oo + kk * (0.5*epsB*vpsq_moment)

        if Gforce != 0:
            # remap omega -> omega - k*G
            # where capital G is dimensionless gravitational drift,
            # allows to include other forces (electric, ...)
            # abs(Omcs) in Gforce norm is to match k normalization scheme
            G = Gforce / self.species.vth_perp / abs(self.species.Omcs(self.B0))
            oo = oo - kk * G

        # construct Bessel sums on grid (k,Re(ω),Im(ω))
        bshape = (self.k_vec.size, self.omega_re_vec.size, self.omega_im_vec.size)
        bsum0 = np.zeros(bshape, dtype='complex128')
        bsum1 = np.zeros(bshape, dtype='complex128')
        if with_prime:
            bsum0p = np.zeros(bshape, dtype='complex128')
            bsum1p = np.zeros(bshape, dtype='complex128')
        if with_bsum2:
            bsum2 = np.zeros(bshape, dtype='complex128')

        ##### another way to sum bessels, similar speed
        ##  #for n in range(0, bessel_nmax+1):
        ##  for n in range(0, bessel_Fprime.shape[0]):
        ##      if n == 0:
        ##          # Fprime contribution is zero b/c of n in numerator
        ##          bsum0 += 0
        ##          bsum1 += 1/oo * bessel_F[n,...]
        ##      else:
        ##          #bsum0 +=  n/(oo - n) * bessel_Fprime[n,...]
        ##          #bsum0 += -n/(oo + n)  * bessel_Fprime[n,...]
        ##          #bsum1 += 1./(oo - n) * bessel_F[n,...]
        ##          #bsum1 += 1./(oo + n)* bessel_F[n,...]
        ##          # micro-optimize the arithmetic operations
        ##          # cuts time from ~19.3 to 10.5 seconds on (20,401,301,202) grid
        ##          # compared to the commented-out preceding code
        ##          # TODO could simplify farther using collapsed bessel sums
        ##          #     1/(oo**2 - n**2)  and n**2/(oo**2 - n**2)
        ##          # from combining terms, going from 2 divisions to 1 division +
        ##          # 1 multiplication (squaring = multiplication, less costly than pow(...))
        ##          # but it has to be benchmarked and regression tested. --ATr,2024june28
        ##          inv_om_minus = 1./(oo - n)
        ##          inv_om_plus  = 1./(oo + n)
        ##          bsum0 += n * (inv_om_minus - inv_om_plus) * bessel_Fprime[n,...]
        ##          bsum1 +=     (inv_om_minus + inv_om_plus) * bessel_F[n,...]
        ##      if verbose:
        ##          print(f'Bessel J_{n:d} sum done, elapsed', datetime.now()-started)
        ##  # normalization factors
        ##  bsum0 *= (1./self.k_vec**2)[:,np.newaxis,np.newaxis]

        #for n in range(1, bessel_nmax+1):
        for n in range(1, bessel_Fprime.shape[0]):
            # micro-optimize the arithmetic operations
            # cuts time from ~12.4 to ~10 seconds on (20,401,301,202) grid
            # when compared to computing denominators
            #     1/((oo/n)**2-1) and 1/(oo**2-n**2)
            # separately.
            invres = 1./(oo*oo - n**2)
            bsum0 += n**2 * invres * bessel_Fprime[n,...]
            bsum1 +=        invres * bessel_F[n,...]

            if with_prime:
                invres2 = invres*invres
                bsum0p += invres2 * 1./n**2 * bessel_Fprime[n,...]
                bsum1p += (
                        invres
                        - (2/n**2)*oo*oo * invres2
                ) * bessel_F[n,...]
            if with_bsum2:
                bsum2 += invres * bessel_Fprime_vpsq[n,...]

            if verbose:
                print(f'Bessel n={n:d} summand done, elapsed', datetime.now()-started)

        # normalization factors
        # hoist outside loop to reduce arithmetic operations
        bsum0 *= 2/kk**2
        bsum1 *= 2*oo
        # handle n=0 term separately
        bsum1 += 1./oo * bessel_F[0,...]

        if with_prime:
            bsum0p *= -4*oo/(kk**2)
            bsum1p *= 2
            # handle n=0 term separately
            bsum1p += -1./oo**2 * bessel_F[0,...]
        if with_bsum2:
            bsum2 *= 2*oo
            # handle n=0 term separately
            bsum2 += bessel_Fprime_vpsq[0,...] / oo

        if verbose:
            print('Bessel n=0 summand done, elapsed', datetime.now()-started)

        # cache for future computation
        self.bsum0 = bsum0
        self.bsum1 = bsum1
        if with_prime:
            self.bsum0p = bsum0p
            self.bsum1p = bsum1p
        if with_bsum2:
            self.bsum2 = bsum2

        return

    # -------------------------------------------------------------------------
    # Susceptibilities
    # -------------------------------------------------------------------------

    def chi_fluid(self, epsilonN, ns, warm=False):
        """
        Compute susceptibility chi on grid (k, Re(ω), Im(ω)).
        Distribution function is either cold or warm Maxwellian.
        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            warm = use thermal corrections from small-k Bessel expansions
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        # notice that epsN/k/omega has omega in denominator,
        # unlike numerator placement in chi_kinetic(...)
        if warm:
            # the double expansion in small k*rhoLe and omega/Omce
            # results in different coefficients for the non-gradient
            # versus the gradient terms; the expansion here matches that
            # in Lindgren, Langdon, Birdsall (1976), Equation (2) discussion.
            lamb = (kk**2)/2  # argument to modified Bessel I_n(...)
            term0 = omps_Omcs**2 * (1 - 3./4 * lamb)
            term1 = -1 * omps_Omcs**2 * epsN/kk/oo * (1 - lamb)
            return term0 + term1
        else:
            term0 = omps_Omcs**2 * (1 - epsN/kk/oo)
            return term0

    def chi_kinetic(self, epsilonN, ns):
        """
        Compute susceptibility chi on grid (k, Re(ω), Im(ω)).
        Distribution function enters via Bessel sums.

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
            self.cache_bessel_sums(...)
        before you can compute kinetic chi.

        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        terms = self.bsum0 - epsN*oo/kk * self.bsum0 - epsN/kk * self.bsum1

        return omps_Omcs**2 * terms

    def chi_oblique_Zfunc_lowk(self, epsilonN, ns, k_parallel):
        """
        Compute low-k limit of susceptibility chi on grid (k, Re(ω), Im(ω))
        for electrostatic waves, inhomogeneous plasma, Maxwellian distribution.
        ... neglects all Bessel terms n >= 2.
        ... takes I_0(...), I_1(...) -> small argument limit.
        ... Zfunc models Maxwellian distributions with finite temperature, to
        allow a simple description of parallel Landau damping

        This method provides the combined perp+prll susceptibility response;
        it supersedes previous use of "chi_prll_Zfunc_lowk" and
        "chi_perp_fluid" which was not exactly correct in a higher-order term.

        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            k_parallel = (scalar) signed parallel angular wavenumber in cm^-1
                         When choosing sign of k_parallel, remember that omega
                         is scaled to SIGNED species cyclotron freq in plasma
                         dispersion function argument.
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        # rescale to species rho_Ls
        kp = k_parallel * self.species.rLs(self.B0)
        # plasma function argument
        zeta0s = oo / kp
        # plasma function evaluated
        Z0 = Zfunc(zeta0s)

        kksq = kk**2
        # NOTE it is tacitly assumed that kperp/kk ~ 1 for the moment...
        # -ATr,2024nov15
        #kperp = np.sqrt(kk**2 - kp**2)

        terms = omps_Omcs**2 * (
            #kperp**2/kksq * (epsN/kperp * Z0/kp - zeta0s*Z0)
            (epsN/kk * Z0/kp - zeta0s*Z0)
            + kp**2/kksq * 2./kp**2 * (1 + zeta0s*Z0)
        )
        # note that in the limit zeta0s->infty,
        # we recover omps_Omcs**2 * (1 - epsN/kk/oo) + ...
        # like in chi_perp_fluid(...)
        return terms

    # -------------------------------------------------------------------------
    # Derivatives of chi with respect to frequency omega, which can be used
    # when estimating complex roots in a weak growth approximation.
    # In practice, more useful to root find on the 3D (k, Re(ω), Im(ω)) grid.
    # -------------------------------------------------------------------------

    def chi_perp_prime_kinetic(self, epsilonN, ns):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω,
        on grid (k, Re(ω), Im(ω)).

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
            self.cache_bessel_sums(..., with_prime=True)
        first.

        Distribution function enters via Bessel sums.
        WARNING = the d(chi)/d(ω) Bessel sums have not been extensively tested
        as of 2024 July 05, use at your own risk and be prepared to debug
        errors.

        Derivative is taken as d/d(ω/Omega_cs) with respect to the current
        species' cyclotron frequency... therefore to convert between this
        species + reference species the caller should multiply result by a
        signed factor Omega_c0/Omega_cs
        """
        assert self.bsum0 is not None
        assert self.bsum1 is not None
        assert self.bsum0p is not None
        assert self.bsum1p is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        term0 = -epsN/kk * self.bsum0
        term1 = (1 - epsN*oo/kk) * self.bsum0p
        term2 = -1 * epsN/kk * self.bsum1p

        return omps_Omcs**2 * (term0 + term1 + term2)

    def chi_perp_prime_fluid(self, epsilonN, ns):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω,
        on grid (k, Re(ω), Im(ω)) for a cold fluid.

        Derivative is taken as d/d(ω/Omega_cs) with respect to the current
        cyclotron frequency... therefore to convert between this
        species + reference species the caller should multiply result by a
        signed factor Omega_c0/Omega_cs
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        return omps_Omcs**2 * epsN/kk/oo**2

    # -------------------------------------------------------------------------
    # Same susceptibility functions, but take (ik,omega) as argument which
    # means we cannot pre-cache the Bessel sums
    # ik = index into k grid to use pre-cached Bessel integrals
    # omega = complex argument
    #
    # Use this costlier calculation to help refine the dispersion relation
    # solve on approximate grid
    # -------------------------------------------------------------------------

    def ikchi_perp_fluid(self, epsilonN, ns, ik, omega):
        """
        Compute chi at one grid point in k, arbitrary complex omega.
        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            ik = index into instance attribute self.k_vec
            omega = complex angular frequency in rad/s
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)

        kk = self.k_vec[ik]  # scaled to species rho_Ls already
        oo = omega / self.species.Omcs(self.B0)  # rescale to Omega_cs

        # notice that eps/k/omega has omega in denominator,
        # unlike numerator placement in chi_kinetic(...)
        return omps_Omcs**2 * (1 - epsN/kk/oo)

    def ikchi_perp_kinetic(self, epsilonN, ns, ik, omega):
        """
        Compute chi at one grid point in k, arbitrary complex omega.
        You must first call
        self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)

        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            ik = index into instance attribute self.k_vec
            omega = complex angular frequency in rad/s
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.k_vec[ik]  # scaled to species rho_Ls already
        oo = omega / self.species.Omcs(self.B0)  # rescale to Omega_cs

        # bessel indices
        nvec = np.arange(self.bessel_Fprime.shape[0])
        # bessel summand prefactor
        invres = 2./(oo*oo - nvec[1:]**2)
        # underscore to emphasize these are not computed on grid
        _bsum0 = 1/kk**2 * np.sum( invres * nvec[1:]**2 * self.bessel_Fprime[1:,ik] )
        _bsum1 = oo      * np.sum( invres               * self.bessel_F[1:,ik]      )
        # handle n=0 term separately
        _bsum1 += 1./oo * self.bessel_F[0,ik]

        terms = (1 - epsN*oo/kk) * _bsum0 - epsN/kk * _bsum1

        return omps_Omcs**2 * terms

    def ikchi_perp_prime_fluid(self, epsilonN, ns, ik, omega):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω, at one grid
        point in k and at arbitrary complex omega, for a cold fluid.

        Derivative is taken as d/d(ω/Omega_cs) with respect to the current
        cyclotron frequency... therefore to convert between this
        species + reference species the caller should multiply result by a
        signed factor Omega_c0/Omega_cs

        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            ik = index into instance attribute self.k_vec
            omega = complex angular frequency in rad/s
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.k_vec[ik]  # scaled to species rho_Ls already
        oo = omega / self.species.Omcs(self.B0)  # rescale to Omega_cs

        return omps_Omcs**2 * epsN/kk/oo**2

    def ikchi_perp_prime_kinetic(self, epsilonN, ns, ik, omega):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω, at one grid
        point in k and at arbitrary complex omega.  You must first call
        self.cache_besselI_integrals(...) or cache_besselJ_integrals(...).

        Derivative is taken as d/d(ω/Omega_cs) with respect to the current
        cyclotron frequency... therefore to convert between this
        species + reference species the caller should multiply result by a
        signed factor Omega_c0/Omega_cs

        Inputs:
            epsilonN = signed density gradient lengthscale in cm^-1
            ns = single-species number density in cm^-3
            ik = index into instance attribute self.k_vec
            omega = complex angular frequency in rad/s
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        kk = self.k_vec[ik]  # scaled to species rho_Ls already
        oo = omega / self.species.Omcs(self.B0)  # rescale to Omega_cs

        # bessel indices
        nvec = np.arange(self.bessel_Fprime.shape[0])
        # bessel summand prefactor
        invres = 2./(oo*oo - nvec[1:]**2)
        # underscore to emphasize these are not computed on grid
        _bsum0 = 1/kk**2 * np.sum( invres * nvec[1:]**2 * self.bessel_Fprime[1:,ik] )
        # derivatives of bessel sums
        _bsum0p = -oo/kk**2 * np.sum( invres*invres * nvec[1:]**2 * self.bessel_Fprime[1:,ik] )
        _bsum1p = (
                      np.sum( invres    * self.bessel_F[1:,ik] )
            - oo**2 * np.sum( invres**2 * self.bessel_F[1:,ik] )
        )
        # handle n=0 term separately
        _bsum1p += -1/oo**2 * self.bessel_F[0,ik]

        term0 = -epsN/kk * _bsum0
        term1 = (1 - epsN*oo/kk) * _bsum0p
        term2 = -1 * epsN/kk * _bsum1p
        return omps_Omcs**2 * (term0 + term1 + term2)

    # -------------------------------------------------------------------------
    # Experimental scheme to compute DCLC stability in a faster way
    # -------------------------------------------------------------------------

    def chi_perp_kinetic_approx_An(self, n, omega_pin, epsilonN, ns):
        """
        Compute coefficient A_n for APPROXIMATE Bessel terms organized in a new
        way, analogous to the dispersion structure of EBWs/IBWs in homogeneous
        plasma, written as:

            B/ω^2 + A_1/(ω^2-1) + A_2(ω^2/4-1) = 0

        for the n=1-2 cyclotron band, to test for DCLC stability using an
        easier-to-handle cubic equation in ω^2 of form:

            (...)*ω^6 + (...)*ω^4 + (...)*ω^2 + (...) = 0,

        which should be simpler than solving the full dispersion relation on a
        grid of (k,Re(ω),Im(ω)).

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
        before using this subroutine.

        Input:
            n = which bessel sum term to use

            omega_pin = choose a constant value of omega (in rad/s) to assume in the
                        coefficients, in order to simplify the omega
                        dependence of the problem at hand.
                        This is key to make the scheme work.

                        Example: to check stability within n=1 to n=2 cyclotron
                        band, it is suggested to use omega_pin=1.5*Omega_cs,
                        but you can refine that guess if you wish.

            rest = same arguments as for chi_kinetic(...) and chi_fluid(...)
        Output:
            A_n coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None
        assert n >= 1

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin / self.species.Omcs(self.B0)  # rescale to Omega_cs

        An = 2 * omps_Omcs**2 * (
            (1 - epsN*oo/kk) * 1/kk**2 * (-1) * self.bessel_Fprime[n,...]
            + epsN*oo/kk * 1/n**2 * self.bessel_F[n,...]
        )
        return An

    def chi_perp_kinetic_approx_B(self, omega_pin, epsilonN, ns):
        """
        Compute coefficient B for APPROXIMATE Bessel terms organized in a new
        way, analogous to the dispersion structure of EBWs/IBWs in homogeneous
        plasma, written as:

            B/ω^2 + A_1/(ω^2-1) + A_2(ω^2/4-1) = 0

        for the n=1-2 cyclotron band, to test for DCLC stability using an
        easier-to-handle cubic equation in ω^2 of form:

            (...)*ω^6 + (...)*ω^4 + (...)*ω^2 + (...) = 0,

        which should be simpler than solving the full dispersion relation on a
        grid of (k,Re(ω),Im(ω)).

        Input:
            omega_pin = choose a constant value of omega (in rad/s) to assume in the
                        coefficients, in order to simplify the omega
                        dependence of the problem at hand.
                        This is key to make the scheme work.

                        Example: to check stability within n=1 to n=2 cyclotron
                        band, it is suggested to use omega_pin=1.5*Omega_cs,
                        but you can refine that guess if you wish.

            rest = same arguments as for chi_kinetic(...) and chi_fluid(...)
        Output:
            B coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin / self.species.Omcs(self.B0)  # rescale to Omega_cs

        # only the n=0 bessel term is needed
        B = omps_Omcs**2 * (-epsN*oo/kk) * self.bessel_F[0,...]
        # adjust for the normalization of 1/omega^2 factor in front
        # of definition of B/omega^2 in the full multi-species dispersion rel
        #B *= (self.qs_q0/self.ms_m0)**2  # TODO may break old code --ATr,2025oct13
        raise Exception('fix me')
        return B

    def chi_perp_fluid_approx_B(self, omega_pin, epsilonN, ns):
        """
        Like chi_kinetic_approx_B, but for cold fluid with J_0^2(...) -> 1.

        Output:
            B coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        #assert self.bessel_Fprime is not None
        #assert self.bessel_F is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin / self.species.Omcs(self.B0)  # rescale to Omega_cs

        # only the n=0 bessel term is needed
        B = omps_Omcs**2 * (-epsN*oo/kk) # * self.bessel_F[0,...]
        # adjust for the normalization of 1/omega^2 factor in front
        # of definition of B/omega^2 in the full multi-species dispersion rel
        #B *= (self.qs_q0/self.ms_m0)**2  # TODO may break old code --ATr,2025oct13
        raise Exception('fix me')
        return B

    # -------------------------------------------------------------------------
    # Susceptibilities with grad(B), finite-beta effects following Tang (1972)
    # -------------------------------------------------------------------------

    def chi_perp_fluid_tang(self, ns, epsilonN=0., epsilonB=0.):
        """
        Compute cold-fluid electrostatic chi_{xx} on grid (k, Re(ω), Im(ω))
        with magnetic gradient effect, following Tang et al. (1972 Phys. Fluids).
        and Tang (1972, PhD thesis).

        Beware, the epsilonB term requires you to keep thermal contributions
        from other chi_{ij} components to obtain a consistent ordering,
        because epsilonB ~ -(beta/2) * epsilonN.
        Within the assumed background equilibrium, beta is contributed by all species.

        Inputs:
            ns = single-species number density in cm^-3
            epsilonN = density gradient (signed) in cm^-1
            epsilonB = magnetic gradient (signed) in cm^-1
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        epsB = epsilonB * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        # notice that eps/k/omega has omega in denominator,
        # unlike numerator placement in chi_kinetic(...)
        # parentheses grouped to minimize number of big array operations
        term0 = omps_Omcs**2
        term1 = (omps_Omcs**2 * (epsB - epsN)) / (kk*oo)
        return term0 + term1

    # TODO this function doesn't really belong here, because
    # the vacuum electromagnetic term does not come from any one species
    def disp_EM_fluid_tang(self, ns, epsilonN=0.):
        """
        Warm-fluid electromagnetic correction to the exactly-perpendicular
        electrostatic slab dispersion relation, as expressed by
        Tang et al. (1972); Callen & Guest (1971, 1973) in the form

            omega^2/k^2/c^2 * chi_{xy} * chi_{yx}

        which is valid for chi_{yy} << k^2*c^2/omega^2.

        Warning: this is NOT a susceptibility; you cannot add contributions
        from multiple species.

        Inputs:
            ns = single-species number density in cm^-3
            epsilonN = signed density gradient lengthscale in cm^-1
        """
        #omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        #epsN = epsilonN * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)
        #vts_c = vth0_c * (self.Ts_T0/self.ms_m0)**0.5
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        epsB = epsilonB * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        vts_c = self.species.vth_perp() / CLIGHT

        term0 = omps_Omcs**4 * vts_c**2 / (kk*kk)
        # this factor has expanded (1 + C*epsilon)^2 ~ 1 + 2*C*epsilon
        # which arises from chi_{xy} * chi_{yx}
        term1 = term0*epsN*kk/oo

        return term0 + term1

    def chi_perp_kinetic_PR1966(self, ns, Freduced, vperp):
        """
        Compute kinetic electrostatic chi_{xx} on grid (k, Re(ω), Im(ω))
        in the limit k >> 1 (normalized to species Larmor radius)
        and neglecting spatial gradients in both density and magnetic field.
        We therefore invoke the following assumptions / procedures:
        * J_n^2(z) -> 1/(pi*z) for z >> 1,
        * Neglect grad(B) drift in resonant denominator
        * Neglect all O(epsilonN^1) and O(epsilonB^1) contributions to susceptibility
        * Convert infinite sum_n 1/(omega + n) = pi * cotangent(pi*omega)

        The calculation follows Post/Rosenbluth (1966) Eqn's (34)--(36),
        and also Tang+ (1972) Eqn's (2)--(4), if Tang's Eqn (2)
        is multiplied by a factor of 2*pi to make the normalization work.

        However, do NOT integrate by parts following Post/Rosenbluth Eqn (35)
        so that we can consider distributions with partly-filled loss cones,
        i.e., F(vperp=0) != 0.

        Inputs:
            ns = single-species number density in cm^-3
        """
        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        # dF/dvperp in species-specific dimensionless units
        Fprime = np.gradient(self.Freduced, self.vperp)
        # inv_a_cubed is 1/a^3 where a \propto Larmor radius
        # for a maxwellian, 1/a^3 = -2*sqrt(pi)*(2*kB*Ts/ms)^(-3/2)
        inv_a_cubed = np.trapz(Fprime/self.vperp * 2*np.pi, self.vperp)

        # parentheses to try to be efficient/smart with the operations
        term0 = (omps_Omcs**2 * inv_a_cubed / kk**3) * (oo/np.tan(np.pi*oo))
        return term0

    def chi_perp_kinetic_epsilonB(self, ns_n0, omp0_Omc0, epsilonN=0., epsilonB=0.):
        """
        Similar to ESPerp_GradRho_Species.chi_perp_kinetic(...) but add extra
        bsum2 term, which we expect to be smaller than epsilonN terms by factor
        of beta... so probably unimportant, unless gradients are steep AND beta
        is large...
        """
        assert self.bsum0 is not None
        assert self.bsum1 is not None
        assert self.bsum2 is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        epsB = epsilonB * self.species.rLs(self.B0)
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        terms = self.bsum0 - epsN*oo/kk * self.bsum0 - epsN/kk * self.bsum1
        terms += -0.5*(epsB/kk) * self.bsum2

        return omps_Omcs**2 * terms

    # -------------------------------------------------------------------------
    # Susceptibilities with external gravity and/or electric field, following
    # Rosenbluth, Krall, Rostoker (1962)
    # -------------------------------------------------------------------------

    def chi_perp_kinetic_Gforce(self, ns, epsilonN=0., Gforce=0.):
        """
        Similar to ESPerp_GradRho_Species.chi_perp_kinetic(...) but add extra
        terms to include drifts caused by an external force field.

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
            self.cache_bessel_sums(...)
        before you can compute kinetic chi.

        Input:
            ns = single-species number density in cm^-3

            epsilonN = signed density gradient lengthscale in cm^-1

            Gforce = 0 or float, external force field acceleration (cm/s^2).
                Gforce is used for both gravity and external electric fields
                (hence capital rather than lowercase G).
                Sign matters; positive G points along the +y axis.

        """
        assert self.bsum0 is not None
        assert self.bsum1 is not None

        omps_Omcs = self.species.omps(ns) / self.species.Omcs(self.B0)
        epsN = epsilonN * self.species.rLs(self.B0)
        #epsB = epsilonB * self.species.rLs(self.B0)
        # abs(Omcs) in Gforce norm is to match k normalization scheme
        G = Gforce / self.species.vth_perp / abs(self.species.Omcs(self.B0))
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        ## DCLC
        #terms = self.bsum0 - epsN*oo/kk * self.bsum0 - epsN/kk * self.bsum1
        ## DCLC with finite grad(B) correction
        #terms += -0.5*(epsB/kk) * self.bsum2

        # DCLC but with gravitational drift
        # recall that bsum0 terms are distributed out to minimize large array
        # operations.
        terms = self.bsum0 - epsN*oo/kk * self.bsum0 - (epsN + 2*G)/kk * self.bsum1

        return omps_Omcs**2 * terms


class BounceAvgESPerp(object):

    def __init__(self,
                 grid: WaveGrid,
                 species: KineticVDFGrid,
                 field: FieldLineVec,
    ):
        """
        Susceptibility for perpendicular electrostatic waves in a paraxial
        mirror plasma, computed for one species on a grid of (k, Re(ω), Im(ω)).

        Coordinate scheme (at midplane):
        * k points along x (poloidal)
        * grad(n) points along y, so epsilon = dn/dy (radial)
        * magnetic field points along z (toroidal)
        * Electron diamagnetic drift towards +k, ion towards -k for epsilon > 0

        WARNING CURRENT GEOMETRY REQUIRES
            Bx = B poloidal ~ 0 for WHAM
            Bz = B axial,
            By = B radial,
        When extracting B field to feed into this code,
        make sure you extract at x=0 and y > 0,
        because internal code is built on Cartesian gradients & doesn't handle
        cylindrical geometry yet.

        Inputs:
            grid = dredge.chi.WaveGrid(...) instance
            species = dredge.species.KineticVDFGrid(...) instance
                      sampled at field-line-following coordinate s = 0
            field = dredge.field.FieldLineVec(...) instance
        """
        self.grid = grid
        self.species = species
        self.field = field
        assert isinstance(species, KineticVDFGrid)
        assert isinstance(field, FieldLineVec)

        # midplane magnetic field for \Omega_{cs} and other normalizations
        self.B0 = field.query_Bmag_at([0.])

        # internal code works in dimensionless units
        # charge sign is not used for k rescaling,
        # but charge sign is included in Omega_{cs}
        self.k_vec        = grid.k_vec        * self.species.rLs (B = self.B0)
        self.omega_re_vec = grid.omega_re_vec / self.species.Omcs(B = self.B0)
        self.omega_im_vec = grid.omega_im_vec / self.species.Omcs(B = self.B0)

        # grids for broadcasting, faster than np.meshgrid(...)
        kk  = self.k_vec       [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        self.kk = kk
        self.oo = omr + 1j*omi

        return

    def setup_bounce_average(self, NS_RESOLUTION=500, TBOUNCE4TH_MAX = 1e99):
        """
        Prepare intermediate variables for bounce averaging
        ALL VARIABLES IN DIMENSIONFUL (CGS) UNITS
        Inputs:
             tbounce4th_max = maximum possible value for tbounce4th to avoid
                NaNs... only really necessary for vprll=0 and vperp=0
                bounce time will be pinned to max(tbounce) anywhere on grid
                    (usually neighboring points) or tbounce4th_max, whichever
                    is less.
        """
        sp, fld = (self.species, self.field)
        # (vperp,vprll) are interpreted to be at s=0 (midplane)
        vperp, vprll = np.meshgrid(sp.vperp_vec, sp.vprll_vec, indexing='ij')
        # compute (E,mu) on grid to do the bounce average
        # and construct useful variables
        E = 0.5*sp.mass*(vperp**2 + vprll**2)
        mu = 0.5*sp.mass*vperp**2 / self.B0

        # save configuration param
        self.TBOUNCE4TH_MAX = TBOUNCE4TH_MAX

        # segment the velocity phase space regions
        trapped = mu*np.amax(fld.Bmag) > E
        passing = np.logical_not(trapped)
        # need to deal with special case of mu=0, call it passing
        # even for mu=0 and E=0  # TODO can handle more elegantly?  -ATr,2025nov06
        trapped[ mu == 0 ] = False
        passing[ mu == 0 ] = True
        assert np.all(np.logical_or(trapped, passing))

        # turning point for all trapped particles
        Bturn = np.empty_like(E)
        Bturn[mu != 0.] = E[mu!=0.] / mu[mu!=0.]
        Bturn[mu == 0.] = np.amax(fld.Bmag)  # avoid divide-by-zero; mu=0 is always pasing except for singularity at origin
        Bturn[passing] = np.amax(fld.Bmag)
        # must enforce Bturn strictly within
        # B_vec range for interpolation;
        # when vprll = 0, E/mu can be < min(B_vec)
        # due to numerical imprecision.
        Bturn = np.minimum(Bturn, np.amax(fld.Bmag))
        Bturn = np.maximum(Bturn, np.amin(fld.Bmag))
        # get bounce-average integral upper limit
        sturn = fld.query_s_at(Bturn)

        # Construct arc-length "s" grid for every point in velocity space,
        # staggering positions to make numerical integral more accurate.
        ssamp = np.empty((NS_RESOLUTION, sp.vperp_vec.size, sp.vprll_vec.size),
                          dtype=np.float64)
        for ii in range(sp.vperp_vec.size):
            for jj in range(sp.vprll_vec.size):
                ssamp_loc = np.linspace(fld.s[0], sturn[ii,jj],
                                         NS_RESOLUTION+1)
                ssamp_loc = ssamp_loc[:-1] + 0.5*np.diff(ssamp_loc)  # offset to get cell centers
                ssamp[:,ii,jj] = ssamp_loc

        self.E = E  # (vperp,vprll) grid
        self.mu = mu  # (vperp,vprll) grid
        self.trapped = trapped  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.passing = passing  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.Bturn = Bturn  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.sturn = sturn  # (vperp,vprll) grid
        self.ssamp = ssamp  # (NS_RESOLUTION,vperp,vprll) grid

        # philosophy, only compute stuff that uses vgrid and Bgeometry info,
        # and keep everything in CGS units for now; let user/caller handle the
        # k gridding and non-dimensionalization

        # LOCAL magnetic field structure at varying "s"
        self.Bmag     = fld.query_Bmag_at    (self.ssamp)  # shape (  NS_RESOLUTION,vperp,vprll)
        self.Bvec     = fld.query_B_at       (self.ssamp)  # shape (3,NS_RESOLUTION,vperp,vprll)
        self.gradB    = fld.query_gradBmag_at(self.ssamp)  # shape (3,NS_RESOLUTION,vperp,vprll)
        self.dbhat_ds = fld.query_dbhat_ds_at(self.ssamp)  # shape (3,NS_RESOLUTION,vperp,vprll)
        self.bhat = self.Bvec / self.Bmag[np.newaxis,...]  # shape (3,NS_RESOLUTION,vperp,vprll)

        # LOCAL signed cyclotron frequency at varying "s"
        # shape (NS_RESOLUTION,vperp,vprll)
        self.Omcs_loc = sp.Omcs(self.Bmag)

        # LOCAL velocities at varying "s";
        # shape (NS_RESOLUTION,vperp,vprll)
        self.vperp_loc = np.sqrt(self.Bmag/self.Bmag[0,:,:]) * vperp[np.newaxis,:,:]
        self.vprll_loc = np.sqrt(  vprll[np.newaxis,:,:]**2
                                 + vperp[np.newaxis,:,:]**2 * (1. - self.Bmag/self.Bmag[0,:,:]) )
        self.vprll_loc *= np.sign( vprll[np.newaxis,:,:] )
        # equivalent rewritings
        # self.vperp_loc = (2 * mu[np.newaxis,:,:] * self.Bmag / sp.m)**0.5
        # self.vprll_loc = (2/sp.m * (E[np.newaxis,:,:] - mu[np.newaxis,:,:] * self.Bmag))**0.5

        # TODO Rahul mentioned something about using E,mu coordinates because then
        # one velocity-space coordinate factors out of vdrift expressions, when
        # you normalize to Bturn or sturn or something... can probably simplify
        # this code --ATr,2025nov11+13

        # LOCAL guiding-center drift velocities at varying s;
        # shape (3,NS_RESOLUTION,vperp,vprll)
        self.v_gradB = (
                0.5 * (self.vperp_loc**2 / self.Omcs_loc / self.Bmag)[np.newaxis,...]
                * np.cross(self.bhat, self.gradB, axisa=0,axisb=0,axisc=0)
        )
        self.v_curv = (
                (self.vprll_loc**2 / self.Omcs_loc)[np.newaxis,...]
                * np.cross(self.bhat, self.dbhat_ds, axisa=0,axisb=0,axisc=0)
        )

        # used to normalize bounce integral.
        # to be honest, the user should probably call this with
        # whatever integration method they want....
        self.tbounce4th = self.bounce_average(
            x = np.ones_like(ssamp),
            norm = False,
        )
        return

    def bounce_average(self, x, norm=True):#, method='trapz'):#, EPS_FUDGE=1e-4):
        """
        Average over one-fourth(!) of an orbit for trapped particles,
        which is equivalent to one-half for passing particles.

        Input:
            x = shape (s, vperp, vprll)
        Return:
            shape (vperp, vprll)
        """
        sp, fld = (self.species, self.field)
        assert x.shape[0] == self.ssamp.shape[0]
        assert len(x.shape) == 3

        # FOLLOWING CALCULATIONS ARE ALL DIMENSIONFUL
        # TODO convert to dimensionless -ATr,2025nov03

        try:
            E = self.E[np.newaxis,...]  # should be shallow copy
            mu = self.mu[np.newaxis,...]
            ssamp = self.ssamp  # (NS_RESOLUTION,vperp,vprll) grid
            Bsamp = self.Bmag   # (NS_RESOLUTION,vperp,vprll) grid
        except:
            print("ERROR: need to call prepare_bounce_average(...)")
            raise

        result = np.empty_like(E)

        # NOTE: the limit vprll -> 0 requires some care to handle singularity
        # Got it with help from ChatGPT, for the case of B''(s=0) nonzero...
        # For case of B''(s=0) zero, B''''(s=0) nonzero, then I need to
        # retrace the singularity-handling procedure...
        # WARNING it still breaks at vperp_vec -> 0
        # --ATr,2025nov06
        pitch90 = (sp.vprll_vec == 0)
        if np.any(pitch90):
            # TODO cleanup the warnings by handling casewise
            # --ATr,2025nov06

            # assume all grids are (s, vperp, vprll)
            integrand = x / ( 2*(E - mu*Bsamp)/sp.mass )**0.5
            result = np.trapz(integrand, ssamp, axis=0)

            # limiting form of bounce-average integral near the singularity,
            # valid for the case x=1, but TODO MAY NOT BE CORRECT FOR x(s)
            # spatially varying........ --ATr,2025nov06
            # MIGHT NEED HIGHER ORDER EDGE STENCILS FOR THIS.
            dB_ds2 = np.gradient(np.gradient(fld.Bmag, fld.s, edge_order=2),
                                 fld.s, edge_order=2)
            if x.shape[2] == self.ssamp.shape[2]:  # computed x on full v_parallel grid
                result[:, pitch90] = (
                    (x[0])[:,pitch90]  # cannot use [0,:,pitch90] b/c mixing selector functions
                    * np.pi/2 * (sp.mass / dB_ds2[0])**0.5 / (self.mu[:,pitch90])**0.5
                )
            elif x.shape[2] == 1:  # x is independent of v_parallel, we broadcast along coordinate
                assert np.where(pitch90)[0].size == 1
                jj = np.where(pitch90)[0][0]  # index explicitly to make code nicer
                # TODO handle ugly divide by zero warning -ATr,2025nov06
                result[:, jj] = (
                    x[0,:,0]
                    * np.pi/2 * (sp.mass / dB_ds2[0])**0.5 / (self.mu[:,jj])**0.5
                )
            else:
                raise Exception('got bad x shape {}'.format(x.shape))
        else:
            # assume all grids are (s, vperp, vprll)
            integrand = x / ( 2*(E - mu*Bsamp)/sp.mass )**0.5
            result = np.trapz(integrand, ssamp, axis=0)

        if norm:
            result /= self.tbounce4th

        # handle the zero point specially
        # important that this comes AFTER the "nominal" norm factor is applied
        # TODO cleanup warning messages related to this!!!!! --ATr,2025nov06
        muzero = (sp.vperp_vec == 0)
        if np.any(pitch90) and np.any(muzero):
            assert np.where(muzero)[0].size == 1
            assert np.where(pitch90)[0].size == 1
            ii = np.where(muzero)[0][0]
            jj = np.where(pitch90)[0][0]

            # handle broadcasting cases
            # TODO ugly ugly ugly -ATr,2025nov06
            if x.shape[1] == self.ssamp.shape[1] and x.shape[2] == self.ssamp.shape[2]:
                x_singular = x[0,ii,jj]
            elif x.shape[1] == 1                 and x.shape[2] == self.ssamp.shape[2]:
                x_singular = x[0,0,jj]
            elif x.shape[1] == self.ssamp.shape[1] and x.shape[2] == 1:
                x_singular = x[0,ii,0]
            elif x.shape[1] == 1                   and x.shape[2] == 1:
                x_singular = x[0,0,0]
            else:
                raise Exception('got bad x shape {}'.format(x.shape))

            if norm:
                # force to the moment's value at s=0 at singular point??
                result[ii,jj] = x_singular
            else:
                # raw calculation of bounce integral which blows up
                # which I currently only use to obtain the bounce time with x=1
                if x_singular == 1.:
                    # we are calculating the bounce time
                    # so just look for max bounce time on grid which should be
                    # from adjoining cells on (vperp,vprll) mesh
                    rmax = np.amax( result[ np.isfinite(result) ] )
                    result[ii,jj] = min(self.TBOUNCE4TH_MAX, rmax)
                else:
                    # explicitly get neighboring cells
                    # vperp=0 is assumed to be the smallest point on mesh
                    assert ii == 0
                    # can we safely index into neighboring points on vprll mesh?
                    assert jj > 0 and jj < len(sp.vprll_vec)-1
                    # TODO stupid stupid hack --ATr,2025nov06

                    # when passing in x != 0, it could have arbitrary
                    # dependence on (s,vperp,vprll) so we don't know where the
                    # maximum lives
                    # THEREFORE, cannot use max(Result) like before
                    # we need to look at the neighboring cells in velocity space
                    # instead.  I am not sure if i am weighting correctly.

                    # I don't expect to use this code branch alot because
                    # usually, when computing bounce averages, we divide out
                    # the normalization.
                    # I think my hacky fix / ceiling should work, but there's
                    # no physics-motivated reason to use it.
                    # expect it may only be used during debugging

                    xmax_neighbor = np.amax([  # don't use nanmax b/c if you get nan, something is wrong
                            x[0,ii,  jj-1],  # left in vprll axis
                            x[0,ii,  jj+1],  # right in vprll axis
                            x[0,ii+1,jj  ],  # up in vperp axis
                            x[0,ii+1,jj-1],  # diagonal up left
                            x[0,ii+1,jj+1],  # diagonal up right
                    ])
                    rmax_neighbor = np.amax([  # don't use nanmax b/c if you get nan, something is wrong
                            result[ii,  jj-1],  # left in vprll axis
                            result[ii,  jj+1],  # right in vprll axis
                            result[ii+1,jj  ],  # up in vperp axis
                            result[ii+1,jj-1],  # diagonal up left
                            result[ii+1,jj+1],  # diagonal up right
                    ])
                    result[ii,jj] = xmax_neighbor * min(self.TBOUNCE4TH_MAX,
                                                        rmax_neighbor/xmax_neighbor)
                    raise Exception('this code branch should only be used for debugging -ATr,2025nov06')

        return result

    def chi_GK(self, epsilonN, ns, Gforce, Teff_ceiling=None, loop=False):
        """
        Compute gyro-averaged, GK-ordered susceptibility for exactly
        perpendicular electrostatic waves

        Input:
            epsilonN = signed density gradient in cm^-1 at midplane z=0
            ns = single-species number density in cm^-3 at midplane z=0
            Gforce = external acceleration (cm/s^2); positive g points along +y axis
            Teff_ceiling = maximum effective temperature in erg,
                to avoid division by zero in regions where distribution is
                near or equal to zero
            loop = compute chi using loop over (Re(omega),Im(omega)) instead of
                doing massive broadcasted arrays?
                Costly, but it helps us handle resonant denominator without
                approximating omega_d/omega << 1.
        """
        sp = self.species
        omps_Omcs = sp.omps(ns) / sp.Omcs(self.B0)
        epsN = epsilonN * sp.rLs(self.B0)
        # abs(Omcs) in Gforce norm is to match k normalization scheme
        G = Gforce / sp.vth_perp / abs(sp.Omcs(self.B0))
        kk = self.kk  # scaled to species rho_Ls
        oo = self.oo  # scaled to species Omega_cs

        # ------------------------------------------
        # start preparing pieces of velocity-space integral

        # Convention: make array shapes like (vperp, vprll, k, Re(om), Im(om))
        # Integrate over (vperp, vprll) in order to evaluate susceptibility,
        # leaving only a 3D array.
        # Be strategic about what arrays to broadcast.

        # EXTEND to (vperp, vprll; k, Re(omega), Im(omega)) grid
        # for broadcasting integral
        kk5d = kk[np.newaxis, np.newaxis, ...]  # scaled to species rho_Ls
        oo5d = oo[np.newaxis, np.newaxis, ...]  # scaled to species Omega_cs
        assert kk5d.ndim == 5
        assert oo5d.ndim == 5

        def _compute_Teff():
            # internal method helps isolate namespaces

            # In GK equation for arbitrary non-Maxwellian F(vperp,vprll), the
            # effective temperature is defined:
            #       1/Teff = -1 * (dF0/dE + 1/B * dF0/dµ) / F0
            # Use chain rule to evaluate:
            #       dF0/dE                  =   dF0/dvprll * 1/(m * vprll)
            #       1/B * dF0/dµ            = - dF0/dvprll * 1/(m * vprll)
            #                                 + dF0/dvperp * 1/(m * vperp)
            #       (dF0_dE + 1/B * dF0/dµ) =   dF0/dvperp * 1/(m * vperp)

            # NOTE edge_order=2 is required to get correct Tperp at vperp=0
            # line, if coordinate array includes vperp=0 exactly.
            # When using edge_order=1 for isotropic Maxwellian,
            # resulting Tperp is 2x larger than true value.
            # --ATr,2026mar04
            df0_dvperp  = np.gradient(sp.df,      sp.vperp_vec, axis=0, edge_order=2)
            df0_dvperp2 = np.gradient(df0_dvperp, sp.vperp_vec, axis=0, edge_order=2)

            # need special handling on µ=0 (vperp=0) line
            # because vperp=0, df/dvperp -> 0 and 1/(m*vperp) -> inf gives
            # indeterminate limit 0/0; apply l'Hopital's rule to bypass
            zeromu = (sp.vperp_vec == 0)
            if np.any(zeromu):
                inv_Teff = np.empty_like(sp.df)
                inv_Teff[ zeromu,:] =   df0_dvperp2[zeromu,:] / sp.m
                inv_Teff[~zeromu,:] = ( df0_dvperp[~zeromu,:]
                                        / (sp.m * sp.vperp_vec[~zeromu,np.newaxis]) )
                inv_Teff /= (-1. * sp.df)
            else:
                inv_Teff = df0_dvperp / (sp.m * sp.vperp_vec[:,np.newaxis])
                inv_Teff /= (-1. * sp.df)

            # regions of low phase space density may have df0/dvperp = 0
            # and Teff->infty; enforce ceiling to avoid dividing by zero
            if Teff_ceiling is not None:
                # when df0/dvperp = 0 exactly, deliberately choose a "positive"
                # temperature for ceiling b/c f=0 should not be driving
                # instability...
                sel = np.logical_and(inv_Teff >= 0.,
                                     inv_Teff < 1./Teff_ceiling)
                inv_Teff[sel] = 1./Teff_ceiling
                # handle negative/positive gradients separately
                # to preserve sign
                sel = np.logical_and(inv_Teff < 0.,
                                     inv_Teff > -1./Teff_ceiling)
                inv_Teff[sel] = -1./Teff_ceiling

            # effective temperature in ergs (CGS unit)
            # defined so that Teff > 0 for Maxwellian
            Teff = 1./inv_Teff

            # normalized to T_perp; for Maxwellian we expect Teff = 1
            # in these dimensionless units.
            Teff /= (0.5 * sp.m * sp.vth_perp**2)
            # broadcast over (vperp, vprll; k, Re(omega), Im(omega)) grid
            Teff = Teff[...,np.newaxis,np.newaxis,np.newaxis]
            assert Teff.ndim == 5
            return Teff

        Teff = _compute_Teff()

        # diamagnetic drift frequency
        # is bounce-average invariant, within our approximation
        # WARNING: epsN is the factor in (1 + epsN*R_y) where R_y is guiding
        # center; epsN is NOT the same as n/(dn/dy) because of boltzmann factor
        # when G is present, see my paper and LaTeX notes
        omega_star = -1 * kk5d * (0.5*epsN*Teff)  # function of (vperp, vprll, k)
        omega_star = omega_star.astype(np.complex128)  # maybe needed for loop=True ? -ATr,2026mar04

        # gyrocenter drift velocities on grid of (s, vperp, vprll)
        v_drift = np.zeros( self.ssamp.shape, dtype=np.float64)

        # multiply all the drifts by sqrt(B/B0) to account for kperp changing
        # along fluxtube, assuming simple flux freezing
        # TODO DOUBLE CHECK CAREFULLY, NEED FEEDBACK ON THIS --ATr,2025nov13

        # external gravitational force drift
        v_drift += (G * np.sqrt(self.Bmag/self.B0))

        # grad(B) drift
        # Pre-cached drift velocity array shape = (3,NS_RESOLUTION,vperp,vprll)
        # in my axisymmetric slab approx, only use x (poloidal) component
        # need charge sign factor to work with k normalization
        v_drift += (self.v_gradB[0,...]/sp.vth_perp) * np.sqrt(self.Bmag/self.B0) * np.sign(sp.Omcs(self.B0))

        # curvature drift
        # Pre-cached drift velocity array shape = (3,NS_RESOLUTION,vperp,vprll)
        # in my axisymmetric slab approx, only use x (poloidal) component
        # need charge sign factor to work with k normalization
        v_drift += (self.v_curv [0,...]/sp.vth_perp) * np.sqrt(self.Bmag/self.B0) * np.sign(sp.Omcs(self.B0))

        # compute the bounce average for every point in velocity space
        v_BAD = self.bounce_average(v_drift, norm=True)  # (vperp, vprll) shape

        # multiply by midplane k_perp(s=0) to get the final bounce-averaged
        # drift frequency on grid (vperp,vprll,k,Re(omega),Im(omega))
        omega_d = kk5d * v_BAD[...,np.newaxis,np.newaxis,np.newaxis]
        omega_d = omega_d.astype(np.complex128)  # maybe needed for loop=True ? -ATr,2026mar04

        # -------------------------------------------
        # to make integral tractable,
        # expand the 1/(omega - omega_drift) term
        # handling both cases of omega_d/omega << 1 and >> 1 separately...
        #
        # to approximate 1/(1-x)...
        # when x < 1, use 1+x+x^2+x^3
        # when x > 1, use -1/x*(1+1/x+1/x^2+1/x^3)
        # for the case x = -1, we should use np.inf
        #
        # PROBLEM is omega_d depends on velocity space and k,
        # so we still end up with a 5D loop...
        # for every omega, scan velocity space
            # for every velocity space point
            #   omega_d = whatever
            #   if abs(omega/omega_d) < 1 use one approx forchi
            #   if abs(omega/omega_d) > 1 use another approx for chi
            #   if abs(omega/omega_d) == 1 use np.inf
        #
        # TEMPORARY RESOLUTION
        # only consider the case of omega_d/omega << 1
        # -------------------------------------------

        # Compute chi using explicit for loop over (Re(omega), Im(omega))
        # costly, but helps us correctly handle the resonant denominator
        # when omega_d \sim omega prevents us from expanding...
        # TODO want to parallelize this
        if loop:

            # MAKE THE J_0^2(...) grid
            # NOT BOTHERING TO ORGANIZE VARIABLES IN A CONSISTENT STYLE YET
            # TODO CLEAN UP THIS GARBAGE... too much going on in this function,
            # there is a risk of inadvertent namespace collisions -ATr,2025nov09
            vperp_vth = sp.vperp_vec / sp.vth_perp
            Jarg = kk5d * vperp_vth[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
            J0sq = scipy.special.jv(0, Jarg)**2

            mom = np.zeros((self.k_vec.size,
                            self.omega_re_vec.size,
                            self.omega_im_vec.size), dtype=np.complex128)

            # SAME IDEA, BUT DONT APPLY THE EXPANSION
            # and forget about Teff, just assume Maxwellian
            # only works for the gravity drift case, so both omega_star and omega_d
            # are independent of velocity space, and we can break the factor
            # (oo - omega_star) / (oo - omega_d) out of the integral
            # result agrees well with fully-kinetic expression
            mom += sp.moment( 1./Teff            )  # <-- this should just return 1 for Maxwellian

            assert J0sq.shape[3] == 1  # J0^2 independent of omega
            assert J0sq.shape[4] == 1
            assert Teff.shape[3] == 1  # Teffective independent of omega
            assert Teff.shape[4] == 1
            assert omega_star.shape[3] == 1  # diamagfreq independent of omega
            assert omega_star.shape[4] == 1
            assert omega_d.shape[3] == 1  #drift freq independent of omega
            assert omega_d.shape[4] == 1

            started = datetime.now()

            # avoid intermediate array alloc
            # when constructing arrays
            prep = np.ones((sp.vperp_vec.size, sp.vprll_vec.size, self.k_vec.size),
                           dtype=np.complex128)
            minus_J0sq_Teff = -1 * J0sq[:,:,:,0,0] / Teff[:,:,:,0,0]
            minus_J0sq_Teff = minus_J0sq_Teff.astype(np.complex128)

            for ii in range(self.omega_re_vec.size):
                for jj in range(self.omega_im_vec.size):

                    #mom[:,ii,jj] += sp.moment(
                    #        (- J0sq[:,:,:,0,0] / Teff[:,:,:,0,0])
                    #        * (oo5d[:,:,:,ii,jj] - omega_star[:,:,:,0,0])
                    #        / (oo5d[:,:,:,ii,jj] - omega_d[:,:,:,0,0])
                    #)

                    # pre-initializing arrays does not make much difference
                    # from quick tests on personal laptop, maybe ~10% faster

                    prep = minus_J0sq_Teff * (oo5d[:,:,:,ii,jj] - omega_star[:,:,:,0,0])
                    prep /= (oo5d[:,:,:,ii,jj] - omega_d[:,:,:,0,0])
                    mom[:,ii,jj] += sp.moment( prep )

                    #print('done omega_im index', jj, 'elapsed', datetime.now()-started)

                print('done omega_re index', ii, 'elapsed', datetime.now()-started)

            chi = (2. * omps_Omcs**2 / kk**2) * mom

            return chi

        # Compute chi using vectorized operations over multi-D numpy arrays
        # be very careful about numpy axis positions and broadcasting
        else:

            # MAKE THE J_0^2(...) grid
            # NOT BOTHERING TO ORGANIZE VARIABLES IN A CONSISTENT STYLE YET
            # TODO CLEAN UP THIS GARBAGE... too much going on in this function,
            # there is a risk of inadvertent namespace collisions -ATr,2025nov09
            vperp_vth = sp.vperp_vec / sp.vth_perp
            Jarg = kk5d * vperp_vth[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
            J0sq = scipy.special.jv(0, Jarg)**2
            del vperp_vth, Jarg

            invoo = 1./oo  # 3D array use outside VDF integral

            mom = np.zeros((self.k_vec.size,
                            self.omega_re_vec.size,
                            self.omega_im_vec.size), dtype=np.complex128)

            mom += sp.moment( 1./Teff * (1. - J0sq)                      )
            mom += sp.moment( 1./Teff * J0sq *   omega_star              ) * invoo
            mom += sp.moment( 1./Teff * J0sq * (-omega_d   )             ) * invoo
            mom += sp.moment( 1./Teff * J0sq *   omega_d    * omega_star ) * invoo**2
            # DROP the higher-order terms in (omega_d/omega)
            # TODO ADDING (omega_d/omega)^2 terms breaks calculation terribly, WHY? --ATr,2025nov09
            # it does not seem to converge like I expect, it only works correctly
            # with the lowest order terms...
            #mom += sp.moment( 1./Teff * J0sq * (-omega_d**2)             ) * invoo**2
            #mom += sp.moment( 1./Teff * J0sq *   omega_d**2 * omega_star ) * invoo**3
            #mom += sp.moment( 1./Teff * J0sq * (-omega_d**3)             ) * invoo**3
            #mom += sp.moment( 1./Teff * J0sq *   omega_d**3 * omega_star ) * invoo**4
            #mom += sp.moment( 1./Teff * J0sq * (-omega_d**4)             ) * invoo**4
            #mom += sp.moment( 1./Teff * J0sq *   omega_d**4 * omega_star ) * invoo**5
            #mom += sp.moment( 1./Teff * J0sq * (-omega_d**5)             ) * invoo**5
            #mom += sp.moment( 1./Teff * J0sq *   omega_d**5 * omega_star ) * invoo**6
            #mom += sp.moment( 1./Teff * J0sq * (-omega_d**6)             ) * invoo**6
            #mom += sp.moment( 1./Teff * J0sq *   omega_d**6 * omega_star ) * invoo**7

            # SAME IDEA, BUT DONT APPLY THE EXPANSION
            # and forget about Teff, just assume Maxwellian
            # only works for the gravity drift case, so both omega_star and omega_d
            # are independent of velocity space, and we can break the factor
            # (oo - omega_star) / (oo - omega_d) out of the integral
            # result agrees well with fully-kinetic expression
            # TODO still troubleshooting why taylor expansion in omegaD/omega is so bad...
            # --ATr,2025nov10
    #        mom += sp.moment( 1./Teff            )  # <-- this should just return 1 for Maxwellian
    #        mom += sp.moment( 1./Teff * (- J0sq) ) * (oo + 0.5*kk*epsN) / (oo - kk*G)

            chi = (2. * omps_Omcs**2 / kk**2) * mom

            return chi
