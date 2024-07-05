#!/usr/bin/env python
"""
Code to calculate plasma susceptibilities for dispersion relation
now optimized for uniform grids on (k, Re(omega), Im(omega)) to help
us quickly sweep parameter space of (Tc/T0, nc/n0, epsilon).

Keep Bessel sum code and chi code packaged together because their normalization
factors are linked; changes to one method affect the other.

The besselI(...) and besselJ(...) sums are defined to agree exactly for a
Maxwellian, up to numerical precision and discretization errors.

Important to use same grids throughout - it makes things simpler...

-ATr, 2024 Jan-Jun
"""

from __future__ import division, print_function

import numpy as np
import scipy as sp

from datetime import datetime

from . import special

# beware... changing temperature, mass, charge,
# ... requires re-computing bessel functions...
# changing density or epsilon does NOT require recomputing bessels
# changing B-field strength is complicated...
# ... it modifies both Omcs/omps and rho_Ls
# ... dont worry about it for now


class ESPerp_GradRho_Species(object):

    def __init__(self, ms_m0, qs_q0, Ts_T0,
                 k0_vec, omega0_re_vec, omega0_im_vec):
        """
        Hot plasma susceptibility for perpendicular electrostatic waves as
        computed for one species, quantities normalized to user's choice of
        some reference species.
        Coordinate scheme:
            k points along x
            grad(n) points along y, so +epsilon = density increaes towards positive y
            magnetic field points along z.
            Electron diamagnetic drift towards +k, ion diamagnetic drift towards -k.
        Inputs:
            ms_m0 = mass
            qs_q0 = signed charge
            Ts_T0 = temperature (mainly for Maxwellian case)
            k0_vec = 1D array of angular wavenumber grid points, normalized to
                     the reference species Larmor radius v_th/Omega_cs where
                     v_th = sqrt(2*kB*T0/ms).
                     The internal attribute is rescaled to the current species'
                     Larmor radius.
            omega0_re_vec = 1D array, real angular frequency Re(omega) grid
                            points normalized to the reference species'
                            cyclotron frequency Omega_c0.
                            The internal attribute is rescaled to the SIGNED
                            current species' cyclotron frequency Omega_cs
            omega0_im_vec = 1D array, imaginary angular frequency Im(omega)
                            grid points scaled to the reference species'
                            cyclotron frequency omega_c0.
                            The internal attribute is rescaled to the SIGNED
                            current species' cyclotron frequency Omega_cs
        """
        self.ms_m0 = ms_m0
        self.qs_q0 = qs_q0
        self.Ts_T0 = Ts_T0
        # notice that charge sign is not used for (k,omega) rescaling
        # charge sign must be specified in chi
        self.k_vec = k0_vec * Ts_T0**0.5 * ms_m0**0.5 / abs(qs_q0)
        self.omega_re_vec = omega0_re_vec * ms_m0 / qs_q0
        self.omega_im_vec = omega0_im_vec * ms_m0 / qs_q0

        # dont allow any fancy/weird gridding
        assert self.k_vec.ndim == 1
        assert self.omega_re_vec.ndim == 1
        assert self.omega_im_vec.ndim == 1

        # calculation breaks at resonant denominators
        # when omega exactly equal to cyclotron harmonics
        # so ensure we only sample non-integer values
        # check for both reference species and the current species
        assert np.all(omega0_re_vec.astype(np.int64) != omega0_re_vec)
        assert np.all(self.omega_re_vec.astype(np.int64) != self.omega_re_vec)

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

    def mesh_extent(self):
        """Helper method for 2D plots of dispersion or susceptibility terms"""
        extent = [-1, 1, -1, 1, -1, 1]
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

    def grid_roots(self,arr):
        """
        Helper method to trace dispersion relation roots in 3D
        (k, omega_re, omega_im) coordinates
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

        Inputs:
            bessel_nmax = largest Bessel index (cyclotron harmonic) to include
            verbose = talk while computing

        Output:
            None, but the following class attributes are updated.
            bessel_Fprime, bessel_F = arrays with shape (bessel_nmax, k)
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

        for n in range(0, bessel_nmax+1):
            arg = np.exp(-lamb) * sp.special.iv(n, lamb)
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

        return

    def cache_besselJ_integrals(
            self,
            bessel_nmax = 20,
            Freduced = None,
            vperp = None,
            verbose = True,
    ):
        """
        Compute Bessel J_n(...) integrals over reduced distribution F(vperp) or
        distribution gradient dF/d(dvperp) * 1/vperp, for electrostatic
        dispersion relation of a slab plasma with a density gradient, for
        linear waves propagating exactly perpendicular to B.

        F(vperp) is defined such that f(v) d^3v = F(vperp) 2*pi*vperp dvperp,
        i.e., F = integral f(v) dv_parallel.

        Inputs:
            bessel_nmax = largest Bessel index (cyclotron harmonic) to include
            Freduced = 1D array of F(vperp)
            vperp = 1D array of vperp sample points for Freduced,
                    normalized to species thermal velocity v_th = sqrt(2*kB*Ts/ms)
                    where Ts is a reference Maxwellian temperature
            verbose = talk while computing

        Output:
            None, but the following class attributes are updated.
            bessel_Fprime, bessel_F = arrays with shape (bessel_nmax, k)
            bessel_Fprime = integral 2*pi*vperp*dvperp * J_n^2 * dF/dvperp / vperp
            bessel_F      = integral 2*pi*vperp*dvperp * J_n^2 * F
        """
        assert Freduced.ndim == 1
        assert vperp.ndim == 1
        assert Freduced.shape == vperp.shape

        started = datetime.now()

        bessel_Jnsq_Fprime = np.empty((bessel_nmax+1, self.k_vec.size))
        bessel_Jnsq_F      = np.empty((bessel_nmax+1, self.k_vec.size))

        # enforce normalization = 1
        # using the same integration scheme that will be used
        # in all the subsequent Bessel-weighted integrals...
        norm = np.trapz(Freduced * 2*np.pi*vperp, vperp)
        Freduced = Freduced/norm

        # dF/dvperp
        Fprime = np.gradient(Freduced, vperp)

        # setup (k, vperp) grid for Bessel Jn-weighted moments of F(vperp)
        kg      = self.k_vec   [...,np.newaxis]
        vperpg  = vperp        [np.newaxis,...]
        Fg      = Freduced     [np.newaxis,...]
        Fprimeg = Fprime       [np.newaxis,...]

        for n in range(0, bessel_nmax+1):

            Jnsq = sp.special.jv(n, kg*vperpg)**2

            # \int dF/dvperp * 1/vperp * J_n^2(z) * 2*pi*vperp dvperp
            bessel_Jnsq_Fprime[n,:] = np.trapz(Fprimeg * Jnsq * 2*np.pi, vperpg, axis=-1)

            # \int F * J_n^2(z) * 2*pi*vperp dvperp
            bessel_Jnsq_F[n,:] = np.trapz(Fg * Jnsq * 2*np.pi*vperpg, vperpg, axis=-1)

            if verbose:
                print(f'Bessel J_{n:d} integral done, elapsed', datetime.now()-started)

        self.bessel_Fprime = bessel_Jnsq_Fprime
        self.bessel_F      = bessel_Jnsq_F

        return

    def cache_bessel_sums(self, verbose=True, with_prime=False):
        """
        Compute sums of Bessel J_n(...) integrals or I_n(...) terms which have
        already been pre-cached by the user, for use in electrostatic
        dispersion relation of a slab plasma with a density gradient, for
        linear waves propagating exactly perpendicular to B.

        The Bessel sums are crafted to omit epsilon (spatial gradient) factors,
        so that the user can cache the Bessel sums, then recompute behavior
        quickly for varying epsilon, Omega_ci/omega_pi ~ vA/c ~ density, etc.

        Inputs:
            verbose = talk while computing
            with_prime = compute additional sums for d(chi)/dω calculation,
                         which we are using to estimate electron Landau damping...

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

        # construct complex omega on grid (k,Re(ω),Im(ω))
        omr, omi = np.meshgrid(self.omega_re_vec, self.omega_im_vec, indexing='ij')
        oo = omr + 1j*omi
        oo = oo[np.newaxis,...]

        # construct k on broadcastable grid (k,Re(ω),Im(ω))
        kk = self.k_vec[:, np.newaxis, np.newaxis]

        # construct Bessel sums on grid (k,Re(ω),Im(ω))
        bshape = (self.k_vec.size, self.omega_re_vec.size, self.omega_im_vec.size)
        bsum0 = np.zeros(bshape, dtype='complex128')
        bsum1 = np.zeros(bshape, dtype='complex128')
        if with_prime:
            bsum0p = np.zeros(bshape, dtype='complex128')
            bsum1p = np.zeros(bshape, dtype='complex128')

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

        if verbose:
            print('Bessel n=0 summand done, elapsed', datetime.now()-started)

        # cache for future computation
        self.bsum0 = bsum0
        self.bsum1 = bsum1
        if with_prime:
            self.bsum0p = bsum0p
            self.bsum1p = bsum1p

        return

    # -------------------------------------------------------------------------
    # Susceptibility
    # -------------------------------------------------------------------------

    def chi_perp_fluid(self, epsilon0, ns_n0, omp0_Omc0, warm=False):
        """
        Compute susceptibility chi on grid (k, Re(ω), Im(ω)).
        Distribution function is either cold or warm Maxwellian.
        Inputs:
            epsilon0 = signed gradient lengthscale, normalized to reference
                       species Larmor radius
            ns_n0 = density ratio
            omp0_Omc0 = plasma/cyclotron frequency ratio for reference species
            warm = apply thermal corrections from small-k Bessel sum expansions
        """
        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # scaled to species rho_Ls, Omega_cs already
        # broadcasting is faster than meshgrid
        #kk, omr, omi = np.meshgrid(self.k_vec, self.omega_re_vec, self.omega_im_vec, indexing='ij')
        kk = self.k_vec        [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        oo = omr + 1j*omi

        # notice that eps/k/omega has omega in denominator,
        # unlike numerator placement in chi_kinetic(...)
        if warm:
            # the double expansion in small k*rhoLe and omega/Omce
            # results in different coefficients for the non-gradient
            # versus the gradient terms; the expansion here matches that
            # in Lindgren, Langdon, Birdsall (1976), Equation (2) discussion.
            lamb = (kk**2)/2  # argument to modified Bessel I_n(...)
            term0 = omps_Omcs**2 * (1 - 3./4 * lamb)
            term1 = -1 * omps_Omcs**2 * eps/kk/oo * (1 - lamb)
            return term0 + term1
        else:
            term0 = omps_Omcs**2 * (1 - eps/kk/oo)
            return term0

    def chi_perp_kinetic(self, epsilon0, ns_n0, omp0_Omc0):
        """
        Compute susceptibility chi on grid (k, Re(ω), Im(ω)).
        Distribution function enters via Bessel sums.

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
            self.cache_bessel_sums(...)
        before you can compute kinetic chi.

        Inputs:
            epsilon0 = signed gradient lengthscale, normalized to reference
                       species Larmor radius
            ns_n0 = density ratio
            omp0_Omc0 = plasma/cyclotron frequency ratio for reference species
        """
        assert self.bsum0 is not None
        assert self.bsum1 is not None

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # scaled to species rho_Ls, Omega_cs already
        # broadcasting is faster than meshgrid
        #kk, omr, omi = np.meshgrid(self.k_vec, self.omega_re_vec, self.omega_im_vec, indexing='ij')
        kk = self.k_vec        [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        oo = omr + 1j*omi

        terms = self.bsum0 - eps*oo/kk * self.bsum0 - eps/kk * self.bsum1

        return omps_Omcs**2 * terms

    def chi_prll_Zfunc_lowk(self, epsilon0, ns_n0, omp0_Omc0, k_parallel):
        """
        Compute very simplified low-k limit of parallel susceptibility which
        ... neglects all Bessel terms n>=1
        ... takes J_0^2(...) = 1 limit as k->0.
        ... uses Zfunc to assume Maxwellian distribution with temperature Ts
        this allows a simple description of parallel Landau damping.

        Inputs:
            epsilon0 = signed gradient lengthscale, normalized to reference
                       species Larmor radius
            ns_n0 = density ratio
            omp0_Omc0 = plasma/cyclotron frequency ratio for reference species
            k_parallel = (scalar) signed parallel angular wavenumber,
                         normalized to reference species Larmor radius.
                         When choosing sign of k_parallel, remember that omega
                         is scaled to SIGNED species cyclotron freq.
        """
        if epsilon0 != 0:
            raise Exception("Error: gradient term not yet added to parallel chi")

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        #eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # scaled to species rho_Ls, Omega_cs already
        # broadcasting is faster than meshgrid
        #kk, omr, omi = np.meshgrid(self.k_vec, self.omega_re_vec, self.omega_im_vec, indexing='ij')
        #kk = self.k_vec        [:, np.newaxis, np.newaxis]  # not needed for approximate parallel susceptibility
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        oo = omr + 1j*omi

        # rescale to species rho_Ls
        kp = k_parallel * (self.Ts_T0*self.ms_m0)**0.5 / abs(self.qs_q0)

        # plasma function argument
        zeta0s = oo / kp

        # ratio of larmor radius to debye length for this species
        rhoLs_lde = 2**0.5 * omps_Omcs

        return 1./kp**2 * rhoLs_lde**2 * (1 + zeta0s * special.Zfunc(zeta0s))

    # -------------------------------------------------------------------------
    # Derivaties of chi with respect to frequency omega, which can be used
    # when estimating complex roots in a weak growth approximation.
    # In practice, more useful to root find on the 3D (k, Re(ω), Im(ω)) grid.
    # -------------------------------------------------------------------------

    def chi_perp_prime_kinetic(self, epsilon0, ns_n0, omp0_Omc0):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω,
        on grid (k, Re(ω), Im(ω)).

        You must call
            self.cache_besselI_integrals(...) or cache_besselJ_integrals(...)
            self.cache_bessel_sums(..., with_prime=True)
        first.

        Distribution function enters via Bessel sums.

        Derivative is taken as d/d(ω/Omega_c0) with respect to the REFERENCE
        cyclotron frequency... therefore to convert between this
        species + reference species you need a signed factor Omega_c0/Omega_cs
        """
        assert self.bsum0 is not None
        assert self.bsum1 is not None
        assert self.bsum0p is not None
        assert self.bsum1p is not None

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        Omc0_Omcs = self.ms_m0 / self.qs_q0

        # scaled to species rho_Ls, Omega_cs already
        # broadcasting is faster than meshgrid
        #kk, omr, omi = np.meshgrid(self.k_vec, self.omega_re_vec, self.omega_im_vec, indexing='ij')
        kk = self.k_vec        [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        oo = omr + 1j*omi

        term0 = -eps/kk * self.bsum0
        term1 = (1 - eps*oo/kk) * self.bsum0p
        term2 = -1 * eps/kk * self.bsum1p

        return omps_Omcs**2 * (term0 + term1 + term2) * Omc0_Omcs

    def chi_perp_prime_fluid(self, epsilon0, ns_n0, omp0_Omc0):
        """
        Compute frequency-derivative of susceptibility, d(chi)/dω,
        on grid (k, Re(ω), Im(ω)) for a cold fluid.

        Derivative is taken as d/d(ω/Omega_c0) with respect to the REFERENCE
        cyclotron frequency... therefore to convert between this
        species + reference species you need a signed factor Omega_c0/Omega_cs
        """
        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        Omc0_Omcs = self.ms_m0 / self.qs_q0  # signed

        # scaled to species rho_Ls, Omega_cs already
        # broadcasting is faster than meshgrid
        #kk, omr, omi = np.meshgrid(self.k_vec, self.omega_re_vec, self.omega_im_vec, indexing='ij')
        kk = self.k_vec        [:, np.newaxis, np.newaxis]
        omr = self.omega_re_vec[np.newaxis, :, np.newaxis]
        omi = self.omega_im_vec[np.newaxis, np.newaxis, :]
        oo = omr + 1j*omi

        return omps_Omcs**2 * eps/kk/oo**2 * Omc0_Omcs

    # -------------------------------------------------------------------------
    # Experimental scheme to compute DCLC stability in a faster way
    # -------------------------------------------------------------------------

    def chi_perp_kinetic_approx_An(self, n, omega_pin, epsilon0, ns_n0, omp0_Omc0):
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

            omega_pin = choose a constant value of omega to assume in the
                        coefficients, in order to simplify the omega
                        dependence of the problem at hand.
                        This is key to make the scheme work.

                        Example: to check stability within n=1 to n=2 cyclotron
                        band, it is suggested to use omega_pin=1.5, but you can
                        refine that guess if you wish.

            rest = same arguments as for chi_kinetic(...) and chi_fluid(...)
        Output:
            A_n coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None
        assert n >= 1

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin * self.ms_m0 / self.qs_q0  # mimic norm of self.omega_re_vec

        An = 2 * omps_Omcs**2 * (
            (1 - eps*oo/kk) * 1/kk**2 * (-1) * self.bessel_Fprime[n,...]
            + eps*oo/kk * 1/n**2 * self.bessel_F[n,...]
        )
        return An

    def chi_perp_kinetic_approx_B(self, omega_pin, epsilon0, ns_n0, omp0_Omc0):
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
            omega_pin = choose a constant value of omega to assume in the
                        coefficients, in order to simplify the omega
                        dependence of the problem at hand.
                        This is key to make the scheme work.

                        Example: to check stability within n=1 to n=2 cyclotron
                        band, it is suggested to use omega_pin=1.5, but you can
                        refine that guess if you wish.

            rest = same arguments as for chi_kinetic(...) and chi_fluid(...)
        Output:
            B coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        assert self.bessel_Fprime is not None
        assert self.bessel_F is not None

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin * self.ms_m0 / self.qs_q0  # mimic norm of self.omega_re_vec

        # only the n=0 bessel term is needed
        B = omps_Omcs**2 * (-eps*oo/kk) * self.bessel_F[0,...]
        # adjust for the normalization of 1/omega^2 factor in front
        # of definition of B/omega^2 in the full multi-species dispersion rel
        B *= (self.qs_q0/self.ms_m0)**2
        return B

    def chi_perp_fluid_approx_B(self, omega_pin, epsilon0, ns_n0, omp0_Omc0):
        """
        Like chi_kinetic_approx_B, but for cold fluid with J_0^2(...) -> 1.

        Output:
            B coefficient for new dispersion approximation scheme, computed
            on grid of shape (k,)
        """
        #assert self.bessel_Fprime is not None
        #assert self.bessel_F is not None

        omps_Omcs = omp0_Omc0 * ns_n0**0.5 * self.ms_m0**0.5
        eps = epsilon0 * self.Ts_T0**0.5 * self.ms_m0**0.5 / abs(self.qs_q0)

        # replace the usual 3D (k,Re(ω),Im(ω)) with a 1D grid in k,
        # because ω is fixed to user-chosen approximation
        kk = self.k_vec
        oo = omega_pin * self.ms_m0 / self.qs_q0  # mimic norm of self.omega_re_vec

        # only the n=0 bessel term is needed
        B = omps_Omcs**2 * (-eps*oo/kk) # * self.bessel_F[0,...]
        # adjust for the normalization of 1/omega^2 factor in front
        # of definition of B/omega^2 in the full multi-species dispersion rel
        B *= (self.qs_q0/self.ms_m0)**2
        return B
