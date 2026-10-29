"""
Code to compute bounce-average integrals
"""

import time

import numba
import numpy as np

# TODO: implement methods from https://arxiv.org/abs/2412.01724


@numba.njit(parallel=True)
def _bounce_average_njit_kernel(
        x_grid, Bsamp_grid, s_grid, E_grid, mu_grid, vperp_vec, vprll_vec,
        mass, dB_ds2_origin, tbounce4th, tbounce4th_max, norm=True
):
    r"""
    Internal function to compute the bounce average, hottest logic

    Strongly encouraged to only supply real (not complex) arguments.
    I hit some issues when passing complex dtypes into x_grid

    Input:
        x_grid = quantity to be averaged; shape (vperp,vprll,s)
        Bsamp_grid = B field magnitude on grid (vperp,vprll,s)
        s_grid = s coordinates on grid (vperp,vprll,s)
                 recall that field-line integration points may differ in
                 velocity space b/c turning-point location varies
        E_grid = energy on grid (vperp,vprll)
        mu_grid = magnetic moment on grid (vperp,vprll)
        vperp_vec = v_\perp coordinates in cm/s, 1D array
        vprll_vec = v_\parallel coordinates in cm/s, 1D array
        mass = particle species mass in grams, scalar
        dB_ds2_origin = d^2(B)/ds^2 evaluated at s=0, scalar, used to help
                        evaluate limit vprll->0 (with vperp=finite)
        tbounce4th = shape (vperp,vprll) grid of bounce times
        tbounce4th_max = upper limit on quarter bounce time
        norm = normalize?
    Return:
        shape (vperp, vprll)
    """
    result = np.empty((vperp_vec.size, vprll_vec.size), dtype=x_grid.dtype)

    for ii in numba.prange(vperp_vec.size):
        for jj in numba.prange(vprll_vec.size):
            x_vec = x_grid    [ii,jj,:]  # NOTE (s,vperp,vprll) shape
            B_vec = Bsamp_grid[ii,jj,:]  # doesn't play nice with numba
            s_vec = s_grid    [ii,jj,:]  # compilation of np.trapezoid
            E     = E_grid    [ii,jj]    # b/c [:,ii,jj] data not contiguous in c ordering
            mu    = mu_grid   [ii,jj]
            # common case; breaks with divide-by-zero or huge number
            # for vprll=0 (pitch angle 90)
            # singular line requires separate handling for vperp=0 or vperp>0
            integrand = x_vec / np.sqrt( (2./mass)*(E - mu*B_vec) )
            result[ii,jj] = np.trapezoid(integrand, s_vec)

    # special case handling
    muzero = (vperp_vec == 0)
    pitch90 = (vprll_vec == 0)

    # handle vprll=0 line with special remainder loop
    if np.any(pitch90):
        assert np.where(pitch90)[0].size == 1
        jj = np.where(pitch90)[0][0]
        for ii in range(vperp_vec.size):
            # prevent exactly zero; the zero case will be dealt with later
            #mu = mu_grid[ii,jj]
            mu = max(1e-99,mu_grid[ii,jj])
            # limiting form of bounce-average integral near the singularity,
            # valid for the case x=1, but TODO MAY NOT BE CORRECT FOR x(s)
            # spatially varying........ --ATr,2025nov06
            result[ii,jj] = x_grid[ii,jj,0] * np.pi/2 * np.sqrt(mass / dB_ds2_origin / mu)

    # apply normalization BEFORE the singular point handling
    if norm:
        result /= tbounce4th

    # handle zero point specially
    if np.any(muzero) and np.any(pitch90):
        assert np.where(muzero)[0].size == 1
        assert np.where(pitch90)[0].size == 1
        ii = np.where(muzero)[0][0]
        jj = np.where(pitch90)[0][0]
        if norm:
            # force to the moment's value at s=0 at singular point??
            # TODO is this correct???
            result[ii,jj] = x_grid[ii,jj,0]
        else:
#            # this treatment is only valid when x=1
#            # as used when computing bounce orbit periods
#            assert x_grid[ii,jj,0] == 1.
#            result[ii,jj] = min(tbounce4th_max, np.nanmax(result))
            # PROBLEM, np.nanmax(...) does not play nice with complex arguments
            # and compiler will balk
            result[ii,jj] = np.nan

    return result


class OrbitAverage(object):
    """
    Orbit average one plasma species' kinetic distribution on (vperp, vprll, s)
    grid where (vperp, vprll) are velocity-space coordinates at s=0 and s is
    arc length along a magnetic field line.
    Abstract base class is not meant for users to initialize.
    """

    def __init__(self, species, field):
        """
        Inputs:
            species = dredge.species.KineticVDFGrid(...) instance
                      sampled at field-line-following coordinate s=0
            field = dredge.field.FieldLine(...) instance
        """
        self.species = species
        self.field = field
        # midplane magnetic field for \Omega_{cs} and other normalizations
        self.B0 = field.query_Bmag_at([0.])

    def _setup_local_quantities(self):
        """
        Prepare intermediate variables for orbit averaging
        ALL VARIABLES IN DIMENSIONFUL (CGS) UNITS.
        Requires self.ssamp to be initialized.
        """
        # philosophy, only compute stuff that uses vgrid and Bgeometry info,
        # and keep everything in CGS units for now; let user/caller handle the
        # k gridding and non-dimensionalization

        sp, fld = (self.species, self.field)
        # (vperp,vprll) are interpreted to be at s=0 (midplane)
        vperp, vprll = np.meshgrid(sp.vperp_vec, sp.vprll_vec, indexing='ij')
        self.E = 0.5*sp.mass*(vperp**2 + vprll**2)         # shape (  vperp,vprll)
        self.mu = 0.5*sp.mass*vperp**2 / self.B0           # shape (  vperp,vprll)

        # LOCAL magnetic field structure at varying "s"
        # ratio B(s)/B(s=0) is useful for factors of Omega_c(s)/Omega_c(s=0)
        self.Bmag     = fld.query_Bmag_at    (self.ssamp)  # shape (  vperp,vprll,s)
        self.Bvec     = fld.query_B_at       (self.ssamp)  # shape (3,vperp,vprll,s)
        self.gradB    = fld.query_gradBmag_at(self.ssamp)  # shape (3,vperp,vprll,s)
        self.dbhat_ds = fld.query_dbhat_ds_at(self.ssamp)  # shape (3,vperp,vprll,s)
        self.bhat = self.Bvec / self.Bmag[np.newaxis,...]  # shape (3,vperp,vprll,s)
        self.B_B0 = self.Bmag / (self.Bmag[:,:,0])[:,:,np.newaxis]  # shape (  vperp,vprll,s)

        # LOCAL signed cyclotron frequency at varying "s"
        self.Omcs_loc = sp.Omcs(self.Bmag)                          # shape (  vperp,vprll,s)

        # LOCAL mapped velocities at varying "s"; shape (  vperp,vprll,s)
        self.vperp_loc = np.sqrt(self.B_B0) * vperp[:,:,np.newaxis]
        self.vprll_loc = np.sqrt(  vprll[:,:,np.newaxis]**2
                                 + vperp[:,:,np.newaxis]**2 * (1. - self.B_B0) )
        self.vprll_loc *= np.sign( vprll[:,:,np.newaxis] )
        # equivalent writings using (E,mu) coordinates
        #self.vperp_loc = (2 * mu[:,:,np.newaxis] * self.Bmag / sp.m)**0.5
        #self.vprll_loc = (2/sp.m * (E[:,:,np.newaxis] - mu[:,:,np.newaxis] * self.Bmag))**0.5

        # TODO implement E,mu coordinates because then one velocity-space
        # coordinate factors out of vdrift expressions, when normalized to
        # Bturn or sturn or similar...  --ATr,2025nov11+13

        # LOCAL guiding-center drift velocities; shape (3,vperp,vprll,s)
        self.v_gradB = (
                0.5 * (self.vperp_loc**2 / self.Omcs_loc / self.Bmag)[np.newaxis,...]
                * np.cross(self.bhat, self.gradB, axisa=0,axisb=0,axisc=0)
        )
        self.v_curv = (
                (self.vprll_loc**2 / self.Omcs_loc)[np.newaxis,...]
                * np.cross(self.bhat, self.dbhat_ds, axisa=0,axisb=0,axisc=0)
        )

        return

    def bounce_average(self, x, norm=True):
        """
        Average a quantity x over particle orbits.

        Input:
            x = shape (vperp, vprll, s)
        Return:
            shape (vperp, vprll)
        """
        raise NotImplementedError

    def bounce_average_norm_raw(self, x, norm=True):
        """Like self.bounce_average(...), but remove safety checks and
        conditionals to go as fast as possible."""
        raise NotImplementedError

    def bounce_average_njit(self, x, norm=True):
        """Like self.bounce_average(...), but with numba JIT compilation."""
        raise NotImplementedError


class UniformOrbitAverage(OrbitAverage):

    def __init__(self, species, field):
        super().__init__(species, field)

        # field-line-following coordinate axis; shape (vperp,vprll,s)
        self.ssamp = np.zeros(
                (species.vperp_vec.size, species.vprll_vec.size, 1)
        )

        self._setup_local_quantities()

        # ratio r(s)/r(s=0) is useful for factors of k(s)/k(s=0) assuming
        # k_perp ~ m/r where m = azimuthal mode number, r = cylindrical radius
        # UniformOrbitAverage --> k_perp does not vary along s
        self.r_r0 = np.ones_like(self.ssamp)

    def bounce_average(self, x, norm=True):
        assert norm  # un-normalized case has no meaning
        return x[...,0]  # allows broadcasting over left axes

    def bounce_average_norm_raw(self, x, norm=True):
        assert norm
        return x[...,0]

    def bounce_average_njit(self, x, norm=True):
        assert norm
        return x[...,0]


class BounceOrbitAverage(OrbitAverage):

    def __init__(self, species, field, NS_RESOLUTION=500, TBOUNCE4TH_MAX=1e99):
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
        super().__init__(species, field)

        sp, fld = (self.species, self.field)
        # (vperp,vprll) are interpreted to be at s=0 (midplane)
        vperp, vprll = np.meshgrid(sp.vperp_vec, sp.vprll_vec, indexing='ij')

        # determine ssamp coordinate extents for each point in velocity space
        # based on particle turning points
        E = 0.5*sp.mass*(vperp**2 + vprll**2)
        mu = 0.5*sp.mass*vperp**2 / self.B0
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
        ssamp = np.empty((sp.vperp_vec.size, sp.vprll_vec.size, NS_RESOLUTION),
                          dtype=np.float64)
        for ii in range(sp.vperp_vec.size):
            for jj in range(sp.vprll_vec.size):
                ssamp_loc = np.linspace(fld.s[0], sturn[ii,jj],
                                         NS_RESOLUTION+1)
                ssamp_loc = ssamp_loc[:-1] + 0.5*np.diff(ssamp_loc)  # offset to get cell centers
                ssamp[ii,jj,:] = ssamp_loc

        self.trapped = trapped  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.passing = passing  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.Bturn = Bturn  # (vperp,vprll) grid  # useful for viz, not needed For integral
        self.sturn = sturn  # (vperp,vprll) grid  # useful for viz, not needed For integral yet
        self.ssamp = ssamp  # (vperp,vprll,s) grid

        self._setup_local_quantities()

        # ratio r(s)/r(s=0) along field line; shape (vperp,vprll,s)
        # useful for factors of k(s)/k(s=0) if we assume k_perp ~ m/r
        # where m is azimuthal mode number and r is cylindrical radius
        self.rsamp = fld.query_r_at(self.ssamp)  # shape (vperp,vprll,s)
        self.r_r0 = self.rsamp / (self.rsamp[:,:,0])[:,:,np.newaxis]

        # Need this second-order gradient to evaluate bounce integral in the
        # limit vprll->0
        # TODO MIGHT NEED HIGHER ORDER EDGE STENCILS FOR THIS?
        dB_ds2 = np.gradient(np.gradient(fld.Bmag, fld.s, edge_order=2),
                             fld.s, edge_order=2)
        self.dB_ds2_origin = dB_ds2[0]  # scalar

        # save configuration param; needed before first call to
        # bounce_average(...)
        self.TBOUNCE4TH_MAX = TBOUNCE4TH_MAX

        # used to normalize bounce integral.
        # to be honest, the user should probably call this with
        # whatever integration method they want....
        self.tbounce4th = self.bounce_average(
            x = np.ones_like(ssamp),
            norm = False,
        )

        # useful for "hot" bounce average integrals
        # pre-cache inverses to avoid costly multiplication!!!
        self.inv_vprll_loc_abs = 1. / np.maximum(1e-99, np.abs(self.vprll_loc))
        self.inv_tbounce4th = 1. / np.maximum(1e-99, np.abs(self.tbounce4th))

        return

    def bounce_average(self, x, norm=True):
        """
        Average over one-fourth(!) of an orbit for trapped particles,
        which is equivalent to one-half for passing particles.

        Input:
            x = shape (vperp, vprll, s)
        Return:
            shape (vperp, vprll)
        """
        try:
            _ = self.ssamp
        except:
            raise Exception("ERROR: need to call _setup_local_quantities(...)")

        sp, fld = (self.species, self.field)
        assert x.shape[-1] == self.ssamp.shape[-1]
        assert len(x.shape) == 3

        # FOLLOWING CALCULATIONS ARE ALL DIMENSIONFUL
        # TODO convert to dimensionless -ATr,2025nov03

        # inherit datatype from integrand...
        result = np.empty(sp.df.shape, dtype=x.dtype)

        E = self.E[...,np.newaxis]  # should be shallow copy
        mu = self.mu[...,np.newaxis]  # for easier broadcasting
        ssamp = self.ssamp  # (vperp,vprll,s) grid
        Bsamp = self.Bmag   # (vperp,vprll,s) grid

        # NOTE: the limit vprll -> 0 requires some care to handle singularity
        # Got it with help from ChatGPT, for the case of B''(s=0) nonzero...
        # For case of B''(s=0) zero, B''''(s=0) nonzero, then I need to
        # retrace the singularity-handling procedure...
        # WARNING it still breaks at vperp_vec -> 0
        # --ATr,2025nov06

        pitch90 = (sp.vprll_vec == 0)
        muzero = (sp.vperp_vec == 0)

        if np.any(pitch90):
            assert np.where(pitch90)[0].size == 1
            jj = np.where(pitch90)[0][0]

            # assume all grids are (vperp, vprll, s)
            #integrand = x / ( 2*(E - mu*Bsamp)/sp.mass )**0.5
            #result[:] = np.trapezoid(integrand, ssamp, axis=-1)

            # enforce sqrt argument >=0
            Eprll = np.maximum(E - mu*Bsamp, 0.)
            denom = ( 2*Eprll/sp.mass )**0.5

            # limiting form of bounce-average integral near the singularity,
            # valid for the case x=1, but TODO MAY NOT BE CORRECT FOR x(s)
            # spatially varying........ --ATr,2025nov06
            if x.shape[1] == self.ssamp.shape[1]:  # computed x on full v_parallel grid

                # handle points "left" of singularity
                if jj > 0:
                    _integrand = x[:,:jj] / denom[:,:jj]
                    result[:,:jj] = np.trapezoid(_integrand, ssamp[:,:jj], axis=-1)

                # handle singularity v_parallel = 0
                if np.any(muzero):
                    assert np.where(muzero)[0].size == 1
                    ii = np.where(muzero)[0][0]

                    result[ii,jj] = np.inf  # to be overriden

                    result[~muzero,jj] = (
                        # cannot use [~muzero,jj,0] b/c mixing selector functions
                        (x[...,0])[~muzero,jj]
                        * np.pi/2 * (sp.mass / self.dB_ds2_origin)**0.5
                        / (self.mu[~muzero,jj])**0.5
                    )

                # handle points "right" of singularity
                if jj < (sp.vprll_vec.size - 1):
                    _integrand = x[:,(jj+1):] / denom[:,(jj+1):]
                    result[:,(jj+1):] = np.trapezoid(_integrand, ssamp[:,(jj+1):], axis=-1)

            # UNUSED code branch will soon be deprecated/eliminated
            # --ATr,2026mar12

            #elif x.shape[1] == 1:  # x is independent of v_parallel, we broadcast along coordinate
            #    # TODO handle ugly divide by zero warning -ATr,2025nov06
            #    result[:,jj] = (
            #        x[:,0,0]
            #        * np.pi/2 * (sp.mass / self.dB_ds2_origin)**0.5 / (self.mu[:,jj])**0.5
            #    )

            else:
                raise Exception('got bad x shape {}'.format(x.shape))
        else:
            # assume all grids are (vperp, vprll, s)
            integrand = x / ( 2*(E - mu*Bsamp)/sp.mass )**0.5
            result[:] = np.trapezoid(integrand, ssamp, axis=-1)

        if norm:
            result *= self.inv_tbounce4th

        # handle the zero point specially
        # important that this comes AFTER the "nominal" norm factor is applied
        # TODO cleanup warning messages related to this!!!!! --ATr,2025nov06
        if np.any(pitch90) and np.any(muzero):
            assert np.where(muzero)[0].size == 1
            assert np.where(pitch90)[0].size == 1
            ii = np.where(muzero)[0][0]
            jj = np.where(pitch90)[0][0]

            # handle broadcasting cases
            # TODO ugly ugly ugly -ATr,2025nov06
            if x.shape[0] == self.ssamp.shape[0] and x.shape[1] == self.ssamp.shape[1]:
                x_singular = x[ii,jj,0]
            elif x.shape[0] == 1                 and x.shape[1] == self.ssamp.shape[1]:
                x_singular = x[0,jj,0]
            elif x.shape[0] == self.ssamp.shape[0] and x.shape[1] == 1:
                x_singular = x[ii,0,0]
            elif x.shape[0] == 1                   and x.shape[1] == 1:
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
                            x[ii,  jj-1,0],  # left in vprll axis
                            x[ii,  jj+1,0],  # right in vprll axis
                            x[ii+1,jj  ,0],  # up in vperp axis
                            x[ii+1,jj-1,0],  # diagonal up left
                            x[ii+1,jj+1,0],  # diagonal up right
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

    def _reset_timers(self):
        # profiling, internal method for dev use
        self.time_bavg_setup = 0
        self.time_bavg_trapz = 0
        self.time_bavg_singu = 0

    def bounce_average_norm_raw(self, x, norm=True):
        """
        Average over one-fourth(!) of an orbit for trapped particles,
        which is equivalent to one-half for passing particles.

        Remove safety checks and conditionals to go as fast as possible

        Input:
            x = np.ndarray shape (vperp, vprll, s)
        Return:
            np.ndarray shape (vperp, vprll) holding normalized bounce-average
            of x
        """

        _t0 = time.perf_counter()

        integrand = x * self.inv_vprll_loc_abs  # (vperp,vprll,s) grid

        _t1 = time.perf_counter()

        result = np.trapezoid(integrand, self.ssamp, axis=-1)
        result *= self.inv_tbounce4th

        _t2 = time.perf_counter()

        pitch90 = (self.species.vprll_vec == 0)
        muzero = (self.species.vperp_vec == 0)
        #assert np.where(pitch90)[0].size == 1
        #assert np.where(muzero)[0].size == 1
        if np.any(pitch90):
            jj = np.where(pitch90)[0][0]

            # limiting form of bounce-average integral near the singularity,
            # valid for the case x=1, but TODO MAY NOT BE CORRECT FOR x(s)
            # spatially varying........ --ATr,2025nov06

            # handle singularity v_parallel = 0
            result[~muzero,jj] = (
                # cannot use [~muzero,jj,0] b/c mixing selector functions
                (x[...,0])[~muzero,jj]
                * np.pi/2 * (self.species.mass / self.dB_ds2_origin)**0.5
                / (self.mu[~muzero,jj])**0.5
            ) * self.inv_tbounce4th[~muzero,jj]

            # handle origin v_parallel = v_perp = 0
            # force to the moment's value at s=0 at singular point??
            #if np.any(muzero):
            #    ii = np.where(muzero)[0][0]
            #    result[ii,jj] = x[ii,jj,0]
            result[muzero,jj] = x[muzero,jj,0]

        _t3 = time.perf_counter()

        self.time_bavg_setup += _t1 - _t0
        self.time_bavg_trapz += _t2 - _t1
        self.time_bavg_singu += _t3 - _t2

        return result

    def bounce_average_njit(self, x, norm=True):
        """
        Average over one-fourth(!) of an orbit for trapped particles,
        which is equivalent to one-half for passing particles.
        Add numba JIT compilation to try to speed up loops.

        Input:
            x = shape (s, vperp, vprll)
        Return:
            shape (vperp, vprll)
        """
        return _bounce_average_njit_kernel(
                x_grid         = x,
                Bsamp_grid     = self.Bmag,
                s_grid         = self.ssamp,
                E_grid         = self.E,
                mu_grid        = self.mu,
                vperp_vec      = self.species.vperp_vec,
                vprll_vec      = self.species.vprll_vec,
                mass           = self.species.mass,
                dB_ds2_origin  = self.dB_ds2_origin,
                tbounce4th     = self.tbounce4th,
                tbounce4th_max = self.TBOUNCE4TH_MAX,
                norm           = norm
        )
