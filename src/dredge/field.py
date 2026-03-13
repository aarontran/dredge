"""
Represent electromagnetic fields on flux tube or surface.
Includes tools to construct fields from analytic functions.
"""

import numpy as np
#import scipy as sp
import scipy  # avoid collision with sp = alias for species

from scipy.interpolate import RegularGridInterpolator


class FieldLine(object):
    """
    Axisymmetric magnetic field line in cylindrical geometry
    Abstract base class is not meant for users to initialize;
    it provides dummy methods and attributes to
    * hint, for code developers, what functionality must be implemented.
    * provide function docstrings for inheriting classes
    """

    # docstring inheritance works for Python version >= 3.5 according to
    # https://stackoverflow.com/a/53858708
    # and https://stackoverflow.com/a/73101317

    def __init__(self):
        pass

    def query_Bmag_at(self, s_points: np.ndarray):
        """
        Query magnetic-field magnitude at arc-length coordinates along flux
        tube, broadcasting over input numpy.ndarray shape.

        Args:
            s_points: arc length in cm, numpy.ndarray instance

        Returns:
            numpy.ndarray of same shape as s_points, holding magnetic-field
            magnitude in Gauss (CGS unit).
        """
        #return np.zeros_like(s_points)
        pass

    def query_B_at(self, s_points: np.ndarray):
        """
        Query magnetic-field vector at arc-length coordinates along flux tube,
        broadcasting over input numpy.ndarray shape.

        Args:
            s_points: arc length in cm, numpy.ndarray instance

        Returns:
            numpy.ndarray of shape (3, *s_points.shape) holding magnetic-field
            vector in Cartesian (x,y,z) components.  Vector component values
            are expressed in Gauss (CGS unit).
        """
        #bx = np.zeros_like(s_points)
        #by = np.zeros_like(s_points)
        #bz = np.zeros_like(s_points)
        #return np.array([bx,by,bz])
        pass

    def query_gradBmag_at(self, s_points: np.ndarray):
        """
        Query gradient of magnetic-field magnitude, grad(|B|), at arc-length
        coordinates along flux tube, broadcasting over input numpy.ndarray
        shape.

        Args:
            s_points: arc length in cm, numpy.ndarray instance

        Returns:
            numpy.ndarray of shape (3, *s_points.shape) holding grad(|B|)
            vector in Cartesian (x,y,z) components.  Vector component values
            are expressed in Gauss/cm (CGS unit).
        """
        #dBx = np.zeros_like(s_points)
        #dBy = np.zeros_like(s_points)
        #dBz = np.zeros_like(s_points)
        #return np.array([dBx, dBy, dBz])
        pass

    def query_dbhat_ds_at(self, s_points):
        r"""
        Query directional derivative of magnetic-field unit vector along
        itself, d/ds( \hat{b} ), at arc-length coordinates along flux tube,
        broadcasting over input numpy.ndarray shape.

        Args:
            s_points: arc length in cm, numpy.ndarray instance

        Returns:
            numpy.ndarray of shape (3, *s_points.shape) holding d/ds(\hat{b})
            vector in Cartesian (x,y,z) components.  Vector component values
            are expressed in 1/cm (CGS unit).
        """
        #kx = np.zeros_like(s_points)
        #ky = np.zeros_like(s_points)
        #kz = np.zeros_like(s_points)
        #return np.array([kx, ky, kz])
        pass

    def query_s_at(self, B_points):
        """
        Query arc-length coordinate corresponding to given magnetic-field
        magnitude values on flux tube, broadcasting over input numpy.ndarray
        shape.  Only works for monotonically ascending or descending |B|(s).

        Args:
            B_points: magnetic-field magnitude in Gauss (CGS unit),
                      numpy.ndarray instance

        Returns:
            numpy.ndarray of same shape as B_points, holding arc-length
            coordinates in cm (CGS unit).
        """
        #return np.zeros_like(B_points)
        pass


class FieldLineConst(FieldLine):
    """
    Axisymmetric CONSTANT magnetic field line in cylindrical geometry
    """
    def __init__(self, B0, axis=2):
        """
        Constant magnetic field line with no spatial gradients
        Inputs:
            B0 = magnetic field strength in Gauss (CGS unit)
            axis = 0,1,2 which way the field points? in Cartesian (x,y,z)
        """
        self.B0 = B0
        self.axis = axis

    def query_Bmag_at(self, s_points):
        return self.B0 * np.ones_like(s_points)

    def query_B_at(self, s_points):
        Bx = np.zeros_like(s_points)
        By = np.zeros_like(s_points)
        Bz = self.B0 * np.ones_like(s_points)
        if self.axis == 0:
            return np.array([Bz, Bx, By])
        if self.axis == 1:
            return np.array([By, Bz, Bx])
        if self.axis == 2:
            return np.array([Bx, By, Bz])

    def query_gradBmag_at(self, s_points):
        return np.zeros((3,*s_points.shape), dtype=s_points.dtype)

    def query_dbhat_ds_at(self, s_points):
        return np.zeros((3,*s_points.shape), dtype=s_points.dtype)

    def query_s_at(self, B_points):
        return np.zeros_like(B_points)


class FieldLineVec(FieldLine):
    """
    Axisymmetric magnetic field line in cylindrical geometry
    """
    def __init__(self,
                 Bx: np.ndarray,
                 By: np.ndarray,
                 Bz: np.ndarray,
                 gradBmagx : np.ndarray,
                 gradBmagy : np.ndarray,
                 gradBmagz : np.ndarray,
                 r_pos: np.ndarray,
                 z_pos: np.ndarray,
                 **kwargs):
        r"""
        Axisymmetric magnetic field line in cylindrical geometry,
        represented by 1D vectors along \hat{b}.

        WARNINGS:
        * Only works for |B| monotonically ascending.
        * All inputs are 1D numpy arrays.

        Inputs:
            Bx = magnetic field Cartesian x-component in Gauss (CGS units)
            By = magnetic field Cartesian y-component in Gauss (CGS units)
            Bz = magnetic field Cartesian z-component in Gauss (CGS units)
            gradBmagx = grad(|B|) x-component in Gauss/cm (CGS units)
            gradBmagy = grad(|B|) y-component in Gauss/cm (CGS units)
            gradBmagz = grad(|B|) z-component in Gauss/cm (CGS units)
            r_pos = r-coordinate positions (cm) for B field on flux surface
            z_pos = z-coordinate positions (cm) for B field on flux surface
                    must be monotonically ascending,
                    must have z=0. as the first point in array.
            **kwargs = passed to scipy.interpolate.RegularGridInterpolator(...)
                       for all subsequent user queries of field values along
                       surface
        """
        # Enforce all 1D numpy arrays of same size
        for dat in [Bx,By,Bz, gradBmagx,gradBmagy,gradBmagz, r_pos,z_pos]:
            assert dat.ndim == 1
            assert dat.size == Bx.size

        # Store all values in physical (CGS) units
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.B = np.array([Bx, By, Bz])
        self.Bmag = (Bx**2 + By**2 + Bz**2)**0.5
        self.bhat = self.B / self.Bmag

        self.gradBmagx = gradBmagx
        self.gradBmagy = gradBmagy
        self.gradBmagz = gradBmagz
        self.gradB = np.array([gradBmagx, gradBmagy, gradBmagz])

        self.r = r_pos
        self.z = z_pos

        # compute arc length along the curve,
        # needed for bounce-average integral
        dl = (np.diff(r_pos)**2 + np.diff(z_pos)**2)**0.5
        dr_dl = np.diff(r_pos) / dl
        dz_dl = np.diff(z_pos) / dl
        ds_dl = (dr_dl**2 + dz_dl**2)**0.5
        self.s = np.cumsum(ds_dl * dl)  # integrate along arc
        self.s = np.insert(self.s, 0, 0.)  # start at s=0

        # compute curvature vector along the flux tube
        # NOTE it's better to take gradients on user provided points, rather
        # than compute gradients upon interpolation points that vary in
        # velocity space; faster lookup table to get gradients and avoids ugly
        # singularities, and user's provided grid resolution determines
        # accuracy of gradients rather than "downstream" numerical resolution
        # choices interfering with accuracy of gradient computations
        self.dbhat_ds_x = np.gradient(self.bhat[0], self.s, edge_order=2)
        self.dbhat_ds_y = np.gradient(self.bhat[1], self.s, edge_order=2)
        self.dbhat_ds_z = np.gradient(self.bhat[2], self.s, edge_order=2)
        self.dbhat_ds = np.array([self.dbhat_ds_x,
                                  self.dbhat_ds_y,
                                  self.dbhat_ds_z])

        # for interpolators, B must be ascending/descending
        # (cannot be EXACTLY const) and requires >=3 points for gradient
        # calculations

        # pre-cache interpolators for B, grad(B), curvature along flux tube
        if 'bounds_error' not in kwargs:
            kws = dict(bounds_error=True, **kwargs)
        else:
            kws = kwargs
        self.Bx_interp        = RegularGridInterpolator((self.s,), self.Bx, **kws)
        self.By_interp        = RegularGridInterpolator((self.s,), self.By, **kws)
        self.Bz_interp        = RegularGridInterpolator((self.s,), self.Bz, **kws)
        self.Bmag_interp      = RegularGridInterpolator((self.s,), self.Bmag, **kws)
        self.gradBmagx_interp = RegularGridInterpolator((self.s,), self.gradBmagx, **kws)
        self.gradBmagy_interp = RegularGridInterpolator((self.s,), self.gradBmagy, **kws)
        self.gradBmagz_interp = RegularGridInterpolator((self.s,), self.gradBmagz, **kws)
        self.dbhat_ds_x_interp = RegularGridInterpolator((self.s,), self.dbhat_ds_x, **kws)
        self.dbhat_ds_y_interp = RegularGridInterpolator((self.s,), self.dbhat_ds_y, **kws)
        self.dbhat_ds_z_interp = RegularGridInterpolator((self.s,), self.dbhat_ds_z, **kws)

        # interpolator to convert from B-field magnitude to arc length s
        # only works if B-field monotonically ascends/descends along s
        assert np.all(np.diff(self.Bmag) > 0.) or np.all(np.diff(self.Bmag) < 0.)
        self.s_interp = RegularGridInterpolator((self.Bmag,), self.s, **kws)

    ## TODO DO WE REALLY NEED THESE INTERPOLATION METHODS?
    ## Maybe more efficient to integrate directly using the user-supplied points;
    ## then if better resolution demanded, user responsible for providing
    ## finer grained B-field... -ATr,2025oct14

    # NOTE YES WE DO NEED because s sample points vary in velocity space
    # ALSO it's better to compute gradients on user provided points,
    # then just interpolate to get result; it avoids numerical issues
    # when taking gradients with s=(0,0) for pathological parts of velociy
    # space
    # --ATr,2025nov13

    def query_Bmag_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        return self.Bmag_interp(points)

    def query_B_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        Bx = self.Bx_interp(points)
        By = self.By_interp(points)
        Bz = self.Bz_interp(points)
        return np.array([Bx, By, Bz])

    def query_gradBmag_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        dBx = self.gradBmagx_interp(points)
        dBy = self.gradBmagy_interp(points)
        dBz = self.gradBmagz_interp(points)
        return np.array([dBx, dBy, dBz])

    def query_dbhat_ds_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        kx = self.dbhat_ds_x_interp(points)
        ky = self.dbhat_ds_y_interp(points)
        kz = self.dbhat_ds_z_interp(points)
        return np.array([kx, ky, kz])

    def query_s_at(self, B_points):
        points = np.asarray(B_points)[...,np.newaxis]
        return self.s_interp(points)


class FieldLineParabolic(FieldLineVec):
    """
    Axisymmetric magnetic field with parabolic (r,z) dependence and curl(B)=0
    in cylindrical geometry, which provides a simple analytic approximation to
    a magnetic mirror device, computed on a discrete grid of (r,z) positions.
    """
    def __init__(self,
                 B0: float,
                 Bt: float,
                 Lp: float,
                 r0: float,
                 z0: float,
                 ds: float,
                 n_steps: float,
                 axis_r = 1,
                 axis_z = 2,
                 ):
        r"""
        Axisymmetric magnetic field with parabolic (r,z) dependence and
        curl(B)=0 in cylindrical geometry, which provides a simple analytic
        approximation to a magnetic mirror device, computed on a discrete grid
        of (r,z) positions.

        Obtain flux surface by a simple forward-Euler ray-tracing method.
        The accuracy can be improved.

        The analytic functions are:

            B_z = B_0 + (B_t - B_0) \left(\frac{z}{L_p}\right)^2
                      - (B_t - B_0) \frac{1}{2} \left(\frac{r}{L_p}\right)^2

        and

            B_r = - (B_t - B_0) \frac{r z}{{L_p}^2}

        where B_0, B_t, L_p are constants.

        Inputs:
            B0: midplane magnetic field at (r,z)=(0,0) in Gauss (CGS unit)
            Bt: throat magnetic field at (r,z)=(0,Lp) in Gauss (CGS unit)
            Lp: magnetic-mirror throat z coordinate in cm (CGS unit)
            r0: radius at midplane to select flux surface, in cm (CGS unit)
            ds: step size in arc-length for field-line tracing
            n_steps: number of steps to trace field line
            axis_r: which Cartesian (x,y,z) coordinate shall be radial?
            axis_z: which Cartesian (x,y,z) coordinate shall be axial?
        """
        self.B0 = B0
        self.Bt = Bt
        self.Lp = Lp
        self.r0 = r0
        self.z0 = z0

        # TODO if we implement more analytic functions, we may want to refactor
        # out the general code framework for converting analytic functions to
        # discrete data --ATr,2025nov15
        r_pos, z_pos = self.trace_field_line(r0, z0, ds=ds, n_steps=n_steps)

        Bvec     = np.zeros((3,r_pos.size), dtype=r_pos.dtype)
        gradBmag = np.zeros((3,r_pos.size), dtype=r_pos.dtype)

        assert axis_r != axis_z
        assert axis_r in [0,1,2]
        assert axis_z in [0,1,2]
        Bvec[axis_r]     = self.Br_func      (r_pos, z_pos)
        Bvec[axis_z]     = self.Bz_func      (r_pos, z_pos)
        gradBmag[axis_r] = self.dBmag_dr_func(r_pos, z_pos)
        gradBmag[axis_z] = self.dBmag_dz_func(r_pos, z_pos)

        super().__init__(
            Bx = Bvec[0],
            By = Bvec[1],
            Bz = Bvec[2],
            gradBmagx = gradBmag[0],
            gradBmagy = gradBmag[1],
            gradBmagz = gradBmag[2],
            r_pos = r_pos,
            z_pos = z_pos,
        )

    # assume parabolic B, then enforce div(B)=0 and Br(z=0) = 0
    # to determine the functional form of Br
    # further enforce curl(B)=0 to get an extra parabolic term in Bz(r,z)

    def Br_func(self, r, z):
        """Compute B_r component at cylindrical (r,z) in cm"""
        B0, Bt, Lp = (self.B0, self.Bt, self.Lp)
        return -(Bt-B0) * (r/Lp) * (z/Lp)

    def Bz_func(self, r, z):
        """Compute B_z component at cylindrical (r,z) in cm"""
        B0, Bt, Lp = (self.B0, self.Bt, self.Lp)
        return B0 + (Bt-B0)*(z/Lp)**2 - 0.5*(Bt-B0)*(r/Lp)**2

    # MATHEMATICA INPUT to get grad(B) expressions
    #   bz[r_, z_] := b + d*(z/lp)^2 - (d/2)*(r/lp)^2
    #   br[r_, z_] := -d*r*z/lp^2
    #   D[Sqrt[bz[r, z]^2 + br[r,z]^2], r]
    #   D[Sqrt[bz[r, z]^2 + br[r,z]^2], z]

    def dBmag_dr_func(self, r, z):
        """Compute d|B|/dr at cylindrical (r,z) in cm"""
        B0, Bt, Lp = (self.B0, self.Bt, self.Lp)
        d = (Bt-B0)
        Bzmag = B0 + d*(z/Lp)**2 - 0.5*d*(r/Lp)**2
        # only numerator changes for d/dz versus d/dr
        num = 2*d**2 * r * z**2 / Lp**4 - 2*d*r*Bzmag/Lp**2
        argsq = d**2 * (r/Lp)**2 * (z/Lp)**2 + Bzmag**2
        return num / (2 * np.sqrt(argsq))

    def dBmag_dz_func(self, r, z):
        """Compute d|B|/dz at cylindrical (r,z) in cm"""
        B0, Bt, Lp = (self.B0, self.Bt, self.Lp)
        d = (Bt-B0)
        Bzmag = B0 + d*(z/Lp)**2 - 0.5*d*(r/Lp)**2
        # only numerator changes for d/dz versus d/dr
        num = 2*d**2 * r**2 * z / Lp**4 + 4*d*z*Bzmag/Lp**2
        argsq = d**2 * (r/Lp)**2 * (z/Lp)**2 + Bzmag**2
        return num / (2 * np.sqrt(argsq))

    def trace_field_line(self, r0, z0, ds=1., n_steps=10):
        """
        Trace magnetic field line using forward-Euler method, starting from
        some initial (r0,z0) position.
        Args:
            r0: starting radius in cm
            z0: starting axial coordinate in cm
            ds: arc-length step size in cm
            n_steps: number of steps to take
        Return:
            two-tuple (r,z) of radius and axial coordinates tracing a magnetic
            field line; r and z are each a 1D numpy.ndarray of shape (n_steps,)
        """
        # TODO if we implement more analytic functions, we may want to refactor
        # out the field-line tracing methods --ATr,2025nov15
        r = np.empty(n_steps, dtype=np.float64)
        z = np.empty(n_steps, dtype=np.float64)
        r[0] = r0
        z[0] = z0
        for ii in range(1, n_steps):
            # extremely crude forward euler method
            # only good for small step size ds
            # TODO improve method to leapfrogged Euler, Crank-Nicholson, RK4,
            # or anything better than forward Euler --ATr,2025nov15
            Br = self.Br_func(r[ii-1], z[ii-1])
            Bz = self.Bz_func(r[ii-1], z[ii-1])
            Bmag = (Br**2 + Bz**2)**0.5
            vr = Br/Bmag
            vz = Bz/Bmag
            r[ii] = r[ii-1] + vr*ds
            z[ii] = z[ii-1] + vz*ds
        return (r,z)
