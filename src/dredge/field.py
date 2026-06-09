"""
Represent electromagnetic fields on flux tube or surface.
Includes tools to construct fields from analytic functions.
"""

import numpy as np
#import scipy as sp
import scipy  # avoid collision with sp = alias for species

from scipy.interpolate import RegularGridInterpolator

from .const import CLIGHT

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


class ConstFieldLine(FieldLine):
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


class VecFieldLine(FieldLine):
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
        # spatial-coordinate interpolators used for wavevector scale factors
        self.r_interp         = RegularGridInterpolator((self.s,), self.r, **kws)
        self.z_interp         = RegularGridInterpolator((self.s,), self.z, **kws)

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

    def query_r_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        return self.r_interp(points)

    def query_z_at(self, s_points):
        points = np.asarray(s_points)[...,np.newaxis]
        return self.z_interp(points)

    def query_s_at(self, B_points):
        points = np.asarray(B_points)[...,np.newaxis]
        return self.s_interp(points)


class ParabolicFieldLine(VecFieldLine):
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
                 n_steps: int,
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
        r_pos, z_pos = self._trace_field_line(r0, z0, ds=ds, n_steps=n_steps)

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

    def _trace_field_line(self, r0, z0, ds=1., n_steps=10):
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


class DipoleFieldLine(VecFieldLine):
    """
    Axisymmetric dipole magnetic field, far-field limit,
    using Equations (2.13a-d) of Mishchenko et al. (2018, JPP).
    Field is computed on a discrete grid of cylindrical (r,z) positions.
    """
    def __init__(self,
                 I: float,
                 r0: float,
                 req: float,
                 ds: float,
                 n_steps: float,
                 axis_r = 1,
                 axis_z = 2,
                 ):
        r"""
        Axisymmetric dipole magnetic field, far-field limit,
        using Equations (2.13a-d) of Mishchenko et al. (2018, JPP).
        Field is computed on a discrete grid of cylindrical (r,z) positions.
        Field trace starts from (r,z)=(req,0) on the equatorial plane, where req
        is a user-specified radius, and is performed using RK4 method.

        Inputs:
            I: current in statAmpere (CGS unit)
            r0: current ring radius in cm (CGS unit)
            req: radius at equatorial plane to select field line, in cm (CGS unit)
            ds: step size in arc-length for field-line tracing, in cm (CGS unit)
            n_steps: number of steps to trace field line
            axis_r: which Cartesian (x,y,z) coordinate shall be radial at equatorial plane?
            axis_z: which Cartesian (x,y,z) coordinate shall be axial at equatorial plane?
        """
        self.I = I
        self.r0 = r0
        self.M = np.pi * self.I * self.r0**2 / CLIGHT  # magnetic moment of current loop in CGS units
        self.req = req
        self.psi = self.M / self.req  # flux function at equatorial plane, identifies field line

        r_pos, z_pos = self._trace_field_line(req, 0, ds=ds, n_steps=n_steps)

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
            Bx        = Bvec[0],
            By        = Bvec[1],
            Bz        = Bvec[2],
            gradBmagx = gradBmag[0],
            gradBmagy = gradBmag[1],
            gradBmagz = gradBmag[2],
            r_pos     = r_pos,
            z_pos     = z_pos,
        )

    def Br_func(self, r, z):
        """Compute B_r component at cylindrical (r,z) in cm"""
        Brho = 2 * self.M * z / (r**2 + z**2)**2
        Btheta = self.M * r / (r**2 + z**2)**2
        return Brho * np.sin(np.arctan2(r,z)) + Btheta * np.cos(np.arctan2(r,z))

    def Bz_func(self, r, z):
        """Compute B_z component at cylindrical (r,z) in cm"""
        Brho = 2 * self.M * z / (r**2 + z**2)**2
        Btheta = self.M * r / (r**2 + z**2)**2
        return Brho * np.cos(np.arctan2(r,z)) - Btheta * np.sin(np.arctan2(r,z))

    def dBmag_dr_func(self, r, z):
        """Compute d|B|/dr at cylindrical (r,z) in cm"""
        # Use Mathematica to check algebra
        #   modB[r_, z_] := absM/(r^2 + z^2)^2*Sqrt[4*z^2 + r^2];
        #   Simplify[D[modB[r, z], r]]
        return -3 * abs(self.M) * r * (r**2 + 5*z**2) / (r**2 + z**2)**3 / np.sqrt(4*z**2 + r**2)

    def dBmag_dz_func(self, r, z):
        """Compute d|B|/dz at cylindrical (r,z) in cm"""
        # Use Mathematica to check algebra
        #   modB[r_, z_] := absM/(r^2 + z^2)^2*Sqrt[4*z^2 + r^2];
        #   Simplify[D[modB[r, z], z]]
        return -12 * abs(self.M) * z**3 / (r**2 + z**2)**3 / np.sqrt(4*z**2 + r**2)

    def _trace_field_line(self, r0, z0, ds, n_steps):
        """
        Trace field line using RK4 method starting from (r0,z0) position.
        Args:
            r0: starting radius in cm
            z0: starting axial coordinate in cm
            ds: arc-length step size in cm
            n_steps: number of steps to take
        Return:
            two-tuple (r,z) of radius and axial coordinates tracing a magnetic
            field line; r and z are each a 1D numpy.ndarray of shape (n_steps,)
        """
        # TODO refactor the field-line tracing methods out of VecFieldLine
        # subclasses --ATr,2025nov15
        r = np.empty(n_steps, dtype=np.float64)
        z = np.empty(n_steps, dtype=np.float64)
        r[0] = r0
        z[0] = z0

        def veloc(ri, zi):
            Br = self.Br_func(ri, zi)
            Bz = self.Bz_func(ri, zi)
            Bmag = np.sqrt(Br**2 + Bz**2)
            return Br / Bmag, Bz / Bmag

        for ii in range(1, n_steps):
            # 4th-order Runge-Kutta method
            k1r, k1z = veloc(r[ii-1],              z[ii-1])
            k2r, k2z = veloc(r[ii-1] + 0.5*ds*k1r, z[ii-1] + 0.5*ds*k1z)
            k3r, k3z = veloc(r[ii-1] + 0.5*ds*k2r, z[ii-1] + 0.5*ds*k2z)
            k4r, k4z = veloc(r[ii-1] +     ds*k3r, z[ii-1] +     ds*k3z)
            r[ii] = r[ii-1] + (ds/6.) * (k1r + 2*k2r + 2*k3r + k4r)
            z[ii] = z[ii-1] + (ds/6.) * (k1z + 2*k2z + 2*k3z + k4z)

        return r, z


class PleiadesFieldLine(VecFieldLine):
    """
    Axisymmetric magnetic field line from HDF5 file output by Pleiades code.
    Field line is traced from (r0, z0) along +z direction.
    """

    def __init__(self,
                 path: str,
                 r0: float,
                 z0: float = 0.,
                 ds: float = 0.5,
                 n_steps: int = 200,
                 axis_r: int = 1,
                 axis_z: int = 2,
                 field_group: str = 'VacuumFields',
                 ):
        r"""
        Axisymmetric magnetic field line from HDF5 file output by Pleiades code.
        Field line is traced from (r0, z0) along +z direction.

        WARNING: base class VecFieldLine requires |B| to be monotonic along the
        traced field line.  For a mirror machine this holds from the midplane
        (z=0) to the first mirror throat, but NOT necessarily over the full
        z range.  Choose n_steps * ds to stay within the monotonic segment.
        Verify after construction: assert np.all(np.diff(field.Bmag) > 0.)

        Expected HDF5 layout:
            Mesh/R            shape (n_z, n_r)  — r-coordinates, meters
            Mesh/Z            shape (n_z, n_r)  — z-coordinates, meters
            VacuumFields/B    shape (n_z, n_r)  — |B| magnitude, Tesla
            VacuumFields/BR   shape (n_z, n_r)  — B_r component, Tesla
            VacuumFields/BZ   shape (n_z, n_r)  — B_z component, Tesla

        User may also request 'Equilibrium' fields, which sum both vacuum and
        plasma diamagnetic fields; the HDF5 layout for Equilibrium/{B,B,BZ}
        should match that of VacuumFields.

        Legacy files without 'Mesh' group are auto-detected; these contain 1D
        R,Z grid vectors and B,B_R,B_Z arrays of shape (n_r,n_z).

        Inputs:
            path:    path to HDF5 file
            r0:      starting radius in cm (CGS)
            z0:      starting axial position in cm (CGS); default 0 (midplane)
            ds:      RK4 arc-length step size in cm (CGS)
            n_steps: number of RK4 steps to trace
            axis_r:  which Cartesian (x,y,z) index maps to cylindrical r
            axis_z:  which Cartesian (x,y,z) index maps to cylindrical z
            field_group: 'VacuumFields' or 'Equilibrium', corresponding to
                         the desired HDF5 group in Pleiades output file
        """
        import h5py

        # Pleiades HDF5 files before ~2026 May used 'Equil' rather than
        # 'Equilibrium'
        assert field_group in ['VacuumFields', 'Equilibrium']

        with h5py.File(path, 'r') as f:
            # current grouped layout
            if 'Mesh' in f:
                R  = f['Mesh/R'][...].T  # (n_z,n_r) -> (n_r,n_z)
                Z  = f['Mesh/Z'][...].T
                B  = f[f'{field_group:s}/B'][...].T
                BR = f[f'{field_group:s}/BR'][...].T
                BZ = f[f'{field_group:s}/BZ'][...].T
                R_vec, Z_vec = R[:, 0], Z[0, :]    # 1D grid vectors from 2D mesh
                assert np.all(R == R_vec[:,np.newaxis]), "mesh must be rectilinear"
                assert np.all(Z == Z_vec[np.newaxis,:]), "mesh must be rectilinear"
            # legacy flat layout (B_R/B_Z, 1D R/Z)
            else:
                assert not any(isinstance(v, h5py.Group) for v in f.values()), \
                    "Unrecognized HDF5 layout (expected groupless legacy file)"
                R_vec = f['R'][...]  # already 1D grid vectors
                Z_vec = f['Z'][...]
                B  = f['B'][...]     # already (n_r, n_z)
                BR = f['B_R'][...]
                BZ = f['B_Z'][...]

        # both layouts store SI; convert to CGS to match dredge convention
        R_vec = R_vec * 100  # meters -> cm
        Z_vec = Z_vec * 100
        B  = B  * 1e4  # Tesla -> Gauss
        BR = BR * 1e4
        BZ = BZ * 1e4

        # Enforce axisymmetry boundary condition Br(r=0) = 0.
        # Pleiades output fills the r=0 column by copying the r=Δr column rather
        # than imposing symmetry.  For Bz, |B| (even in r) this copy is harmless
        # to O(Δr^2), but for Br (odd in r) a nonzero on-axis value gives a
        # spurious *constant* inward Br across the first radial cell, which
        # drags near-axis traced field lines into r=0.  See git history /
        # diagnosis for wham-r_field.h5.  --ATr,2026jun
        if R_vec[0] == 0.:
            BR[0, :] = 0.

        # evaluate gradients on user-input mesh, and interpolate gradients,
        # instead of taking gradients of interpolated B values
        dBdR_2d, dBdZ_2d = np.gradient(B, R_vec, Z_vec)

        # setup for field line tracing
        interp_kws = dict(bounds_error=True)
        self._BR_interp = RegularGridInterpolator((R_vec, Z_vec), BR, **interp_kws)
        self._BZ_interp = RegularGridInterpolator((R_vec, Z_vec), BZ, **interp_kws)
        self._dBdR_interp = RegularGridInterpolator((R_vec, Z_vec), dBdR_2d, **interp_kws)
        self._dBdZ_interp = RegularGridInterpolator((R_vec, Z_vec), dBdZ_2d, **interp_kws)

        r_pos, z_pos = self._trace_field_line(r0, z0, ds, n_steps)

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
            Bx        = Bvec[0],
            By        = Bvec[1],
            Bz        = Bvec[2],
            gradBmagx = gradBmag[0],
            gradBmagy = gradBmag[1],
            gradBmagz = gradBmag[2],
            r_pos     = r_pos,
            z_pos     = z_pos,
        )

    def Br_func(self, r, z):
        """Compute B_r component at cylindrical (r,z) in cm"""
        # Claude's advice: .ravel() and .reshape(...) idiom for
        # interpolator-backed functions preserves numpy's scalar broadcasting
        # contract, matching how pure-math implementations like
        # DipoleFieldLine behave automatically.
        r, z = np.asarray(r), np.asarray(z)
        pts = np.column_stack([r.ravel(), z.ravel()])
        return self._BR_interp(pts).reshape(r.shape)

    def Bz_func(self, r, z):
        """Compute B_z component at cylindrical (r,z) in cm"""
        r, z = np.asarray(r), np.asarray(z)
        pts = np.column_stack([r.ravel(), z.ravel()])
        return self._BZ_interp(pts).reshape(r.shape)

    def dBmag_dr_func(self, r, z):
        """Compute d|B|/dr at cylindrical (r,z) in cm"""
        r, z = np.asarray(r), np.asarray(z)
        pts = np.column_stack([r.ravel(), z.ravel()])
        return self._dBdR_interp(pts).reshape(r.shape)

    def dBmag_dz_func(self, r, z):
        """Compute d|B|/dz at cylindrical (r,z) in cm"""
        r, z = np.asarray(r), np.asarray(z)
        pts = np.column_stack([r.ravel(), z.ravel()])
        return self._dBdZ_interp(pts).reshape(r.shape)

    def _trace_field_line(self, r0, z0, ds, n_steps):
        """
        Trace field line using RK4 method starting from (r0,z0) position.
        Args:
            r0: starting radius in cm
            z0: starting axial coordinate in cm
            ds: arc-length step size in cm
            n_steps: number of steps to take
        Return:
            two-tuple (r,z) of radius and axial coordinates tracing a magnetic
            field line; r and z are each a 1D numpy.ndarray of shape (n_steps,)
        """
        # TODO refactor the field-line tracing methods out of VecFieldLine
        # subclasses --ATr,2025nov15
        r = np.empty(n_steps, dtype=np.float64)
        z = np.empty(n_steps, dtype=np.float64)
        r[0] = r0
        z[0] = z0

        def veloc(ri, zi):
            Br = self.Br_func(ri, zi)
            Bz = self.Bz_func(ri, zi)
            Bmag = np.sqrt(Br**2 + Bz**2)
            return Br / Bmag, Bz / Bmag

        for ii in range(1, n_steps):
            # 4th-order Runge-Kutta method
            k1r, k1z = veloc(r[ii-1],              z[ii-1])
            k2r, k2z = veloc(r[ii-1] + 0.5*ds*k1r, z[ii-1] + 0.5*ds*k1z)
            k3r, k3z = veloc(r[ii-1] + 0.5*ds*k2r, z[ii-1] + 0.5*ds*k2z)
            k4r, k4z = veloc(r[ii-1] +     ds*k3r, z[ii-1] +     ds*k3z)
            r[ii] = r[ii-1] + (ds/6.) * (k1r + 2*k2r + 2*k3r + k4r)
            z[ii] = z[ii-1] + (ds/6.) * (k1z + 2*k2z + 2*k3z + k4z)

        return r, z
