"""
Code to construct uniform grids of (k, Re(ω), Im(ω)) for linear plasma
response calculation
"""

import numpy as np


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
