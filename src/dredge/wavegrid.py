"""
Code to construct uniform grids of (k, Re(ω), Im(ω)) for linear plasma
response calculation
"""

import numpy as np
from mpi4py import MPI

from .util import print0, printn

class WaveGrid(object):
    """
    Grid of (k, Re(ω), Im(ω)) for dispersion relation calculations with
    MPI domain decomposition.
    """

    def __init__(self, k_vec_global, omega_re_vec_global, omega_im_vec_global,
                 proc_layout=(1,1,1)):
        """
        Initialize local sub-grid of global (k, Re(ω), Im(ω)) grid
        for dispersion relation calculations, based on a manual MPI processor
        layout.

        Inputs:
            k_vec_global: 1D array of angular wavenumber grid points in cm^{-1}.
            omega_re_vec_global: 1D array, Re(omega) grid points in rad/s.
            omega_im_vec_global: 1D array, Im(omega) grid points in rad/s.
            proc_layout: tuple (px,py,pz) for (k, omega_re, omega_im) axes.
                         product px*py*pz must equal MPI COMM_WORLD size
        """

        # Does requested layout match the available resources?
        if np.prod(proc_layout) != MPI.COMM_WORLD.Get_size():
            raise ValueError(
                f"Layout {proc_layout} requires {np.prod(proc_layout)} ranks, "
                f"but COMM_WORLD size is {MPI.COMM_WORLD.Get_size()}."
            )

        # Create Cartesian communicator for coordinate mapping;
        # wrapper around MPI_Cart_create(...), which organizes MPI ranks for
        # arch/network topology; overkill, but convenient
        # https://web.cels.anl.gov/~thakur/sc15-mpi-tutorial/slides.pdf
        # https://www.cs.kent.edu/~farrell/dist/ref/mpitut/node67.html
        self.cart_comm = MPI.COMM_WORLD.Create_cart(proc_layout,
                                                    periods=[False,False,False], reorder=True)
        assert self.cart_comm != MPI.COMM_NULL
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.cart_rank = self.cart_comm.Get_rank()
        self.world_size = self.cart_comm.Get_size()
        self.proc_layout = proc_layout
        # Get my MPI topology coordinates in the 3D grid.
        # E.g., for size=4
        # rank=0 -> block_idxs=[0, 0, 0]
        # rank=1 -> block_idxs=[0, 1, 0]
        # rank=2 -> block_idxs=[1, 0, 0]
        # rank=3 -> block_idxs=[1, 1, 0]
        self.block_idxs = self.cart_comm.Get_coords(self.cart_rank)

        # Decompose the global indices into local ranges
        # NX,NY,NZ = global number of grid points in each dimension
        # lx,ly,lz = local number of grid points in each dimension
        # sx,sy,sz = global start index
        self.NX = len(k_vec_global)
        self.NY = len(omega_re_vec_global)
        self.NZ = len(omega_im_vec_global)
        self.sx, self.lx = self.span(self.NX, proc_layout[0], self.block_idxs[0])
        self.sy, self.ly = self.span(self.NY, proc_layout[1], self.block_idxs[1])
        self.sz, self.lz = self.span(self.NZ, proc_layout[2], self.block_idxs[2])

        # TODO DEV/DEBUGGING
        printn("MPI block", self.block_idxs,
               f"global idx ({self.sx:d},{self.sy:d},{self.sz:d})",
               f"offset ({self.lx:d},{self.ly:d},{self.lz:d})",
               f"Nvoxels {self.lx*self.ly*self.lz}")

        # Global domain vectors
        self.k_vec_global        = k_vec_global
        self.omega_re_vec_global = omega_re_vec_global
        self.omega_im_vec_global = omega_im_vec_global
        self._validate_monotony("k",         k_vec_global)
        self._validate_monotony("Re(omega)", omega_re_vec_global)
        self._validate_monotony("Im(omega)", omega_im_vec_global)

        # Local domain vectors: individual ranks operate ONLY on these
        self.k_vec        = k_vec_global       [self.sx : self.sx + self.lx]
        self.omega_re_vec = omega_re_vec_global[self.sy : self.sy + self.ly]
        self.omega_im_vec = omega_im_vec_global[self.sz : self.sz + self.lz]

        # dont allow any fancy/weird gridding
        assert self.k_vec.ndim == 1
        assert self.omega_re_vec.ndim == 1
        assert self.omega_im_vec.ndim == 1
        # grid points must ascend monotonically, with no duplicates
        # be careful because sign of charge enters into omega
        assert np.all(np.diff(self.omega_re_vec) > 0)
        assert np.all(np.diff(self.omega_im_vec) > 0)
        assert np.all(np.diff(self.k_vec) > 0)

    @staticmethod
    def _validate_monotony(name, vec):
        """Ensures global vectors are 1D and strictly increasing."""
        assert vec.ndim == 1, f"{name} coordinate vector must be 1D"
        if vec.size > 1:
            assert np.all(np.diff(vec) > 0), f"{name} coordinates must be strictly increasing"

    @staticmethod
    def span(N_global, M_ranks, pos):
        """
        Get number of grid points, starting global grid index for 1D array
        of N_global points decomposed amongst M_ranks, for current rank 'pos'.
        Args:
            N_global = global number of grid points
            M_ranks = total number MPI ranks along this dimension
            pos = my local MPI rank in range [0, M_ranks)
        Returns:
            (start, offset) where start = starting global grid index for local
            domain, offset = local number of grid points,
        """
        avg, rest = divmod(N_global, M_ranks)  # N_global = avg * M_ranks + rest
        # distribute remainder cells as evenly as possible
        N_local = avg + 1 if pos < rest else avg
        start = pos * avg + min(pos, rest)
        return start, N_local

    def gather(self, local_arr):
        """
        Gathers distributed sub-blocks into a global array on Rank 0.
        Args:
            local_arr: array of shape (lx,ly,lz)
        Returns:
            global_arr: array of shape (NX, NY, NZ) on rank 0, None on others
        """
        NX, sx, lx = self.NX, self.sx, self.lx
        NY, sy, ly = self.NY, self.sy, self.ly
        NZ, sz, lz = self.NZ, self.sz, self.lz

        # Why not MPI_Gatherv?  Don't prematurely optimize...  --ATr,2026mar13

        if self.cart_rank == 0:

            global_arr = np.zeros((NX,NY,NZ), dtype=local_arr.dtype)

            # copy my own chunk directly
            global_arr[sx:sx+lx, sy:sy+ly, sz:sz+lz] = local_arr

            # Collect chunks from all other ranks
            for r in range(1, self.world_size):
                r_block_idxs = self.cart_comm.Get_coords(r)
                r_sx, r_lx = self.span(NX, self.proc_layout[0], r_block_idxs[0])
                r_sy, r_ly = self.span(NY, self.proc_layout[1], r_block_idxs[1])
                r_sz, r_lz = self.span(NZ, self.proc_layout[2], r_block_idxs[2])

                # Receive into intermediate buffer
                buf = np.empty((r_lx, r_ly, r_lz), dtype=local_arr.dtype)
                self.cart_comm.Recv(buf, source=r, tag=99)  # blocking
                global_arr[r_sx:r_sx+r_lx,
                           r_sy:r_sy+r_ly,
                           r_sz:r_sz+r_lz] = buf

            return global_arr
        else:
            self.cart_comm.Send(local_arr, dest=0, tag=99)
            return None

    def mesh_extent(self):
        """
        Helper method for plots of global coordinate grid
        (e.g., for pyplot.imshow(..., extent=...)).
        """
        extent = np.array([-1, 1, -1, 1, -1, 1])
        lbnd = lambda arr: arr[0] - np.diff(arr)[0]/2
        ubnd = lambda arr: arr[-1] + np.diff(arr)[-1]/2
        if self.k_vec.size > 1:
            extent[0] = lbnd(self.k_vec_global)
            extent[1] = ubnd(self.k_vec_global)
        if self.omega_re_vec_global.size > 1:
            extent[2] = lbnd(self.omega_re_vec_global)
            extent[3] = ubnd(self.omega_re_vec_global)
        if self.omega_im_vec_global.size > 1:
            extent[4] = lbnd(self.omega_im_vec_global)
            extent[5] = ubnd(self.omega_im_vec_global)
        return extent

    def grid_roots(self, arr: np.ndarray):
        """
        Finds the global coordinate indices of local minima of the input array,
        meant to be gathered array of abs(D) on global (k, Re(ω), Im(ω)) grid.

        Helper method to trace approximate abs(D) roots in (k, Re(ω), Im(ω)).

        Input:
            arr = abs(D) to minimize
        Output:
            inds = (3,) tuple of indices into k, Re(ω), Im(ω) mesh vectors
        """
        assert arr.ndim == 3
        assert arr.shape[0] == self.k_vec_global.size
        assert arr.shape[1] == self.omega_re_vec_global.size
        assert arr.shape[2] == self.omega_im_vec_global.size
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
        Returns physical (k, Re(omega), Im(omega)) coordinates of local minima
        of the input (global-sized) array.  The returned values are approximate
        roots of the dispersion relation, which can be used as initial guesses
        for root-finding methods.

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
