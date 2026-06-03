#!/usr/bin/env python

import numpy as np
import warnings

import dredge as dr
from dredge.const import (
    CLIGHT, ERG_PER_EV, Q_ELEMENTARY, GAUSS_PER_TESLA,
    M_ELECTRON, M_PROTON,
)

# How to setup a "dredge" calculation?
# You must initialize:
#
#   field = dr.field.FieldLine object (or a subclass thereof)
#   species = dr.species.Species object (or a subclass thereof)
#   grid = dr.wavegrid.WaveGrid object (or a subclass thereof)
#   calc = dr.chi.SlabESPerp or dr.chi.BounceAvgESPerp, one (or more) for
#          each species whose susceptibility contribution you wish to
#          evaluate
#
# Use the methods in "calc" object to compute susceptibility.  Then, sum them
# together to craft a dispersion relation, whose roots you can find with the
# help of methods in "grid" object.

def main():

    # ----------------------------------------------------------------
    # Define the magnetic geometry

    B0 = 0.26 * GAUSS_PER_TESLA  # WHAM-HTS Phase 1
    BT = 17 * GAUSS_PER_TESLA  # WHAM-HTS
    LP = 98  # WHAM-HTS throat position (cm)

    # TESTING: concentrate plasma at z=0 to recover slab limit,
    # choose tiny (ds, n_steps) so that no magnetic geometry variation along s
    # is seen by particles
    ns_resolution = 10  # number of sample points for bounce-average integral

    field = dr.field.FieldLineParabolic(
        B0      = B0,       # central magnetic field strength (Gauss)
        Bt      = BT,       # throat magnetic field strength (Gauss)
        Lp      = LP,       # z position (cm) of magnetic throat
        r0      = 0.233,    # start position (cm) for field line trace
        z0      = 0,        # start position (cm) for field line trace
        ds      = 1e-6,     # step size (cm) to trace field line
        n_steps = 20,       # number of steps to trace field line
        axis_r  = 1,  # cartesian coord to align with radial direction
        axis_z  = 2,  # cartesian coord to align with axial direction
    )

    # ----------------------------------------------------------------
    # Define the plasma parameters and species

    n0 = 1e13  # ion density assuming Z=1 in cm^-3

    # plasma temperature
    Ti_perp = 100 * ERG_PER_EV  # ion temperature
    Te_perp = 100 * ERG_PER_EV  # electron temperature
    Ti_aniso = 2    # ratio of Tperp/Tprll for bi-Maxwellian
    Te_aniso = 2

    # species mass, charge
    mi = M_PROTON  # ion mass
    me = M_ELECTRON  # electron mass
    qi = Q_ELEMENTARY  # ion charge
    qe = -Q_ELEMENTARY  # electron charge

    # inverse density scale length ~ (WHAM Phase 1 plasma radius)^-1 in cm^-1
    # etaN < 0 for coordinate system with dn/dy along y, k along x, B along z
    etaN = -1./15  # signed, cm^-1

    # numerical mesh for VDF calculation
    # shared by both ions and electrons
    vperp_vth_vec = np.linspace(0, 4, 100)
    vprll_vth_vec = np.linspace(-4, 4, 201)

    def make_aniso_df(vperp_vth_vec, vprll_vth_vec, vth, aniso=1, floor=1e-99):
        vperp, vprll = np.meshgrid(vperp_vth_vec * vth,
                                   vprll_vth_vec * vth, indexing='ij')
        df = dr.vdf.bimaxwellian(vperp, vprll, vth, vth/aniso**0.5)
        df[ df<floor ] = floor
        return df

    # thermal velocities with sqrt(2) factors
    vthi_perp = np.sqrt(2*Ti_perp/mi)
    vthe_perp = np.sqrt(2*Te_perp/me)

    ion = dr.species.KineticVDFGrid(
        mass = mi,
        charge = qi,
        vperp_vec = vperp_vth_vec * vthi_perp,
        vprll_vec = vprll_vth_vec * vthi_perp,
        df = make_aniso_df(
            vperp_vth_vec,
            vprll_vth_vec,
            vthi_perp,
            aniso = Ti_aniso,
        ),
    )

    lec = dr.species.KineticVDFGrid(
        mass = me,
        charge = qe,
        vperp_vec = vperp_vth_vec * vthe_perp,
        vprll_vec = vprll_vth_vec * vthe_perp,
        df = make_aniso_df(
            vperp_vth_vec,
            vprll_vth_vec,
            vthe_perp,
            aniso = Te_aniso,
        ),
    )

    del vthi_perp, vthe_perp

    # ----------------------------------------------------------------
    # Define the (k,\omega) mesh for susceptibility calculation

    # good for baseline interchange test
    #k_vec_global        = np.linspace(1e-5, 2, 121, dtype=np.float64)
    #omega_re_vec_global = np.linspace(1e-8, 0.05, 201, dtype=np.float64)
    #omega_im_vec_global = np.linspace(0, 0.04, 181, dtype=np.float64)

    # for quick testing with "loop" setup
    k_vec_global = np.array([0.08])
    omega_re_vec_global = np.linspace(1e-8, 0.008, 41, dtype=np.float64)
    omega_im_vec_global = np.linspace(0, 0.016, 41, dtype=np.float64)
    #omega_re_vec_global = np.linspace(1e-8, 0.008, 21, dtype=np.float64)  # EVEN QUICKER TEST
    #omega_im_vec_global = np.linspace(0, 0.016, 21, dtype=np.float64)

    # TODO move this logic into "WaveGrid"
    # Do you want to compute damped modes?
    # Comment or uncomment the relevant block of code as needed.
    # WARNING I don't treat the resonant integral rigorously,
    # so my damped mode calculation may not be correct...

    # unstable+normal modes only
    if not np.any(omega_im_vec_global < 0.):
        print('Get normal modes; add Im(omega)<0 point')
        omega_im_vec_global = np.insert(omega_im_vec_global, 0, -1*omega_im_vec_global[1])

    # all damped+unstable+normal modes
    #if not np.any(omega_im_vec_global < 0.):
    #    print('Get normal+damped modes; add Im(omega)<0 points to case={}'.format(label))
    #    case['omega_im_vec_global'] = np.concatenate((-1*omega_im_vec_global[:0:-1],
    #                                                  omega_im_vec_global))

    # ----------------------------------------------------------------
    # Prepare the calculation

    # convert user input to dimension-ful CGS units
    # and prepare MPI domain decomposition
    solve_grid = dr.wavegrid.WaveGrid( k_vec_global / ion.rLs(B0),
                                       omega_re_vec_global * ion.Omcs(B0),
                                       omega_im_vec_global * ion.Omcs(B0),
                                       proc_layout = (1,4,2), )

    calc_i = dr.chi.BounceAvgESPerp( grid = solve_grid,
                                     species = ion,
                                     field = field, )

    calc_e = dr.chi.BounceAvgESPerp( grid = solve_grid,
                                     species = lec,
                                     field = field, )

    calc_i.setup_bounce_average(NS_RESOLUTION=ns_resolution)
    calc_e.setup_bounce_average(NS_RESOLUTION=ns_resolution)

    # ----------------------------------------------------------------

    chi_i = calc_i.chi_GK(
        ns = n0,  # midplane value in cm^-3
        epsilonN = etaN,  # signed midplane value in cm^-1
        Gforce = 0,  # signed midplane value, in cm/s^2
        Teff_ceiling = 1e5 * ERG_PER_EV,  # 100 keV
        method = 'loop6d',
        enable_Upsilon = False,
    )

    chi_e = calc_e.chi_GK(
        ns = n0,
        epsilonN = etaN,
        Gforce = 0,
        Teff_ceiling = 1e5 * ERG_PER_EV,  # 100 keV
        method = 'expand',
    )

    # dispersion relation computed on 3D grid of (k,Re(omega),Im(omega))
    dd = 1. + chi_i + chi_e

    if solve_grid.world_size > 1:
        dd_loc = dd
        dd = solve_grid.gather(dd_loc)

    # ----------------------------------------------------------------

    if solve_grid.rank != 0:
        return

    # dispersion relation roots
    k_root, omega_re_root, omega_im_root, absd_root = solve_grid.roots(np.abs(dd))
    stable = omega_im_root == 0.
    unstab = omega_im_root > 0
    damped = omega_im_root < 0
    # probably better to define weak/strong damping w.r.t. Re(omega)...
    wkdamp = np.logical_and(damped, np.abs(omega_im_root) < 0.5*ion.Omcs(B0))  # weakly damped
    strdamp = np.logical_and(damped, np.abs(omega_im_root) >= 0.5*ion.Omcs(B0))  # strongly damped

    k_root *= ion.rLs(B0)  # convert to dimensionless coordinates
    omega_re_root /= ion.Omcs(B0)  # convert to dimensionless coordinates
    omega_im_root /= ion.Omcs(B0)  # convert to dimensionless coordinates

    print("found local minima (possible roots)")
    print("k = ", k_root[0])
    print("Re(omega) =", omega_re_root[0])
    print("Im(omega) =", omega_im_root[0])
    print("expected",  omega_re_vec_global[2]) # 0.0008
    print("expected",  omega_im_vec_global[10])  # 0.00072

    print("delta", omega_re_root[0] - omega_re_vec_global[2] )
    print("delta", omega_im_root[0] / omega_im_vec_global[2] )
    print("relative delta",
          np.abs( omega_re_root[0] - omega_re_vec_global[2] ) / omega_re_vec_global[2] )
    print("relative delta",
          np.abs( omega_im_root[0] - omega_im_vec_global[10] ) / omega_im_vec_global[10] )

    # ----------------------------------------------------------------

    #plt.imshow(
    #    np.abs(dd[0,:,:]).T, origin='lower',
    #    extent=(omega_re_vec[0], omega_re_vec[-1],
    #           omega_im_vec[0], omega_im_vec[-1]),
    #    norm=mpl.colors.LogNorm(),
    #    cmap='turbo',
    #)
    #plt.title(r'$|D|$')
    #plt.xlabel(r'Re($\omega$)')
    #plt.ylabel(r'Im($\omega$)')
    #plt.colorbar()
    #plt.savefig('interchange_test.png', dpi=300, bbox_inches='tight')
    #plt.clf()
    #plt.close()

    return k_root, omega_re_root, omega_im_root



if __name__ == "__main__":
    main()
