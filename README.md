README
======

"dredge" is a linear dispersion solver for perpendicular electrostatic waves in
plasma with spatial gradient in the local slab approximation (Stix Chapter 14).

This code can compute and has been used to study:
* DCLC in kparallel=0 limit with Maxwellians
* DCLC in kparallel=0 limit with any distribution
* DCLC in kparallel=0 limit with multiple species (should work but not tested
* DCLC weak kparallel!=0 limit with electron Landau damping
* All sub-cases of DCLC in kparallel=0 limit:
  + Perp ES in homogeneous plasma (Harris dispersion, Bernsteins, UH mode, ...)
  + Collisionless drift wave

This code is similar to NHDS/ALPS/etc, and in many respects simpler and cruder,
but it also has some useful features:
* Spatial gradient terms
* Python interface with efficient numpy broadcasting
* Modular structure in susceptibility chi to enable fast parameter sweeps
  + Compute D on 3D (k,omega) mesh to get coarse, inexact solution instead of
    tracing roots.
  + Cache Bessel integrals + sums to allow on demand re-calculation of
    dispersion when varying (omega, epsilon) and varying "chi" pieces (e.g.,
    fix ion chi but sweep fluid electron Te).


Usage
-----

The user interface is NOT stable and continuously evolving, no guarantees of
backwards compatibility.

See `example/dclc.ipynb` for a calculation of DCLC slab dispersion, for a
subtracted Maxwellian plasma.


Developer notes
---------------

Compute bottleneck is arithmetic with big (kperp, Re(omega), Im(omega)) arrays.
Bessels are cheap because they're only computed on (kperp, vperp) grid, then
integrals are cached as 1D (kperp,) vectors.

Short-term possible improvements:
* Root tracing / refinement of compute on grid (in work)
* Dispersion branch segmentation/topology for more user-friendly plotting
* DCLC cotangent(...) approx just to compare
* DCLC in kparallel!=0 limit with EM effects
* DCLC with radial geometry effects
* DCLC with axial geometry effects
* HFCLC in kparallel!=0 limit
* Dory-Guest-Harris (should work already, but not tested/studied)

Long-term/stretch goals:
* Dispersion for any k angle with gradient (maybe faster to fork ALPS instead)
* Hua-Sheng Xie's matrix solve method
* Parallelize (multiprocessing, pandas/dask, MPI, something else?)

Philosophy: human time is costly, computer time and memory is cheap, so use
simple, easy-to-debug, brute force methods whenever possible (except in
time-sensitive pieces of code).

Thanks to:
* Xinyi Guo for a parallel EM dispersion solver, pieces of which were spun off
  over the years and informed the design of this code
* Kris Klein for some advice on writing dispersion solvers
* Many others for discussion of DCLC physics
* NASA, NSF, DOE, and Realta Fusion for support
* NASA/HECC Pleiades, DOE/NERSC Perlmutter for compute
