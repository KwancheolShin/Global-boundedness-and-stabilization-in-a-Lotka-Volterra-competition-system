Numerical package accompanying main_revision1.tex, revised 2026-10-18
===================================================================

Run commands from this code_revision2 directory. The results/ directory contains
the numerical output used for the revised manuscript. The figures/ directory is
one level above this directory. Original solver filenames are retained, although
their historical experiment letters do not always match the manuscript.

Environment used: Windows, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0,
Matplotlib 3.11.0. Exact package versions are in requirements.txt. The original
scripts use Windows path separators. The added scripts use pathlib.

For reproducible performance, limit BLAS threads before starting Python.
PowerShell:
  $env:OPENBLAS_NUM_THREADS='1'
  $env:OMP_NUM_THREADS='1'
  $env:MKL_NUM_THREADS='1'

Manuscript experiment / script mapping
------------------------------------
A, 1D coexistence and spatial order:
  python sim.py A results\resA.json
A, 2D coexistence:
  python run2dA.py results
B, exclusion at negative and zero invasion rate:
  python expC.py results
C, original 1D heterogeneous runs and mesh refinement:
  python sim.py B results\resB.json
  python checkB.py results
C, original 2D runs, tolerance checks and auxiliary-mass drift:
  python run2dB.py results
  python long2d.py results
  python drift2d.py results
D, finite-threshold growth and bounded diagonal cases:
  python expE.py results
E, Holling type II uptake:
  python additional_experiments.py holling

Additional heterogeneous diagnostics
------------------------------------
1. Integrate local M2/M3 to T=1e7 on N=100 and N=200, and compute the leading
   eigenvalues of the full semidiscrete linearization:
     python check_long_1d.py
   Output: results/long1d_diagnostics.json, including full final states.
2. Solve stationary equations independently on N=200 and N=400, initialized
   from the N=100 output of step 1:
     python refine_coexistence.py
   Output: results/coexistence_refinement.json.
3. Resident-only invasion eigenvalues at c=0 and 48 x 48 checks to T=1e7:
     python extra_numerical_checks.py
   Output: results/extra_checks.json. The long 2D mesh checks use long2d_*.json
   from the original 32 x 32 integrations as references.
4. Continue resident-only states over the common nonlocal coefficient c in
   [0,0.5], evaluate principal invasion eigenvalues, and bracket sign changes:
     python additional_experiments.py invasion
   Output: results/invasion_continuation.json.

The added one-dimensional solver supports p['F1'] and p['F2']; both default to
linear uptake, so the original experiments retain their original equations.
The Holling II experiment supplies F1(w)=F2(w)=w/(1+0.2w).

Figures
-------
Original figure filenames remain unchanged; their order in the revised PDF is
controlled by LaTeX, not by the numbers in the filenames.
  python figs2.py results ..\figures pdf all
  python figs3.py results ..\figures pdf
  python figs4.py results ..\figures pdf
New figures from the saved diagnostics:
  python additional_experiments.py plots
This writes rev2_fig6_coexistence, rev2_fig7_invasion and rev2_fig8_hollingII
as both PDF and PNG. The three computations/plots can also be requested with:
  python additional_experiments.py all
This command expects the supplied long1d_diagnostics and coexistence_refinement
files; it does not recompute those long runs.

Numerical conventions and interpretation
---------------------------------------
* sim.py uses conservative finite volumes and variable-order SciPy BDF.
* sim2d_bdf2.py also uses SciPy's variable-order BDF, despite its historical
  filename. Its exact sparse Jacobian includes auxiliary total-mass variables.
  sim2d.py supplies parameter sets and spatial discretization; its older
  first-order time-stepping scheme is not used for the reported 2D results.
* Exclusion times in the table come from checkB.py and drift2d.py, using log-log
  interpolation between output times at a smaller-species mass of 1e-3.
* The legacy JSON field T_blowup denotes first hitting of CAP=1e6. It is a
  finite-threshold event time, not a rigorous PDE blow-up time.
* BDF does not preserve positivity. Values below the reporting resolution and
  tiny negative roundoff/tolerance values are not biologically resolved data.
* The rare positive masses around 0.005 in local M2/M3 are much larger than
  solver error. Time integration, stationary solves, grid refinement and
  eigenvalue calculations support coexistence, but do not constitute a proof
  for the continuous PDE or a global-attraction/bifurcation theorem.
* Crossing locations of the invasion eigenvalue describe local invasibility
  of a resident-only state; other attractors are not ruled out by that check.

Tolerances
----------
Original A (1D): rtol=1e-12, atol=1e-15 for the convergence curve; see sim.py
for the separate spatial-convergence runs.
Original C (1D): rtol=1e-7, atol=1e-10.
Extended C (1D): N=100 uses rtol=1e-10, atol=1e-13;
                  N=200 uses rtol=1e-8, atol=1e-11.
Long C (2D): rtol=1e-6, atol=1e-12.
E (Holling II): rtol=1e-10, atol=1e-13 on both meshes.
Every new diagnostic JSON records its numerical residual or tolerances.
Large computation times are possible if the BLAS thread limits are omitted.

manifest.sha256.json records hashes of this package's files except itself.
