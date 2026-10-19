Near-ideal-free invasion threshold and additional mesh checks (v1.2.0)
=====================================================================

Notation
--------
The scripts keep the variable names of nonlocal_robustness.py. In the manuscript:
  eta   -> zeta   (resource amplitude, r(x) = 1 + zeta*cos(pi*x))
  kappa -> k      (suppression strength, responsive motility prefactor*exp(-k*w))
  s     -> sigma  (nonlocal self-competition a4 = b4)
  q     -> tau    (nonlocal cross-competition a5 = b5)
M2: d1 = 0.1, d2 = exp(-k*w) (responsive resident v, rare invader u).
M3: d1 = 0.1*exp(-k*w), d2 = 1 (responsive resident u, rare invader v).

Purpose
-------
For the uniform resource (zeta = 0) the resident density is Rbar = 1/(2+sigma), and
the invasion rate on the common-coefficient path sigma = tau is zero for every motility.
If the responsive resident had constant fitness W - R - sigma*mean(R) and constant
d(W)*R, it would be an ideal free distribution and every rare invader would have
invasion rate zero. Linearizing in zeta, both conditions hold when k*Rbar = 1,
i.e. k = 2 + sigma. ideal_free_threshold.py locates the value k_*(zeta, sigma) in
[1.5, 3.2] at which the rare-species invasion rate on sigma = tau changes sign and
compares it with 2 + sigma. In M2 the rate has a further sign change below k = 1,
where the slow constant-motility species u invades as in M1; it is not bracketed.

Reproduction
------------
Set BLAS threads to one before Python starts and run from code_revision2:
  python ideal_free_threshold.py compute   (about 10 minutes)
  python ideal_free_threshold.py plot
  python spatial_order_check.py            (about 1 minute)
  python expC_mesh_1d.py                   (about 20 seconds)
The calculations used Windows, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0 and
Matplotlib 3.11.0 (Anaconda). Resident states and invasion eigenvalues are computed
with the functions of nonlocal_robustness.py (N cells, analytic Jacobian,
symmetric tridiagonal eigenvalue problem).

Outputs
-------
results/ideal_free_threshold.json
  profiles        rare-species and reverse invasion rates for k in [1.4, 3.0],
                  zeta = 0.5, sigma = tau in {0, 0.5}, N = 200
  threshold_eta   k_* for zeta in {0.05, 0.1, 0.2, ..., 0.9}, sigma in {0, 0.1, 0.25, 0.5}
  threshold_self  k_* at zeta = 0.5 for 26 values of sigma in [0, 0.5]
  threshold_mesh  k_* on N = 100, 200, 400 at zeta in {0.1, 0.5, 0.9}, sigma in {0, 0.5}
  near_baseline   rates for k in {1.9, 1.95, 2, 2.05, 2.1}, zeta = 0.5, sigma = 0
  crossings       common coefficient c_* with k_*(0.5, c_*) = k for several k
  fitness_range   spatial range of W - R in the responsive resident state
  grid_consistency comparison of k_* with the invasion signs archived in
                  results/nonlocal_robustness.json (N = 200, k >= 1, zeta > 0)
results/spatial_order.json   early-time mesh convergence for Experiment A (1D and 2D)
results/expC_mesh_1d.json    N = 100 versus N = 200 for the 1D runs of Experiment C
../figures/rev4_fig12_threshold.pdf (and PNG preview)

Interpretation limits
---------------------
The ideal-free condition k*Rbar = 1 is a first-order statement in zeta. The thresholds
k_* are computed for the finite-volume equations on the stated grids; they are not
proofs for the PDE, and they do not classify all attractors.
