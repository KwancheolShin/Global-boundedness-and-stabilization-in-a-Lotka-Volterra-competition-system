Separate nonlocal competition and robustness experiments (v1.1.0)
================================================================

Purpose and model
-----------------
These additional one-dimensional experiments distinguish population-wide
self-competition s=a4=b4 from cross-competition q=a5=b5. They use the original
finite-volume equations in sim.make_rhs, on the unit interval with
zero flux, linear uptake, all local competition/conversion coefficients one,
zero baseline mortality, and resource growth coefficient one.

The resource carrying capacity is r(x)=1+eta*cos(pi*x). M2 uses
d1(w)=0.1 and d2(w)=exp(-kappa*w); M3 uses d1(w)=0.1*exp(-kappa*w)
and d2(w)=1. The original examples have eta=0.5, kappa=2.
Changing kappa changes both motility and its resource sensitivity; this is
not a test of independently adjustable diffusion and taxis coefficients.

Reproduction
------------
Use requirements.txt. The new calculations used Windows, Python 3.12.14,
NumPy 2.4.6, SciPy 1.18.0, and Matplotlib 3.11.0. The older saved experiments
used the Python version recorded in README_revision2.txt. Set thread limits
before Python starts, then run from code_revision2:

  $env:OPENBLAS_NUM_THREADS='1'
  $env:OMP_NUM_THREADS='1'
  $env:MKL_NUM_THREADS='1'
  python nonlocal_robustness.py all

Individual stages: sweep, trajectories, checks, plots. The checks stage
requires the saved sweep and trajectory outputs; plots reads those outputs.
The pilot stage prints a few baseline invasion values without changing data.
Commands overwrite their corresponding generated outputs. No simulation
uses external measurements or personal information.

Resident states and exact dependence on cross-competition
--------------------------------------------------------
For a resident R with resource W, the stationary equations contain s but not q:
  L(d_R(W)*R) + R*(W-R-s*mean(R)) = 0,
  L W + W*(r-R-W) = 0.
The finite-volume matrix for a rare invader is
  A(s,q)=L diag(d_A(W))+diag(W-R-q*mean(R)).
Thus lambda(s,q)=lambda(s,0)-q*mean(R), exactly for a fixed resident branch.
We solve both resident problems and both invasion directions. The reported
qcritical=lambda(s,0)/mean(R) gives the whole q-dependence without repeating
identical resident solves on a cross-competition grid. Negative qcritical
values are retained, rather than clipped to zero. Self-competition changes
the resident/resource profiles and is not a fixed-environment spectral shift.

Positive stationary states are obtained with scipy.optimize.root and an
analytic Jacobian, using continuation in s where stated. Their residuals and
minimum values are stored. The invasion matrix is similar to a symmetric
tridiagonal matrix; its largest eigenvalue and eigenvector defect are saved.
These stationary solves select a computed positive resident branch. They do
not establish uniqueness or classify all resident equilibria of the PDE.

Sampling and diagnostics
------------------------
At each N=100 and N=200, the Cartesian grid is:
  eta   = 0, 0.1, 0.25, 0.5, 0.75, 0.9
  kappa = 0, 1, 1.5, 2, 2.5, 3
  s     = 0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5
for both M2 and M3 and both invasion directions (504 parameter sets per mesh).
Additionally, the original eta=0.5, kappa=2 setting is continued at 51 equally
spaced s values in [0,0.5], on N=100,200,400, and the rare-species crossing
along q=s is located with a bracketed scalar solve. Selected heterogeneous
and motility settings also receive N=400 checks; see the JSON for every case.
The invasion sign plot displays sampled points, not interpolated boundaries.
A 2e-9 absolute band labels near-zero invasion rates neutral/unresolved.
Mesh differences are reported independently of that neutral band.

Time integrations use an algebraically equivalent face-flux evaluation of
the original sim.make_rhs equations, with an analytic Jacobian
and SciPy's variable-order BDF method. Initial consumers are constant,
u0=f and v0=1-f, with w0=r. The baseline cases include self-only (s=0.1,q=0),
cross-only (s=0,q=0.1), and common (s=q=0.1) competition. Cross-only runs use
f=0.1,0.5,0.9; the other baseline cases use f=0.5. Additional low/high
heterogeneity and sensitivity cases use (eta,kappa)=(0.1,1) and (0.9,3),
self-only f=0.5 and cross-only f=0.1,0.9. All are run on N=100 and N=200.
Baseline refinements use N=200 and N=400 with tighter time tolerances.
Common-coefficient runs use T=1e7; separated-coefficient runs use T=1e5.
At (eta,kappa)=(0.5,2.5) and (0.9,3), both M2 and M3 also receive
s=q=0.5 coexistence integrations on N=100 and N=200 (68 runs in total).
Face-flux differences reduce roundoff pollution of the spatial mean; a direct
comparison with the legacy stencil is included in the independent checks.
Every output records T, tolerances, masses over time, full final state,
residual, minimum saved value, independent stationary-solve distance and
the leading eigenvalue of the full semidiscrete stationary linearization.

Independent checks
------------------
- Uniform resource exact resident R=1/(2+s), W=(1+s)/(2+s), with principal
  invasion rate (s-q)/(2+s), on three meshes and multiple motility choices.
- Direct nonsymmetric eigenvalue versus the symmetric similarity calculation.
- Direct eigenvalue at q>0 versus the exact scalar spectral shift.
- Resident and full analytic Jacobians versus centered directional differences.
- Integrated original finite-volume RHS versus reaction-only mass balance.
- Mesh refinement, time-tolerance refinement, positivity/residual diagnostics.
- Reproduction of the previously archived baseline invasion crossings.

Files
-----
results/nonlocal_robustness.json  : parameter grid and threshold continuations
results/nonlocal_trajectories.json: integrations and stationary diagnostics
results/nonlocal_checks.json     : independent checks and refinement summary
../figures/rev3_fig9_separation.pdf  (and PNG preview)
../figures/rev3_fig10_robustness.pdf (and PNG preview)
../figures/rev3_fig11_trajectories.pdf (and PNG preview)

Interpretation limits
---------------------
Mutual invasibility is an invasion diagnostic, not a proof of stable or
globally attracting PDE coexistence. Mutual non-invasibility alone is not
a proof of bistability; selected different-initial-composition integrations
and semidiscrete stability checks provide additional numerical evidence.
Small negative values at the time-integrator tolerance are saved unchanged;
BDF is not positivity preserving. Reported numerical exclusion uses a stated
resolution, not an exact finite-time zero. A finite parameter grid does not
establish a universal monotonic effect of heterogeneity or motility.
The algebraic spectral shift is exact; the computed resident profiles,
thresholds, attraction and parameter dependence remain numerical findings.
