Historical instructions for revision 1. See README_revision2.txt for the current
manuscript, new experiments, numerical caveats and reproduction instructions.

Code for the numerical experiments of main_revision1.tex (Section 6)
===================================================================

Requirements: Python 3 with numpy, scipy and matplotlib.
<dir> is the folder in which the JSON results are written and from which the
figure scripts read them (the scripts use Windows-style paths).

One dimension  (sim.py: finite-volume method of lines, SciPy BDF)
  python sim.py A <dir>\resA.json     Experiment A (n = 1): convergence and observed order (Fig. 1a)
  python expC.py <dir>                Experiment B: competitive exclusion, Gamma_v < 0 and = 0 (Fig. 2) -> resC.json
  python sim.py B <dir>\resB.json     Experiment C (n = 1): heterogeneous resource, N = 100 (Fig. 3, Table 1)
  python checkB.py <dir>              Experiment C (n = 1): N = 200 repetition and exclusion times -> checkB.json
  python expE.py <dir>                Experiment D: finite-time blow-up, (a2,b3) = (2,3) for N = 50..400 and the
                                      diagonal a2 = b3 = 2.8, 2.9, 3.1, 3.2 (Fig. 5) -> resE.json

Two dimensions  (sim2d_bdf2.py: five-point finite volumes, BDF with exact sparse Jacobian)
  python run2dA.py <dir>              Experiment A (n = 2): 48 x 48, rtol 1e-11 (Fig. 1b) -> res2dA.json
  python run2dB.py <dir>              Experiment C (n = 2): checks up to t = 2e5 (32 x 32, 48 x 48, rtol 1e-8, x-only) -> bdf2d_*.json
  python long2d.py <dir>              Experiment C (n = 2): runs up to t = 1e7 (Fig. 4) -> long2d_*.json
  python drift2d.py <dir>             Experiment C (n = 2): drift of the auxiliary masses and the exclusion times of Table 1 -> drift2d.json
  sim2d.py                            Laplacian and parameter sets only (its own first-order scheme is not used)

Figures
  python figs2.py <dir> <outdir> pdf all    Figures 1-3 (rev1_fig1_convergence, rev1_fig2_exclusion, rev1_fig3_masses1d)
  python figs3.py <dir> <outdir> pdf        Figure 4 (rev1_fig4_masses2d)
  python figs4.py <dir> <outdir> pdf        Figure 5 (rev1_fig5_blowup)
  figs.py                                   figures of the first draft of the revision (superseded by figs2.py)

Exclusion times are the first times at which min(mass_u, mass_v) drops below the
threshold, interpolated log-log between output times.
Blow-up times (Experiment D) are the first times at which max(||u||_inf, ||v||_inf)
reaches 1e6, located by a terminal event of solve_ivp.
