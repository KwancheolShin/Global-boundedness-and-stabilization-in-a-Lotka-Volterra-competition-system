# Numerical code and data for a Lotka–Volterra competition system

Reproducibility materials for **Global boundedness and stabilization in a Lotka–Volterra competition system with density-suppressed motility, dynamical resource and nonlocal competition**, by K. Kavitha, Gurusamy Arumugam and Kwancheol Shin.

This repository contains the numerical implementation, saved outputs and figure files associated with the manuscript revision of 19 October 2026. The manuscript is being prepared for submission; this repository does not assert journal acceptance or publication.

## Contents

- `code_revision2/`: complete numerical package, including the original experiment scripts and the additional revision experiments.
- `code_revision2/results/`: saved JSON outputs used for the numerical claims and plots.
- `code_revision2/README_revision2.txt`: experiment-by-experiment commands, mesh sizes, tolerances and interpretation limits.
- `code_revision2/README_robustness.txt` and `code_revision2/README_ideal_free.txt`: design of the v1.1.0 and v1.2.0 additions.
- `code_revision2/requirements.txt`: exact package versions used for the archived calculations.
- `figures/`: figure files. The manuscript uses eight of them: `rev1_fig1` to `rev1_fig5`, `rev2_fig6`, `rev2_fig8` and `rev4_fig12`. The v1.1.0 figures `rev2_fig7`, `rev3_fig9`, `rev3_fig10` and `rev3_fig11` are kept for reference but are no longer in the manuscript.
- `MANIFEST.sha256.json` and `verify_repository.py`: integrity and basic consistency checks.
- `CITATION.cff`: software citation metadata.

## Environment and installation

The original archived calculations and the v1.2.0 additions used Windows and Python 3.14.6. The v1.1.0 self/cross-competition and robustness experiments used Windows and Python 3.12.14. All use NumPy 2.4.6, SciPy 1.18.0 and Matplotlib 3.11.0. Several original scripts use Windows path separators; use Windows for the full original command sequence. The additional experiments resolve their paths with `pathlib`.

From a fresh clone, create a virtual environment and install the recorded requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r code_revision2/requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python verify_repository.py
```

Thread limits should be set **before** starting Python. Long integrations can be computationally expensive. Saved outputs allow the figures to be regenerated without repeating those integrations.

## Regenerate the manuscript figures

Run these commands from `code_revision2`:

```powershell
python figs2.py results ../figures pdf all
python figs3.py results ../figures pdf
python figs4.py results ../figures pdf
python additional_experiments.py plots
python nonlocal_robustness.py plots
python ideal_free_threshold.py plot
```

These commands overwrite the corresponding generated figures. The `rev1_` to `rev4_` prefixes identify the development round, not the order of figures in the final manuscript.

| Manuscript content | Computation | Saved output |
| --- | --- | --- |
| A: one-dimensional coexistence and mesh convergence | `sim.py A results/resA.json` | `resA.json` |
| A: two-dimensional coexistence | `run2dA.py results` | `res2dA.json` |
| A: early-time spatial convergence (1D and 2D) | `spatial_order_check.py` | `spatial_order.json` |
| B: negative and zero invasion-rate exclusion | `expC.py results` | `resC.json` |
| C: heterogeneous local/nonlocal competition | `sim.py B results/resB.json`, `checkB.py results` | `resB.json`, `checkB.json` |
| C: one-dimensional mesh refinement | `expC_mesh_1d.py` | `expC_mesh_1d.json` |
| C: two-dimensional long-time runs | `run2dB.py results`, `long2d.py results`, `drift2d.py results` | `bdf2d_*.json`, `long2d_*.json`, `drift2d.json` |
| C: rare-species coexistence diagnostics | `check_long_1d.py`, then `refine_coexistence.py` | `long1d_diagnostics.json`, `coexistence_refinement.json` |
| C: resident invasion continuation | `additional_experiments.py invasion` | `invasion_continuation.json` |
| C: near-ideal-free invasion threshold | `ideal_free_threshold.py compute` | `ideal_free_threshold.json` |
| C: separated self/cross competition, heterogeneity and motility | `nonlocal_robustness.py all` | `nonlocal_robustness.json`, `nonlocal_trajectories.json`, `nonlocal_checks.json` |
| D: finite-threshold growth | `expE.py results` | `resE.json` |
| E: Holling type II response | `additional_experiments.py holling` | `hollingII.json` |

Prefix each computation with `python`. Some checks depend on previously saved results; the supplied results satisfy these dependencies. See the detailed README before running the full sequence.

## Additional computations in v1.2.0

See `code_revision2/README_ideal_free.txt`. For a uniform resource the resident density is `1/(2+s)`, and on the common-coefficient path `s=q` every invasion rate vanishes. If the resident with motility `exp(-kappa*w)` had constant fitness and constant `d(W)*R`, it would be an ideal free distribution; to first order in the resource amplitude `eta` both conditions hold when `kappa = 2+s`. `ideal_free_threshold.py` locates the value `kappa_*(eta,s)` at which the rare species' invasion rate changes sign. In all computed cases `kappa_* < 2+s` and `(2+s-kappa_*)/eta^2` lies between 0.079 and 0.100. The original parameters `eta=0.5`, `kappa=2`, `s=q=0` lie about 0.02 above the threshold, and the common-coefficient crossing near `c=0.021` is the value of `c` with `kappa_*(0.5,c)=2`. All 350 points of the v1.1.0 grid with `kappa>=1` and `eta>0` agree with `kappa_*`. In the manuscript these parameters are written `zeta` (for `eta`), `k` (for `kappa`), `sigma` (for `s`) and `tau` (for `q`).

`spatial_order_check.py` repeats the mesh-convergence test of Experiment A at `t=0.1`, before the non-uniform modes decay, and `expC_mesh_1d.py` refines the one-dimensional runs of Experiment C from N=100 to N=200.

## Additional experiments in v1.1.0

See `code_revision2/README_robustness.txt` for the complete design. These experiments separate self-competition from cross-competition, compute both invasion directions, and cover 504 resource/motility/self-competition combinations per mesh on N=100 and 200, with selected N=400 refinements. Sixty-eight integrations test convergence, initial composition, mesh size and time tolerances.

For a fixed resident branch, cross-competition produces the exact shift `lambda(s,q) = lambda(s,0) - q * resident_mass(s)`. Selected self-only runs approach coexistence; cross-only runs select different stable semidiscrete resident states from different initial compositions. The dependence of the common-coefficient outcome on the motility sensitivity is explained by the v1.2.0 threshold computation above.

`python nonlocal_robustness.py checks` reruns independent algebraic checks and compares the archived results across meshes and tolerances. It does not repeat every integration. `python nonlocal_robustness.py all` recomputes these experiments and their figures.

## Numerical interpretation

The two-dimensional solver named `sim2d_bdf2.py` uses SciPy's variable-order BDF method; its filename does not mean fixed BDF2. The legacy `T_blowup` field records first hitting of a numerical cap of 1e6, not a proved PDE blow-up time. BDF does not preserve positivity, and very small masses must be interpreted using the resolution limits stated in the manuscript.

The heterogeneous coexistence states and invasion thresholds are numerical findings for the stated parameter sets. The ideal-free condition is a first-order statement in the resource amplitude. A repository integrity check does not validate the analytical proofs.

## Citation and contact

Cite the repository using `CITATION.cff`, and include the release tag `v1.2.0` or an exact commit identifier for reproducibility. No archival DOI has been assigned. The GitHub source snapshot is the versioned record provided here.

Corresponding author: Kwancheol Shin, Korea University Sejong Campus, kcshin3623@korea.ac.kr.

Claude/Claude Code (Anthropic) and Codex (OpenAI) assisted with code development, checks and documentation. Scientific responsibility remains with the authors, as described in the manuscript.
