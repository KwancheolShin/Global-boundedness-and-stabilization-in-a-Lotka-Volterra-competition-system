# Numerical code and data for a Lotka–Volterra competition system

Reproducibility materials for **Global boundedness and stabilization in a Lotka–Volterra competition system with density-suppressed motility, dynamical resource and nonlocal competition**, by K. Kavitha, Gurusamy Arumugam and Kwancheol Shin.

This repository contains the numerical implementation, saved outputs and eight figure files associated with the manuscript revision of 18 October 2026. The manuscript is being prepared for submission; this repository does not assert journal acceptance or publication.

## Contents

- `code_revision2/`: complete numerical package, including the original experiment scripts and the additional revision experiments.
- `code_revision2/results/`: saved JSON outputs used for the numerical claims and plots.
- `code_revision2/README_revision2.txt`: experiment-by-experiment commands, mesh sizes, tolerances and interpretation limits.
- `code_revision2/requirements.txt`: exact package versions used for the archived calculations.
- `figures/`: the eight PDF figure files used in the manuscript.
- `MANIFEST.sha256.json` and `verify_repository.py`: integrity and basic consistency checks.
- `CITATION.cff`: software citation metadata.

## Environment and installation

The archived calculations used Windows, Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0 and Matplotlib 3.11.0. Several original scripts use Windows path separators; use Windows for the full original command sequence. The additional experiments resolve their paths with `pathlib`.

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
```

These commands overwrite the corresponding generated figures. The `rev1_` and `rev2_` prefixes identify the development round, not the order of figures in the final manuscript.

| Manuscript content | Computation | Saved output |
| --- | --- | --- |
| A: one-dimensional coexistence and mesh convergence | `sim.py A results/resA.json` | `resA.json` |
| A: two-dimensional coexistence | `run2dA.py results` | `res2dA.json` |
| B: negative and zero invasion-rate exclusion | `expC.py results` | `resC.json` |
| C: heterogeneous local/nonlocal competition | `sim.py B results/resB.json`, `checkB.py results` | `resB.json`, `checkB.json` |
| C: two-dimensional long-time runs | `run2dB.py results`, `long2d.py results`, `drift2d.py results` | `bdf2d_*.json`, `long2d_*.json`, `drift2d.json` |
| C: rare-species coexistence diagnostics | `check_long_1d.py`, then `refine_coexistence.py` | `long1d_diagnostics.json`, `coexistence_refinement.json` |
| C: resident invasion continuation | `additional_experiments.py invasion` | `invasion_continuation.json` |
| D: finite-threshold growth | `expE.py results` | `resE.json` |
| E: Holling type II response | `additional_experiments.py holling` | `hollingII.json` |

Prefix each computation with `python`. Some checks depend on previously saved results; the supplied results satisfy these dependencies. See the detailed README before running the full sequence.

## Numerical interpretation

The two-dimensional solver named `sim2d_bdf2.py` uses SciPy's variable-order BDF method; its filename does not mean fixed BDF2. The legacy `T_blowup` field records first hitting of a numerical cap of 1e6, not a proved PDE blow-up time. BDF does not preserve positivity, and very small masses must be interpreted using the resolution limits stated in the manuscript.

The heterogeneous coexistence and invasion thresholds are numerical findings for the stated parameter sets. They do not establish a continuous-PDE bifurcation theorem or global attraction. A repository integrity check does not validate the analytical proofs.

## Citation and contact

Cite the repository using `CITATION.cff`, and include the release tag `v1.0.1` or an exact commit identifier for reproducibility. No archival DOI has been assigned. The GitHub source snapshot is the versioned record provided here.

Corresponding author: Kwancheol Shin, Korea University Sejong Campus, kcshin3623@korea.ac.kr.

Claude/Claude Code (Anthropic) and Codex (OpenAI) assisted with code development, checks and documentation. Scientific responsibility remains with the authors, as described in the manuscript.
