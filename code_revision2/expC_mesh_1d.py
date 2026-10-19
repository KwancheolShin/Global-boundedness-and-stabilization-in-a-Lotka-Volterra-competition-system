"""Mesh refinement for the one-dimensional runs of Experiment C (Figure 3, Table 1).

Repeats sim.experiment_B with N = 200 and compares the total masses with the archived
N = 100 results in results/resB.json (same output times and tolerances).
Run from code_revision2:  python expC_mesh_1d.py
"""
import json
import time
from pathlib import Path
import numpy as np
import sim

RESULTS = Path(__file__).resolve().parent / "results"

if __name__ == "__main__":
    t0 = time.time()
    coarse = json.loads((RESULTS / "resB.json").read_text())
    fine = sim.experiment_B(N=200)
    out = {}
    for tag, rec in fine.items():
        a = coarse[tag]
        hist = max(float(np.max(np.abs(np.array(a["mass_" + s]) - np.array(rec["mass_" + s])))) for s in "uv")
        final = max(abs(a["mass_" + s][-1] - rec["mass_" + s][-1]) for s in "uv")
        out[tag] = dict(final_masses_N100=[a["mass_u"][-1], a["mass_v"][-1]],
                        final_masses_N200=[rec["mass_u"][-1], rec["mass_v"][-1]],
                        max_final_difference=final, max_history_difference=hist)
        print(tag, out[tag], flush=True)
    out["seconds"] = time.time() - t0
    (RESULTS / "expC_mesh_1d.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("done %.1f s" % out["seconds"])
