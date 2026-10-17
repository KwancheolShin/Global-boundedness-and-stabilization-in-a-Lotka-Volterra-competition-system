import json, sys, time
import numpy as np
R = sys.argv[1]
sys.path.insert(0, R)
import run2dB as rb
import sim2d_bdf2 as b
T = 1.0e7
te = np.concatenate(([0.0], np.logspace(-1, np.log10(T), 400)[:-1], [T]))
for mode in ("local", "nonlocal"):
    for key in ("M1", "M2"):
        p = rb.setup(mode, key)
        x, y, sol = b.run(32, p, T, te, rtol=1e-6, atol=1e-12)
        M = 32 * 32; h = 1 / 32
        mu = h * h * sol.y[:M].sum(axis=0); mv = h * h * sol.y[M:2 * M].sum(axis=0)
        mi = np.minimum(mu, mv)
        first = lambda th: (float(te[np.where(mi < th)[0][0]]) if (mi < th).any() else None)
        out = {"t": te.tolist(), "mass_u": mu.tolist(), "mass_v": mv.tolist(), "u": sol.y[:M, -1].tolist(),
               "v": sol.y[M:2 * M, -1].tolist(), "w": sol.y[2 * M:3 * M, -1].tolist(), "N": 32,
               "t_excl_1e-3": first(1e-3), "t_excl_1e-2": first(1e-2)}
        json.dump(out, open(f"{R}\\long2d_{mode}_{key}.json", "w"))
        print(mode, key, "final (%.6f, %.3e)" % (mu[-1], mv[-1]) if mu[-1] > mv[-1] else "final (%.3e, %.6f)" % (mu[-1], mv[-1]),
              "t(min<1e-2)=", out["t_excl_1e-2"], "t(min<1e-3)=", out["t_excl_1e-3"], flush=True)
