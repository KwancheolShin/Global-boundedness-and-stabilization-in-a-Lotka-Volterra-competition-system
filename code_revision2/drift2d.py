"""Re-run the 2D heterogeneous cases and record the drift of both auxiliary masses P, Q
and the threshold-crossing times (log-log interpolation).  Usage: drift2d.py <dir>"""
import json, sys
import numpy as np
R = sys.argv[1]
sys.path.insert(0, R)
import sim2d_bdf2 as b
from sim2d import base_B
MOT = {"M1": (lambda w: 0.1 + 0 * w, lambda w: 0 * w, lambda w: 1.0 + 0 * w, lambda w: 0 * w),
       "M2": (lambda w: 0.1 + 0 * w, lambda w: 0 * w, lambda w: np.exp(-2.0 * w), lambda w: -2.0 * np.exp(-2.0 * w))}
def setup(mode, key):
    p = base_B(mode == "nonlocal"); p["d1"], p["dd1"], p["d2"], p["dd2"] = MOT[key]; return p
def cross(t, m, th):
    i = np.where(m < th)[0]
    if len(i) == 0: return None
    i = i[0]
    if i == 0 or m[i - 1] <= 0: return float(t[i])
    a = (np.log(th) - np.log(m[i - 1])) / (np.log(m[i]) - np.log(m[i - 1]))
    return float(np.exp(np.log(t[i - 1]) + a * (np.log(t[i]) - np.log(t[i - 1]))))
res = {}
for T, npts in ((2.0e5, 300), (1.0e7, 400)):
    te = np.concatenate(([0.0], np.logspace(-1, np.log10(T), npts)[:-1], [T]))
    for mode in ("local", "nonlocal"):
        for key in ("M1", "M2"):
            N = 32; M = N * N; h = 1.0 / N
            x, y, sol = b.run(N, setup(mode, key), T, te, rtol=1e-6, atol=1e-10 if T < 1e6 else 1e-12)
            mu = h * h * sol.y[:M].sum(axis=0); mv = h * h * sol.y[M:2 * M].sum(axis=0)
            dP = float(np.abs(sol.y[3 * M] - mu).max()); dQ = float(np.abs(sol.y[3 * M + 1] - mv).max())
            mi = np.minimum(mu, mv)
            r = {"T": T, "drift_P": dP, "drift_Q": dQ, "final": [float(mu[-1]), float(mv[-1])],
                 "minU": float(sol.y[:M].min()), "minV": float(sol.y[M:2 * M].min()),
                 "cross_1e-2": cross(te, mi, 1e-2), "cross_1e-3": cross(te, mi, 1e-3)}
            res[f"{mode}_{key}_T{T:.0e}"] = r
            print(f"{mode:8s} {key} T={T:.0e} drift P {dP:.2e} Q {dQ:.2e} final ({mu[-1]:.6f}, {mv[-1]:.3e}) "
                  f"cross1e-2 {r['cross_1e-2']} cross1e-3 {r['cross_1e-3']} minU {r['minU']:.1e} minV {r['minV']:.1e}", flush=True)
json.dump(res, open(R + r"\drift2d.json", "w"), indent=1)
