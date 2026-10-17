import json, sys, time
import numpy as np
R = sys.argv[1]
sys.path.insert(0, R)
import sim2d_bdf2 as b
from sim2d import base_B

T = 2.0e5
te = np.concatenate(([0.0], np.logspace(-1, np.log10(T), 300)[:-1], [T]))
MOT = {"M1": (lambda w: 0.1 + 0 * w, lambda w: 0 * w, lambda w: 1.0 + 0 * w, lambda w: 0 * w),
       "M2": (lambda w: 0.1 + 0 * w, lambda w: 0 * w, lambda w: np.exp(-2.0 * w), lambda w: -2.0 * np.exp(-2.0 * w))}


def setup(mode, key, xonly=False):
    p = base_B(mode == "nonlocal")
    p["d1"], p["dd1"], p["d2"], p["dd2"] = MOT[key]
    if xonly:
        p["r"] = lambda x, y: 1.0 + 0.5 * np.cos(np.pi * x)
        p["w0"] = lambda x, y: 1.0 + 0.5 * np.cos(np.pi * x)
    return p


def go(tag, N, p, rtol):
    t0 = time.time()
    x, y, sol = b.run(N, p, T, te, rtol=rtol, atol=1e-10)
    M = N * N; h = 1.0 / N
    U, V, W = sol.y[:M], sol.y[M:2 * M], sol.y[2 * M:3 * M]
    drift = float(np.abs(sol.y[3 * M] - h * h * U.sum(axis=0)).max())
    out = {"tag": tag, "N": N, "rtol": rtol, "t": te.tolist(), "mass_u": (h * h * U.sum(axis=0)).tolist(),
           "mass_v": (h * h * V.sum(axis=0)).tolist(), "u": U[:, -1].tolist(), "v": V[:, -1].tolist(),
           "w": W[:, -1].tolist(), "min_uvw": [float(U.min()), float(V.min()), float(W.min())],
           "ydev": float(max(np.abs(Z.reshape(N, N) - Z.reshape(N, N).mean(axis=0, keepdims=True)).max()
                             for Z in (U[:, -1], V[:, -1], W[:, -1]))),
           "mass_drift": drift, "nfev": int(sol.nfev), "njev": int(sol.njev), "_seconds": time.time() - t0}
    json.dump(out, open(f"{R}\\bdf2d_{tag}.json", "w"))
    print(tag, "N=%d rtol=%g  final masses (%.6f, %.6f)  min %s  drift %.1e  %.1fs" % (
        N, rtol, out["mass_u"][-1], out["mass_v"][-1], ["%.1e" % m for m in out["min_uvw"]], drift, out["_seconds"]), flush=True)


if __name__ == "__main__":
    for mode in ("local", "nonlocal"):
        for key in ("M1", "M2"):
            go(f"{mode}_{key}", 32, setup(mode, key), 1e-6)
    for mode in ("local", "nonlocal"):
        go(f"{mode}_M2_N48", 48, setup(mode, "M2"), 1e-6)
        go(f"{mode}_M2_rtol1e-8", 32, setup(mode, "M2"), 1e-8)
    for mode in ("local", "nonlocal"):
        go(f"{mode}_M2_xonly", 32, setup(mode, "M2", xonly=True), 1e-6)
    print("all done")
