"""2D linearly implicit finite-volume solver on (0,1)^2 with homogeneous Neumann conditions.

u_t = Lap(d1(w) u) + u k_u,  k_u = a1 w - a2 (th1 + u) - a3 v - a4 I(u) - a5 I(v)
v_t = Lap(d2(w) v) + v k_v,  k_v = b1 w - b2 u - b3 (th2 + v) - b4 I(v) - b5 I(u)
w_t = Lap(w)       + w k_w,  k_w = -u - v + mu (r - w)

Time step:  (I - dt L D(w^n) + dt diag(k^-)) z^{n+1} = (1 + dt k^+) z^n  for z = u, v,
            (I - dt L + dt diag(k_w^-)) w^{n+1} = (1 + dt k_w^+) w^n,
with k^{+/-} the positive/negative parts of k evaluated at time n.  The matrices are M-matrices
(column-diagonally dominant), so the scheme preserves nonnegativity and the diffusion is conservative.

NOTE: this first-order scheme is NOT used for the results in the paper (it is inaccurate for large dt
in the long runs).  All reported 2D results use sim2d_bdf2.py; this file only provides laplacian(N)
and the parameter sets A (Experiment A) and base_B (Experiment C, n = 2).
"""
import json, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def laplacian(N):
    h = 1.0 / N
    e = np.ones(N)
    T = sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1], format="lil")
    T[0, 0] = -1.0; T[N - 1, N - 1] = -1.0          # zero-flux (reflecting) ends
    T = T.tocsr() / h**2
    I = sp.identity(N, format="csr")
    return (sp.kron(I, T) + sp.kron(T, I)).tocsr()


def run2d(N, p, schedule, rec_times, rec_every=None):
    h = 1.0 / N
    xc = (np.arange(N) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="xy")
    x, y = X.ravel(), Y.ravel()
    L = laplacian(N)
    Id = sp.identity(N * N, format="csr")
    r = p["r"](x, y)
    u, v, w = p["u0"](x, y), p["v0"](x, y), p["w0"](x, y)
    t = 0.0
    rec = {"t": [], "mass_u": [], "mass_v": [], "err": []}
    rec_times = sorted(rec_times); ri = 0

    def record():
        rec["t"].append(t); rec["mass_u"].append(h * h * u.sum()); rec["mass_v"].append(h * h * v.sum())
        if "steady" in p:
            us, vs, ws = p["steady"]
            rec["err"].append([float(np.abs(u - us).max()), float(np.abs(v - vs).max()), float(np.abs(w - ws).max())])

    record()
    for t_end, dt in schedule:
        while t < t_end - 1e-12:
            dt_ = min(dt, t_end - t)
            Iu, Iv = h * h * u.sum(), h * h * v.sum()
            ku = p["a1"] * w - p["a2"] * (p["th1"] + u) - p["a3"] * v - p["a4"] * Iu - p["a5"] * Iv
            kv = p["b1"] * w - p["b2"] * u - p["b3"] * (p["th2"] + v) - p["b4"] * Iv - p["b5"] * Iu
            kw = -u - v + p["mu"] * (r - w)
            Au = Id - dt_ * (L @ sp.diags(p["d1"](w))) + dt_ * sp.diags(np.maximum(-ku, 0))
            Av = Id - dt_ * (L @ sp.diags(p["d2"](w))) + dt_ * sp.diags(np.maximum(-kv, 0))
            Aw = Id - dt_ * L + dt_ * sp.diags(np.maximum(-kw, 0))
            u_new = spsolve(Au.tocsc(), u * (1 + dt_ * np.maximum(ku, 0)))
            v_new = spsolve(Av.tocsc(), v * (1 + dt_ * np.maximum(kv, 0)))
            w_new = spsolve(Aw.tocsc(), w * (1 + dt_ * np.maximum(kw, 0)))
            u, v, w = u_new, v_new, w_new
            t += dt_
            while ri < len(rec_times) and t >= rec_times[ri] - 1e-9:
                record(); ri += 1
    return {"N": N, **rec, "x": x.tolist(), "y": y.tolist(),
            "u": u.tolist(), "v": v.tolist(), "w": w.tolist(),
            "min_uvw": [float(u.min()), float(v.min()), float(w.min())]}


A = dict(a1=2.0, a2=2.0, a3=0.5, a4=0.2, a5=0.2, b1=2.0, b2=0.5, b3=2.0, b4=0.2, b5=0.2,
         th1=0.0, th2=0.0, mu=1.0,
         r=lambda x, y: 2.0 + 0 * x,
         d1=lambda w: np.exp(-0.5 * w), d2=lambda w: 1.0 / (1.0 + 0.5 * w),
         u0=lambda x, y: 0.5 + 0.3 * np.cos(np.pi * x) * np.cos(np.pi * y),
         v0=lambda x, y: 0.5 + 0.3 * np.cos(2 * np.pi * x),
         w0=lambda x, y: 1.0 + 0.5 * np.cos(np.pi * y),
         steady=(40 / 69, 40 / 69, 58 / 69))


def base_B(nonlocal_on):
    c = 0.5 if nonlocal_on else 0.0
    return dict(a1=1.0, a2=1.0, a3=1.0, a4=c, a5=c, b1=1.0, b2=1.0, b3=1.0, b4=c, b5=c,
                th1=0.0, th2=0.0, mu=1.0,
                r=lambda x, y: 1.0 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y),
                u0=lambda x, y: 0.5 + 0 * x, v0=lambda x, y: 0.5 + 0 * x,
                w0=lambda x, y: 1.0 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y))


MOT = {"M1": (lambda w: 0.1 + 0 * w, lambda w: 1.0 + 0 * w),
       "M2": (lambda w: 0.1 + 0 * w, lambda w: np.exp(-2.0 * w))}

if __name__ == "__main__":
    which, outp = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if which == "A":
        recs = list(np.round(np.arange(0, 25.0001, 0.05), 6))
        out = run2d(48, A, [(25.0, 0.005)], recs)
    else:   # which = "B:<local|nonlocal>:<M1|M2>:<N>:<dtmax>"
        _, mode, key, N, dtmax = which.split(":")
        p = base_B(mode == "nonlocal"); p["d1"], p["d2"] = MOT[key]
        dtmax = float(dtmax)
        sched = [(1.0, 0.01), (100.0, 0.1), (1e4, min(1.0, dtmax)), (2e5, dtmax)]
        recs = list(np.unique(np.concatenate(([0.0], np.logspace(-1, np.log10(2e5), 200)[:-1], [2e5]))))
        out = run2d(int(N), p, sched, recs)
        out["tag"] = which
    out["_seconds"] = time.time() - t0
    json.dump(out, open(outp, "w"))
    print("done", which, round(out["_seconds"], 1), "s; min(u,v,w) =", out["min_uvw"])
