"""Experiment D of the paper (1D): finite-time blow-up.

Kinetics of Fig. 3 of the original manuscript, a1 = b1 = 1, a3 = a4 = a5 = b2 = b4 = b5 = -1,
mu = 1, r = 1 and constant motilities d1 = 1, d2 = 5, now with the nonnegative initial
data of Experiment A.
  (a) (a2, b3) = (2, 3): blow-up time T_N for N = 50, 100, 200, 400, and the spatially
      homogeneous ODE started from the means of the initial data;
  (b) diagonal a2 = b3 = a with a = 2.8, 2.9 (blow-up) and 3.1, 3.2 (convergence to
      the coexistence state u* = v* = 1/(a-1), w* = (a-3)/(a-1)).
T_N is the first time at which max(||u||_inf, ||v||_inf) reaches CAP = 1e6.
Usage: python expE.py <dir>   ->  <dir>\resE.json
"""
import json, sys, time
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, sys.argv[1])
import sim

R = sys.argv[1]
CAP = 1.0e6


def params(a2, b3):
    return dict(a1=1.0, a2=a2, a3=-1.0, a4=-1.0, a5=-1.0, b1=1.0, b2=-1.0, b3=b3, b4=-1.0, b5=-1.0,
                th1=0.0, th2=0.0, mu=1.0, r=lambda x: 1.0 + 0 * x,
                d1=lambda w: 1.0 + 0 * w, d2=lambda w: 5.0 + 0 * w,
                u0=lambda x: 0.5 + 0.3 * np.cos(np.pi * x),
                v0=lambda x: 0.5 + 0.3 * np.cos(2 * np.pi * x),
                w0=lambda x: 1.0 + 0.5 * np.cos(np.pi * x))


def run_pde(N, p, T, rtol=1e-8, atol=1e-10):
    x, rhs = sim.make_rhs(N, p)
    y0 = np.concatenate((p["u0"](x), p["v0"](x), p["w0"](x)))

    def big(t, y):
        return CAP - max(y[:N].max(), y[N:2 * N].max())
    big.terminal = True
    sol = solve_ivp(rhs, (0.0, T), y0, method="BDF", rtol=rtol, atol=atol, events=big)
    if sol.status < 0:
        raise RuntimeError(sol.message)
    U, V, W = sol.y[:N], sol.y[N:2 * N], sol.y[2 * N:]
    return {"t": sol.t.tolist(), "u_inf": U.max(axis=0).tolist(), "v_inf": V.max(axis=0).tolist(),
            "u_min": U.min(axis=0).tolist(), "w_min": float(W.min()),
            "T_blowup": float(sol.t_events[0][0]) if len(sol.t_events[0]) else None,
            "final": [U[:, -1].tolist(), V[:, -1].tolist(), W[:, -1].tolist()]}


def run_ode(p, y0, T):
    A2, A3 = p["a2"] + p["a4"], p["a3"] + p["a5"]      # |Omega| = 1
    B2, B3 = p["b2"] + p["b5"], p["b3"] + p["b4"]

    def f(t, y):
        U, V, W = y
        return [U * (p["a1"] * W - A2 * U - A3 * V), V * (p["b1"] * W - B2 * U - B3 * V),
                -W * (U + V) + p["mu"] * W * (1.0 - W)]

    def big(t, y):
        return CAP - max(y[0], y[1])
    big.terminal = True
    s = solve_ivp(f, (0.0, T), y0, method="LSODA", rtol=1e-11, atol=1e-13, events=big)
    return {"t": s.t.tolist(), "U": s.y[0].tolist(), "V": s.y[1].tolist(), "W": s.y[2].tolist(),
            "T_blowup": float(s.t_events[0][0]) if len(s.t_events[0]) else None}


t0 = time.time()
out = {"CAP": CAP}
# (a) (a2, b3) = (2, 3): mesh refinement of the blow-up time and the homogeneous ODE
p = params(2.0, 3.0)
out["a"] = {}
for N in (50, 100, 200, 400):
    res = run_pde(N, p, 50.0)
    out["a"][str(N)] = res if N == 200 else {"T_blowup": res["T_blowup"], "w_min": res["w_min"]}
    print("(2,3) N=%d  T_N=%.6f  min w=%.3e" % (N, res["T_blowup"], res["w_min"]), flush=True)
xf = (np.arange(10 ** 5) + 0.5) / 10 ** 5
means = [float(p[k](xf).mean()) for k in ("u0", "v0", "w0")]
out["a"]["ode"] = run_ode(p, means, 50.0)
out["a"]["ode_y0"] = means
print("(2,3) ODE from means %s: T=%.6f" % (means, out["a"]["ode"]["T_blowup"]), flush=True)
# (b) diagonal a2 = b3 = a
out["b"] = {}
for a in (2.8, 2.9, 3.1, 3.2):
    res = run_pde(200, params(a, a), 200.0)
    rec = {"t": res["t"], "u_inf": res["u_inf"], "v_inf": res["v_inf"], "T_blowup": res["T_blowup"]}
    if res["T_blowup"] is None:
        us, ws = 1.0 / (a - 1.0), (a - 3.0) / (a - 1.0)
        U, V, W = (np.array(z) for z in res["final"])
        rec["steady"] = [us, us, ws]
        rec["dist_final"] = float(max(np.abs(U - us).max(), np.abs(V - us).max(), np.abs(W - ws).max()))
    out["b"]["%.1f" % a] = rec
    print("diag a=%.1f  T_blowup=%s  %s" % (a, rec["T_blowup"],
          "dist to (u*,v*,w*) at t=200: %.2e" % rec["dist_final"] if "dist_final" in rec else ""), flush=True)
out["_seconds"] = time.time() - t0
json.dump(out, open(R + r"\resE.json", "w"))
print("done %.1f s" % out["_seconds"])
