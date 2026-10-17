"""1D finite-volume / method-of-lines solver for the revised manuscript.

u_t = (d1(w) u)_xx + u (a1 F1(w) - a2 (th1 + u) - a3 v - a4 I(u) - a5 I(v))
v_t = (d2(w) v)_xx + v (b1 F2(w) - b2 u - b3 (th2 + v) - b4 I(v) - b5 I(u))
w_t = w_xx - F1(w) u - F2(w) v + mu w (r(x) - w)
on (0,1) with homogeneous Neumann conditions, I(f) = int_0^1 f dx.
F1 and F2 may be supplied in p; both default to linear uptake.
"""
import json, sys, time
import numpy as np
from scipy.integrate import solve_ivp


def make_rhs(N, p):
    h = 1.0 / N
    x = (np.arange(N) + 0.5) * h
    r = p["r"](x)
    F1 = p.get("F1", lambda w: w)
    F2 = p.get("F2", lambda w: w)

    def lap(phi):
        ext = np.concatenate(([phi[0]], phi, [phi[-1]]))  # zero-flux ghosts
        return (ext[2:] - 2.0 * ext[1:-1] + ext[:-2]) / h**2

    def rhs(t, y):
        u, v, w = y[:N], y[N:2 * N], y[2 * N:]
        Iu, Iv = h * u.sum(), h * v.sum()
        fu = u * (p["a1"] * F1(w) - p["a2"] * (p["th1"] + u) - p["a3"] * v - p["a4"] * Iu - p["a5"] * Iv)
        fv = v * (p["b1"] * F2(w) - p["b2"] * u - p["b3"] * (p["th2"] + v) - p["b4"] * Iv - p["b5"] * Iu)
        fw = -F1(w) * u - F2(w) * v + p["mu"] * w * (r - w)
        return np.concatenate((lap(p["d1"](w) * u) + fu, lap(p["d2"](w) * v) + fv, lap(w) + fw))

    return x, rhs


def run(N, p, T, t_eval, rtol=1e-8, atol=1e-10):
    x, rhs = make_rhs(N, p)
    y0 = np.concatenate((p["u0"](x), p["v0"](x), p["w0"](x)))
    sol = solve_ivp(rhs, (0.0, T), y0, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return x, sol


def coarsen(f, factor):
    return f.reshape(-1, factor).mean(axis=1)


# ---------------- Experiment A: convergence to the coexistence state ----------------
A = dict(a1=2.0, a2=2.0, a3=0.5, a4=0.2, a5=0.2, b1=2.0, b2=0.5, b3=2.0, b4=0.2, b5=0.2,
         th1=0.0, th2=0.0, mu=1.0, r=lambda x: 2.0 + 0 * x,
         d1=lambda w: np.exp(-0.5 * w), d2=lambda w: 1.0 / (1.0 + 0.5 * w),
         u0=lambda x: 0.5 + 0.3 * np.cos(np.pi * x),
         v0=lambda x: 0.5 + 0.3 * np.cos(2 * np.pi * x),
         w0=lambda x: 1.0 + 0.5 * np.cos(np.pi * x))


def steady_A(p):
    M = np.array([[p["a2"] + p["a4"], p["a3"] + p["a5"], -p["a1"]],
                  [p["b2"] + p["b5"], p["b3"] + p["b4"], -p["b1"]],
                  [1.0, 1.0, p["mu"]]])
    rhs = np.array([-p["a2"] * p["th1"], -p["b3"] * p["th2"], p["mu"] * 2.0])
    return np.linalg.solve(M, rhs)


def experiment_A():
    us, vs, ws = steady_A(A)
    out = {"steady": [us, vs, ws]}
    # long-time decay on the finest production mesh
    T = 25.0
    te = np.linspace(0, T, 501)
    x, sol = run(200, A, T, te, rtol=1e-12, atol=1e-15)
    N = 200
    err = {k: np.abs(sol.y[i * N:(i + 1) * N] - s).max(axis=0).tolist()
           for i, (k, s) in enumerate(zip("uvw", (us, vs, ws)))}
    out["decay"] = {"t": te.tolist(), **err}
    # exponential rate fitted to the envelope max_{s in [t, t+3]} max(err_u, err_v, err_w), t in [2, 18]
    tt = np.array(te); e = np.max(np.vstack([err["u"], err["v"], err["w"]]), axis=0)
    env = np.array([e[(tt >= s) & (tt <= s + 3)].max() for s in tt])
    m = (tt >= 2) & (tt <= 18)
    slope, icpt = np.polyfit(tt[m], np.log(env[m]), 1)
    out["rate"] = float(-slope); out["rate_icpt"] = float(icpt)
    out["final_err"] = float(e[-1])
    # eigenvalues of the linearisation of the kinetics about the constant state (spatially homogeneous mode)
    a1, a2, a3, a4, a5 = (A[k] for k in ("a1", "a2", "a3", "a4", "a5"))
    b1, b2, b3, b4, b5 = (A[k] for k in ("b1", "b2", "b3", "b4", "b5"))
    J = np.array([[-us * (a2 + a4), -us * (a3 + a5), a1 * us],
                  [-vs * (b2 + b5), -vs * (b3 + b4), b1 * vs],
                  [-ws, -ws, -A["mu"] * ws]])
    out["eig_k0"] = [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(J)]
    # mesh self-convergence at t = 2
    Ns = [50, 100, 200, 400]
    sols = {}
    for n in Ns:
        _, s = run(n, A, 2.0, [2.0], rtol=1e-10, atol=1e-12)
        sols[n] = s.y[:, -1]
    diffs = []
    for n in Ns[:-1]:
        fine = sols[2 * n]
        c = np.concatenate([coarsen(fine[i * 2 * n:(i + 1) * 2 * n], 2) for i in range(3)])
        diffs.append(float(np.abs(sols[n] - c).max()))
    out["mesh"] = {"N": Ns[:-1], "diff": diffs,
                   "order": [float(np.log2(diffs[i] / diffs[i + 1])) for i in range(len(diffs) - 1)]}
    return out


# ---------------- Experiment C (n = 1): heterogeneous resource, dispersal strategies ----------------
def base_B(nonlocal_on):
    c = 0.5 if nonlocal_on else 0.0
    return dict(a1=1.0, a2=1.0, a3=1.0, a4=c, a5=c, b1=1.0, b2=1.0, b3=1.0, b4=c, b5=c,
                th1=0.0, th2=0.0, mu=1.0, r=lambda x: 1.0 + 0.5 * np.cos(np.pi * x),
                u0=lambda x: 0.5 + 0 * x, v0=lambda x: 0.5 + 0 * x,
                w0=lambda x: 1.0 + 0.5 * np.cos(np.pi * x))


MOT = {
    "M1": (lambda w: 0.1 + 0 * w, lambda w: 1.0 + 0 * w),
    "M2": (lambda w: 0.1 + 0 * w, lambda w: np.exp(-2.0 * w)),
    "M3": (lambda w: 0.1 * np.exp(-2.0 * w), lambda w: 1.0 + 0 * w),
}


def experiment_B(T=2.0e5, N=100):
    res = {}
    for nl in (False, True):
        for key, (d1, d2) in MOT.items():
            p = base_B(nl); p["d1"], p["d2"] = d1, d2
            te = np.concatenate(([0.0], np.logspace(-1, np.log10(T), 400)[:-1], [T]))
            x, sol = run(N, p, T, te, rtol=1e-7, atol=1e-10)
            h = 1.0 / N
            mu_ = h * sol.y[:N].sum(axis=0); mv_ = h * sol.y[N:2 * N].sum(axis=0)
            tag = ("nonlocal" if nl else "local") + "_" + key
            res[tag] = {"t": te.tolist(), "mass_u": mu_.tolist(), "mass_v": mv_.tolist(),
                        "x": x.tolist(), "u": sol.y[:N, -1].tolist(), "v": sol.y[N:2 * N, -1].tolist(),
                        "w": sol.y[2 * N:, -1].tolist(),
                        "dmass_u": float((mu_[-1] - mu_[-11]) / (te[-1] - te[-11])),
                        "dmass_v": float((mv_[-1] - mv_[-11]) / (te[-1] - te[-11]))}
            print(tag, "u:", round(mu_[-1], 6), "v:", round(mv_[-1], 6), flush=True)
    return res


if __name__ == "__main__":
    which = sys.argv[1]
    t0 = time.time()
    out = experiment_A() if which == "A" else experiment_B()
    out["_seconds"] = time.time() - t0
    json.dump(out, open(sys.argv[2], "w"))
    print("done", which, round(out["_seconds"], 1), "s")
