"""Experiment B of the paper (1D): convergence to the semi-trivial state (u_hat, 0, w_hat)."""
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import sim

R = sys.argv[1]
base = dict(a1=2.0, a2=2.0, a3=0.5, a4=0.2, a5=0.2, b1=1.0, b3=2.0, b4=0.2, b5=0.2,
            th1=0.0, th2=0.0, mu=1.0, r=lambda x: 2.0 + 0 * x,
            d1=lambda w: np.exp(-0.5 * w), d2=lambda w: 1.0 / (1.0 + 0.5 * w),
            u0=lambda x: 0.5 + 0.3 * np.cos(np.pi * x),
            v0=lambda x: 0.5 + 0.3 * np.cos(2 * np.pi * x),
            w0=lambda x: 1.0 + 0.5 * np.cos(np.pi * x))
r0 = 2.0
A11 = base["a2"] + base["a4"]
uh = base["mu"] * (base["a1"] * r0 - base["a2"] * base["th1"]) / (base["mu"] * A11 + base["a1"])
wh = r0 - uh / base["mu"]
out = {"u_hat": uh, "w_hat": wh}
N = 200
for tag, b2, T, te in (("neg", 1.5, 40.0, np.linspace(0, 40.0, 801)),
                        ("zero", 0.9, 1.0e5, np.concatenate(([0.0], np.logspace(-2, 5, 400)[:-1], [1.0e5])))):
    p = dict(base); p["b2"] = b2
    Gamma = p["b1"] * wh - p["b3"] * p["th2"] - (b2 + p["b5"]) * uh
    x, sol = sim.run(N, p, T, te, rtol=1e-11, atol=1e-14)
    U, V, W = sol.y[:N], sol.y[N:2 * N], sol.y[2 * N:]
    eu = np.abs(U - uh).max(axis=0); ev = V.max(axis=0); ew = np.abs(W - wh).max(axis=0)
    rec = {"b2": b2, "Gamma": float(Gamma), "t": te.tolist(), "eu": eu.tolist(), "ev": ev.tolist(), "ew": ew.tolist(),
           "mass_v": (V.sum(axis=0) / N).tolist()}
    if tag == "neg":
        m = (te >= 10) & (te <= 30)
        for k, e in (("u", eu), ("v", ev), ("w", ew)):
            rec["rate_" + k] = float(-np.polyfit(te[m], np.log(e[m]), 1)[0])
        # linearisation of the homogeneous kinetics at (u_hat, 0, w_hat)
        a1, a2, a3, a4, a5 = (p[k] for k in ("a1", "a2", "a3", "a4", "a5"))
        J = np.array([[-uh * (a2 + a4), -uh * (a3 + a5), a1 * uh],
                      [0.0, Gamma, 0.0],
                      [-wh, -wh, -p["mu"] * wh]])
        rec["eig"] = [[float(z.real), float(z.imag)] for z in np.linalg.eigvals(J)]
    else:
        m = (te >= 1e3) & (te <= 1e5)
        rec["loglog_slope_v"] = float(np.polyfit(np.log(te[m]), np.log(ev[m]), 1)[0])
        rec["t_times_v_end"] = float(te[-1] * ev[-1])
    out[tag] = rec
    print(tag, "Gamma=%.4f" % Gamma, {k: v for k, v in rec.items() if k.startswith(("rate", "eig", "loglog", "t_times"))}, flush=True)
json.dump(out, open(R + r"\resC.json", "w"))
print("u_hat=%.6f w_hat=%.6f" % (uh, wh))
