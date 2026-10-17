import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import sim

R = sys.argv[1]
B = json.load(open(R + r"\resB.json"))


def excl_time(t, mu, mv, tol=1e-3):
    """First time at which min(mass_u, mass_v) drops below tol (log-log interpolation)."""
    t, mu, mv = map(np.asarray, (t, mu, mv))
    minority = np.minimum(mu, mv)
    idx = np.where(minority < tol)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0 or minority[i] <= 0 or minority[i - 1] <= 0:
        return float(t[i])
    a = (np.log(tol) - np.log(minority[i - 1])) / (np.log(minority[i]) - np.log(minority[i - 1]))
    return float(np.exp(np.log(t[i - 1]) + a * (np.log(t[i]) - np.log(t[i - 1]))))


out = {}
for k, v in B.items():
    if k.startswith("_"):
        continue
    out[k] = {"T_excl_N100": excl_time(v["t"], v["mass_u"], v["mass_v"]),
              "mass_u": v["mass_u"][-1], "mass_v": v["mass_v"][-1]}

# refinement check N = 200 for every scenario
T = 2.0e5
for nl in (False, True):
    for key, (d1, d2) in sim.MOT.items():
        p = sim.base_B(nl); p["d1"], p["d2"] = d1, d2
        te = np.concatenate(([0.0], np.logspace(-1, np.log10(T), 400)[:-1], [T]))
        x, sol = sim.run(200, p, T, te, rtol=1e-7, atol=1e-10)
        h = 1.0 / 200
        mu_ = h * sol.y[:200].sum(axis=0); mv_ = h * sol.y[200:400].sum(axis=0)
        tag = ("nonlocal" if nl else "local") + "_" + key
        out[tag].update({"T_excl_N200": excl_time(te, mu_, mv_), "mass_u_N200": float(mu_[-1]), "mass_v_N200": float(mv_[-1])})
json.dump(out, open(R + r"\checkB.json", "w"), indent=1)
for k, v in out.items():
    print(k, v)
