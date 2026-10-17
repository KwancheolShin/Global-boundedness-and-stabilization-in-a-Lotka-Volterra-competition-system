import json, sys
import numpy as np
R = sys.argv[1]
sys.path.insert(0, R)
import sim2d_bdf2 as b
from sim2d import A
p = dict(A)
p["dd1"] = lambda w: -0.5 * np.exp(-0.5 * w)
p["dd2"] = lambda w: -0.5 / (1.0 + 0.5 * w) ** 2
te = np.round(np.arange(0, 25.0001, 0.05), 6)
N = 48; M = N * N
x, y, sol = b.run(N, p, 25.0, te, rtol=1e-11, atol=1e-14)
us, vs, ws = p["steady"]
err = np.vstack([np.abs(sol.y[:M] - us).max(axis=0), np.abs(sol.y[M:2 * M] - vs).max(axis=0),
                 np.abs(sol.y[2 * M:3 * M] - ws).max(axis=0)]).T
out = {"t": te.tolist(), "err": err.tolist(), "N": N}
json.dump(out, open(R + r"\res2dA.json", "w"))
e = err.max(axis=1); t = te
env = np.array([e[(t >= s) & (t <= s + 3)].max() for s in t]); m = (t >= 2) & (t <= 12)
print("early envelope rate %.4f; error at t=25: %.2e; nfev %d" % (-np.polyfit(t[m], np.log(env[m]), 1)[0], e[-1], sol.nfev))
