"""Mesh self-convergence for Experiment A at early times, when the non-uniform modes are present.

One dimension: N = 50, 100, 200, 400, 800; two dimensions: 24x24, 48x48, 96x96 cells.
Each solution is compared with the next finer one after averaging pairs (1D) or
2x2 blocks (2D) of fine cells. Run from code_revision2:  python spatial_order_check.py
"""
import json
import time
import numpy as np
import sim
import sim2d_bdf2 as b2
from sim2d import A as A2

TIMES = [0.1, 0.5, 2.0]
TIMES_2D = [0.1, 0.5]


def coarsen_1d(fine, n):
    return np.concatenate([sim.coarsen(fine[i * 2 * n:(i + 1) * 2 * n], 2) for i in range(3)])


def coarsen_2d(field, n):
    return field.reshape(n, 2, n, 2).mean(axis=(1, 3)).ravel()


def orders(diffs):
    return [float(np.log2(diffs[i] / diffs[i + 1])) for i in range(len(diffs) - 1)]


def one_dimension():
    Ns = [50, 100, 200, 400, 800]
    sols = {n: sim.run(n, sim.A, TIMES[-1], TIMES, rtol=1e-11, atol=1e-14)[1].y for n in Ns}
    out = {}
    for j, t in enumerate(TIMES):
        diffs = [float(np.abs(sols[n][:, j] - coarsen_1d(sols[2 * n][:, j], n)).max()) for n in Ns[:-1]]
        spread = float(max(np.ptp(sols[800][i * 800:(i + 1) * 800, j]) for i in range(3)))
        out[str(t)] = dict(N=Ns[:-1], diff=diffs, order=orders(diffs), max_spatial_range=spread)
    return out


def two_dimensions():
    p = dict(A2)
    p["dd1"] = lambda w: -0.5 * np.exp(-0.5 * w)
    p["dd2"] = lambda w: -0.5 / (1.0 + 0.5 * w) ** 2
    Ns = [24, 48, 96]
    sols = {}
    for n in Ns:
        t0 = time.time()
        sols[n] = b2.run(n, p, TIMES_2D[-1], TIMES_2D, rtol=1e-10, atol=1e-13)[2].y
        print("2D N=%d %.1f s" % (n, time.time() - t0), flush=True)
    out = {}
    for j, t in enumerate(TIMES_2D):
        diffs = []
        for n in Ns[:-1]:
            m, mf = n * n, 4 * n * n
            coarse = sols[n][:, j]
            fine = sols[2 * n][:, j]
            d = max(float(np.abs(coarse[i * m:(i + 1) * m] - coarsen_2d(fine[i * mf:(i + 1) * mf], n)).max())
                    for i in range(3))
            diffs.append(d)
        out[str(t)] = dict(N=Ns[:-1], diff=diffs, order=orders(diffs))
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = dict(one_dimension=one_dimension())
    print(json.dumps(res["one_dimension"], indent=1), flush=True)
    res["two_dimensions"] = two_dimensions()
    res["seconds"] = time.time() - t0
    (sim_dir := __import__("pathlib").Path(__file__).resolve().parent / "results").mkdir(exist_ok=True)
    (sim_dir / "spatial_order.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["two_dimensions"], indent=1), "\ndone %.1f s" % res["seconds"], flush=True)
