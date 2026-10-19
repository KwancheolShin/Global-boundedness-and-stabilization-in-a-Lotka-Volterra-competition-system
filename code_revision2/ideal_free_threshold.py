"""Near-ideal-free invasion threshold in the heterogeneous examples of Experiment C.

The responsive resident (v in M2, u in M3) has motility prefactor*exp(-k*w).
On the common-coefficient path q=s we locate the suppression strength k_*(eta,s)
at which the rare species' principal invasion eigenvalue changes sign, and compare
it with the first-order ideal-free value k = 1/mean(R) = 2+s of the uniform resident.

Paper notation: zeta = eta (resource amplitude), k = kappa (suppression strength),
sigma = s (nonlocal self-competition), tau = q (nonlocal cross-competition).
All quantities are finite-volume computations on (0,1), not PDE statements.
Run from code_revision2 with BLAS threads set to one (see README_ideal_free.txt):

  python ideal_free_threshold.py compute
  python ideal_free_threshold.py plot
"""
import argparse
import json
import time
import numpy as np
from scipy.optimize import brentq
import nonlocal_robustness as nr

RARE = {"M2": ("u", "v"), "M3": ("v", "u")}  # (rare invader, responsive resident)
K_LO, K_HI = 1.5, 3.2  # bracket of the threshold; the other sign change lies below k=1
ETA_GRID = [.05, .1, .2, .3, .4, .5, .6, .7, .8, .9]
SELF_GRID = [0., .1, .25, .5]
OUT = "ideal_free_threshold.json"


def rare_rate(n, case, eta, k, s, guess=None):
    """Rare-species invasion eigenvalue at the responsive resident, with q=s."""
    rare, resident = RARE[case]
    try:
        z, diag = nr.resident(n, case, resident, eta, k, s, guess)
    except RuntimeError:
        z, diag = nr.resident(n, case, resident, eta, k, s, None)
    lam, _ = nr.invasion(n, case, rare, z, k, s)
    return lam, z, diag


def other_rate(n, case, eta, k, s):
    """Responsive species invading the resident with constant motility, with q=s."""
    rare, resident = RARE[case]
    z, _ = nr.resident(n, case, rare, eta, k, s)
    lam, _ = nr.invasion(n, case, resident, z, k, s)
    return lam


def kstar(n, case, eta, s, xtol=1e-10):
    state = {}

    def f(k):
        lam, z, diag = rare_rate(n, case, eta, k, s, state.get("z"))
        state["z"] = z
        state["residual"] = max(state.get("residual", 0.), diag["residual"])
        return lam

    lo, hi = f(K_LO), f(K_HI)
    if not lo < 0 < hi:
        raise RuntimeError(("no bracket", n, case, eta, s, lo, hi))
    root = brentq(f, K_LO, K_HI, xtol=xtol)
    return dict(N=n, case=case, eta=eta, self=s, kstar=root, ifd_value=2 + s,
                deficit=2 + s - root, max_resident_residual=state["residual"])


def crossing(n, case, eta, k, xtol=1e-11):
    """Common coefficient c=s=q at which the rare-species rate vanishes for fixed k."""
    f = lambda c: rare_rate(n, case, eta, k, c)[0]
    return dict(N=n, case=case, eta=eta, kappa=k, ccritical=brentq(f, 0., .7, xtol=xtol))


def compute():
    t0 = time.time()
    out = dict(environment=nr.environment(), bracket=[K_LO, K_HI])
    # (a) Rate profiles across the threshold, both invasion directions.
    profiles = []
    for case in ("M2", "M3"):
        for s in (0., .5):
            guess = None
            for k in np.round(np.linspace(1.4, 3.0, 81), 10):
                lam, guess, diag = rare_rate(200, case, .5, float(k), s, guess)
                profiles.append(dict(N=200, case=case, eta=.5, self=s, kappa=float(k), rare_rate=lam,
                                     other_rate=other_rate(200, case, .5, float(k), s),
                                     residual=diag["residual"]))
        print("profiles", case, round(time.time() - t0, 1), flush=True)
    out["profiles"] = profiles
    # (b) Threshold versus resource amplitude and self-competition.
    out["threshold_eta"] = [kstar(200, case, eta, s) for case in ("M2", "M3")
                            for s in SELF_GRID for eta in ETA_GRID]
    print("threshold_eta", round(time.time() - t0, 1), flush=True)
    out["threshold_self"] = [kstar(200, case, .5, float(s)) for case in ("M2", "M3")
                             for s in np.round(np.linspace(0, .5, 26), 10)]
    print("threshold_self", round(time.time() - t0, 1), flush=True)
    # (c) Mesh dependence of the threshold.
    out["threshold_mesh"] = [kstar(n, case, eta, s) for n in (100, 200, 400) for case in ("M2", "M3")
                             for eta in (.1, .5, .9) for s in (0., .5)]
    print("threshold_mesh", round(time.time() - t0, 1), flush=True)
    # (d) Rates near the original value k=2 (eta=0.5, s=q=0) on three meshes.
    near = []
    for n in (100, 200, 400):
        for case in ("M2", "M3"):
            for k in (1.9, 1.95, 2., 2.05, 2.1):
                lam, _, _ = rare_rate(n, case, .5, k, 0.)
                near.append(dict(N=n, case=case, kappa=k, rare_rate=lam,
                                 other_rate=other_rate(n, case, .5, k, 0.)))
    out["near_baseline"] = near
    # (e) Common-coefficient crossing c_*(k) at eta=0.5.
    out["crossings"] = [crossing(200, case, .5, k) for case in ("M2", "M3")
                        for k in (2., 2.05, 2.1, 2.25, 2.5)]
    out["crossings"] += [crossing(400, case, .5, 2.) for case in ("M2", "M3")]
    print("crossings", round(time.time() - t0, 1), flush=True)
    # (f) Spatial variation of the resident fitness W-R-s*mean(R).
    fitness = []
    for case in ("M2", "M3"):
        for k in (1.5, 2., 2.5):
            z, _ = nr.resident(200, case, RARE[case][1], .5, k, 0.)
            R, W = z[:200], z[200:]
            g = W - R
            fitness.append(dict(case=case, eta=.5, self=0., kappa=k, fitness_range=float(g.max() - g.min()),
                                resident_mean=float(R.mean())))
    out["fitness_range"] = fitness
    # (g) Consistency with the archived invasion-sign grid of nonlocal_robustness.json.
    grid = json.loads((nr.RESULTS / "nonlocal_robustness.json").read_text())
    thresholds = {}
    for case in ("M2", "M3"):
        for eta in grid["eta_grid"]:
            if eta == 0:
                continue
            for s in grid["self_grid"]:
                thresholds[(case, eta, s)] = kstar(200, case, eta, s)["kstar"]
    responsive_only = {"M2": "v only", "M3": "u only"}
    agree, neutral, disagree = 0, 0, []
    for r in grid["rows"]:
        if r["N"] != 200 or r["eta"] == 0 or r["kappa"] < 1:
            continue
        ks = thresholds[(r["case"], r["eta"], r["self"])]
        predicted = "both invade" if r["kappa"] > ks else responsive_only[r["case"]]
        if r["diagonal_class"] == predicted:
            agree += 1
        elif r["diagonal_class"] == "neutral/unresolved":
            neutral += 1
        else:
            disagree.append(dict(case=r["case"], eta=r["eta"], kappa=r["kappa"], self=r["self"],
                                 archived=r["diagonal_class"], kstar=ks))
    out["grid_consistency"] = dict(points_with_kappa_ge_1=agree + neutral + len(disagree), agree=agree,
                                   neutral=neutral, disagree=disagree,
                                   thresholds=[dict(case=c, eta=e, self=s, kstar=v)
                                               for (c, e, s), v in thresholds.items()])
    out["seconds"] = time.time() - t0
    nr.save(OUT, out)
    print("done", round(out["seconds"], 1), "s; grid agree/neutral/disagree", agree, neutral, len(disagree),
          flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Same style and categorical colours as figs2.py (u: slot 1, v: slot 2).
    C_U, C_V, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": .6, "lines.linewidth": 1.4,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    d = json.loads((nr.RESULTS / OUT).read_text())
    color = {"M2": C_U, "M3": C_V}  # colour of the rare species
    style = {"M2": "-", "M3": "--"}
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.55), layout="constrained")
    # (a) rare-species invasion rate across the threshold
    ax = axs[0]
    for case in ("M2", "M3"):
        for s in (0., .5):
            rows = [r for r in d["profiles"] if r["case"] == case and r["self"] == s]
            ax.plot([r["kappa"] for r in rows], [1e4 * r["rare_rate"] for r in rows], color=color[case],
                    ls=style[case])
    ax.axhline(0, color=MUTED, lw=.6)
    ax.axvline(2, color=MUTED, ls=":", lw=.8)
    ax.text(2.0, -2.05, r" $k=2$", color=MUTED, fontsize=7.5, va="bottom")
    for s, xytext in ((0., (1.45, 1.1)), (.5, (2.56, -1.25))):
        ks = [r["kstar"] for r in d["threshold_self"] if r["case"] == "M2" and abs(r["self"] - s) < 1e-12][0]
        ax.plot([ks], [0], "o", color=INK, ms=3)
        ax.annotate(rf"$\sigma=\tau={s:g}$" + "\n" + rf"$k_*={ks:.3f}$", xy=(ks, 0), xytext=xytext, fontsize=7.5,
                    color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))
    ax.set(xlabel="suppression strength $k$", ylabel=r"rare-species rate $\times10^{4}$", xlim=(1.4, 3.0))
    ax.set_title(r"(a) $\zeta=0.5$", fontsize=8, color=INK)
    ax.grid(True, color=GRID, lw=.5)
    # (b) threshold versus resource amplitude
    ax = axs[1]
    for s in SELF_GRID:
        ax.axhline(2 + s, color=MUTED, ls=":", lw=.8)
        for case in ("M2", "M3"):
            rows = [r for r in d["threshold_eta"] if r["case"] == case and r["self"] == s]
            ax.plot([0.] + [r["eta"] for r in rows], [2 + s] + [r["kstar"] for r in rows], color=color[case],
                    ls=style[case], lw=1.1)
        ax.text(.02, 2 + s + .01, rf"$\sigma={s:g}$", color=INK, fontsize=7.5, va="bottom")
    ax.set(xlabel=r"resource amplitude $\zeta$", ylabel=r"threshold $k_*$", xlim=(0, .9), ylim=(1.88, 2.57))
    ax.set_title(r"(b) $k_*(\zeta,\sigma)$, dotted: $2+\sigma$", fontsize=8, color=INK)
    ax.grid(True, color=GRID, lw=.5)
    # (c) threshold versus self-competition at zeta = 0.5
    ax = axs[2]
    for case in ("M2", "M3"):
        rows = [r for r in d["threshold_self"] if r["case"] == case]
        ax.plot([r["self"] for r in rows], [r["kstar"] for r in rows], color=color[case], ls=style[case])
    ax.plot([0, .5], [2, 2.5], color=MUTED, ls=":", lw=.8)
    ax.axhline(2, color=MUTED, ls="-.", lw=.8)
    c = [r["ccritical"] for r in d["crossings"] if r["N"] == 200 and r["kappa"] == 2. and r["case"] == "M2"][0]
    ax.annotate(rf"$c_*\approx{c:.3f}$", xy=(c, 2.), xytext=(.13, 1.96), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6))
    ax.text(.5, 2.003, r"$k=2$", color=MUTED, fontsize=7.5, ha="right", va="bottom")
    ax.set(xlabel=r"self-competition $\sigma=\tau$", ylabel=r"threshold $k_*$", xlim=(0, .5), ylim=(1.94, 2.52))
    ax.set_title(r"(c) $\zeta=0.5$, dotted: $2+\sigma$", fontsize=8, color=INK)
    ax.grid(True, color=GRID, lw=.5)
    handles = [Line2D([], [], color=C_U, ls="-", label=r"M2: $u$ invading the $v$-resident state"),
               Line2D([], [], color=C_V, ls="--", label=r"M3: $v$ invading the $u$-resident state")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
    for ext in ("pdf", "png"):
        fig.savefig(nr.FIGURES / f"rev4_fig12_threshold.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["compute", "plot", "all"])
    mode = parser.parse_args().mode
    if mode in ("compute", "all"):
        compute()
    if mode in ("plot", "all"):
        plot()
