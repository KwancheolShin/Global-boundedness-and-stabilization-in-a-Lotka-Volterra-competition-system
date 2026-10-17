"""Figures for the second revision round.  Usage: figs2.py <resdir> <outdir> <ext> <which>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT, EXT, WHICH = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
C_U, C_V, C_W = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs, light)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(f"{OUT}\\{name}.{EXT}", dpi=150 if EXT == "png" else None)
    plt.close(fig)


def ref_line(ax, t_all, e_all, ta, tb, lam, label, up=20, dx=1.2):
    tt = np.array([ta, tb]); i = np.argmin(abs(t_all - ta))
    y0 = up * max(e_all[i:i + 12].max(), 1e-300)
    ax.semilogy(tt, y0 * np.exp(-lam * (tt - ta)), color=MUTED, lw=0.8)
    ax.text(tb - dx, y0 * np.exp(-lam * (tb - ta)) * 6, label, color=MUTED, fontsize=8)


if WHICH in ("1", "all"):
    A1 = json.load(open(R + r"\resA.json")); A2 = json.load(open(R + r"\res2dA.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    t1 = np.array(A1["decay"]["t"])
    curves1 = [np.maximum(np.array(A1["decay"][k]), 1e-16) for k in "uvw"]
    t2 = np.array(A2["t"]); E2 = np.maximum(np.array(A2["err"]), 1e-16)
    curves2 = [E2[:, 0], E2[:, 1], E2[:, 2]]
    eig = sorted(A1["eig_k0"], key=lambda z: z[0])
    lam_fast, lam_slow = -eig[0][0], -eig[-1][0]
    for ax, t, cs, title in ((axes[0], t1, curves1, r"(a) $n=1$, $N=200$"), (axes[1], t2, cs2 if False else curves2, r"(b) $n=2$, $48\times48$ cells")):
        for c, col, ls, lab in zip(cs, (C_U, C_V, C_W), ("-", "--", ":"),
                                   (r"$\|u-u_*\|_{L^\infty}$", r"$\|v-v_*\|_{L^\infty}$", r"$\|w-w_*\|_{L^\infty}$")):
            ax.semilogy(t, c, color=col, ls=ls, label=lab)
        e_all = np.max(np.vstack(cs), axis=0)
        ref_line(ax, t, e_all, 1.0, 11.0, lam_fast, r"$e^{-%.2f\,t}$" % lam_fast)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel(r"$t$"); ax.grid(True, color=GRID, lw=0.5); ax.set_xlim(0, 25); ax.set_ylim(1e-14, 1)
    ref_line(axes[0], t1, np.max(np.vstack(curves1), axis=0), 16.0, 24.0, lam_slow, r"$e^{-%.2f\,t}$" % lam_slow)
    axes[0].set_ylabel("distance to steady state")
    axes[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, "rev1_fig1_convergence")

if WHICH in ("2", "all"):
    C = json.load(open(R + r"\resC.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    d = C["neg"]; t = np.array(d["t"])
    cs = [np.maximum(np.array(d[k]), 1e-16) for k in ("eu", "ev", "ew")]
    for c, col, ls, lab in zip(cs, (C_U, C_V, C_W), ("-", "--", ":"),
                               (r"$\|u-\hat u\|_{L^\infty}$", r"$\|v\|_{L^\infty}$", r"$\|w-\hat w\|_{L^\infty}$")):
        axes[0].semilogy(t, c, color=col, ls=ls, label=lab)
    g = -d["Gamma"]
    ref_line(axes[0], t, np.max(np.vstack(cs), axis=0), 8.0, 36.0, g, r"$e^{-%.3f\,t}$" % g, dx=4)
    axes[0].set_title(r"(a) $\Gamma_v=%.3f$" % d["Gamma"], fontsize=8, color=INK)
    axes[0].set_xlabel(r"$t$"); axes[0].set_ylabel("distance to $(\\hat u,0,\\hat w)$")
    axes[0].grid(True, color=GRID, lw=0.5); axes[0].set_xlim(0, 40); axes[0].set_ylim(1e-14, 1)
    axes[0].legend(loc="upper right", fontsize=7.5)
    d = C["zero"]; t = np.array(d["t"]); m = t > 0
    ev = np.array(d["ev"]); eu = np.array(d["eu"]); ew = np.array(d["ew"])
    axes[1].loglog(t[m], eu[m], color=C_U, ls="-", label=r"$\|u-\hat u\|_{L^\infty}$")
    axes[1].loglog(t[m], ev[m], color=C_V, ls="--", label=r"$\|v\|_{L^\infty}$")
    axes[1].loglog(t[m], ew[m], color=C_W, ls=":", label=r"$\|w-\hat w\|_{L^\infty}$")
    tt = np.array([1e2, 1e5]); i = np.argmin(abs(t - 1e2))
    axes[1].loglog(tt, 3 * ev[i] * (tt / 1e2) ** -1.0, color=MUTED, lw=0.8)
    axes[1].text(2e3, 3 * ev[i] * (2e3 / 1e2) ** -1.0 * 2.2, r"$\propto t^{-1}$", color=MUTED, fontsize=8)
    axes[1].set_title(r"(b) $\Gamma_v=0$", fontsize=8, color=INK)
    axes[1].set_xlabel(r"$t$"); axes[1].grid(True, which="major", color=GRID, lw=0.5)
    axes[1].set_xlim(1e-1, 1e5)
    fig.tight_layout(); save(fig, "rev1_fig2_exclusion")

if WHICH in ("3", "all"):
    B = json.load(open(R + r"\resB.json"))
    titles = {"M1": r"(M1) $d_1=0.1,\ d_2=1$", "M2": r"(M2) $d_1=0.1,\ d_2(w)=e^{-2w}$",
              "M3": r"(M3) $d_1(w)=0.1e^{-2w},\ d_2=1$"}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, sharey=True)
    for i, mode in enumerate(("local", "nonlocal")):
        for j, key in enumerate(("M1", "M2", "M3")):
            ax = axes[i, j]; d = B[f"{mode}_{key}"]; tt = np.array(d["t"]); m = tt > 0
            ax.semilogx(tt[m], np.array(d["mass_u"])[m], color=C_U, ls="-", label=r"$\int_\Omega u$")
            ax.semilogx(tt[m], np.array(d["mass_v"])[m], color=C_V, ls="--", label=r"$\int_\Omega v$")
            ax.grid(True, color=GRID, lw=0.5); ax.set_ylim(-0.02, 0.6); ax.set_xlim(1e-1, 2e5)
            if i == 0: ax.set_title(titles[key], fontsize=8, color=INK)
            if j == 0: ax.set_ylabel(mode + " competition\nmass", fontsize=8)
            if i == 1: ax.set_xlabel(r"$t$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "rev1_fig3_masses1d")
print("ok", WHICH)
