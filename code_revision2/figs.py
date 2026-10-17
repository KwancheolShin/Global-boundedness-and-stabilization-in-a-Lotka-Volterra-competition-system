import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1]
OUT = sys.argv[2]
A = json.load(open(R + r"\resA.json"))
B = json.load(open(R + r"\resB.json"))

C_U, C_V, C_W = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs, light)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---- Figure 1: exponential convergence (Experiment A) ----
t = np.array(A["decay"]["t"])
fig, ax = plt.subplots(figsize=(4.6, 2.9))
for key, c, ls, lab in (("u", C_U, "-", r"$\|u-u_*\|_{L^\infty}$"),
                        ("v", C_V, "--", r"$\|v-v_*\|_{L^\infty}$"),
                        ("w", C_W, ":", r"$\|w-w_*\|_{L^\infty}$")):
    e = np.maximum(np.array(A["decay"][key]), 1e-16)
    ax.semilogy(t, e, color=c, ls=ls, label=lab)
eig = sorted(A["eig_k0"], key=lambda z: z[0])            # most negative first
lam_fast, lam_slow = -eig[0][0], -eig[-1][0]               # 1.26 (complex pair), 0.87 (real)
e_all = np.max(np.vstack([A["decay"][k] for k in "uvw"]), axis=0)
def anchor(t0): return e_all[np.argmin(abs(t - t0))]
for (ta, tb), lam_ in (((1.0, 11.0), lam_fast), ((16.0, 24.0), lam_slow)):
    tt_ = np.array([ta, tb]); y0 = 20 * max(anchor(ta), anchor(ta + 0.6))
    ax.semilogy(tt_, y0 * np.exp(-lam_ * (tt_ - ta)), color=MUTED, lw=0.8)
    ax.text(tb - 1.2, y0 * np.exp(-lam_ * (tb - ta)) * 6, r"$e^{-%.2f\,t}$" % lam_, color=MUTED, fontsize=8)
ax.set_xlabel(r"$t$"); ax.set_ylabel("distance to steady state")
ax.grid(True, which="major", color=GRID, lw=0.5)
ax.set_xlim(0, 25); ax.set_ylim(1e-14, 1)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT + r"\rev1_fig1_convergence.pdf"); plt.close(fig)

# ---- Figure 2: total masses, heterogeneous resource (Experiment B) ----
titles = {"M1": r"(M1) $d_1=0.1,\ d_2=1$",
          "M2": r"(M2) $d_1=0.1,\ d_2(w)=e^{-2w}$",
          "M3": r"(M3) $d_1(w)=0.1e^{-2w},\ d_2=1$"}
fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, sharey=True)
for i, mode in enumerate(("local", "nonlocal")):
    for j, key in enumerate(("M1", "M2", "M3")):
        ax = axes[i, j]; d = B[f"{mode}_{key}"]
        tt = np.array(d["t"]); m = tt > 0
        ax.semilogx(tt[m], np.array(d["mass_u"])[m], color=C_U, ls="-", label=r"$\int_\Omega u$")
        ax.semilogx(tt[m], np.array(d["mass_v"])[m], color=C_V, ls="--", label=r"$\int_\Omega v$")
        ax.grid(True, which="major", color=GRID, lw=0.5)
        ax.set_ylim(-0.02, 0.6); ax.set_xlim(1e-1, 2e5)
        if i == 0: ax.set_title(titles[key], fontsize=8, color=INK)
        if j == 0: ax.set_ylabel(("local" if mode == "local" else "nonlocal") + " competition\nmass", fontsize=8)
        if i == 1: ax.set_xlabel(r"$t$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT + r"\rev1_fig2_masses.pdf"); plt.close(fig)
print("figures written")
