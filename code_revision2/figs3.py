"""Figure 4: total masses for the 2D heterogeneous-resource runs.  Usage: figs3.py <resdir> <outdir> <ext>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT, EXT = sys.argv[1], sys.argv[2], sys.argv[3]
C_U, C_V = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})
titles = {"M1": r"(M1) $d_1=0.1,\ d_2=1$", "M2": r"(M2) $d_1=0.1,\ d_2(w)=e^{-2w}$"}
fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.9), sharex=True, sharey=True)
for i, m in enumerate(("local", "nonlocal")):
    for j, k in enumerate(("M1", "M2")):
        ax = axes[i, j]; d = json.load(open(f"{R}\\long2d_{m}_{k}.json"))
        t = np.array(d["t"]); s = t > 0
        ax.semilogx(t[s], np.array(d["mass_u"])[s], color=C_U, ls="-", label=r"$\int_\Omega u$")
        ax.semilogx(t[s], np.array(d["mass_v"])[s], color=C_V, ls="--", label=r"$\int_\Omega v$")
        ax.grid(True, color=GRID, lw=0.5); ax.set_ylim(-0.02, 0.6); ax.set_xlim(1e-1, 1e7)
        ax.set_xticks([1e-1, 1e1, 1e3, 1e5, 1e7])
        if i == 0: ax.set_title(titles[k], fontsize=8, color=INK)
        if j == 0: ax.set_ylabel(m + " competition\nmass", fontsize=8)
        if i == 1: ax.set_xlabel(r"$t$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{OUT}\\rev1_fig4_masses2d.{EXT}", dpi=150 if EXT == "png" else None); plt.close(fig)
print("figure 4 written")
