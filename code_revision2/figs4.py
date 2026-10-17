"""Figure 5 of the revision (rev1_fig5_blowup, Experiment D).  Usage: figs4.py <resdir> <outdir> <ext>"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT, EXT = sys.argv[1], sys.argv[2], sys.argv[3]
C_U, C_V, C_W = "#2a78d6", "#eb6834", "#1baf7a"     # same palette as figs2.py
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})

E = json.load(open(R + r"\resE.json"))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)

# (a) (a2, b3) = (2, 3): PDE norms and the homogeneous ODE from the means of the data
ax = axes[0]
d = E["a"]["200"]; t = np.array(d["t"])
o = E["a"]["ode"]; to = np.array(o["t"])
ax.semilogy(t, d["u_inf"], color=C_U, ls="-", label=r"$\|u\|_{L^\infty}$")
ax.semilogy(t, d["v_inf"], color=C_V, ls="--", label=r"$\|v\|_{L^\infty}$")
ax.semilogy(to, o["U"], color=MUTED, lw=0.8, ls=":", label=r"$U,\ V$ (homogeneous)")
ax.semilogy(to, o["V"], color=MUTED, lw=0.8, ls=":")
T = d["T_blowup"]
ax.axvline(T, color=MUTED, lw=0.6, ls="-.")
ax.text(T - 0.04, 3e3, r"$T_{200}=%.4f$" % T, ha="right", color=MUTED, fontsize=8)
ax.set_xlim(0, 2.25); ax.set_ylim(0.2, 2e6)
ax.set_title(r"(a) $(a_2,b_3)=(2,3)$", fontsize=8, color=INK)
ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$L^\infty$ norm")
ax.grid(True, color=GRID, lw=0.5)
ax.legend(loc="upper left", fontsize=7.5)

# (b) diagonal a2 = b3 = a: blow-up for a < 3, convergence for a > 3
ax = axes[1]
C_4 = "#eda100"                                      # categorical slot 4 of the palette
for a, ls, c in (("2.8", "-", C_U), ("2.9", "--", C_V), ("3.1", "-.", C_W), ("3.2", ":", C_4)):
    r = E["b"][a]
    ax.semilogy(r["t"], r["u_inf"], color=c, ls=ls, label=r"$a=%s$" % a)
ax.set_xlim(0, 20)
ax.set_title(r"(b) $a_2=b_3=a$: $\|u\|_{L^\infty}$", fontsize=8, color=INK)
ax.set_xlabel(r"$t$"); ax.grid(True, color=GRID, lw=0.5)
ax.legend(loc="upper right", fontsize=7.5)

fig.tight_layout()
fig.savefig(f"{OUT}\\rev1_fig5_blowup.{EXT}", dpi=150 if EXT == "png" else None)
print("ok")
