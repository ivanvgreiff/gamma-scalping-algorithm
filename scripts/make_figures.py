"""Regenerate the figures in docs/figures used by the README.

Usage (from the repo root):  python scripts/make_figures.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulation.monte_carlo import simulate_delta_hedge_paths_mc, call_price_bs

OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})
S0 = K = 100.0; T = 1.0; r = 0.0

# 1) single path: RV 30% vs IV 20%
res = simulate_delta_hedge_paths_mc(S0=S0, K=K, r=r, sigma_real=0.30, sigma_imp=0.20, T=T,
                                    n_steps=365, n_paths=1, seed=36, record_timeseries=True)
ts = res["timeseries"]
fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(ts["time"], ts["S"], color=BLUE); ax[0].set_title("Underlying price")
ax[1].plot(ts["time"], ts["delta"], color=BLUE); ax[1].set_title("Option delta (= size of the short hedge)")
ax[2].plot(ts["time"], ts["pnl_cum"], color=AQUA); ax[2].axhline(0, color=INK2, lw=0.8)
ax[2].set_title("Delta-hedged P&L (mark-to-market)"); ax[2].set_xlabel("Time (years)")
fig.suptitle("One simulated path: long ATM call bought at 20% IV, realized vol 30%, daily re-hedging",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/single_path.png", dpi=150); plt.close(fig)

# 2) P&L distributions for three RV/IV regimes
fig, ax = plt.subplots(figsize=(9, 4.2))
bins = np.linspace(-12, 16, 90)
for rv, col, lab in [(0.10, ORANGE, "RV 10% < IV"), (0.20, BLUE, "RV 20% = IV"), (0.30, AQUA, "RV 30% > IV")]:
    p = simulate_delta_hedge_paths_mc(S0=S0, K=K, r=r, sigma_real=rv, sigma_imp=0.20, T=T,
                                      n_steps=365, n_paths=20000, seed=11)["pnl"]
    ax.hist(p, bins=bins, histtype="step", color=col, lw=2, density=True, label=f"{lab}  (mean {p.mean():+.2f})")
    print(lab, p.mean(), p.std())
ax.set_title("Final P&L of a delta-hedged long call, IV = 20% (20,000 paths each)")
ax.set_xlabel("P&L (option on underlying at 100)"); ax.set_ylabel("Density"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/pnl_distributions.png", dpi=150); plt.close(fig)

# 3) mean P&L vs RV - IV, with theory C(RV) - C(IV)
iv = 0.5; diffs = np.linspace(-0.3, 0.3, 13); m, lo, hi = [], [], []
for d in diffs:
    p = simulate_delta_hedge_paths_mc(S0=S0, K=K, r=r, sigma_real=iv + d, sigma_imp=iv, T=T,
                                      n_steps=365, n_paths=5000, seed=5)["pnl"]
    m.append(p.mean()); lo.append(np.percentile(p, 5)); hi.append(np.percentile(p, 95))
theory = [float(np.atleast_1d(call_price_bs(S0, K, r, iv + d, T))[0] - np.atleast_1d(call_price_bs(S0, K, r, iv, T))[0]) for d in diffs]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.fill_between(diffs * 100, lo, hi, color=BLUE, alpha=0.15, lw=0, label="5th–95th percentile")
ax.plot(diffs * 100, theory, color=ORANGE, lw=2, ls="--", label="Theory: C(RV) − C(IV)")
ax.plot(diffs * 100, m, color=BLUE, marker="o", ms=5, label="Simulated mean")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_title("The edge is realized minus implied volatility (IV = 50%)")
ax.set_xlabel("Realized vol − implied vol (vol points)"); ax.set_ylabel("P&L"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/pnl_vs_vol_spread.png", dpi=150); plt.close(fig)
print("theory vs sim max abs diff", np.max(np.abs(np.array(theory) - np.array(m))))

# 4) hedge frequency: dispersion shrinks ~ 1/sqrt(n)
steps = [12, 52, 120, 365, 1000, 2500]; sd = []
for n in steps:
    p = simulate_delta_hedge_paths_mc(S0=S0, K=K, r=r, sigma_real=0.2, sigma_imp=0.2, T=T,
                                      n_steps=n, n_paths=8000, seed=3)["pnl"]
    sd.append(p.std())
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(steps, sd, color=BLUE, marker="o", ms=5, label="Simulated std of P&L")
ax.plot(steps, sd[0] * np.sqrt(steps[0] / np.array(steps)), color=ORANGE, ls="--", label="∝ 1/√n")
ax.set_xscale("log"); ax.set_title("Hedging error vs re-hedge frequency (RV = IV = 20%, no costs)")
ax.set_xlabel("Number of re-hedges over the option's life (log scale)"); ax.set_ylabel("Std of P&L"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/hedge_frequency.png", dpi=150); plt.close(fig)
print(dict(zip(steps, np.round(sd, 3))))
