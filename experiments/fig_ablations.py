"""Ablation figure: 10k training molecules of the final dataset, final settings, 3 seeds per variant.

Dot plot: rows = variants (labelled on the axis), panels = test sets, x = S1-S6 excitation-energy MAE (eV; the
states every variant supervises). Small dots = individual seeds, large marker = seed mean; the thin vertical line
marks the LEH mean for comparison. One hue per head family (LEH blue, sum orange, softmin aqua, mean yellow;
validated row order). Inputs: experiments/wb97/runs_abl/*/predictions.pkl. Output: paper/figs/ablations.{pdf,png,csv}.
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa

fs.setup()
VARIANTS = [("leh", "LEH", "leh"), ("leh_diag", "LEH, no pair coupling", "leh"), ("leh_sup6", "LEH, 6 supervised states", "leh"),
            ("leh_noexc", "LEH, no S$_1$-sampled data", "leh"), ("sum", "Direct-sum", "direct_sum"),
            ("softmin", "Direct-softmin", "softmin"), ("mean", "Direct-mean", "direct_mean")]
SETS = [("test_random_offeq", "Random\nthermal"), ("test_scaffold_offeq", "Scaffold\nthermal"),
        ("test_random_exc", "Random\nS$_1$-sampled"), ("test_large_offeq", "Larger\nthermal"),
        ("test_large_exc", "Larger\nS$_1$-sampled")]


def s16(P):
    gp = np.array([np.asarray(x["gaps_pred"], float)[:6] for x in P]); gr = np.array([np.asarray(x["gaps_ref"], float)[:6] for x in P])
    m = np.isfinite(gp) & np.isfinite(gr)
    return float(np.mean(np.abs(gp[m] - gr[m])))


vals = {}
for key, _, _ in VARIANTS:
    for s in range(3):
        P = pickle.load(open(ROOT / f"experiments/wb97/runs_abl/abl_{key}_s{s}/predictions.pkl", "rb"))
        for st, _ in SETS:
            vals.setdefault((key, st), []).append(s16(P[st]))
fig, axes = plt.subplots(1, len(SETS), figsize=(7.0, 2.6), sharey=True)
y = np.arange(len(VARIANTS))[::-1]
rows = []
for ax, (st, title) in zip(axes, SETS):
    ref = np.mean(vals[("leh", st)])
    ax.axvline(ref, color=fs.AXIS, lw=0.8, zorder=0)
    for yy, (key, lab, col) in zip(y, VARIANTS):
        v = np.array(vals[(key, st)])
        ax.scatter(v, [yy] * len(v), s=9, color=fs.MODEL_COLOR[col], alpha=0.55, linewidths=0, zorder=2)
        ax.scatter([v.mean()], [yy], s=34, color=fs.MODEL_COLOR[col], edgecolors="white", linewidths=0.8, zorder=3)
        rows.append(dict(set=st, variant=lab.replace("$_1$", "1"), mean=float(v.mean()), std=float(v.std()),
                         seeds=";".join(f"{x:.4f}" for x in v)))
    ax.text(0.97, 0.99, title, transform=ax.transAxes, fontsize=6, color=fs.INK2, va="top", ha="right")
    lo, hi = ax.get_xlim(); ax.set_xlim(lo - 0.01, hi + 0.01)
axes[0].set_yticks(y); axes[0].set_yticklabels([lab for _, lab, _ in VARIANTS])
fig.supxlabel("Mean absolute error of S$_1$–S$_6$ energies (eV)", fontsize=7, color=fs.INK, y=0.02)
fig.tight_layout()
fs.panel_labels(fig, list(zip(axes, "abcde")))
out = ROOT / "paper/figs/ablations"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
with open(out.with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written", out)
