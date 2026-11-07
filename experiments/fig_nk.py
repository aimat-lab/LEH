"""Fixed-budget sampling test: N molecules x K geometries per molecule at constant N*K (experiments/wb97/
make_nk_splits.py, runs_nk; LEH, ablation settings, 3 seeds; thermal geometries only).

Panels: S1-S6 excitation-energy MAE on random, scaffold and larger off-equilibrium test molecules, and S1 force MAE
(random off-eq.), vs K (geometries per molecule). One line per budget (10k and 20k training geometries); markers =
seed mean, bars = seed std. Input: experiments/wb97/summary_runs_nk.json. Output: paper/figs/nk.{pdf,png,csv}.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa

fs.setup()
S = json.load(open(ROOT / "experiments/wb97/summary_runs_nk.json"))
BUD = [("G10k", "10k geometries", [1, 2, 3], dict(linestyle=(0, (4, 2)), marker="s", alpha=0.8)),
       ("G20k", "20k geometries", [1, 2], dict(linestyle="-", marker="o"))]
PAN = [("test_random_offeq", "S1-6 MAE", "a", "Mean absolute error of\nS$_1$–S$_6$ energies (eV)"),
       ("test_scaffold_offeq", "S1-6 MAE", "b", None),
       ("test_large_offeq", "S1-6 MAE", "c", None),
       ("test_random_offeq", "F_S1 MAE", "d", "Mean absolute error of\nS$_1$ forces (eV/Å)")]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3))
for ax in axes[1:3]:
    ax.sharey(axes[0])  # energy panels a-c on one y range
rows = []
for ax, (st, q, title, ylab) in zip(axes, PAN):
    for key, lab, Ks, sty in BUD:
        m = [S[f"nk_{key}_K{k}"][st][q][0] for k in Ks]
        e = [S[f"nk_{key}_K{k}"][st][q][1] for k in Ks]
        ax.errorbar(Ks, m, yerr=e, color=fs.MODEL_COLOR["leh"], lw=1.3, ms=4, capsize=2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.6, label=lab, **sty)
        rows += [dict(set=st, metric=q, budget=lab, K=k, molecules=int((10000 if key == "G10k" else 20000) / k), mean=a, std=b)
                 for k, a, b in zip(Ks, m, e)]
    ax.set_xticks([1, 2, 3]); ax.set_xlim(0.7, 3.3)
    ax.set_xlabel("Geometries per molecule")
    if ylab:
        ax.set_ylabel(ylab)
    lo, hi = ax.get_ylim(); pad = 0.25 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13), handlelength=2.4)
fig.tight_layout()
fs.panel_labels(fig, list(zip(axes, "abcd")))
out = ROOT / "paper/figs/nk"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
with open(out.with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written", out)
