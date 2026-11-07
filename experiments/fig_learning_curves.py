"""Learning-curve figure: all models on identical nested training subsets of the final dataset.

2x2 panels: top = excitation energies S1-S3 (MAE, eV; the range shared by every model incl. X-MACE), bottom = S1
forces (MAE, eV/A); left = random test molecules, off-equilibrium (in distribution), right = larger molecules
(11-20 heavy atoms, off-equilibrium; size extrapolation). Log-log axes, mean over seeds with min-max bars.
The pretrained OMNI-P2x (zero-shot, best of raw / validation-recalibrated per set) is a dotted horizontal line:
it uses none of our training data (its forces are raw). OMNI-P2x fine-tuned: "authors' protocol" = MLatom's OMNI-P2x
fine-tuning defaults (100 epochs, layers 0 and 4 of every element network frozen, gap weight 0.01, per member, mean
of the 3 members); "adapted" = the same with all layers trainable and gap weight 1. All lines except the pretrained
one are raw predictions (no recalibration). Inputs: experiments/wb97/learning_curves.json (learning_curves.py) and
experiments/wb97/runs_zeroshot/omnip2x/predictions.pkl. Output: paper/figs/learning_curves.{pdf,png,csv} (main text:
our models, X-MACE and the OMNI-P2x architecture trained from scratch; MS-ANI, the same architecture without the
level-of-theory input, is indistinguishable and only named in the caption) and learning_curves_si.* (SI: additionally
MS-ANI, OMNI-P2x pretrained and both fine-tuning protocols).
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter, NullLocator  # noqa

fs.setup()
R = json.load(open(ROOT / "experiments/wb97/learning_curves.json"))
SIZES = ["1k", "3k", "10k", "30k", "all"]
_sub = json.load(open(ROOT / "experiments/wb97/subsets_100k_s1.json"))["subsets"]
NGEO = {z: _sub[z]["train_entries"] for z in SIZES}      # training geometries (x axis)
NMOL = {z: _sub[z]["molecules"] for z in SIZES}          # training molecules (CSV / caption)
MODELS = ["leh", "leh_L2", "softmin", "xmace", "omnift", "omniftA", "vecmsani", "msani"]
MODELS_MAIN = ["leh", "leh_L2", "softmin", "xmace", "vecmsani"]  # MS-ANI = same model as vecmsani (caption; SI)
SETS = [("test_random_offeq", "Random molecules, off-equilibrium"),
        ("test_large_offeq", "Larger molecules (11–20 heavy atoms), off-equilibrium")]
METRICS = [("S1-3 MAE", "Mean absolute error of\nS$_1$–S$_3$ energies (eV)"), ("F_S1 MAE", "Mean absolute error of\nS$_1$ forces (eV/Å)")]


def zero_shot():
    """OMNI-P2x pretrained: S1-S3 MAE (best of raw / shift / linear recalibration fitted on val) and raw S1 forces."""
    f = ROOT / "experiments/wb97/runs_zeroshot/omnip2x/predictions.pkl"
    if not f.exists():
        return {}
    P = pickle.load(open(f, "rb"))
    g = lambda p: (np.array([np.asarray(x["gaps_pred"], float)[:3] for x in p]), np.array([np.asarray(x["gaps_ref"], float)[:3] for x in p]))
    gpv, grv = g(P["val"])
    shift = np.nanmean(grv - gpv, 0)
    lin = np.array([np.polyfit(gpv[:, n], grv[:, n], 1) for n in range(3)])
    out = {}
    for s, _ in SETS:
        gp, gr = g(P[s])
        maes = [np.nanmean(np.abs(x - gr)) for x in (gp, gp + shift, gp * lin[:, 0] + lin[:, 1])]
        fs1 = np.nanmean(np.concatenate([np.abs(x["Fex_pred"][:, 0] - x["Fex_ref"][:, 0]).ravel() for x in P[s] if x.get("Fex_ref") is not None]))
        out[s] = {"S1-3 MAE": float(min(maes)), "F_S1 MAE": float(fs1)}
    return out


ZS = zero_shot()
ZS_LABEL = "OMNI-P2x pretrained (energies recal.)"  # zero-shot; energies recalibrated on val.; all other lines raw
def make(models, with_zs, name):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey="row")  # same y range side by side
    rows = []
    for c, (s, title) in enumerate(SETS):
        for r, (met, ylab) in enumerate(METRICS):
            ax = axes[r, c]
            for m in models:
                pts = [(NGEO[z], R.get(m, {}).get(z, {}).get(s, {}).get(met)) for z in SIZES]
                pts = [(x, v) for x, v in pts if v]
                if not pts:
                    continue
                x = np.array([p[0] for p in pts]); y = np.array([p[1][0] for p in pts]); sd = np.array([p[1][1] for p in pts])
                st = dict(fs.STYLE.get(m, {}))
                ax.errorbar(x, y, yerr=sd, color=fs.MODEL_COLOR[m], label=fs.LABEL[m], lw=1.5, ms=4.5, capsize=0,
                            markeredgecolor="white", markeredgewidth=0.8, **st)
                inv = {v: k for k, v in NGEO.items()}
                rows += [dict(set=s, metric=met, model=fs.LABEL[m], n_train_geometries=int(xx), n_train_molecules=NMOL[inv[int(xx)]],
                              mean=float(yy), std=float(ss)) for xx, yy, ss in zip(x, y, sd)]
            if with_zs and s in ZS:
                ax.axhline(ZS[s][met], color=fs.MODEL_COLOR["omni_zeroshot"], lw=1.2, linestyle=(0, (1, 2)),
                           label=ZS_LABEL)
                rows.append(dict(set=s, metric=met, model=fs.LABEL["omni_zeroshot"], n_train_geometries=0,
                                 n_train_molecules=0, mean=ZS[s][met], std=0.0))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1.5e3, 2.5e5)
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", which="minor", length=2)
            ticks = [0.15, 0.2, 0.3, 0.5, 1, 2] if r == 0 else [0.15, 0.2, 0.3, 0.5, 1]
            ax.tick_params(axis="y", labelleft=c == 0)
            lo, hi = ax.get_ylim()
            ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi * 1.001]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(NullLocator())
            if r == 1:
                ax.set_xlabel("Number of training geometries")
            if c == 0:
                ax.set_ylabel(ylab)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    order = [fs.LABEL[m] for m in models] + [ZS_LABEL]
    hl = sorted(zip(handles, labels), key=lambda t: order.index(t[1]) if t[1] in order else 99)
    handles, labels = [h for h, _ in hl], [l for _, l in hl]
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07), handlelength=2.6, columnspacing=1.0,
               fontsize=6)
    fig.tight_layout(); fig.subplots_adjust(hspace=0.22)
    fs.panel_labels(fig, [(axes[0, 0], "a"), (axes[0, 1], "b"), (axes[1, 0], "c"), (axes[1, 1], "d")])
    out = ROOT / f"paper/figs/{name}"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", out.with_suffix(".pdf"), out.with_suffix(".png"), out.with_suffix(".csv"))


# main text: our models and the from-scratch baselines; SI: additionally the pretrained and fine-tuned OMNI-P2x
make(MODELS_MAIN, False, "learning_curves")
make(MODELS, True, "learning_curves_si")
