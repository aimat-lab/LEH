"""Paper figures from experiment results. Each figure is skipped when its inputs do not exist yet.

python experiments/figures.py [--exp EXPERIMENTS_DIR] [--out paper/figs]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa

plt = fs.plt
MAIN = ["latent_k1", "direct_sum", "direct_softmin", "direct_mean"]


def seed_stats(res, name, get):
    """Mean and (min, max) over a run and its seed replicas name_s1, name_s2, ..."""
    vals = [get(res[k]) for k in res if k == name or (k.startswith(name + "_s") and k[len(name) + 2:].isdigit())]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def shade_extrapolation(ax, x0, label="held-out sizes"):
    ax.axvspan(x0, ax.get_xlim()[1], color="#f0efec", zorder=0, lw=0)
    ax.text(x0 + 0.15, 0.03, label, va="bottom", ha="left", fontsize=6.5, color=fs.INK2,
            transform=ax.get_xaxis_transform())


def fig_qcdge_size(exp, out):
    f = exp / "qcdge/results.json"
    if not f.exists():
        return print("skip qcdge size: no results")
    res = json.load(open(f))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, key, title in [(axs[0], "mae_S1", "$S_1$"), (axs[1], "mae_sup", "$S_1$–$S_6$ (mean)")]:
        for m in MAIN:
            if m not in res:
                continue
            xs, ys, lo, hi = [], [], [], []
            for size, split in [(s, "val_by_size") for s in range(5, 9)] + [(9, "test_9_by_size"), (10, "test_10_by_size")]:
                def get(r, split=split, size=size):
                    return r.get(split, {}).get(str(size), {}).get(key)
                mu, a, b = seed_stats(res, m, get)
                xs.append(size); ys.append(mu); lo.append(mu - a); hi.append(b - mu)
            fs.line(ax, xs, ys, m)
            if np.any(np.array(hi) > 0):
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=fs.MODEL_COLOR[m], elinewidth=0.8, capsize=0)
        ax.set_xlabel("heavy atoms"); ax.set_ylabel("MAE (eV)"); ax.set_title(title, loc="left", color=fs.INK)
        ax.set_xlim(4.6, 10.4); ax.set_ylim(bottom=0)
        shade_extrapolation(ax, 8.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.22, wspace=0.3)
    fig.savefig(out / "qcdge_size.pdf"); plt.close(fig)
    print("wrote qcdge_size.pdf")


def fig_fragments(exp, out):
    f = exp / "qcdge/results.json"
    if not f.exists():
        return print("skip fragments: no results")
    res = json.load(open(f))
    models = [m for m in MAIN + ["latent_diag"] if m in res and "fragments" in res[m]]
    if not models:
        return print("skip fragments: no fragment results")
    kinds = ["AA", "AB", "AAB"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    w = 0.6 / len(models)
    for j, m in enumerate(models):
        y = [max(np.mean(res[m]["fragments"][k]["mae_vs_union_ref"]), 1e-6) for k in kinds]
        fs.line(ax, np.arange(3) + (j - (len(models) - 1) / 2) * w, y, m, ls="none", markersize=6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["A+A", "A+B", "A+A+B"]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("MAE vs. union of\nmonomer references (eV)"); ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.savefig(out / "fragments.pdf"); plt.close(fig)
    print("wrote fragments.pdf")


def fig_wb97_size(exp, out):
    f = exp / "wb97/results.json"
    if not f.exists():
        return print("skip wb97 size: no results")
    res = json.load(open(f))
    panels = [("mae_S1", "$S_1$ excitation energy", "MAE (eV)"), ("mae_sup", "$S_1$–$S_{10}$ (mean)", "MAE (eV)"),
              ("fS1_mae", "$S_1$ forces", "MAE (eV/Å)")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, (key, title, yl) in zip(axs, panels):
        for m in MAIN:
            if m not in res:
                continue
            pts = []
            for split in ["test_random_offeq_by_size", "test_random_by_size", "test_large_by_size"]:
                for s, v in res[m].get(split, {}).items():
                    if split == "test_random_by_size" and "test_random_offeq_by_size" in res[m]:
                        continue
                    if v.get("n", 0) >= 5 and v.get(key) is not None:
                        pts.append((int(s), v[key]))
            pts.sort()
            if pts:
                fs.line(ax, [p[0] for p in pts], [p[1] for p in pts], m, markersize=3.5)
        ax.set_xlabel("heavy atoms"); ax.set_ylabel(yl); ax.set_title(title, loc="left", color=fs.INK)
        ax.set_ylim(bottom=0)
        shade_extrapolation(ax, 10.5, "ZINC test")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.22, wspace=0.35)
    fig.savefig(out / "wb97_size.pdf"); plt.close(fig)
    print("wrote wb97_size.pdf")


def fig_scans(exp, out, models=("latent_k1", "direct_softmin", "direct_sum")):
    rd = exp / "md/results"
    have = [m for m in models if (rd / f"{m}.pkl").exists()]
    if not have:
        return print("skip scans: no results")
    R = {m: pickle.load(open(rd / f"{m}.pkl", "rb")).get("scans", {}) for m in have}
    scans = [s for s in ["stilbene", "biphenyl", "acroleinCO"] if any(s in R[m] for m in have)]
    if not scans:
        return print("skip scans: empty")
    xl = {"stilbene": "C=C torsion (deg)", "biphenyl": "inter-ring torsion (deg)", "acroleinCO": "r(C=O) (Å)"}
    fig, axs = plt.subplots(len(have), len(scans), figsize=(2.3 * len(scans), 1.9 * len(have)), squeeze=False,
                            sharex="col")
    for i, m in enumerate(have):
        for j, s in enumerate(scans):
            ax = axs[i, j]
            d = R[m].get(s)
            if d is None:
                continue
            x = d["coord"]
            e0r = d["E0_ref"] - d["E0_ref"].min()
            e0p = d["E0_pred"] - d["E0_ref"].min()
            for n, (yr, yp) in enumerate([(e0r, e0p)] + [(e0r + d["gaps_ref"][:, k], e0p + d["gaps_pred"][:, k])
                                                          for k in range(2)]):
                ax.plot(x, yr, color=fs.INK, lw=0.9, ls="--", marker=None, label="TDDFT" if n == 0 else None)
                fs.line(ax, x, yp, m, markersize=2.5, label=None)
            if i == len(have) - 1:
                ax.set_xlabel(xl[s])
            if j == 0:
                ax.set_ylabel(f"{fs.LABEL[m]}\nenergy (eV)")
            if i == 0:
                ax.set_title({"stilbene": "Stilbene", "biphenyl": "Biphenyl", "acroleinCO": "Acrolein C=O"}[s]
                             + ": $S_0$, $S_1$, $S_2$", loc="left", color=fs.INK)
    axs[0, 0].legend(loc="best", fontsize=6)
    fig.savefig(out / "scans.pdf"); plt.close(fig)
    print("wrote scans.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(ROOT / "experiments"))
    ap.add_argument("--out", default=str(ROOT / "paper/figs"))
    a = ap.parse_args()
    fs.setup()
    exp, out = Path(a.exp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in [fig_qcdge_size, fig_fragments, fig_wb97_size, fig_scans]:
        fn(exp, out)
