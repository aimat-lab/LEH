"""Size-dependence and parity figures of the full-set models (inputs from collect_size_parity.py on HoreKa2 plus the
local zero-shot OMNI-P2x predictions).

size.{pdf,png,csv}: S1-S3 MAE (log scale) vs heavy atoms per molecule. (a) off-equilibrium random (QCDGE-derived,
  <= 10) and larger (ZINC, 11-20) test molecules, training size range shaded; (b) S-free dyes and (c) ladders at
  equilibrium (bins of 3 heavy atoms; models never saw conjugated dyes or ladders). OMNI-P2x zero-shot is shown raw in all panels:
  the linear recalibration fitted on our validation set (omnip2x_analyze.py) lowers (a) to ~0.6 eV but does not transfer
  to dyes. Bins with < 20 molecules omitted.
Main text: LEH, softmin, X-MACE and the OMNI-P2x architecture from scratch (MS-ANI, the same model without the
level-of-theory input, only in the SI); SI: size_si (+ MS-ANI, OMNI-P2x pretrained raw, both fine-tuning protocols) and
parity_all (all).
parity.{pdf,png}: S1 excitation energy predicted vs TDDFT for larger molecules (off-eq.) and S-free dyes (eq.),
  density hexbins (single-hue sequential ramp, log counts, one colour scale per row), MAE / R2 over all points in each panel title (also in parity.csv); each row's axes
  span the 0.5-99.5 % range of the TDDFT reference (+8 %), identical for all models of the row; the share of points
  outside the axes is stated in the panel (they are included in MAE / R2). parity_all.*: the same for all models (SI).
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa

fs.setup()
SP = json.load(open(ROOT / "experiments/wb97/size_parity.json"))
PAR = dict(np.load(ROOT / "experiments/wb97/parity.npz"))
ORDER = ["leh", "leh_L2", "softmin", "xmace", "omnift", "omniftA", "vecmsani", "msani"]
XM = SP.get("xmace_size", "30k")
lab = lambda m: fs.LABEL[m] + (f" ({XM})" if m == "xmace" and XM != "all" else "")

# zero-shot OMNI-P2x (raw; the small-molecule recalibration does not transfer to dyes, see omnip2x_analyze.py)
ZS = pickle.load(open(ROOT / "experiments/wb97/runs_zeroshot/omnip2x/predictions.pkl", "rb"))


def zs_rows(key):
    P = ZS.get(key, [])
    gp = np.array([np.asarray(x["gaps_pred"], float)[:3] for x in P]); gr = np.array([np.asarray(x["gaps_ref"], float)[:3] for x in P])
    return gp, gr, np.array([int(x["n_heavy"]) for x in P])


def binned(per_seed, sname, width=1, lo=None, hi=None, minn=20):
    """Mean over seeds of per-bin S1-S3 MAE, rebinned to `width` heavy atoms (n-weighted)."""
    acc = {}
    for res in per_seed:
        for h, v in res.get(sname, {}).get("bins", {}).items():
            b = int(h) // width * width
            a = acc.setdefault(b, [0.0, 0])
            a[0] += v["s13"] * v["n"]; a[1] += v["n"]
    xs = sorted(b for b, (s, n) in acc.items() if n / len(per_seed) >= minn and (lo is None or b >= lo) and (hi is None or b <= hi))
    return np.array(xs) + (width - 1) / 2, np.array([acc[b][0] / acc[b][1] for b in xs])


from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa


def make_size(order, with_zs, name):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    ax.axvspan(0.5, 10.5, color=fs.GRID, alpha=0.5, lw=0)
    ax.text(5.5, 0.31, "training\nsize range", ha="center", va="center", fontsize=6, color=fs.INK2, transform=ax.get_xaxis_transform())
    for m in order:
        if m not in SP["models"]:
            continue
        st = dict(fs.STYLE.get(m, {}))
        for k, sname in enumerate(["random_offeq", "larger_offeq"]):
            x, y = binned(SP["models"][m], sname)
            ax.plot(x, y, color=fs.MODEL_COLOR[m], lw=1.4, ms=3.5, markeredgecolor="white", markeredgewidth=0.6,
                    label=lab(m) if k == 0 else None, **st)
            rows += [dict(panel="a", set=sname, model=lab(m), heavy_atoms=float(a), s13_mae=float(b)) for a, b in zip(x, y)]
    for k, zkey in enumerate(["test_random_offeq", "test_large_offeq"] if with_zs else []):
        gp, gr, nh = zs_rows(zkey)
        xs = [v for v in sorted(set(nh)) if (nh == v).sum() >= 20]
        y = [np.nanmean(np.abs(gp[nh == v] - gr[nh == v])) for v in xs]
        ax.plot(xs, y, color=fs.MODEL_COLOR["omni_zeroshot"], lw=1.2, linestyle=(0, (1, 2)))
        rows += [dict(panel="a", set=zkey, model=fs.LABEL["omni_zeroshot"], heavy_atoms=float(v), s13_mae=float(yy)) for v, yy in zip(xs, y)]
    ax.set_xlabel("Heavy atoms per molecule"); ax.set_ylabel("Mean absolute error of\nS$_1$\u2013S$_3$ energies (eV)")
    ax.set_xlim(0.5, 20.5)
    for ax, (sname, key, title) in zip(axes[1:], [("dyes_eq", "test_dyeall_noS_eq", "b  Dyes (S-free), equilibrium"),
                                                  ("ladders_eq", "test_lad_noS_eq", "c  Ladders (S-free), equilibrium")]):
        for m in order:
            if m not in SP["models"]:
                continue
            x, y = binned(SP["models"][m], sname, width=3, lo=6)
            ax.plot(x, y, color=fs.MODEL_COLOR[m], lw=1.3, ms=3.5, markeredgecolor="white", markeredgewidth=0.6, **dict(fs.STYLE.get(m, {})))
            rows += [dict(panel=title[0], set=sname, model=lab(m), heavy_atoms=float(a), s13_mae=float(b)) for a, b in zip(x, y)]
        gp, gr, nh = zs_rows(key)
        if with_zs and len(nh):
            b = nh // 3 * 3
            xs = [v for v in sorted(set(b)) if (b == v).sum() >= 20 and v >= 6]
            y = [np.nanmean(np.abs(gp[b == v] - gr[b == v])) for v in xs]
            ax.plot(np.array(xs) + 1, y, color=fs.MODEL_COLOR["omni_zeroshot"], lw=1.2, linestyle=(0, (1, 2)),
                    label=fs.LABEL["omni_zeroshot"] + ", raw" if title[0] == "b" else None)
            rows += [dict(panel=title[0], set=sname, model=fs.LABEL["omni_zeroshot"], heavy_atoms=float(v + 1), s13_mae=float(yy)) for v, yy in zip(xs, y)]
        ax.set_xlabel("Heavy atoms per molecule")
    for ax in axes:
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.yaxis.set_major_locator(FixedLocator([t for t in [0.1, 0.15, 0.2, 0.3, 0.5, 1, 2] if lo <= t <= hi]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.12), handlelength=2.4, columnspacing=1.0,
               fontsize=6)
    fig.tight_layout()
    fs.panel_labels(fig, list(zip(axes, "abc")))
    out = ROOT / f"paper/figs/{name}"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


# main text: without the pretrained / fine-tuned OMNI-P2x (SI: size_si with everything)
make_size([m for m in ORDER if m not in ("omnift", "omniftA", "msani")], False, "size")
make_size(ORDER, True, "size_si")


# parity
cmap = LinearSegmentedColormap.from_list("ink", ["#d6d4cc", fs.INK])
SETP = [("larger_offeq", "test_large_offeq", "Larger molecules, off-eq."), ("dyes_eq", "test_dyeall_noS_eq", "Dyes (S-free), equilibrium")]
from matplotlib.ticker import MultipleLocator  # noqa
plab = lambda m: fs.SHORT.get(m, lab(m))


def parity_fig(models, name, ncols):
    PM = [m for m in models if m == "omni_zeroshot" or f"{m}|larger_offeq|pred" in PAR]
    ncols = min(ncols, len(PM)); nr = -(-len(PM) // ncols)
    fig, axes = plt.subplots(len(SETP) * nr, ncols, figsize=(min(7.0, 1.45 * ncols + 1.2), (1.75 * nr) * len(SETP) + 0.5),
                             squeeze=False, layout="constrained")
    pr_rows = []
    for r, (sname, zkey, stitle) in enumerate(SETP):
        _, grz, _ = zs_rows(zkey)  # identical reference set for every model -> one axis range per set
        lo, hi = np.nanpercentile(grz[:, 0], [0.5, 99.5]); pad = 0.08 * (hi - lo); lo, hi = lo - pad, hi + pad
        step = 2 if hi - lo > 5 else 1
        hbs, block = [], axes[r * nr:(r + 1) * nr]
        for ax in block.ravel()[len(PM):]:
            ax.set_visible(False)  # unused grid cells of the last row of a block
        for c, m in enumerate(PM):
            ax = block[c // ncols, c % ncols]
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            for a in (ax.xaxis, ax.yaxis):
                a.set_major_locator(MultipleLocator(step))
            if c % ncols == 0:
                ax.set_ylabel("Predicted S$_1$ (eV)", fontsize=6.5)
            else:
                ax.tick_params(labelleft=False)
            if r == len(SETP) - 1 and c + ncols >= len(PM):
                ax.set_xlabel("TD-DFT S$_1$ (eV)")
            if m == "omni_zeroshot":
                gp, gr, _ = zs_rows(zkey); p, t = gp[:, 0], gr[:, 0]
            else:
                p, t = PAR.get(f"{m}|{sname}|pred", np.array([])), PAR.get(f"{m}|{sname}|ref", np.array([]))
            ok = np.isfinite(p) & np.isfinite(t)
            ax.plot([lo, hi], [lo, hi], color=fs.MODEL_COLOR.get(m, fs.INK), lw=1.0, zorder=3)
            if ok.sum() < 10:
                ax.text(0.04, 0.96, plab(m), transform=ax.transAxes, fontsize=5.5, color=fs.INK, va="top")
                ax.text(0.5, 0.8, "no predictions", transform=ax.transAxes, ha="center", va="center", fontsize=6.5, color=fs.MUTED)
                continue
            p, t = p[ok], t[ok]
            hb = ax.hexbin(t, p, gridsize=30, extent=(lo, hi, lo, hi), cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0)
            hbs.append(hb)
            mae = np.mean(np.abs(p - t)); r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
            outside = np.mean((p < lo) | (p > hi) | (t < lo) | (t > hi))
            stat = f"MAE {mae:.2f} eV, R$^2$ {r2:.2f}" + (f"\n{100 * outside:.0f} % of points off-axes" if outside >= 0.005 else "\n")
            ax.text(0.04, 0.96, f"{plab(m)}\n{stat.rstrip()}", transform=ax.transAxes, fontsize=5.5, color=fs.INK, va="top",
                    linespacing=1.15)
            pr_rows.append(dict(set=sname, model=plab(m), n=int(ok.sum()), mae_eV=float(mae), r2=float(r2),
                                frac_outside_axes=float(outside), axis_lo=float(lo), axis_hi=float(hi)))
        if hbs:  # one log norm per set so grey levels compare across models
            norm = LogNorm(1, max(h.get_array().max() for h in hbs))
            for h in hbs:
                h.set_norm(norm)
            cb = fig.colorbar(hbs[0], ax=list(block.ravel()), shrink=0.7 if nr == 1 else 0.5, aspect=14, pad=0.02)
            cb.set_label("Geometries per bin", fontsize=6.5); cb.ax.tick_params(labelsize=6)
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    fs.panel_labels(fig, [(axes[k * nr, 0], "ab"[k]) for k in range(len(SETP))])
    out = ROOT / f"paper/figs/{name}"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pr_rows[0])); w.writeheader(); w.writerows(pr_rows)


parity_fig(["leh", "softmin", "xmace", "vecmsani"], "parity", ncols=4)
parity_fig(["leh", "leh_L2", "softmin", "xmace", "msani", "vecmsani", "omnift", "omniftA", "omni_zeroshot"], "parity_all",
           ncols=3)
print("written size and parity figures")
