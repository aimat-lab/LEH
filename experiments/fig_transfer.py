"""Transferability figure (main Fig. 4) and its SI companion (error ratios, S1-S3 and S1 alone).

Main (paper/figs/transfer.*):
(a) S1-S3 MAE of the full-data models on test sets ordered by their distance from the training distribution (training
    molecules: <= 10 heavy atoms, one chromophore, equilibrium + thermal + S1-sampled geometries): random and unseen-
    scaffold molecules (off-eq.), S1-sampled geometries, larger molecules (11-20 heavy atoms, off-eq. and S1-sampled),
    conjugated dyes and ladder molecules (up to 30 heavy atoms, equilibrium, never trained on) and non-bonded dimers of
    two test molecules (vs TD-wB97X-D dimer spectra, 2.5-15 A; details in the separability figure).
(b, c) S1-S3 MAE vs size for dyes and ladders (bins of three heavy atoms, >= 20 molecules per bin).
SI (paper/figs/transfer_si.*): error of Direct-softmin, X-MACE and the OMNI-P2x architecture relative to LEH on the
    same test sets (ratio of MAEs; > 1: LEH better), (a) S1-S3, (b) S1 alone.
Models: LEH, Direct-softmin (same encoder and training), X-MACE, OMNI-P2x architecture trained from scratch.
Inputs: learning_curves.json, size_parity.json, frag_tddft/results_pairs.json and the full-set predictions on the
fragment pairs (hpc/hk_frag_full.sh; falls back to the 10k models with a note if missing).
Output: paper/figs/transfer.{pdf,png,csv}, paper/figs/transfer_si.{pdf,png,csv}.
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
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa

fs.setup()
W = ROOT / "experiments/wb97"
L = json.load(open(W / "learning_curves.json"))
SP = json.load(open(W / "size_parity.json"))
TD = json.load(open(ROOT / "experiments/frag_tddft/results_pairs.json"))
G = pickle.load(open(ROOT / "data/frag_pairs.pkl", "rb")); GID = {m["id"]: m for m in G}
MODELS = ["leh", "softmin", "xmace", "vecmsani", "omnift"]
LBL = {**fs.LABEL, "omnift": "OMNI-P2x, fine-tuned"}
OPEN = set()  # (fine-tuned OMNI-P2x: diamond marker, figstyle.STYLE)
OPTIONAL = {"omnift"}  # full-set dimer predictions of the fine-tuned model (hpc/hk_frag_omnift_full.sh)
CATS = [("test_random_offeq", "Random"), ("test_scaffold_offeq", "Unseen\nscaffolds"), ("test_random_exc", "S$_1$-sampled"),
        ("test_large_offeq", "Larger"), ("test_large_exc", "Larger,\nS$_1$-sampled"), ("dyes_eq", "Dyes"),
        ("ladders_eq", "Ladders"), ("dimers", "Dimers")]


def frag_preds(full=True):
    """{model: [{id: gaps} per seed]} on the fragment pairs; full-set models (both seeds where available) or the 10k models."""
    out = {}
    if full:
        src = {"leh": [W / f"runs_main/main_leh_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
               "softmin": [W / f"runs_main/main_softmin_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
               "xmace": [ROOT / "experiments/fragments/full_xmace/predictions.pkl"],
               "vecmsani": [ROOT / "experiments/fragments/full_vecmsani/predictions.pkl"],
               "omnift": [ROOT / "experiments/fragments/full_omnift/predictions.pkl"]}
    else:
        src = {"xmace": [ROOT / "experiments/fragments/xmace128_sub10k/predictions.pkl"],
               "vecmsani": [ROOT / "experiments/fragments/sub10k_vecmsani/predictions.pkl"]}
        H = pickle.load(open(ROOT / "experiments/fragments/heads_cache.pkl", "rb"))
        out = {"leh": [H["leh"]], "softmin": [H["softmin"]]}
    for m, fs_ in src.items():
        if not fs_[0].exists():
            if m in OPTIONAL:
                print(f"note: {m} dimer predictions missing ({fs_[0]})"); continue
            return None
        out[m] = []
        for f in fs_:
            if f.exists():
                P = pickle.load(open(f, "rb"))
                out[m].append({x["id"]: np.asarray(x["gaps_pred"], float) for k in ("test_frag_mono", "test_frag_dimer") for x in P.get(k, [])})
    return out


FP = frag_preds(True); FULL = FP is not None
if not FULL:
    FP = frag_preds(False)


def dimer_err(m, n=3):
    """MAE (S1-Sn) of the model's dimer spectrum vs TD-wB97X-D, per distance (both pair kinds pooled; seed mean)."""
    per_seed = []
    for S in FP.get(m, []):
        by_d = {}
        for r in TD:
            g = S.get(r["id"])
            if g is not None:
                by_d.setdefault(r["d"], []).append(np.mean(np.abs(g[:n] - np.asarray(r["dimer"][:n]))))
        per_seed.append({d: float(np.mean(v)) for d, v in by_d.items()})
    ds = sorted(set.intersection(*[set(e) for e in per_seed])) if per_seed else []
    return {d: float(np.mean([e[d] for e in per_seed])) for d in ds}


def value(m, key, n=3):
    if key == "dimers":
        e = dimer_err(m, n)
        return float(np.mean(list(e.values()))) if e else np.nan
    if key in ("dyes_eq", "ladders_eq"):
        runs = SP["models"].get(m, [])
        return float(np.mean([r[key]["all"]["s13" if n == 3 else "s1"] for r in runs])) if runs else np.nan
    return L[m]["all"][key][f"S1-{n} MAE"][0]


V = {m: [value(m, k) for k, _ in CATS] for m in MODELS}
V1 = {m: [value(m, k, 1) for k, _ in CATS] for m in MODELS}
rows = [dict(panel="a", set=k, model=LBL[m], s13_mae=V[m][i], s1_mae=V1[m][i]) for m in MODELS for i, (k, _) in enumerate(CATS)]


def mstyle(m):
    return dict(markerfacecolor="white", markeredgecolor=fs.MODEL_COLOR[m], markeredgewidth=1.0) if m in OPEN else \
        dict(markeredgecolor="white", markeredgewidth=0.7)


def logticks(ax, ticks):
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())


def save(fig, name, rows):
    out = ROOT / f"paper/figs/{name}"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def category_axis(ax, labels=True):
    ax.set_xticks(x); ax.set_xlim(-0.6, len(CATS) - 0.4); ax.grid(False, axis="x")
    ax.axvspan(-0.6, 2.5, color=fs.GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xticklabels([c for _, c in CATS] if labels else [], fontsize=6.8)


# main figure
fig = plt.figure(figsize=(7.0, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.1], hspace=0.5, wspace=0.25, top=0.86)
axA = fig.add_subplot(gs[0, :])
x = np.arange(len(CATS)); off = {m: (j - 2) * 0.15 for j, m in enumerate(MODELS)}
for m in MODELS:
    st = dict(fs.STYLE.get(m, {})); mk = st.get("marker", "o") or "o"
    axA.plot(x + off[m], V[m], linestyle="none", marker=mk, ms=5.5, color=fs.MODEL_COLOR[m], label=LBL[m], zorder=3,
             **mstyle(m))
category_axis(axA)
axA.text(1.0, 0.97, "molecules similar to the training molecules\n(≤ 10 heavy atoms, one chromophore)", fontsize=6,
         color=fs.INK2, ha="center", va="top", transform=axA.get_xaxis_transform())
axA.set_yscale("log"); axA.set_ylabel("Mean absolute error of\nS$_1$–S$_3$ energies (eV)"); logticks(axA, [0.15, 0.2, 0.3, 0.5, 1, 2, 5])
hA, lA = axA.get_legend_handles_labels()
fig.legend(hA, lA, loc="upper center", ncol=5, fontsize=6, handletextpad=0.3, columnspacing=1.5, bbox_to_anchor=(0.5, 0.925),
           frameon=False)


def binned(m, sname, width=3, lo=6, minn=20):
    acc = {}
    for res in SP["models"].get(m, []):
        for h, v in res.get(sname, {}).get("bins", {}).items():
            b = int(h) // width * width
            a = acc.setdefault(b, [0.0, 0]); a[0] += v["s13"] * v["n"]; a[1] += v["n"]
    xs = sorted(b for b, (s, n) in acc.items() if n / max(len(SP["models"].get(m, [1])), 1) >= minn and b >= lo)
    return np.array(xs) + (width - 1) / 2, np.array([acc[b][0] / acc[b][1] for b in xs])


axBC = []
for j, (sname, title) in enumerate([("dyes_eq", "b"), ("ladders_eq", "c")]):
    ax = fig.add_subplot(gs[1, j], sharey=axBC[0] if axBC else None); axBC.append(ax)
    for m in MODELS:
        xx, yy = binned(m, sname)
        ax.plot(xx, yy, color=fs.MODEL_COLOR[m], lw=1.3, ms=3.2, **{**mstyle(m), "markeredgewidth": 0.5 if m not in OPEN else 0.8},
                **dict(fs.STYLE.get(m, {"marker": "o"})))
        rows += [dict(panel=title[0], set=sname, model=LBL[m], heavy_atoms=float(a), s13_mae=float(b)) for a, b in zip(xx, yy)]
    ax.set_yscale("log"); ax.set_xlabel("Heavy atoms per molecule")
    if j == 0:
        ax.set_ylabel("Mean absolute error of\nS$_1$–S$_3$ energies (eV)")
for ax in axBC:
    logticks(ax, [0.15, 0.2, 0.3, 0.5, 1, 2])
axBC[1].tick_params(axis="y", labelleft=False)
fs.panel_labels(fig, [(axA, "a"), (axBC[0], "b"), (axBC[1], "c")])
save(fig, "transfer", rows)

# SI: ratios to LEH, S1-S3 and S1 alone
fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True, gridspec_kw=dict(hspace=0.3))
rows_si = []
for ax, VV, what in ((axs[0], V, "S$_1$–S$_3$"), (axs[1], V1, "S$_1$")):
    for m in MODELS[1:]:
        st = dict(fs.STYLE.get(m, {})); mk = st.get("marker", "o") or "o"
        r = np.array(VV[m]) / np.array(VV["leh"])
        ax.plot(x + off[m], r, linestyle="none", marker=mk, ms=5.5, color=fs.MODEL_COLOR[m], label=LBL[m], zorder=3,
                **mstyle(m))
        rows_si += [dict(panel="a" if VV is V else "b", set=k, model=LBL[m], ratio_to_leh=float(v))
                    for (k, _), v in zip(CATS, r)]
    category_axis(ax, ax is axs[1])
    ax.set_yscale("log"); ax.axhline(1, color=fs.MODEL_COLOR["leh"], lw=1.0); logticks(ax, [0.8, 1, 1.5, 2, 3, 5, 10, 20, 40])
    ax.set_ylabel(f"Error relative to LEH,\n{what}")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.0), frameon=False)
fs.panel_labels(fig, [(axs[0], "a"), (axs[1], "b")])
save(fig, "transfer_si", rows_si)
print("written transfer, transfer_si", "| full-set dimers" if FULL else "| dimers from 10k models (full-set predictions missing)")
for m in MODELS:
    print(f"{m:9s} S1-3 " + " ".join(f"{v:.3f}" for v in V[m]) + " | S1 " + " ".join(f"{v:.3f}" for v in V1[m]))
