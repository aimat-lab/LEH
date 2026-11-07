"""Separability figure (main Fig. 5): two non-bonded molecules, compared with the union of the monomer spectra.

(a) Level-correlation diagram for two different molecules A and B at 10 A minimum interatomic distance (beyond
    r_c = 5 A). For TD-wB97X-D and each model: left, the union of the levels of A and B computed separately; right, the
    levels of A + B computed together (S1-S6; X-MACE predicts S1-S3 only). Union levels are coloured by the molecule
    they belong to; dimer levels of TDDFT and LEH by the union level they match (LEH reproduces the union of its own
    monomer predictions exactly), those of the pooled readouts in the model colour. Lines connect matched levels (TDDFT,
    LEH) or the n-th levels (pooled readouts). Levels closer than 30 meV are drawn side by side (degenerate pairs).
(b) The same for two identical molecules (A + A, A from panel a): every level must appear twice.
    Pairs: the AB and AA pairs of molecule pair 8, whose Direct-softmin S1-S3 dimer errors at 10 A are the median of the
    ten AB pairs and the fifth of the ten AA pairs; seed 0 of LEH and Direct-softmin.
(c) S1-S3 MAE of the dimer spectrum vs TD-wB97X-D as a function of the minimum interatomic distance (all 20 pairs,
    seed mean).
(d) Dimer MAE by state group (all pairs and distances; filled) and for the union of the model's monomer predictions
    (open). X-MACE predicts S1-S3 only.
Models: LEH, Direct-softmin (benchmark, full training set, two seeds), X-MACE, OMNI-P2x architecture from scratch.
Inputs: data/frag_pairs.pkl, experiments/frag_tddft/results_pairs.json, runs_main/main_*_all_s*/predictions_frag.pkl,
experiments/fragments/full_{xmace,vecmsani}/predictions.pkl.
Output: paper/figs/separability.{pdf,png,csv}; separability_numbers.json (for make_numbers.py).
"""
import csv
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.lines import Line2D  # noqa
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa
from rdkit import Chem  # noqa
from rdkit.Chem import AllChem, rdDetermineBonds  # noqa
from rdkit.Chem.Draw import rdMolDraw2D  # noqa

fs.setup()
W = ROOT / "experiments/wb97"
TD = json.load(open(ROOT / "experiments/frag_tddft/results_pairs.json"))
G = pickle.load(open(ROOT / "data/frag_pairs.pkl", "rb")); GID = {g["id"]: g for g in G}
MODELS = ["leh", "softmin", "xmace", "vecmsani"]
SRC = {"leh": [W / f"runs_main/main_leh_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
       "softmin": [W / f"runs_main/main_softmin_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
       "xmace": [ROOT / "experiments/fragments/full_xmace/predictions.pkl"],
       "vecmsani": [ROOT / "experiments/fragments/full_vecmsani/predictions.pkl"],
       "omnift": [ROOT / "experiments/fragments/full_omnift/predictions.pkl"]}  # panels c, d only (if present)
LBL = {**fs.LABEL, "omnift": "OMNI-P2x, fine-tuned"}
NAME = {"leh": "LEH", "softmin": "Direct-\nsoftmin", "xmace": "X-MACE", "vecmsani": "OMNI-P2x\narch."}
COL_A, COL_B = fs.SLOTS[1], fs.INK2
PAIR, DIST, NS = 8, 10.0, 6


def load(f):
    P = pickle.load(open(f, "rb"))
    return {x["id"]: np.asarray(x["gaps_pred"], float) for k in ("test_frag_mono", "test_frag_dimer") for x in P.get(k, [])}


FP = {m: [load(f) for f in fl if f.exists()] for m, fl in SRC.items()}
MODELS_CD = MODELS + (["omnift"] if FP["omnift"] else [])  # fine-tuned OMNI-P2x behaves like the architecture in a, b
rows = []


def union(m, S, rid, n=NS):
    """Lowest n levels of the union of the model's monomer predictions, with their origin (0: A, 1: B)."""
    a, b = GID[rid]["mono"]
    ga, gb = S[G[a]["id"]], S[G[b]["id"]]
    lv = np.concatenate([ga, gb]); org = np.r_[np.zeros(len(ga), int), np.ones(len(gb), int)]
    o = np.argsort(lv)[:n]
    return lv[o], org[o]


def ref_union(rid, n=NS):
    """TDDFT union of the monomer levels from the same calculations as the dimer (frag_tddft/results_pairs.json);
    origin (A/B) by matching to the monomer labels of the dataset (identical level of theory, within 25 meV)."""
    a, b = GID[rid]["mono"]
    lv = np.r_[G[a]["gaps"][:n], G[b]["gaps"][:n]]; org = np.r_[np.zeros(n, int), np.ones(n, int)]
    o = np.argsort(lv)[:n]
    u = np.asarray(next(t for t in TD if t["id"] == rid)["union"][:n])
    return u, assign(u, lv[o], org[o])


def assign(levels, ulev, uorg):
    """Origin of each level: the nearest unused union level."""
    used, out = set(), []
    for v in levels:
        k = min((i for i in range(len(ulev)) if i not in used), key=lambda i: abs(ulev[i] - v))
        used.add(k); out.append(uorg[k])
    return np.array(out)


def sticks(ax, x, levels, colors, w=0.62, tol=0.03, lw=2.0):
    o = np.argsort(levels); lv = np.asarray(levels)[o]; cl = [colors[i] for i in o]
    i = 0
    while i < len(lv):
        j = i + 1
        while j < len(lv) and lv[j] - lv[j - 1] < tol:
            j += 1
        k = j - i; seg = w / k
        for q in range(k):
            x0 = x - w / 2 + q * seg + (0.04 if k > 1 else 0)
            ax.plot([x0, x0 + seg - (0.08 if k > 1 else 0)], [lv[i + q]] * 2, color=cl[i + q], lw=lw,
                    solid_capstyle="butt", zorder=3)
        i = j


def structure(mono_idx, size=(360, 260)):
    g = G[mono_idx]; pt = Chem.GetPeriodicTable()
    xyz = f"{len(g['Z'])}\n\n" + "\n".join(f"{pt.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}"
                                          for z, (x, y, w) in zip(g["Z"], g["R"]))
    m = Chem.MolFromXYZBlock(xyz); rdDetermineBonds.DetermineBonds(m, charge=0)
    m = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(m))); AllChem.Compute2DCoords(m)
    dr = rdMolDraw2D.MolDraw2DCairo(*size); o = dr.drawOptions(); o.bondLineWidth = 2; o.padding = 0.05
    dr.DrawMolecule(m); dr.FinishDrawing()
    img = plt.imread(io.BytesIO(dr.GetDrawingText()), format="png")
    ys, xs = np.where(img[..., :3].min(-1) < 0.95)
    return img[max(ys.min() - 5, 0):ys.max() + 6, max(xs.min() - 5, 0):xs.max() + 6], Chem.MolToSmiles(m)


def spectrum_panel(ax, kind, letter):
    rid = f"frag_{kind}_{PAIR}_d{DIST:g}"
    r = next(t for t in TD if t["id"] == rid)
    groups = ["ref"] + MODELS
    xg = {k: 1.25 * j for j, k in enumerate(groups)}
    dl, dr = 0.27, 0.27  # sub-column offsets (separate | together)
    levels = {}
    for k in groups:
        n = 3 if k == "xmace" else NS
        if k == "ref":
            ul, uo = ref_union(rid, n); dim = np.asarray(r["dimer"][:n])
        else:
            S = FP[k][0]; ul, uo = union(k, S, rid, n); dim = S[rid][:n]
        levels[k] = (ul, uo, dim)
    vis = np.concatenate([np.r_[v[0], v[2]] for k, v in levels.items() if k != "vecmsani"] + [levels["vecmsani"][0]])
    lo, hi = vis.min() - 0.3, vis.max() + 0.35
    for k in groups:
        ul, uo, dim = levels[k]
        x1, x2 = xg[k] - dl, xg[k] + dr
        if kind == "AA":  # both copies are molecule A
            uo = np.zeros(len(uo), int)
        sticks(ax, x1, ul, [COL_A if o == 0 else COL_B for o in uo], w=0.36)
        exact = k in ("ref", "leh")
        if exact:  # match each dimer level to a union level
            org = assign(dim, ul, uo) if kind == "AB" else np.zeros(len(dim), int)
            used, pairs = set(), []
            for v in dim:
                j = min((i for i in range(len(ul)) if i not in used), key=lambda i: abs(ul[i] - v)); used.add(j)
                pairs.append((ul[j], v))
            cl = [COL_A if o == 0 else COL_B for o in org]
        else:
            pairs = list(zip(np.sort(ul), np.sort(dim))); cl = [fs.MODEL_COLOR[k]] * len(dim)
        inside = [i for i, v in enumerate(dim) if v <= hi - 0.05]
        if inside:
            sticks(ax, x2, dim[inside], [cl[i] for i in inside], w=0.36)
        for u, v in pairs:
            vv = min(v, hi)
            ax.plot([x1 + 0.18, x2 - 0.18], [u, vv], color=fs.MUTED if exact else fs.MODEL_COLOR[k], lw=0.6,
                    linestyle="-" if exact else (0, (2, 1.5)), zorder=2, alpha=0.9)
        if len(inside) < len(dim):
            ax.annotate("", xy=(x2, hi - 0.03), xytext=(x2, hi - 0.33),
                        arrowprops=dict(arrowstyle="-|>", color=fs.MODEL_COLOR[k], lw=1.2))
            ax.text(x2 - 0.38, hi - 0.06, f"S$_1$ = {dim.min():.1f} eV", ha="right", va="top", fontsize=5.8, color=fs.INK2)
        rows.extend(dict(panel=letter, method=k, column="separate (union)", level=float(v), origin="AB"[o])
                    for v, o in zip(ul, uo))
        rows.extend(dict(panel=letter, method=k, column="together (dimer)", level=float(v)) for v in dim)
        if k == "leh":
            rows.append(dict(panel=letter, method=k, column="max |dimer - union| (eV)",
                             level=float(np.max(np.abs(np.sort(dim) - np.sort(ul))))))
    ax.set_ylim(lo, hi)
    ax.set_xticks([xg[k] for k in groups])
    ax.set_xticklabels(["TD-DFT"] + [NAME[m] for m in MODELS], fontsize=6)
    ax.tick_params(axis="x", length=0, pad=10)
    tr = ax.get_xaxis_transform()
    for k in groups:  # sub-columns: A and B computed separately | A + B computed together
        ax.text(xg[k] - dl, -0.015, "A, B" if kind == "AB" else "A, A", transform=tr, ha="center", va="top", fontsize=5.5,
                color=fs.INK2)
        ax.text(xg[k] + dr, -0.015, "A+B" if kind == "AB" else "A+A", transform=tr, ha="center", va="top", fontsize=5.5,
                color=fs.INK2)
    ax.set_xlim(xg["ref"] - 0.62, xg["vecmsani"] + 0.62); ax.grid(False, axis="x")
    ax.axvspan(xg["ref"] - 0.62, xg["ref"] + 0.62, color=fs.GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_ylabel("Excitation energy (eV)")


fig = plt.figure(figsize=(7.2, 6.05))
outer = fig.add_gridspec(2, 1, height_ratios=[2.02, 1.0], hspace=0.13)
gs = outer[0].subgridspec(2, 2, height_ratios=[1.35, 0.3], hspace=0.45, wspace=0.26)
gs_cd = outer[1].subgridspec(1, 2, wspace=0.26)
axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
for ax, kind, letter in ((axa, "AB", "a"), (axb, "AA", "b")):
    spectrum_panel(ax, kind, letter)
# structures of A and B, legend
sub = gs[1, :].subgridspec(1, 4, width_ratios=[0.8, 0.8, 0.15, 1.6])
for k, (c, mi) in enumerate(zip(["A", "B"], GID[f"frag_AB_{PAIR}_d10"]["mono"])):
    ia = fig.add_subplot(sub[0, k])
    img, smi = structure(mi); ia.imshow(img); ia.set_axis_off()
    ia.set_title(f"molecule {c}", fontsize=6, color=fs.INK, pad=1)
    rows.append(dict(panel="a", column=f"structure {c}", smiles=smi, mol=G[mi]["mol"]))
la = fig.add_subplot(sub[0, 3]); la.set_axis_off()
la.legend([Line2D([], [], color=COL_A, lw=2.2), Line2D([], [], color=COL_B, lw=2.2),
           Line2D([], [], color=fs.MUTED, lw=0.8), Line2D([], [], color=fs.INK2, lw=0.8, linestyle=(0, (2, 1.5)))],
          ["molecule A", "molecule B", "matched levels", "n-th levels"], loc="center left", ncol=2, fontsize=6,
          handlelength=1.8, columnspacing=1.2)


def dimer_mae(m, S, n0, n1, which="dimer", dmin=None):
    """MAE over states n0..n1-1 of the dimer prediction (or the union of the monomer predictions) vs TDDFT dimer."""
    e = {}
    for r in TD:
        if dmin is not None and r["d"] < dmin:
            continue
        g = S[r["id"]] if which == "dimer" else union(m, S, r["id"])[0]
        if len(g) < n1:
            continue
        e.setdefault(r["d"], []).append(np.mean(np.abs(g[n0:n1] - np.asarray(r["dimer"][n0:n1]))))
    return {d: float(np.mean(v)) for d, v in sorted(e.items())}


def seedmean(m, n0, n1, which):
    per = [dimer_mae(m, S, n0, n1, which) for S in FP[m]]
    per = [p for p in per if p]
    if not per:
        return {}
    return {d: float(np.mean([p[d] for p in per])) for d in per[0]}


def logticks(ax, ticks):
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())


axc = fig.add_subplot(gs_cd[0, 0])
NUM = {}
for m in MODELS_CD:
    st = dict(fs.STYLE.get(m, {"marker": "o"}))
    e = seedmean(m, 0, 3, "dimer"); u = seedmean(m, 0, 3, "union")
    axc.plot(list(e), list(e.values()), color=fs.MODEL_COLOR[m], lw=1.3, ms=3.4, markeredgecolor="white",
             markeredgewidth=0.5, label=LBL[m], **st)
    rows += [dict(panel="c", model=LBL[m], distance_A=d, s13_mae_dimer=v, s13_mae_union_of_monomers=u.get(d))
             for d, v in e.items()]
    NUM[m] = {"dimer_s13": float(np.mean(list(e.values()))), "union_s13": float(np.mean(list(u.values())))}
axc.axvline(5, color=fs.MUTED, lw=0.8, linestyle=(0, (2, 2)))
axc.text(5.2, 0.03, "r$_c$", transform=axc.get_xaxis_transform(), fontsize=6, color=fs.INK2, va="bottom")
axc.set_yscale("log"); axc.set_xlabel("Minimum interatomic distance (Å)")
axc.set_ylabel("Mean absolute error of dimer\nS$_1$–S$_3$ energies (eV)")
axc.set_ylim(0.1, 40)
logticks(axc, [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10])
axc.legend(loc="center", fontsize=6, ncol=2, handlelength=1.6, bbox_to_anchor=(0.55, 0.5))

axd = fig.add_subplot(gs_cd[0, 1])
GROUPS = [("S$_1$", 0, 1), ("S$_2$–S$_3$", 1, 3), ("S$_4$–S$_6$", 3, 6)]
off = {m: (j - (len(MODELS_CD) - 1) / 2) * 0.15 for j, m in enumerate(MODELS_CD)}
for m in MODELS_CD:
    st = dict(fs.STYLE.get(m, {})); mk = st.get("marker", "o") or "o"
    for gi, (lab, n0, n1) in enumerate(GROUPS):
        e = seedmean(m, n0, n1, "dimer"); u = seedmean(m, n0, n1, "union")
        if not e:
            continue
        ve, vu = float(np.mean(list(e.values()))), float(np.mean(list(u.values())))
        axd.plot(gi + off[m], ve, marker=mk, ms=6, color=fs.MODEL_COLOR[m], markeredgecolor="white", markeredgewidth=0.7,
                 linestyle="none", zorder=3)
        axd.plot(gi + off[m], vu, marker=mk, ms=6, markerfacecolor="white", markeredgecolor=fs.MODEL_COLOR[m],
                 markeredgewidth=1.0, linestyle="none", zorder=2)
        axd.plot([gi + off[m]] * 2, [vu, ve], color=fs.MODEL_COLOR[m], lw=0.7, zorder=1)
        rows.append(dict(panel="d", model=LBL[m], states=lab, mae_dimer=ve, mae_union_of_monomers=vu))
        NUM[m][f"dimer_{n0 + 1}_{n1}"] = ve; NUM[m][f"union_{n0 + 1}_{n1}"] = vu
axd.set_xticks(range(len(GROUPS))); axd.set_xticklabels([g[0] for g in GROUPS]); axd.grid(False, axis="x")
axd.set_xlim(-0.5, len(GROUPS) - 0.5)
axd.set_yscale("log"); axd.set_ylim(0.1, 40); axd.set_ylabel("Mean absolute error of\ndimer energies (eV)")
logticks(axd, [0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10])
axd.set_xlabel("Excited states")
axd.legend([Line2D([], [], marker="o", color=fs.INK2, linestyle="none", ms=5),
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=fs.INK2, linestyle="none", ms=5)],
           ["A + B computed together", "A and B computed separately"], loc="upper left", fontsize=6, ncol=1)
fs.panel_labels(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d")])
out = ROOT / "paper/figs/separability"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
with open(out.with_suffix(".csv"), "w", newline="") as f:
    keys = sorted({k for r in rows for k in r})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
json.dump(NUM, open(ROOT / "experiments/separability_numbers.json", "w"), indent=1)
print("written", out); print(json.dumps(NUM, indent=1))
