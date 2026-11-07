"""Dataset overview figure (released dataset v4 = v1 + conjugated test sets + active-learning rounds 2-4).

(a) Molecule size (heavy atoms) of the QCDGE-derived training molecules (<= 10), the larger test molecules (ZINC,
    11-20), the dyes (training scaffolds enter the production training set, held-out scaffolds are test) and the ladder
    molecules (test only; up to 30 heavy atoms), log-scaled counts.
(b) S1 excitation energy of the training geometries (production split) by type: equilibrium (QCDGE minima), thermal
    (xTB MD, 300-1000 K), S1-sampled in round 1 (model-driven S1/S2 MD) and in the active-learning rounds 2-4. The
    excited-state-sampled geometries fill the low-gap region that ground-state sampling barely reaches.
(c) S0 energy of each training geometry above the minimum of its own molecule (thermal and excited-state-sampled).
(d) Error of the sampling models of each active-learning round on the frames they picked, before labelling (S1 MAE vs
    TDDFT, by the largest double-bond twist of the frame; experiments/al/prelabel_s1r{2,3,4}.json): twisted geometries
    are where the models fail, and each round reduces the error.
Every panel has a legend; identity is never carried by color alone. Output: paper/figs/dataset.{pdf,png,csv}.
"""
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.colors import to_rgb  # noqa

fs.setup()
D = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
sp = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
tr = [D[i] for i in sp["train"]]
RED, AMBER = fs.SLOTS[7], fs.SLOTS[3]
AL_RE = re.compile(r"_r\ds\dmd\d")
rows = []


def tint(c, f):  # mix a colour with white (f = 1: the colour itself)
    return tuple(1 - f * (1 - x) for x in to_rgb(c))


fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9))
axes = axes.ravel()
# (a) molecule sizes
ax = axes[0]
sets = [("Training (QCDGE-derived)", {m["mol"]: m["n_heavy"] for m in tr if m["source"] in ("A9", "A10", "B9", "B10")}, fs.INK, "-"),
        ("Larger (ZINC)", {m["mol"]: m["n_heavy"] for m in D if m["source"] == "ZINC" and m["kind"] == "eq"}, fs.INK2, "-"),
        ("Dyes (training and test scaffolds)", {m["mol"]: m["n_heavy"] for m in D if m.get("conj") == "dye"}, RED, "-"),
        ("Ladders (test only)", {m["mol"]: m["n_heavy"] for m in D if m.get("conj") == "ladder"}, RED, (0, (3, 1.5)))]
bins = np.arange(0.5, 31.5, 1)
for lab, d, col, ls in sets:
    h, _ = np.histogram(list(d.values()), bins=bins)
    ax.step(bins[:-1] + 0.5, np.where(h > 0, h, np.nan), where="mid", color=col, lw=1.3, linestyle=ls, label=lab)
    rows += [dict(panel="a", series=lab, x=int(b + 0.5), count=int(c)) for b, c in zip(bins[:-1], h)]
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
ax.set_yscale("log"); ax.set_ylim(1, 1e6)
ax.set_xlabel("Heavy atoms per molecule"); ax.set_ylabel("Molecules")

# (b) S1 excitation energies by geometry type
ax = axes[1]
kinds = [("Equilibrium", lambda m: m["kind"] == "qcdge_eq", fs.MUTED, "-"),
         ("Thermal (S$_0$ MD)", lambda m: m["kind"] == "offeq", fs.INK, "-"),
         ("S$_1$-sampled, round 1", lambda m: m["kind"] == "excmd" and not AL_RE.search(m["id"]), RED, "-"),
         ("Active learning, rounds 2–4", lambda m: bool(AL_RE.search(m["id"])), AMBER, (0, (3, 1.2)))]
eb = np.arange(0, 10.01, 0.2)
for lab, f, col, ls in kinds:
    g = np.array([m["gaps"][0] for m in tr if f(m) and np.isfinite(m["gaps"][0])])
    h, _ = np.histogram(g, bins=eb, density=True)
    ax.step(eb[:-1] + 0.1, h, where="mid", color=col, lw=1.3, linestyle=ls, label=f"{lab} ({len(g):,})")
    rows += [dict(panel="b", series=lab, x=float(b + 0.1), density=float(c), n=len(g),
                  share_below_3eV=float(np.mean(g < 3))) for b, c in zip(eb[:-1], h)]
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
ax.set_ylim(0, 0.78)
ax.set_xlabel("S$_1$ excitation energy (eV)"); ax.set_ylabel("Density (1/eV)")

# (c) S0 energy above the own minimum
ax = axes[2]
e_min = {m["mol"]: m["E0"] for m in tr if m["kind"] == "qcdge_eq" and "E0" in m}
rb = np.arange(0, 8.01, 0.2)
for lab, f, col, ls in kinds[1:]:
    r = np.array([m["E0"] - e_min[m["mol"]] for m in tr if f(m) and m["mol"] in e_min and "E0" in m])
    h, _ = np.histogram(r, bins=rb, density=True)
    ax.step(rb[:-1] + 0.1, h, where="mid", color=col, lw=1.3, linestyle=ls, label=lab)
    rows += [dict(panel="c", series=lab, x=float(b + 0.1), density=float(c), n=len(r), median=float(np.median(r)))
             for b, c in zip(rb[:-1], h)]
ax.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
ax.set_xlabel("S$_0$ energy above own minimum (eV)"); ax.set_ylabel("Density (1/eV)")

# (d) pre-label error of the active-learning frames by double-bond twist
ax = axes[3]
TW = [("0-30", 15), ("30-60", 45), ("60-75", 67.5), ("75-91", 83)]
for k, (r, mk) in enumerate((("2", "o"), ("3", "s"), ("4", "D"))):
    P = json.load(open(ROOT / f"experiments/al/prelabel_s1r{r}.json"))
    y = [P["by_twist"][b]["mae"] for b, _ in TW]
    ax.plot([x for _, x in TW], y, color=tint(RED, (0.4, 0.7, 1.0)[k]), marker=mk, ms=4.5, lw=1.3,
            markeredgecolor="white", markeredgewidth=0.6, label=f"Round {r} ({P['n']:,} geometries)")
    rows += [dict(panel="d", series=f"round {r}", x=b, mae=P["by_twist"][b]["mae"], n=P["by_twist"][b]["n"],
                  signed=P["by_twist"][b]["signed"], median_tddft_s1=P["by_twist"][b]["median_tddft_s1"]) for b, _ in TW]
ax.set_xticks([x for _, x in TW]); ax.set_xticklabels(["0–30", "30–60", "60–75", "75–90"])
ax.set_ylim(0, None)
ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.2)
ax.set_xlabel("Largest double-bond twist of the geometry (°)"); ax.set_ylabel("Mean absolute S$_1$ error\nbefore labelling (eV)")
fig.tight_layout()
fs.panel_labels(fig, list(zip(axes, "abcd")))
out = ROOT / "paper/figs/dataset"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
keys = sorted({k for r in rows for k in r})
with open(out.with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("written", out)
