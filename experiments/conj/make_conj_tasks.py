"""Tasks and cost estimate for the conjugated large-molecule dataset (parts A: dyes, B: ladders).

Cost model: wall time on 4 threads ~ exp(c) * nbf^k, fitted to the 1000 ZINC single points (11-20 heavy atoms);
nbf = 6-31G* spherical basis functions (H 2, C/N/O/F 14, S 18); core-h = 4 * time * OVERHEAD (idle and tails).
Writes data/conj/tasks/{dyes_eq,dyes_md,ladders_eq,ladders_md}.pkl for the selected configuration and prints costs.
"""
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
OVERHEAD = 1.25
NBF = {1: 2, 6: 14, 7: 14, 8: 14, 9: 14, 16: 18}


def nbf_of(Z):
    return sum(NBF[int(z)] for z in Z)


# fit the cost model
rows = []
for f in glob.glob(str(ROOT / "data/wb97/large/*.npz")):
    r = np.load(f)
    rows.append((nbf_of(r["Z"]), float(r["time"])))
a = np.array(rows)
k, c = np.polyfit(np.log(a[:, 0]), np.log(a[:, 1]), 1)
print(f"cost model: t[s] = {np.exp(c):.3e} * nbf^{k:.2f}  (fit on {len(a)} ZINC single points)")


def coreh(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    nbf = nbf_of([at.GetAtomicNum() for at in m.GetAtoms()])
    return 4 * np.exp(c) * nbf ** k / 3600 * OVERHEAD


dyes = pd.read_csv(ROOT / "experiments/conj/dyes_all.csv")
lad = pd.read_csv(ROOT / "experiments/conj/ladders.csv")
dyes["coreh"] = [coreh(s) for s in dyes.smiles]
lad["coreh"] = [coreh(s) for s in lad.smiles]
print(f"dyes {len(dyes)}: one geometry each {dyes.coreh.sum():,.0f} core-h (median {dyes.coreh.median():.2f}/geometry)")
print(f"ladders {len(lad)}: one geometry each {lad.coreh.sum():,.0f} core-h (median {lad.coreh.median():.2f}/geometry)")

# selected configuration: all dyes at the xTB minimum, a stratified 40 % of dyes with one MD geometry,
# all ladders at the minimum and one MD geometry
rng = np.random.default_rng(11)
dyes["md_subset"] = False
for (_, s), g in dyes.groupby([pd.cut(dyes.ha, [10, 15, 20, 25, 30]), "has_S"], observed=True):
    pick = rng.choice(g.index, size=int(round(0.4 * len(g))), replace=False)
    dyes.loc[pick, "md_subset"] = True
cost = dict(dyes_eq=dyes.coreh.sum(), dyes_md=dyes[dyes.md_subset].coreh.sum(), ladders_eq=lad.coreh.sum(),
            ladders_md=lad.coreh.sum())
for kk, v in cost.items():
    print(f"  {kk:11s} {v:9,.0f} core-h")
print(f"  total       {sum(cost.values()):9,.0f} core-h")

out = ROOT / "data/conj/tasks"
out.mkdir(parents=True, exist_ok=True)


def tasks(df, prefix, mode, extra_cols):
    t = []
    for i, r in df.iterrows():
        mid = f"{prefix}{i:05d}"
        d = dict(id=f"{mid}_{mode}", mol=mid, smiles=r.smiles, mode=mode, source=prefix.rstrip("_"),
                 n_heavy=int(r.ha), gaps_eq=np.full(10, np.nan), has_S=bool(r.has_S), coreh_est=float(r.coreh))
        d.update({c_: r[c_] for c_ in extra_cols if c_ in r})
        t.append(d)
    t.sort(key=lambda x: -x["coreh_est"])  # largest first: fewer tails at the end of the jobs
    return t


print(f"  dyes_md_all {dyes.coreh.sum():9,.0f} core-h  (alternative: an MD geometry for every dye)")
T = dict(dyes_eq=tasks(dyes, "DYE_", "eq", ["source", "lam_abs_median"]),
         dyes_md=tasks(dyes[dyes.md_subset], "DYE_", "md", ["source", "lam_abs_median"]),
         dyes_md_all=tasks(dyes, "DYE_", "md", ["source", "lam_abs_median"]),
         ladders_eq=tasks(lad, "LAD_", "eq", ["family", "n", "ends", "holdout_longest"]),
         ladders_md=tasks(lad, "LAD_", "md", ["family", "n", "ends", "holdout_longest"]))
for name, t in T.items():
    for x in t:  # keep provenance of dyes ('joung'/'beard') separate from the task source tag
        if "source" in x and x["source"] in ("joung", "beard"):
            x["db"] = x.pop("source"); x["source"] = "DYE"
    pickle.dump(t, open(out / f"{name}.pkl", "wb"))
    print(name, len(t), "tasks")
dyes.to_csv(ROOT / "experiments/conj/dyes_selected.csv", index=False)
