"""Composition of the final dataset (v4 = frozen v1 + conjugated test sets + active-learning rounds 2-4), as released.

Counts geometries and molecules by geometry type, active-learning round and element/size, the number of geometries
with S0 and S1-S3 force labels, the TDDFT S1 distribution of the active-learning frames and the share of low-gap
geometries in the production training split.
Writes experiments/al/dataset_stats_v4.json (used by experiments/make_numbers.py and the SI dataset table).
"""
import collections
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
EL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}


def kind(m):
    i = m["id"]
    r = re.search(r"_r(\d)s\dmd\d", i)
    if r:
        return f"al_r{r.group(1)}"
    if i.startswith(("DYE_", "LAD_")):
        return "conj_eq" if "_eq" in i else "conj_offeq"
    return {"eq": "eq", "md": "offeq", "excmd": "excmd"}.get(m["kind"], m["kind"])


out = {"n_geometries": len(D), "n_molecules": len({m["mol"] for m in D}), "by_type": {}, "elements": {}, "heavy_atoms": {}}
by = collections.defaultdict(list)
for m in D:
    by[kind(m)].append(m)
for k, ms in sorted(by.items()):
    fex = [np.asarray(m["Fex"]) if m.get("Fex") is not None else None for m in ms]
    out["by_type"][k] = dict(n=len(ms), n_molecules=len({m["mol"] for m in ms}),
                             n_F0=int(sum(m.get("F0") is not None and np.all(np.isfinite(m["F0"])) for m in ms)),
                             n_Fex_S1=int(sum(f is not None and np.all(np.isfinite(f[:, 0])) for f in fex)))
al = [m for k, ms in by.items() if k.startswith("al_") for m in ms]
s1 = np.array([m["gaps"][0] for m in al])
out["al"] = dict(n=len(al), n_molecules=len({m["mol"] for m in al}), s1_below_1=int((s1 < 1).sum()),
                 s1_below_2=int((s1 < 2).sum()), s1_median=float(np.median(s1)))
exc = [m for k in ("excmd",) for m in by.get(k, [])] + al
out["n_excited_sampled"] = len(exc)
out["n_with_Fex_S1"] = int(sum(v["n_Fex_S1"] for v in out["by_type"].values()))
cnt = collections.Counter()
for m in {m["mol"]: m for m in D}.values():
    for z in set(int(z) for z in m["Z"]):
        cnt[EL.get(z, str(z))] += 1
out["elements"] = dict(cnt)
out["heavy_atoms"] = dict(sorted(collections.Counter(int(m["n_heavy"]) for m in {m["mol"]: m for m in D}.values()).items()))
tr = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))["train"]
g = np.array([D[i]["gaps"][0] for i in tr]); g = g[np.isfinite(g)]
out["train"] = dict(n=len(tr), s1_below_1=int((g < 1).sum()), s1_below_2=int((g < 2).sum()), share_below_1=float(np.mean(g < 1)),
                    share_below_2=float(np.mean(g < 2)))
json.dump(out, open(ROOT / "experiments/al/dataset_stats_v4.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "heavy_atoms"}, indent=1))
