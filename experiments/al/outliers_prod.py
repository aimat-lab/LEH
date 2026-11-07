"""Outliers of the production models, as experiments/wb97/outliers.json for the unbounded benchmark LEH models: predicted
S1 below 0 and below -5 eV over the validation and all test sets, and the largest |S1 error|. LEH (bounded blocks,
c tanh(x/c), c = 12 / 5 eV) and Direct-softmin, all production runs with predictions.pkl (uniform and weighted loss,
all seeds); the per-architecture summary covers the production models (weighted loss, four seeds).
Writes experiments/al/outliers_prod.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
res = {"runs": {}}
for d in sorted((ROOT / "experiments/al/runs").glob("prod*_v4_s*")):
    f = d / "predictions.pkl"
    if not f.exists():
        continue
    P = pickle.load(open(f, "rb"))
    s1 = np.concatenate([[x["gaps_pred"][0] for x in X] for k, X in P.items() if X])
    ref = np.concatenate([[x["gaps_ref"][0] for x in X] for k, X in P.items() if X])
    e = s1 - ref
    res["runs"][d.name] = dict(n=int(len(s1)), s1_negative=int((s1 < 0).sum()), s1_below_m5=int((s1 < -5).sum()),
                               min_s1=float(np.nanmin(s1)), max_abs_err=float(np.nanmax(np.abs(e))),
                               n_err_above_3=int((np.abs(e) > 3).sum()))
    del P
for a in ("leh", "softmin"):
    R = [r for k, r in res["runs"].items() if k.startswith(f"prodw_{a}_")]
    res[a] = dict(n=int(sum(r["n"] for r in R)), s1_negative=int(sum(r["s1_negative"] for r in R)),
                  s1_below_m5=int(sum(r["s1_below_m5"] for r in R)), n_runs=len(R), min_s1=float(min(r["min_s1"] for r in R)),
                  max_abs_err=float(max(r["max_abs_err"] for r in R)), n_err_above_3=int(sum(r["n_err_above_3"] for r in R)))
json.dump(res, open(ROOT / "experiments/al/outliers_prod.json", "w"), indent=1)
print(json.dumps(res, indent=1))
