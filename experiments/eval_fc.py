"""Franck-Condon test: S1-S3 forces at equilibrium geometries (data/wb97/fc_test_labels.pkl from
experiments/dataset/build_fc_labels.py) for the frozen benchmark models. The geometries are existing test entries
(test_random_eq: 2,000 of them; test_large_eq: all 500), so each model's stored predictions are reused (joined by entry
id); no model is re-run. Metrics per set: S0 force MAE, S1/S2/S3 force MAE and R2, and the gap-gradient MAE
(F_Sn - F_S0, the part that drives the initial excited-state relaxation). Writes experiments/wb97/fc_eval.json.
Usage: python experiments/eval_fc.py RUN_DIR [RUN_DIR ...]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAB = pickle.load(open(ROOT / "data/wb97/fc_test_labels.pkl", "rb"))
SETS = {"random_eq": "test_random_eq", "larger_eq": "test_large_eq"}


def r2(p, r):
    return float(1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2))


res = {}
for rd in map(Path, sys.argv[1:]):
    f = rd / "predictions.pkl"
    if not f.exists():
        print("skip", rd); continue
    P = pickle.load(open(f, "rb"))
    out = {}
    for name, key in SETS.items():
        X = [(x, LAB[x["id"]]) for x in P.get(key, []) if x.get("id") in LAB and x.get("Fex_pred") is not None
             and x.get("F0_pred") is not None]
        if not X:
            continue
        o = dict(n=len(X))
        f0p = np.concatenate([x["F0_pred"].ravel() for x, _ in X]); f0r = np.concatenate([l["F0"].ravel() for _, l in X])
        o["F0 MAE"] = float(np.mean(np.abs(f0p - f0r)))
        for s in range(3):
            fp = np.concatenate([x["Fex_pred"][:, s].ravel() for x, _ in X]); fr = np.concatenate([l["Fex"][:, s].ravel() for _, l in X])
            gp = np.concatenate([(x["Fex_pred"][:, s] - x["F0_pred"]).ravel() for x, _ in X])
            gr = np.concatenate([(l["Fex"][:, s] - l["F0"]).ravel() for _, l in X])
            m = np.isfinite(fr) & np.isfinite(fp)
            o[f"F_S{s+1} MAE"] = float(np.mean(np.abs(fp[m] - fr[m]))); o[f"F_S{s+1} R2"] = r2(fp[m], fr[m])
            o[f"Fgap{s+1} MAE"] = float(np.mean(np.abs(gp[m] - gr[m]))); o[f"Fgap{s+1} R2"] = r2(gp[m], gr[m])
        o["|F_S1| ref mean"] = float(np.mean(np.abs(np.concatenate([l["Fex"][:, 0].ravel() for _, l in X]))))
        out[name] = o
    res[str(rd)] = out
    print(rd.name, {k: {q: round(v, 3) for q, v in o.items() if q in ("n", "F0 MAE", "F_S1 MAE", "F_S1 R2", "Fgap1 MAE")}
                    for k, o in out.items()}, flush=True)
json.dump(res, open(ROOT / "experiments/wb97/fc_eval.json", "w"), indent=1)
