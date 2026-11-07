"""Diagnostic: are a model's predicted forces systematically mis-scaled for larger molecules?
For each heavy-atom bin: S0 / S1 force MAE, least-squares slope of predicted vs reference force components (through
the origin; 1 = correct scale), and correlation. Usage: python experiments/diag_force_scaling.py RUN_DIR [RUN_DIR ...]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

out = {}
for rd in sys.argv[1:]:
    P = pickle.load(open(Path(rd) / "predictions.pkl", "rb"))
    res = {}
    for key in ["test_random_offeq", "test_large_offeq"]:
        rows = [x for x in P.get(key, []) if x.get("F0_ref") is not None and x.get("F0_pred") is not None]
        bins = {}
        for x in rows:
            b = int(x["n_heavy"]) if key == "test_random_offeq" else 5 * (int(x["n_heavy"]) // 5)
            bins.setdefault(b, []).append(x)
        for b, xs in sorted(bins.items()):
            if len(xs) < 20:
                continue
            for nm, getp, getr in [("F0", lambda x: x["F0_pred"], lambda x: x["F0_ref"]),
                                   ("F_S1", lambda x: x["Fex_pred"][:, 0], lambda x: x["Fex_ref"][:, 0])]:
                p = np.concatenate([np.asarray(getp(x), float).ravel() for x in xs if x.get("Fex_ref") is not None or nm == "F0"])
                r = np.concatenate([np.asarray(getr(x), float).ravel() for x in xs if x.get("Fex_ref") is not None or nm == "F0"])
                m = np.isfinite(p) & np.isfinite(r)
                p, r = p[m], r[m]
                res[f"{key}|{b}|{nm}"] = dict(n=len(xs), mae=float(np.mean(np.abs(p - r))), slope=float(p @ r / (r @ r)),
                                              corr=float(np.corrcoef(p, r)[0, 1]))
    out[Path(rd).name] = res
    print(f"== {Path(rd).name}")
    for k, v in res.items():
        print(f"  {k:32s} n {v['n']:5d}  MAE {v['mae']:.3f}  slope {v['slope']:.3f}  corr {v['corr']:.3f}")
json.dump(out, open("experiments/wb97/diag_force_scaling.json", "w"), indent=1)
