"""Effect of the state-ordering fix of msani_train.py predictions (2026-10-16): before, the excitation energies were
sorted but the excited-state forces were taken in model-output order. For each re-predicted run: share of geometries
whose excited states came out of order, and S1/S2/S3 force MAE before (superseded_unsorted_forces/predictions.pkl)
vs after (predictions.pkl) on the off-equilibrium and S1-sampled test sets. Writes <run>/order_fix.json and prints.
Usage: python experiments/baselines/msani_order_check.py RUN_DIR [RUN_DIR ...]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SETS = ["test_random_offeq", "test_random_exc", "test_scaffold_offeq", "test_large_offeq", "test_large_exc"]


def fmae(P, s):
    X = [x for x in P if x.get("Fex_ref") is not None and x.get("Fex_pred") is not None]
    if not X:
        return None
    return float(np.nanmean(np.concatenate([np.abs(x["Fex_pred"][:, s] - x["Fex_ref"][:, s]).ravel() for x in X])))


for rd in map(Path, sys.argv[1:]):
    new_f, old_f = rd / "predictions.pkl", rd / "superseded_unsorted_forces/predictions.pkl"
    if not (new_f.exists() and old_f.exists()):
        continue
    new, old = pickle.load(open(new_f, "rb")), pickle.load(open(old_f, "rb"))
    res = {}
    for k in SETS:
        if k not in new or k not in old:
            continue
        res[k] = dict(n=len(new[k]), frac_permuted=float(np.mean([x.get("state_order_permuted", False) for x in new[k]])),
                      **{f"FS{s + 1}_mae_old": fmae(old[k], s) for s in range(3)},
                      **{f"FS{s + 1}_mae_new": fmae(new[k], s) for s in range(3)})
    json.dump(res, open(rd / "order_fix.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{rd.name:22s} {k:20s} permuted {100 * v['frac_permuted']:5.1f} %  F_S1 {v['FS1_mae_old']:.3f} -> "
              f"{v['FS1_mae_new']:.3f}  F_S2 {v['FS2_mae_old']:.3f} -> {v['FS2_mae_new']:.3f}  "
              f"F_S3 {v['FS3_mae_old']:.3f} -> {v['FS3_mae_new']:.3f}", flush=True)
