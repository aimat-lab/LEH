"""Seed-averaged summary of a run directory analysed by experiments/wb97/analyze.py (results_<runs>.json).

Groups runs by name without the trailing _s<seed>, reports mean +- std over seeds for the main metrics per test set,
and S1-S6 MAE/R2 recomputed from the predictions so that variants with different numbers of supervised states are
comparable. Usage: python experiments/wb97/summarize_seeds.py experiments/wb97/runs_abl
"""
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1])
res = json.load(open(RUNS.parent / f"results_{RUNS.name}.json"))
SETS = ["test_random_offeq", "test_random_eq", "test_random_exc", "test_scaffold_offeq", "test_scaffold_exc",
        "test_large_offeq", "test_large_eq", "test_large_exc"]


def r2(p, r):
    return 1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2)


def s16(pred_file, key, n=6):
    P = pickle.load(open(pred_file, "rb")).get(key, [])
    if not P:
        return np.nan, np.nan
    gp = np.array([np.asarray(p["gaps_pred"], float)[:n] for p in P])
    gr = np.array([np.asarray(p["gaps_ref"], float)[:n] for p in P])
    m = np.isfinite(gp) & np.isfinite(gr)
    return float(np.mean(np.abs(gp[m] - gr[m]))), float(r2(gp[m], gr[m]))


groups = defaultdict(list)
for name in res:
    groups[re.sub(r"_s\d+$", "", name)].append(name)
rows = {}
for g, names in sorted(groups.items()):
    rows[g] = {}
    for s in SETS:
        vals = defaultdict(list)
        for n in names:
            v = res[n].get(s, {})
            if not v:
                continue
            mae, rr = s16(RUNS / n / "predictions.pkl", s)
            vals["S1-6 MAE"].append(mae); vals["S1-6 R2"].append(rr)
            for k, lab in [("mae_E0_per_atom_meV", "E0 meV/at"), ("mae_ES1", "E_S1 MAE"), ("mae_FS1", "F_S1 MAE"),
                           ("mae_F0", "F0 MAE"), ("mae_dE1", "dE1 MAE")]:
                if k in v:
                    vals[lab].append(v[k])
        rows[g][s] = {k: (float(np.mean(x)), float(np.std(x)), len(x)) for k, x in vals.items()}
json.dump(rows, open(RUNS.parent / f"summary_{RUNS.name}.json", "w"), indent=1)
for s in SETS:
    if not any(rows[g].get(s) for g in rows):
        continue
    print(f"\n== {s}  (mean +- std over seeds)")
    keys = ["S1-6 MAE", "S1-6 R2", "dE1 MAE", "E0 meV/at", "E_S1 MAE", "F0 MAE", "F_S1 MAE"]
    print(f"{'variant':16s}" + "".join(f"{k:>17s}" for k in keys))
    for g in rows:
        v = rows[g].get(s, {})
        print(f"{g:16s}" + "".join(f"{v[k][0]:10.3f}+-{v[k][1]:.3f}" if k in v else f"{'-':>17s}" for k in keys))
