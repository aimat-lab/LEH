"""Catastrophic eigenvalue predictions of the full-data LEH benchmark models (unbounded blocks): number of test
predictions with S1 < -5 eV and with S1 < 0, over all frozen test sets of both seeds (experiments/wb97/outliers.json)."""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = dict(n=0, s1_below_m5=0, s1_negative=0, worst=[])
for s in (0, 1):
    P = pickle.load(open(ROOT / f"experiments/wb97/runs_main/main_leh_all_s{s}/predictions.pkl", "rb"))
    for k, X in P.items():
        if not k.startswith("test"):
            continue
        gp = np.array([x["gaps_pred"][0] for x in X])
        out["n"] += len(X); out["s1_below_m5"] += int((gp < -5).sum()); out["s1_negative"] += int((gp < 0).sum())
        out["worst"] += [(f"seed{s}", k, X[i]["id"], float(gp[i])) for i in np.where(gp < -5)[0]]
json.dump(out, open(ROOT / "experiments/wb97/outliers.json", "w"), indent=1)
print(out)
