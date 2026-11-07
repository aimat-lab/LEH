"""Write one tranche of tasks: QCDGE minimum geometry + its 10 reference singlets (eV)."""
import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--qcdge", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
out = Path(a.out)
order = json.load(open(out / "pool_order.json"))[a.start : a.start + a.n]
files = {}
tasks, skipped = [], 0
for mid in order:
    src, key = mid.split("_", 1)
    fn = {"A9": "A_9", "A10": "A_10", "B9": "B_9", "B10": "B_10"}[src]
    if fn not in files:
        files[fn] = h5py.File(Path(a.qcdge) / f"{fn}.hdf5", "r")
    g = files[fn][key]["excited_state"]
    info = json.loads(g["Info_of_AllExcitedStates"][0])
    s = sorted(float(v["excitation_e_eV"].split()[0]) for v in info.values() if v["state_type"].startswith("Singlet"))
    Z = g["labels"][0].astype(np.int64)
    if len(s) != 10 or min(s) <= 0 or not set(Z.tolist()) <= {1, 6, 7, 8, 9}:
        skipped += 1
        continue
    tasks.append(dict(id=mid, Z=Z, R_eq=g["coords"][()].astype(float), gaps_eq=np.array(s), source=src,
                      n_heavy=int((Z > 1).sum())))
(out / "tasks").mkdir(exist_ok=True)
pickle.dump(tasks, open(out / "tasks" / f"{a.name}.pkl", "wb"))
print(a.name, "tasks", len(tasks), "skipped", skipped)
