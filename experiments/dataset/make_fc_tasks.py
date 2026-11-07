"""Test-only S1-S3 forces at equilibrium (Franck-Condon) geometries (user approved 2026-10-17).

The frozen dataset has no excited-state forces at equilibrium geometries (45 % of the entries), so the S1 gradient at
the Franck-Condon point (the initial direction of excited-state relaxation) is never tested. This computes full
TD-wB97X-D/6-31G* single points with S1-S3 gradients (same protocol as all tranches) at the existing equilibrium
geometries of 2,000 random test molecules (md5 order) and all 500 larger (ZINC) test molecules. The results are a
separate test file; the frozen dataset and models are not touched. Because the geometries are identical to existing
test entries, the frozen models' predictions (which include forces for every test entry) are reused, and the
recomputed E0 and excitation energies double as a reproducibility check of the dataset.
Writes data/wb97/tasks/fc_test.pkl and data/wb97/tasks/fc_test_index.json ({task id: dataset entry index}).
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
sp = json.load(open(ROOT / "experiments/wb97/split_100k_s1_sub10k.json"))
rand = sorted(sp["test_random_eq"], key=lambda i: hashlib.md5(f"fc:{D[i]['id']}".encode()).hexdigest())[:2000]
large = list(sp["test_large_eq"])
tasks, index = [], {}
for i in rand + large:
    m = D[i]
    tid = f"{m['id']}_fc"
    R = np.asarray(m["R"], float)
    tasks.append(dict(id=tid, mol=m["mol"], Z=np.asarray(m["Z"]), R_eq=R, R=R, T=0, source=m["source"],
                      gaps_eq=np.asarray(m["gaps"], float), n_heavy=int(m["n_heavy"]), mode="given",
                      sampling=dict(round="fc_test", entry=int(i), kind=m["kind"])))
    index[tid] = int(i)
out = ROOT / "data/wb97/tasks"
out.mkdir(parents=True, exist_ok=True)
pickle.dump(tasks, open(out / "fc_test.pkl", "wb"))
json.dump(index, open(out / "fc_test_index.json", "w"))
print(len(tasks), "tasks:", len(rand), "random eq,", len(large), "larger eq; kinds",
      {k: sum(D[i]["kind"] == k for i in index.values()) for k in {D[i]["kind"] for i in index.values()}})
