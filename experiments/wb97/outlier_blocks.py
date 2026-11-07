"""Size of the learned Hamiltonian blocks of the unbounded full-data LEH benchmark models on all frozen test sets.

For every test geometry: the largest |pair coupling| H_ij (i != j, real rows only) and the largest |onsite element - shift|
(eV), together with the predicted S1. Compares the outlier geometries (S1 < -5 eV, experiments/wb97/outliers.json)
with all other test geometries (maximum and 99.9th percentile). Supports the statement on the cause of the outliers in
the theory section and Supplementary Note "Outliers and bounded blocks".
Writes experiments/wb97/outlier_blocks.json.
Usage: python experiments/wb97/outlier_blocks.py [--threads 8]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.batching import collate  # noqa
from exmlip.training import train as T  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
OUT = json.load(open(ROOT / "experiments/wb97/outliers.json"))
outlier_ids = {(w[0], w[2]) for w in OUT["worst"]}
res = {"per_seed": {}, "outliers": []}
for s in (0, 1):
    rd = ROOT / f"experiments/wb97/runs_main/main_leh_all_s{s}"
    model, cfg = T.build_model_from_dir(rd)
    cfg = {**T.DEFAULT, **cfg}
    T.DEV = torch.device("cpu")
    model = model.to(T.DEV).eval()
    shift = float(model.head.shift)
    _, _, tests = T.load(cfg)
    pair_max, onsite_max, s1, ids, sets = [], [], [], [], []
    for k, ds in tests.items():
        for mols in T.iterate(ds, 64, False):
            b = T.to_dev(collate(mols, cfg["n_states"], 0))
            with torch.no_grad():
                out = model(b["Z"], b["R"], b["batch"], b["nmol"])
            H, M = out["H"], out["M"]
            for i, m in enumerate(mols):
                Mi = int(M[i]); h = H[i, :Mi, :Mi].numpy()
                off = np.abs(h - np.diag(np.diag(h)))
                pair_max.append(float(off.max()) if Mi > 1 else 0.0)
                onsite_max.append(float(np.abs(np.diag(h) - shift).max()))
                s1.append(float(out["eigvals"][i, 0])); ids.append(m["id"]); sets.append(k)
        print(f"seed {s} {k} done ({len(ids)})", flush=True)
    pair_max, onsite_max, s1 = map(np.asarray, (pair_max, onsite_max, s1))
    bad = np.array([(f"seed{s}", i) in outlier_ids for i in ids])
    res["per_seed"][f"seed{s}"] = dict(
        n=int(len(ids)), n_outliers=int(bad.sum()),
        pair_max_other=float(pair_max[~bad].max()), pair_p999_other=float(np.percentile(pair_max[~bad], 99.9)),
        onsite_max_other=float(onsite_max[~bad].max()), onsite_p999_other=float(np.percentile(onsite_max[~bad], 99.9)),
        id_pair_max_other=ids[int(np.argmax(np.where(bad, -1, pair_max)))])
    res["outliers"] += [dict(seed=s, id=ids[j], set=sets[j], s1=float(s1[j]), pair_max=float(pair_max[j]),
                             onsite_max=float(onsite_max[j])) for j in np.where(bad)[0]]
json.dump(res, open(ROOT / "experiments/wb97/outlier_blocks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
