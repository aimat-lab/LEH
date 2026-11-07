"""Supplement to seed_consistency.py: element-preserving null for the S1-S3 subspace weight, and the per-seed-pair
shared weight of S1 (non-degenerate states, all geometries of each set). Reads the cache of seed_consistency.py."""
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C = pickle.load(open(ROOT / "experiments/eigvec/eigvecs_prodw_leh.pkl", "rb"))
pos = {i: k for k, i in enumerate(C["idx"])}
rng = np.random.default_rng(2)
pairs = list(combinations(range(4), 2))
for name, idxs in C["sets"].items():
    sub, subnull, pair = [], [], {p: [] for p in pairs}
    for gi in idxs:
        k = pos[gi]
        Z = np.asarray(C["meta"][k]["Z"])
        if len(Z) < 4:
            continue
        lams = np.array([C["seeds"][s][k]["lam"][:4] for s in range(4)])
        W = [(C["seeds"][s][k]["V"] ** 2).sum(1) / 3 for s in range(4)]
        if (lams[:, 3] - lams[:, 2]).min() >= 0.1:
            sub.append(np.mean([np.minimum(W[a], W[b]).sum() for a, b in pairs]))
            nl = []
            for a, b in pairs:
                p = np.arange(len(Z))
                for z in np.unique(Z):
                    ii = np.where(Z == z)[0]; p[ii] = rng.permutation(ii)
                nl.append(np.minimum(W[a], W[b][p]).sum())
            subnull.append(np.mean(nl))
        gap1 = (lams[:, 1] - lams[:, 0]).min()
        if gap1 >= 0.1:
            for a, b in pairs:
                va, vb = C["seeds"][a][k]["V"][:, 0], C["seeds"][b][k]["V"][:, 0]
                pair[(a, b)].append(np.minimum(va ** 2, vb ** 2).sum())
    print(f"{name:16s} S1-3 subspace: shared {np.median(sub):.2f}  null {np.median(subnull):.2f}   "
          "S1 per pair: " + "  ".join(f"s{a}s{b} {np.median(v):.2f}" for (a, b), v in pair.items()))
