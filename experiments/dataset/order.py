"""Fixed, source-balanced random order over all QCDGE molecules (A_9, A_10, B_9, B_10).

Batches of the new dataset are consecutive slices of this order, so the dataset can grow in tranches.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

Q = Path(sys.argv[1])  # directory with the four QCDGE HDF5 files
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
lists, stats = [], {}
for src in ["A_9", "A_10", "B_9", "B_10"]:
    with h5py.File(Q / f"{src}.hdf5", "r") as f:
        keys = sorted(f.keys())
    perm = [f"{src.replace('_', '')}_{keys[i]}" for i in rng.permutation(len(keys))]
    lists.append(perm)
    stats[src] = len(keys)
order = []
for i in range(max(len(l) for l in lists)):
    for l in lists:
        if i < len(l):
            order.append(l[i])
json.dump(order, open(OUT / "pool_order.json", "w"))
print(stats, len(order))
