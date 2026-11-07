"""split_offeq.json: split.json with train/val restricted to entries that carry E0 (off-equilibrium geometries).

Used for matched comparisons with baselines that need absolute state energies (MS-ANI / OMNI-P2x architecture,
X-MACE), which cannot use the QCDGE equilibrium spectra. Test sets are unchanged.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
mols = pickle.load(open(ROOT / "data/wb97_all.pkl", "rb"))
split = json.load(open(ROOT / "experiments/wb97/split.json"))
has = lambda i: "E0" in mols[i] and np.isfinite(mols[i]["E0"])  # noqa: E731
s = dict(split)
s["train"] = [i for i in split["train"] if has(i)]
s["val"] = [i for i in split["val"] if has(i)]
json.dump(s, open(ROOT / "experiments/wb97/split_offeq.json", "w"))
print({k: len(v) for k, v in s.items()})
