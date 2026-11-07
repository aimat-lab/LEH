"""Production datasets of the active-learning loop (S1 sampling rounds >= 2; production models only, the v1 benchmark
stays frozen).

Base: model B's data (data/wb97_all_100k_s1_conj.pkl = frozen v1 entries in their original order + conjugated dyes and
ladders) and its split (experiments/conj/split_B.json: v1 training + training-dye scaffolds). New TD-wB97X-D frames of
the given rounds (data/wb97/s1rN/*.npz) are appended, so every index of the frozen splits stays valid.
Filters as for v1 (experiments/wb97/prepare.py --drop_reactive): bonding changed during the sampling, or S0 energy
below the minimum of the molecule's own equilibrium geometry (a reaction). Molecules of the reference trajectories
(test: acrolein, uracil, trans-stilbene, biphenyl scans + experiments/al/test_trajectories_extra.json; validation:
experiments/al/validation_trajectories.json) never enter train/val (base entries of such molecules are removed there).
Assignment by the molecule's role in the frozen split: training molecules -> train, validation molecules -> val,
test molecules -> new keys test_{random,scaffold,large}_exc{N} (frozen test keys unchanged). ZINC (larger) frames are
test only.
Writes data/wb97_prod_<tag>.pkl, experiments/al/split_prod_<tag>.json and experiments/al/info_prod_<tag>.json.
Usage: python experiments/al/build_dataset.py --rounds s1r2 [s1r3 ...] --tag v2
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.load_wb97 import load_dirs  # noqa
from exmlip.data.splits import graph_key, graph_key_smiles  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", nargs="+", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--base", default="data/wb97_all_100k_s1_conj.pkl")
ap.add_argument("--base_split", default="experiments/conj/split_B.json")
a = ap.parse_args()

TEST_SMILES = ["C=CC=O", "O=C1C=CNC(=O)N1", "c1ccc(cc1)/C=C/c1ccccc1", "c1ccc(cc1)-c1ccccc1"]
vf = ROOT / "experiments/al/validation_trajectories.json"
VAL_SMILES = [v["smiles"] for v in json.load(open(vf))] if vf.exists() else []
tf = ROOT / "experiments/al/test_trajectories_extra.json"
VAL_SMILES += [v["smiles"] for v in json.load(open(tf))] if tf.exists() else []
excl = {graph_key_smiles(s) for s in TEST_SMILES + VAL_SMILES}

D = pickle.load(open(ROOT / a.base, "rb"))
S = json.load(open(ROOT / a.base_split))
n_base = len(D)
role = {}
for k, idx in S.items():
    r = "train" if k == "train" else "val" if k == "val" else ("large" if "large" in k or "lad" in k else
                                                             "scaffold" if "scaffold" in k else "random" if "random" in k else None)
    if r is None:
        continue
    for i in idx:
        role.setdefault(D[i]["mol"], r)
e_min = {m["mol"]: m["E0"] for m in D if m["kind"] in ("qcdge_eq", "eq") and m.get("E0") is not None}
mol_key = {m["mol"]: graph_key(np.asarray(m["Z"]), np.asarray(m["R"])) for m in D if m["kind"] in ("qcdge_eq", "eq")}

new, stats = [], Counter()
for rnd in a.rounds:
    mols, st = load_dirs([ROOT / "data/wb97" / rnd], add_eq=False)
    stats[f"{rnd}_files"] = st["files"]; stats[f"{rnd}_failed"] = st["failed"]; stats[f"{rnd}_masked_roots"] = st["masked_roots"]
    for m in mols:
        if m.get("conn_changed"):
            stats["dropped_conn_changed"] += 1; continue
        if m["mol"] in e_min and m["E0"] < e_min[m["mol"]]:
            stats["dropped_below_minimum"] += 1; continue
        if mol_key.get(m["mol"]) in excl:
            stats["dropped_reference_molecule"] += 1; continue
        if m["mol"] not in role:
            stats["dropped_no_role"] += 1; continue
        new.append(m)
# trajectory molecules that are part of the base data (e.g. styrene in QCDGE, azobenzene among the dyes) are removed
# from train and val (frozen test keys stay unchanged), so the trajectory criterion and the test trajectories are
# never in the training distribution
excl_mols = {mm for mm, k in mol_key.items() if k in excl}
split = {k: ([i for i in v if D[i]["mol"] not in excl_mols] if k in ("train", "val") else list(v)) for k, v in S.items()}
stats["base_entries_removed_reference_molecules"] = sum(len(S[k]) - len(split[k]) for k in ("train", "val"))
stats["reference_molecules_in_base"] = sorted(excl_mols)
for j, m in enumerate(new):
    i = n_base + j
    r, n = role[m["mol"]], m.get("al_round", 0)
    if r in ("train", "val"):
        split[r].append(i)
    else:
        split.setdefault(f"test_{r}_exc{n}", []).append(i)
D = D + new
tag = a.tag
pickle.dump(D, open(ROOT / f"data/wb97_prod_{tag}.pkl", "wb"))
json.dump(split, open(ROOT / f"experiments/al/split_prod_{tag}.json", "w"))
info = dict(rounds=a.rounds, n_base=n_base, n_new=len(new), n_total=len(D), stats=dict(stats),
            new_by_role=dict(Counter(role[m["mol"]] for m in new)), split_sizes={k: len(v) for k, v in split.items()},
            excluded_smiles=TEST_SMILES + VAL_SMILES)
json.dump(info, open(ROOT / f"experiments/al/info_prod_{tag}.json", "w"), indent=1)
print(json.dumps({k: info[k] for k in ("n_base", "n_new", "n_total", "stats", "new_by_role")}, indent=1))
