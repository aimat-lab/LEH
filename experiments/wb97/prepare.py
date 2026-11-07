"""Build data/wb97_all.pkl and molecule-level splits for the new dataset.

train/val/test_random: hash-based (growth-stable) split of QCDGE-derived molecules (both their equilibrium and off-equilibrium entries).
test_scaffold: whole held-out Bemis-Murcko scaffold families (~10 % of molecules; list in
    experiments/wb97/scaffold_holdout.json, see resplit_scaffold.py); without that file: off-equilibrium entries
    of test molecules whose generic Murcko scaffold does not occur in training (old, tiny subset).
test_large / test_large_eq / test_large_offeq: ZINC molecules with 11-20 heavy atoms (size extrapolation).
--eq_extra TASKS: additional QCDGE equilibrium spectra (gaps only, no E0/forces) appended to the data and written to
  split_eqplus.json, which equals split.json with these entries added to train.
"""
import argparse
import json
import pickle
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.load_wb97 import eq_labels, load_dirs  # noqa
from exmlip.data.splits import graph_key, graph_key_smiles, scaffold_of  # noqa

# molecules used for E5 (reference S1 trajectories and PES scans) never enter any split
E5_SMILES = ["C=CC=O", "O=C1C=CNC(=O)N1", "c1ccc(cc1)/C=C/c1ccccc1", "c1ccc(cc1)-c1ccccc1"]

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="*", help="subdirectories of data/wb97 (default: all except tasks)")
ap.add_argument("--eq_extra", help="task pkl with R_eq/gaps_eq of additional QCDGE molecules")
ap.add_argument("--eq_dir", default=str(ROOT / "data/wb97_eq"), help="compute_e0.py output (S0 at QCDGE minima)")
ap.add_argument("--tag", default="", help="suffix of output files (e.g. _25k), so several dataset versions coexist")
ap.add_argument("--drop_reactive", action="store_true",
                help="drop off-eq geometries whose bonding changed during MD (a different molecule than the minimum) and "
                     "QCDGE minima with max |F0| > 2 eV/A at our level (not a minimum here)")
a = ap.parse_args()
dirs = [ROOT / "data/wb97" / d for d in a.dirs] if a.dirs else \
    [p for p in (ROOT / "data/wb97").iterdir() if p.is_dir() and p.name not in ("tasks",)]
print("dirs", [d.name for d in dirs])
osc_file = ROOT / "data/wb97/osc_eq.pkl"  # QCDGE oscillator strengths (experiments/dataset/extract_qcdge_osc.py)
eq_osc = pickle.load(open(osc_file, "rb")) if osc_file.exists() else None
mols, stats = load_dirs(dirs, eq_dir=a.eq_dir if Path(a.eq_dir).exists() else None, eq_osc=eq_osc)
excl_keys = {graph_key_smiles(smi) for smi in E5_SMILES}
key = {}
for m in mols:
    if m["kind"] in ("qcdge_eq", "eq") or m["mol"] not in key:
        key[m["mol"]] = graph_key(m["Z"], m["R"])
excluded = sorted({mm for mm, k in key.items() if k in excl_keys})
mols = [m for m in mols if m["mol"] not in excluded]
stats["excluded_e5"] = excluded
if a.drop_reactive:
    n0 = len(mols)
    bad_eq = {m["mol"] for m in mols if m["kind"] == "qcdge_eq" and "F0" in m
              and np.linalg.norm(m["F0"], axis=1).max() > 2.0}
    # a distorted geometry below the S0 minimum of its own molecule means the hot MD reached a more stable structure
    # (a reaction the bond check misses); our minimum energies agree with QCDGE's, so these are dropped as reactive
    e_min = {m["mol"]: m["E0"] for m in mols if m["kind"] == "qcdge_eq" and "E0" in m}
    below = {m["id"] for m in mols if m["kind"] == "offeq" and m["mol"] in e_min and m["E0"] < e_min[m["mol"]]}
    stats["dropped_below_minimum"] = len(below)
    mols = [m for m in mols if not m.get("conn_changed", False) and m["id"] not in below
            and not (m["kind"] == "qcdge_eq" and m["mol"] in bad_eq)]
    stats["dropped_reactive_or_bad_eq"] = n0 - len(mols)
    stats["bad_eq_minima"] = sorted(bad_eq)
qc = sorted({m["mol"] for m in mols if m["source"] != "ZINC"})
n = len(qc)


def role_of(mol_id):
    """Stable under dataset growth: 10 % test, 5 % val, 85 % train by a hash of the molecule id."""
    h = int(hashlib.md5(("wb97split:" + mol_id).encode()).hexdigest()[:8], 16) % 1000
    return "test" if h < 100 else ("val" if h < 150 else "train")


role = {mm: role_of(mm) for mm in qc}
scaf = {}
for m in mols:
    if m["source"] != "ZINC" and m["kind"] == "qcdge_eq":
        scaf[m["mol"]] = scaffold_of(Z=m["Z"], R=m["R"])
train_scaf = {scaf.get(mm) for mm, r in role.items() if r == "train"}
split = {k: [] for k in ["train", "val", "test_random", "test_random_offeq", "test_random_exc", "test_scaffold",
                         "test_large", "test_large_eq", "test_large_offeq", "test_large_exc"]}
for i, m in enumerate(mols):
    if m["source"] == "ZINC":
        split["test_large"].append(i)
        split[{"eq": "test_large_eq", "excmd": "test_large_exc"}.get(m["kind"], "test_large_offeq")].append(i)
        continue
    r = role[m["mol"]]
    if r == "test":
        split["test_random"].append(i)
        if m["kind"] == "offeq":
            split["test_random_offeq"].append(i)
        if m["kind"] == "excmd":
            split["test_random_exc"].append(i)
        if scaf.get(m["mol"]) not in train_scaf and m["kind"] == "offeq":
            split["test_scaffold"].append(i)
    else:
        split[r].append(i)
info = dict(stats=stats, n_entries=len(mols), n_molecules_qcdge=n, **{k: len(v) for k, v in split.items()})
if a.eq_extra:
    have = set(key.values()) | excl_keys
    extra, dup = [], 0
    for t in pickle.load(open(a.eq_extra, "rb")):
        k = graph_key(t["Z"], t["R_eq"])
        if k in have:
            dup += 1
            continue
        have.add(k)
        e = dict(Z=t["Z"], R=t["R_eq"], gaps=np.asarray(t["gaps_eq"], float), id=t["id"] + "_eq", mol=t["id"],
                 source=t["source"], kind="qcdge_eq_extra", T=0.0, conn_changed=False, n_heavy=t["n_heavy"])
        lab = eq_labels(a.eq_dir if Path(a.eq_dir).exists() else None, t["id"], t["R_eq"])
        if eq_osc is not None and t["id"] in eq_osc:
            e["osc"] = np.asarray(eq_osc[t["id"]], float)
        if lab is not None:
            e.update(lab)
        extra.append(e)
    split_plus = dict(split)
    split_plus["train"] = split["train"] + list(range(len(mols), len(mols) + len(extra)))
    mols = mols + extra
    json.dump(split_plus, open(ROOT / f"experiments/wb97/split_eqplus{a.tag}.json", "w"))
    info.update(n_eq_extra=len(extra), eq_extra_dropped_duplicates=dup, train_eqplus=len(split_plus["train"]),
                eq_extra_with_E0=sum("E0" in e for e in extra))
# consistency: S0 at a distorted geometry must lie above S0 at the minimum of the same molecule
e_eq = {m["mol"]: m["E0"] for m in mols if m["kind"] == "qcdge_eq" and "E0" in m}
d = np.array([m["E0"] - e_eq[m["mol"]] for m in mols if m["kind"] == "offeq" and m["mol"] in e_eq])
if len(d):
    info.update(dE0_md_minus_eq_min=float(d.min()), dE0_md_minus_eq_median=float(np.median(d)),
                n_dE0_negative=int((d < 0).sum()))
# balanced scaffold hold-out (experiments/wb97/resplit_scaffold.py defines the scaffold list once, on the 100k set):
# whole held-out scaffold families go to test_scaffold*, replacing the old novel-scaffold subset of the random test
HOLD = ROOT / "experiments/wb97/scaffold_holdout.json"
if HOLD.exists():
    from exmlip.data.splits import apply_scaffold_holdout, scaffolds_parallel  # noqa
    eqg = {}
    for m in mols:
        if m["source"] != "ZINC" and m["kind"] in ("qcdge_eq", "qcdge_eq_extra") and m["mol"] not in eqg:
            eqg[m["mol"]] = (m["Z"], m["R"])
    cf = ROOT / f"experiments/wb97/scaffolds{a.tag}.json"
    cache = json.load(open(cf)) if cf.exists() else {}
    todo = [k for k in eqg if k not in cache]
    cache.update(zip(todo, scaffolds_parallel([eqg[k] for k in todo])))
    json.dump(cache, open(cf, "w"))
    held = set(json.load(open(HOLD))["scaffolds"])
    held_mols = {k for k in eqg if cache[k] in held}
    split = apply_scaffold_holdout(mols, split, held_mols)
    if a.eq_extra:
        fp = ROOT / f"experiments/wb97/split_eqplus{a.tag}.json"
        json.dump(apply_scaffold_holdout(mols, json.load(open(fp)), held_mols), open(fp, "w"))
    info.update({k: len(v) for k, v in split.items()}, scaffold_holdout_molecules=len(held_mols))
pickle.dump(mols, open(ROOT / f"data/wb97_all{a.tag}.pkl", "wb"))
json.dump(split, open(ROOT / f"experiments/wb97/split{a.tag}.json", "w"))
json.dump(info, open(ROOT / f"experiments/wb97/data_info{a.tag}.json", "w"), indent=1)
print(info)
