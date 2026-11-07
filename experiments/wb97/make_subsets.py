"""Nested training subsets of the final 100k split (hyperparameter tests, ablations, learning curves).

All subsets share the same validation molecules and the same test sets, so every run is evaluated identically:
- train: the first N training molecules (all their geometries) in a fixed md5 order -> nested (1k in 3k in ... in all);
- val: entries of 1500 validation molecules (md5 order), used for checkpoint/model selection only;
- tests, non-overlapping: test_random_eq/_offeq/_exc, test_scaffold_eq/_offeq/_exc, test_large_eq/_offeq/_exc.
Writes experiments/wb97/split{tag}_sub<N>.json (N in molecules, 'all' = full training split) and subsets{tag}.json.

Usage: python experiments/wb97/make_subsets.py --tag _100k --sizes 1000 3000 10000 30000 all
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_100k")
ap.add_argument("--sizes", nargs="+", default=["1000", "3000", "10000", "30000", "all"])
ap.add_argument("--n_val", type=int, default=1500)
ap.add_argument("--wdir", default=str(ROOT / "experiments/wb97"))
ap.add_argument("--noexc", nargs="*", default=[], help="subset names (e.g. 10k) for which a variant without the "
                "excited-state-MD training geometries (kind excmd) is written as ..._noexc.json (ablation)")
a = ap.parse_args()
W = Path(a.wdir)

mols = pickle.load(open(ROOT / f"data/wb97_all{a.tag}.pkl", "rb"))
split = json.load(open(W / f"split{a.tag}.json"))
EQ = ("qcdge_eq", "qcdge_eq_extra", "eq")


def by_mol(idx):
    d = {}
    for i in idx:
        d.setdefault(mols[i]["mol"], []).append(i)
    return d


def ranked(d, salt):
    return sorted(d, key=lambda m: hashlib.md5((salt + m).encode()).hexdigest())


def kind_sub(idx):
    out = {"eq": [], "offeq": [], "exc": []}
    for i in idx:
        k = mols[i]["kind"]
        out["eq" if k in EQ else ("exc" if k == "excmd" else "offeq")].append(i)
    return out


tr, va = by_mol(split["train"]), by_mol(split["val"])
tr_order, va_order = ranked(tr, "lc:"), ranked(va, "val:")
val_idx = sorted(i for m in va_order[: a.n_val] for i in va[m])
tests = {}
for name in ("test_random", "test_scaffold", "test_large"):
    for k, v in kind_sub(split.get(name, [])).items():
        if v:
            tests[f"{name}_{k}"] = sorted(v)
info = dict(n_train_molecules_total=len(tr), n_val_molecules=min(a.n_val, len(va)), n_val_entries=len(val_idx),
            tests={k: len(v) for k, v in tests.items()}, subsets={})
for s in a.sizes:
    n = len(tr_order) if s == "all" else int(s)
    sel = tr_order[:n]
    train_idx = sorted(i for m in sel for i in tr[m])
    name = "all" if s == "all" else (f"{n // 1000}k" if n % 1000 == 0 else str(n))
    json.dump(dict(train=train_idx, val=val_idx, **tests), open(W / f"split{a.tag}_sub{name}.json", "w"))
    info["subsets"][name] = dict(molecules=len(sel), train_entries=len(train_idx),
                                 train_excmd=sum(mols[i]["kind"] == "excmd" for i in train_idx))
    if name in a.noexc:
        tr_no = [i for i in train_idx if mols[i]["kind"] != "excmd"]
        json.dump(dict(train=tr_no, val=val_idx, **tests), open(W / f"split{a.tag}_sub{name}_noexc.json", "w"))
        info["subsets"][name + "_noexc"] = dict(molecules=len(sel), train_entries=len(tr_no))
json.dump(info, open(W / f"subsets{a.tag}.json", "w"), indent=1)
print(json.dumps(info, indent=1))
