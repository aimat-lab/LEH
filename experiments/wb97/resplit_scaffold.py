"""Add a proper scaffold hold-out test set to an existing new-dataset version (data/wb97_all{tag}.pkl).

The earlier 'test_scaffold' was only the subset of random-test molecules whose generic Murcko scaffold is absent from
training (8 entries at 5k, 20 at 100k). This script replaces it by a balanced Bemis-Murcko scaffold hold-out:
typed scaffolds from the equilibrium geometries, ~10 % of the molecules (whole scaffold families, all their
geometries) removed from train/val/test_random and placed in test_scaffold / test_scaffold_eq / test_scaffold_offeq.

The held-out scaffold list is defined once (on the 100k set) and stored in experiments/wb97/scaffold_holdout.json;
other versions (25k, later tranches) reuse it, so the hold-out is identical across dataset versions.
The previous split is kept as split{tag}_v1_noscaffold.json.

Usage: python experiments/wb97/resplit_scaffold.py --tag _100k [--define] [--holdout path]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.splits import apply_scaffold_holdout, scaffolds_parallel, select_scaffold_holdout  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--holdout", default=str(ROOT / "experiments/wb97/scaffold_holdout.json"))
ap.add_argument("--define", action="store_true", help="(re)define the held-out scaffold list on this dataset")
ap.add_argument("--frac", type=float, default=0.10)
ap.add_argument("--cap", type=float, default=0.002)
ap.add_argument("--nproc", type=int, default=16)
ap.add_argument("--wdir", default=str(ROOT / "experiments/wb97"), help="directory with split/data_info files")
a = ap.parse_args()
W = Path(a.wdir)

mols = pickle.load(open(ROOT / f"data/wb97_all{a.tag}.pkl", "rb"))
eq = {}
for m in mols:
    if m.get("source") != "ZINC" and m.get("kind") in ("qcdge_eq", "qcdge_eq_extra") and m["mol"] not in eq:
        eq[m["mol"]] = (m["Z"], m["R"])
cache_f = W / f"scaffolds{a.tag}.json"
cache = json.load(open(cache_f)) if cache_f.exists() else {}
todo = [k for k in eq if k not in cache]
for k, s in zip(todo, scaffolds_parallel([eq[k] for k in todo], a.nproc)):
    cache[k] = s
json.dump(cache, open(cache_f, "w"))
scaf = {k: cache[k] for k in eq}

hf = Path(a.holdout)
if a.define or not hf.exists():
    held, stats = select_scaffold_holdout(scaf, a.frac, a.cap)
    json.dump(dict(defined_on=a.tag, stats=stats, scaffolds=sorted(held)), open(hf, "w"), indent=1)
    print("defined hold-out:", stats)
held = set(json.load(open(hf))["scaffolds"])
held_mols = {k for k, s in scaf.items() if s in held}

for name in (f"split{a.tag}.json", f"split_eqplus{a.tag}.json"):
    f = W / name
    if not f.exists():
        continue
    back = W / name.replace(".json", "_v1_noscaffold.json")
    split = json.load(open(back if back.exists() else f))  # always start from the original random split
    if not back.exists():
        json.dump(split, open(back, "w"))
    new = apply_scaffold_holdout(mols, split, held_mols)
    json.dump(new, open(f, "w"))
    print(name, {k: len(v) for k, v in new.items()})

info_f = W / f"data_info{a.tag}.json"
if info_f.exists():
    info = json.load(open(info_f))
    new = json.load(open(W / f"split{a.tag}.json"))
    info["scaffold_holdout"] = dict(file=str(hf),
                                    n_heldout_molecules=len(held_mols), n_molecules_with_scaffold=len(scaf),
                                    n_scaffold_failed=sum(s is None for s in scaf.values()),
                                    n_acyclic=sum(s == "" for s in scaf.values()))
    info.update({k: len(v) for k, v in new.items()})
    json.dump(info, open(info_f, "w"), indent=1)
