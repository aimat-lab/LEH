"""Model-B datasets: conjugated dyes and ladders appended to the final dataset (same TDDFT protocol).

Inputs: data/conj/{dyes_eq,dyes_md,ladders_eq,ladders_md}/*.npz, data/conj/tasks/*.pkl (metadata),
data/wb97_all_100k_s1.pkl and experiments/wb97/split_100k_s1_suball.json (final split incl. scaffold hold-out).

Outputs (indices refer to data/wb97_all_100k_s1_conj.pkl = the 100k_s1 entries, unchanged and in the same order,
followed by the conjugated entries, so every 100k split index stays valid):
- experiments/conj/split_B.json      model B: 100k training split + dye training scaffolds (incl. S); validation =
                                     shared 100k validation subset + dye validation scaffolds; tests = all 100k tests,
                                     held-out dye scaffolds and all ladders (never trained on), separated by S content.
- experiments/conj/split_B_noS.json  control: as model B but without any S compound (train, val and tests).
- experiments/conj/split_A_conj.json evaluation of model A (trained on 100k only, no S): all S-free dyes and ladders
                                     (model A never saw dyes), plus the held-out dye scaffolds for a like-for-like
                                     comparison with model B.
- experiments/conj/conj_info.json    counts and filters.
Dye split: balanced Bemis-Murcko scaffold split on typed scaffolds from the SMILES: scaffold families with at most
1 % of the dyes are eligible; md5 order; 10 % of the dyes test, 10 % validation, rest training.
Filters as for the 100k set: MD geometries with a connectivity change or with E0 below the S0 energy at the
molecule's own minimum (a reaction during the hot xTB MD) are dropped.
Test keys are non-overlapping: test_{dye,lad}_{noS,S}_{eq,offeq}; test_dyeall_noS_{eq,offeq} (split_A_conj only).

Usage: python experiments/conj/prepare_conj.py
"""
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.load_wb97 import load_dirs  # noqa

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "experiments/conj"
DIRS = ["dyes_eq", "dyes_md", "ladders_eq", "ladders_md"]
TASKS = {"dyes_eq": "dyes_eq", "dyes_md": "dyes_md_all", "ladders_eq": "ladders_eq", "ladders_md": "ladders_md"}
META = ["smiles", "has_S", "db", "lam_abs_median", "family", "n", "ends", "holdout_longest"]

# --- conjugated entries with metadata ---------------------------------------------------------------------------
conj, stats = load_dirs([ROOT / "data/conj" / d for d in DIRS], add_eq=False)
meta = {}
for d, t in TASKS.items():
    for task in pickle.load(open(ROOT / f"data/conj/tasks/{t}.pkl", "rb")):
        meta.setdefault(task["mol"], {k: task[k] for k in META if k in task})
for m in conj:
    m.update(meta[m["mol"]])
    m["conj"] = "dye" if m["source"] == "DYE" else "ladder"
    m["has_S"] = bool(m.get("has_S", False)) or bool((np.asarray(m["Z"]) == 16).any())
n0 = len(conj)
e_min = {m["mol"]: m["E0"] for m in conj if m["kind"] == "eq"}
below = {m["id"] for m in conj if m["kind"] == "offeq" and m["mol"] in e_min and m["E0"] < e_min[m["mol"]]}
conj = [m for m in conj if not m["conn_changed"] and m["id"] not in below]
stats.update(n_loaded=n0, dropped_conn_changed=stats["conn_changed"],
             dropped_below_minimum=len(below), n_kept=len(conj))


# --- dye scaffold split -------------------------------------------------------------------------------------------
def scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return None


dye_mols = sorted({m["mol"] for m in conj if m["conj"] == "dye"})
scaf = {mm: scaffold(meta[mm]["smiles"]) for mm in dye_mols}
cnt = Counter(scaf.values())
N = len(dye_mols)
elig = [s for s, c in cnt.items() if s and c <= 0.01 * N]


def take(salt, exclude, frac):
    order = sorted((s for s in elig if s not in exclude), key=lambda s: hashlib.md5((salt + s).encode()).hexdigest())
    held, n = set(), 0
    for s in order:
        if n >= frac * N:
            break
        held.add(s); n += cnt[s]
    return held


test_scaf = take("dyetest:", set(), 0.10)
val_scaf = take("dyeval:", test_scaf, 0.10)
role = {mm: "test" if scaf[mm] in test_scaf else ("val" if scaf[mm] in val_scaf else "train") for mm in dye_mols}

# --- combined data --------------------------------------------------------------------------------------------------
base = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
off = len(base)
data = base + conj
sp = json.load(open(ROOT / "experiments/wb97/split_100k_s1_suball.json"))
tests100 = {k: v for k, v in sp.items() if k.startswith("test_")}
idx = {}  # (set, S, kind, role) -> indices
for i, m in enumerate(conj, start=off):
    r = role.get(m["mol"], "test") if m["conj"] == "dye" else "test"
    key = ("dye" if m["conj"] == "dye" else "lad", "S" if m["has_S"] else "noS", "eq" if m["kind"] == "eq" else "offeq", r)
    idx.setdefault(key, []).append(i)
get = lambda *k: idx.get(tuple(k), [])


def test_keys(S_options, roles=("test",)):
    out = {}
    for s in ("dye", "lad"):
        for S in S_options:
            for kind in ("eq", "offeq"):
                v = sorted(i for r in roles for i in get(s, S, kind, r))
                if v:
                    out[f"test_{s}_{S}_{kind}"] = v
    return out


split_B = dict(train=sp["train"] + sorted(i for S in ("noS", "S") for k in ("eq", "offeq") for i in get("dye", S, k, "train")),
               val=sp["val"] + sorted(i for S in ("noS", "S") for k in ("eq", "offeq") for i in get("dye", S, k, "val")),
               **tests100, **test_keys(("noS", "S")))
split_B_noS = dict(train=sp["train"] + sorted(i for k in ("eq", "offeq") for i in get("dye", "noS", k, "train")),
                   val=sp["val"] + sorted(i for k in ("eq", "offeq") for i in get("dye", "noS", k, "val")),
                   **tests100, **test_keys(("noS",)))
split_A = dict(train=sp["train"], val=sp["val"], **tests100, **test_keys(("noS",)))
for kind in ("eq", "offeq"):
    split_A[f"test_dyeall_noS_{kind}"] = sorted(i for r in ("train", "val", "test") for i in get("dye", "noS", kind, r))

pickle.dump(data, open(ROOT / "data/wb97_all_100k_s1_conj.pkl", "wb"))
for name, s in [("split_B", split_B), ("split_B_noS", split_B_noS), ("split_A_conj", split_A)]:
    json.dump(s, open(OUT / f"{name}.json", "w"))
info = dict(stats={k: v for k, v in stats.items() if not isinstance(v, (list, dict))}, n_base=off, n_conj=len(conj),
            n_dye_molecules=N, n_dye_scaffold_families=len(cnt), dye_eligible_families=len(elig),
            dye_roles=Counter(role.values()), dye_roles_with_S=Counter(r for mm, r in role.items() if meta[mm]["has_S"]),
            n_ladder_molecules=len({m["mol"] for m in conj if m["conj"] == "ladder"}),
            n_ladder_S=len({m["mol"] for m in conj if m["conj"] == "ladder" and m["has_S"]}),
            splits={name: {k: len(v) for k, v in s.items()} for name, s in
                    [("split_B", split_B), ("split_B_noS", split_B_noS), ("split_A_conj", split_A)]})
json.dump(info, open(OUT / "conj_info.json", "w"), indent=1, default=int)
print(json.dumps(info, indent=1, default=int))
