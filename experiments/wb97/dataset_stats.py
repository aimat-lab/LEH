"""Counts of the frozen v1 dataset for the paper (experiments/wb97/dataset_stats.json; read by make_numbers.py)."""
import json
import pickle
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = pickle.load(open(ROOT / "data/wb97_all_100k_s1_conj.pkl", "rb"))
S = json.load(open(ROOT / "experiments/wb97/split_100k_s1_suball.json"))
C = json.load(open(ROOT / "experiments/conj/split_A_conj.json"))
v1 = D[:222172]
kinds = Counter(m["kind"] for m in v1)
used = set(i for v in S.values() for i in v)
mols_used = {v1[i]["mol"] for i in used}
st = dict(n_entries_v1=len(v1), n_molecules_v1=len({m["mol"] for m in v1}), n_molecules_used=len(mols_used),
          n_geometries_used=len(used), kinds=dict(kinds),
          n_train_molecules=len({v1[i]["mol"] for i in S["train"]}), n_train_geometries=len(S["train"]),
          n_train_excmd=sum(v1[i]["kind"] == "excmd" for i in S["train"]),
          n_val_molecules=len({v1[i]["mol"] for i in S["val"]}), n_val_geometries=len(S["val"]),
          tests={k: len(v) for k, v in S.items() if k.startswith("test")},
          n_zinc_molecules=len({m["mol"] for m in v1 if m["source"] == "ZINC"}),
          n_conj_entries=len(D) - len(v1), n_dyes=len({m["mol"] for m in D[len(v1):] if m.get("conj") == "dye"}),
          n_ladders=len({m["mol"] for m in D[len(v1):] if m.get("conj") == "ladder"}),
          conj_tests={k: len(v) for k, v in C.items() if "dye" in k or "lad" in k},
          max_heavy_conj=max(int(m["n_heavy"]) for m in D[len(v1):]))
json.dump(st, open(ROOT / "experiments/wb97/dataset_stats.json", "w"), indent=1)
print(json.dumps(st, indent=1))
