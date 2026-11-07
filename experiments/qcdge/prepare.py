"""Build data/qcdge_sub.pkl plus size-separated and random splits."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.load_qcdge import scan_heavy_counts, read_molecules  # noqa
from exmlip.data.splits import scaffold_of  # noqa

rng = np.random.default_rng(0)
k9, nh9 = scan_heavy_counts(ROOT / "raw_data/qcdge/A_9.hdf5")
k10, nh10 = scan_heavy_counts(ROOT / "raw_data/qcdge/A_10.hdf5")
print("A_9 heavy-atom histogram", dict(zip(*np.unique(nh9, return_counts=True))), "A_10", len(k10), flush=True)
small = [k for k, n in zip(k9, nh9) if n <= 8]
nine = [k for k, n in zip(k9, nh9) if n == 9]
n_train_total = len(small)
nine_sel = list(rng.choice(nine, size=5000 + n_train_total, replace=False))
test9_keys, rand_pool9 = nine_sel[:5000], nine_sel[5000:]
ten_keys = list(rng.choice(k10, size=5000, replace=False))
# random-split control: same total size as the small-molecule pool, drawn from all sizes <= 9
# with the natural QM9 size distribution, excluding test_9 molecules
frac_small = len(small) / (len(small) + len(nine))
n_small_rand = int(round(frac_small * n_train_total))
rand_keys = list(rng.choice(small, n_small_rand, replace=False)) + rand_pool9[: n_train_total - n_small_rand]
keys9 = sorted(set(small) | set(test9_keys) | set(rand_keys))
mols = read_molecules(ROOT / "raw_data/qcdge/A_9.hdf5", keys9, "A9")
mols += read_molecules(ROOT / "raw_data/qcdge/A_10.hdf5", ten_keys, "A10")
print("read", len(mols), flush=True)
bad = [i for i, m in enumerate(mols) if len(m["gaps"]) != 10 or not np.all(m["gaps"] > 0)]
print("molecules with <10 singlets or nonpositive energies:", len(bad))
badids = {mols[i]["id"] for i in bad}
for m in mols:
    m["scaffold"] = scaffold_of(Z=m["Z"], R=m["R"])
idx = {m["id"]: i for i, m in enumerate(mols)}
ok = lambda i: mols[i]["id"] not in badids
small_idx = [i for i in rng.permutation([idx["A9_" + k] for k in small]).tolist() if ok(i)]
nval = len(small_idx) // 10
train_scaf = {mols[i]["scaffold"] for i in small_idx[nval:]}
t9 = [idx["A9_" + k] for k in test9_keys if ok(idx["A9_" + k])]
t10 = [idx["A10_" + k] for k in ten_keys if ok(idx["A10_" + k])]
split_size = dict(train=small_idx[nval:], val=small_idx[:nval], test_9=t9, test_10=t10,
                  test_9_novelscaf=[i for i in t9 if mols[i]["scaffold"] not in train_scaf],
                  test_10_novelscaf=[i for i in t10 if mols[i]["scaffold"] not in train_scaf])
r = [i for i in rng.permutation([idx["A9_" + k] for k in rand_keys]).tolist() if ok(i)]
split_rand = dict(train=r[nval:], val=r[:nval], test_9=t9, test_10=t10)
pickle.dump(mols, open(ROOT / "data/qcdge_sub.pkl", "wb"))
json.dump(split_size, open(ROOT / "experiments/qcdge/split_size.json", "w"))
json.dump(split_rand, open(ROOT / "experiments/qcdge/split_rand.json", "w"))
info = {k: len(v) for k, v in split_size.items()}
info["rand_train_n_heavy"] = dict(zip(*[x.tolist() for x in np.unique([mols[i]["n_heavy"] for i in split_rand["train"]], return_counts=True)]))
info["train_n_heavy"] = dict(zip(*[x.tolist() for x in np.unique([mols[i]["n_heavy"] for i in split_size["train"]], return_counts=True)]))
json.dump(info, open(ROOT / "experiments/qcdge/split_info.json", "w"), indent=1)
print(info)
