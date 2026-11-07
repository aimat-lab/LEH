"""Fixed-budget sampling test (user request 2026-10-16): N molecules x K geometries per molecule at constant N*K.

Tells whether new TDDFT budget is better spent on new molecules or on more geometries of known molecules.
Only thermal sampling is compared: geometries are the QCDGE minimum (eq) and xTB-MD snapshots (offeq) of the training
molecules; S1-sampled geometries (excmd) are left out so that K is the only difference between configurations.
- pool: training molecules with an eq and >= 1 offeq geometry, fixed md5 order; K=2 molecules are a subset of the
  K=1 molecules at the same budget;
- K=1: one geometry per molecule, eq or offeq chosen by an md5 bit (~50/50, the same kind mix as K=2);
- K=2: eq + one offeq; K=3: eq + two offeq (only the ~3.7k molecules with two MD snapshots, md5 order; 33/67 mix);
- val and all test sets are those of the learning-curve subsets (split_100k_s1_sub10k.json), so the runs compare
  directly with the learning curves.
Configurations: G20k: 20k x 1, 10k x 2; G10k: 10k x 1, 5k x 2, 3.33k x 3.
Writes experiments/wb97/split_100k_s1_nk_<name>.json and nk_info.json.
"""
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "experiments/wb97"
h = lambda s: hashlib.md5(f"nk:{s}".encode()).hexdigest()

D = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
full = json.load(open(W / "split_100k_s1_suball.json"))
ref = json.load(open(W / "split_100k_s1_sub10k.json"))
geo = defaultdict(lambda: dict(eq=[], offeq=[]))
for i in full["train"]:
    k = D[i]["kind"]
    if k == "qcdge_eq":
        geo[D[i]["mol"]]["eq"].append(i)
    elif k == "offeq":
        geo[D[i]["mol"]]["offeq"].append(i)
pool = sorted((m for m, g in geo.items() if g["eq"] and g["offeq"]), key=h)
pool3 = sorted((m for m, g in geo.items() if g["eq"] and len(g["offeq"]) >= 2), key=h)


def pick(m, K):
    g = geo[m]
    if K == 1:
        return [g["eq"][0]] if int(h(m + ":k1"), 16) % 2 == 0 else [g["offeq"][0]]
    return [g["eq"][0]] + g["offeq"][: K - 1]


CONF = {"G20k_K1": (pool, 20000, 1), "G20k_K2": (pool, 10000, 2),
        "G10k_K1": (pool, 10000, 1), "G10k_K2": (pool, 5000, 2), "G10k_K3": (pool3, 3333, 3)}
info = dict(n_pool=len(pool), n_pool3=len(pool3), configs={})
for name, (pl, n, K) in CONF.items():
    assert len(pl) >= n, (name, len(pl), n)
    tr = [i for m in pl[:n] for i in pick(m, K)]
    sp = {k: v for k, v in ref.items() if k != "train"}
    sp["train"] = tr
    json.dump(sp, open(W / f"split_100k_s1_nk_{name}.json", "w"))
    n_eq = sum(D[i]["kind"] == "qcdge_eq" for i in tr)
    info["configs"][name] = dict(molecules=n, K=K, geometries=len(tr), eq=n_eq, offeq=len(tr) - n_eq)
    print(name, info["configs"][name])
json.dump(info, open(W / "nk_info.json", "w"), indent=1)
