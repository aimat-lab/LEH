"""Geometries for TD-DFT excitation populations as training labels: training and validation entries of the 10k ablation.

Dataset data/wb97_all_100k_s1.pkl with split experiments/wb97/split_100k_s1_sub10k.json (the abl_leh_* runs). Each
entry gets its raw TURBOMOLE archive if one exists (thermal and S1-MD geometries: data/wb97/{t1,t2,s1r*}/raw; QCDGE
minima: data/wb97_eq/raw), otherwise archive = null and run_escf.py recomputes the ground state with the dataset
settings (pilot/t0 tranches and part of the QCDGE minima were stored as npz only).

Usage: python experiments/eigvec/tddft/select_train.py
Output: experiments/eigvec/tddft/selection_abl10k.json (id, idx, split, kind, archive, Z, R, gaps_ref, E0_ref [eV])
"""
import collections
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/eigvec/tddft"


def main():
    mols = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/wb97/split_100k_s1_sub10k.json"))
    arc = {}
    for l in open(OUT / "raw_index.txt"):
        p = l.strip()
        arc.setdefault(p.split("/")[-1].replace(".tar.gz", ""), []).append(p)
    sel, count = [], collections.Counter()
    for s in ("train", "val"):
        for i in split[s]:
            m = mols[i]
            stem = re.sub(r"_md$", "", m["id"])
            if m["kind"] == "qcdge_eq":
                cand = [a for a in arc.get(re.sub(r"_eq$", "", stem), []) if a.startswith("wb97_eq/")]
            else:
                cand = [a for a in arc.get(stem, []) if a.startswith("wb97/")]
            a = cand[0] if cand else None
            count[(s, m["kind"], a is not None)] += 1
            sel.append(dict(id=m["id"], idx=int(i), split=s, kind=m["kind"], archive=a, Z=m["Z"].tolist(),
                            R=m["R"].tolist(), gaps_ref=[float(g) for g in m["gaps"]], E0_ref=float(m["E0"])))
    assert len({e["id"] for e in sel}) == len(sel), "duplicate ids"
    for k, v in sorted(count.items()):
        print(k, v)
    json.dump(sel, open(OUT / "selection_abl10k.json", "w"))
    print(f"written {len(sel)} entries to {OUT / 'selection_abl10k.json'}")


if __name__ == "__main__":
    main()
