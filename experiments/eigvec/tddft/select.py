"""Geometries for the TD-DFT check of the LEH eigenvectors.

- random: training and random-test geometries of the seed-consistency analysis (experiments/eigvec/seed_consistency.py)
  per set and kind (QCDGE minima, thermal, excited-state sampled), the N_LOW with the lowest seed disagreement
  (max over S1-S3 of the seed std of the eigenvalues) among those with a raw TURBOMOLE archive.
- ladders: sulfur-free ladder molecules at their minima (test_lad_noS_eq), N_LAD per heavy-atom bin.
- dyes: sulfur-free test dyes at their minima (test_dye_noS_eq), N_DYE at random.

Usage: python experiments/eigvec/tddft/select.py
Output: experiments/eigvec/tddft/selection.json (id, dataset index, group, archive path relative to data/, Z, R)
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/eigvec/tddft"
N_LOW, N_LAD, N_DYE = 40, 40, 60
LAD_BINS = ((0, 10), (10, 16), (16, 22), (22, 99))


def archive_lookup():
    arc = {}
    for l in open(OUT / "raw_index.txt"):
        p = l.strip()
        arc.setdefault(p.split("/")[-1].replace(".tar.gz", ""), []).append(p)
    return arc


def archive_of(mid, arc):
    stem = re.sub(r"_md$", "", mid)
    if stem in arc:
        return arc[stem][0]
    if stem.endswith("_eq"):  # QCDGE minima: ground-state archive of compute_e0 (wb97_eq/raw/<mol>.tar.gz)
        return next((a for a in arc.get(stem[:-3], []) if a.startswith("wb97_eq/")), None)
    return None


def main():
    mols = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    C = pickle.load(open(ROOT / "experiments/eigvec/eigvecs_prodw_leh.pkl", "rb"))
    arc = archive_lookup()
    pos = {i: k for k, i in enumerate(C["idx"])}
    sel = []
    for name, idxs in C["sets"].items():
        cand = []
        for gi in idxs:
            k = pos[gi]
            if len(C["meta"][k]["Z"]) < 4:
                continue
            a = archive_of(mols[gi]["id"], arc)
            if a is None:
                continue
            lams = np.array([C["seeds"][s][k]["lam"][:3] for s in range(len(C["seeds"]))])
            cand.append((float(lams.std(0).max()), gi, a))
        cand.sort()
        print(f"{name:16s} candidates with archive {len(cand):5d}, chosen dis {cand[0][0]:.3f}-{cand[N_LOW - 1][0]:.3f} eV")
        sel += [dict(group=name, idx=gi, archive=a, dis=d) for d, gi, a in cand[:N_LOW]]
    rng = np.random.default_rng(0)
    lad = split["test_lad_noS_eq"]
    for lo, hi in LAD_BINS:
        pool = [i for i in lad if lo <= mols[i]["n_heavy"] < hi]
        pick = sorted(rng.choice(pool, size=min(N_LAD, len(pool)), replace=False).tolist())
        print(f"ladders heavy {lo}-{hi - 1}: {len(pool)} available, {len(pick)} chosen")
        sel += [dict(group="ladder", idx=i, archive=archive_of(mols[i]["id"], arc)) for i in pick]
    dyes = sorted(rng.choice(split["test_dye_noS_eq"], size=N_DYE, replace=False).tolist())
    sel += [dict(group="dye", idx=i, archive=archive_of(mols[i]["id"], arc)) for i in dyes]
    for s in sel:
        m = mols[s["idx"]]
        assert s["archive"] is not None, m["id"]
        s.update(id=m["id"], Z=np.asarray(m["Z"]).tolist(), R=np.asarray(m["R"]).tolist(),
                 gaps_ref=np.asarray(m["gaps"]).tolist())
    json.dump(sel, open(OUT / "selection.json", "w"))
    print(f"{len(sel)} geometries, {len({s['id'] for s in sel})} unique ids")


if __name__ == "__main__":
    main()
