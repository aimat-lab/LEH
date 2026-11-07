"""Monomer and dimer geometries of the fragment (size-consistency) test, shared by all models and the TDDFT reference.

60 random pairs of equilibrium test molecules (A+B) and 60 identical pairs (A+A, degenerate union) at minimum
interatomic distances D; B is randomly rotated and shifted along x (bisection on the minimum distance). Same pool,
rotations and placement as the first version of fig_fragments.py, so its curves are unchanged.
Dimers carry dummy labels (E0 = 0, gaps = 0) so that every baseline exporter accepts them; monomers keep their TDDFT
labels. Writes data/frag_pairs.pkl (list in exmlip format, ids frag_mono_<j> / frag_<kind>_<pair>_d<d>) and
experiments/fragments/split_frag.json (val: 10 monomers, only to satisfy exporters; test_frag_mono, test_frag_dimer).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.training.fragments import rand_rot  # noqa

D = [2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20]
NPAIR = 60


def place(RA, RB, dmin, rng):
    RA = RA - RA.mean(0)
    RB = (RB - RB.mean(0)) @ rand_rot(rng).T
    lo, hi = 0.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        dd = np.linalg.norm(RA[:, None] - (RB + [mid, 0, 0])[None], axis=-1).min()
        lo, hi = (mid, hi) if dd < dmin else (lo, mid)
    return RA, RB + [hi, 0, 0]


if __name__ == "__main__":
    mols = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/wb97/split_100k_s1_sub10k.json"))
    rng = np.random.default_rng(7)
    pool_idx = rng.choice(split["test_random_eq"], 2 * NPAIR, replace=False)
    out, mono, dim = [], [], []
    for j, i in enumerate(pool_idx):
        m = mols[i]
        out.append(dict(id=f"frag_mono_{j}", mol=m["mol"], src_index=int(i), Z=np.asarray(m["Z"]), R=np.asarray(m["R"], float),
                        E0=m["E0"], gaps=np.asarray(m["gaps"], float), F0=m.get("F0"), kind="frag_mono", source=m["source"],
                        n_heavy=int(m["n_heavy"])))
        mono.append(len(out) - 1)
    for kind in ("AB", "AA"):
        for k, d in enumerate(D):
            for p in range(NPAIR):
                a, b = 2 * p, (2 * p + 1 if kind == "AB" else 2 * p)
                RA, RB = place(out[a]["R"], out[b]["R"], d, np.random.default_rng(1000 * p + k))
                Z = np.concatenate([out[a]["Z"], out[b]["Z"]])
                out.append(dict(id=f"frag_{kind}_{p}_d{d:g}", mol=f"{out[a]['mol']}+{out[b]['mol']}", pair=p, pair_kind=kind,
                                d=float(d), mono=(a, b), Z=Z, R=np.concatenate([RA, RB]), E0=0.0, gaps=np.zeros(10),
                                F0=np.zeros((len(Z), 3)), kind="frag_dimer", source="frag", n_heavy=int((Z > 1).sum())))
                dim.append(len(out) - 1)
    pickle.dump(out, open(ROOT / "data/frag_pairs.pkl", "wb"))
    json.dump(dict(train=[], val=mono[:10], test_frag_mono=mono, test_frag_dimer=dim),
              open(ROOT / "experiments/fragments/split_frag.json", "w"))
    print(len(mono), "monomers,", len(dim), "dimers")
