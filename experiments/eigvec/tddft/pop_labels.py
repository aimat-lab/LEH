"""Training labels for the population loss (train.py w_pop) from the escf population files of run_escf.py.

Per dataset entry: TD-DFT hole and electron populations of the 10 roots (Loewdin, populations.py), hydrogens folded onto
their nearest heavy atom (as in compare.py), and a mask of the roots separated from both neighbouring roots by >= DEG
(root 10 has no computed upper neighbour and is never used). Entries without a population file, or without heavy atoms,
get no labels.

Usage: python experiments/eigvec/tddft/pop_labels.py <pop dir> [--selection selection_abl10k.json] [--out pop_labels_abl10k.pkl]
Output: pickle {dataset index: dict(pop_hole [n,10], pop_elec [n,10], pop_fold [n], pop_mask [10])} and a summary
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
DEG = 0.1  # eV


def fold_map(Z, R):
    heavy = np.where(Z > 1)[0]
    dist = np.linalg.norm(R[:, None] - R[None, heavy], axis=-1)
    return heavy[np.argmin(dist, 1)]  # atom -> heavy atom it belongs to (heavy atoms map to themselves)


def root_mask(exc):
    lo = np.r_[np.inf, np.diff(exc)]
    hi = np.r_[np.diff(exc), -np.inf]  # last root: upper neighbour unknown
    return (np.minimum(lo, hi) >= DEG) & (hi > -np.inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop_dir")
    ap.add_argument("--selection", default=str(D / "selection_abl10k.json"))
    ap.add_argument("--out", default=str(D / "pop_labels_abl10k.pkl"))
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    labels, miss, noheavy, nmask = {}, 0, 0, []
    for e in sel:
        f = Path(a.pop_dir) / f"{e['id']}.npz"
        if not f.exists():
            miss += 1
            continue
        t = np.load(f)
        Z, R = np.asarray(e["Z"]), np.asarray(e["R"])
        assert np.abs(t["R"] - R).max() < 1e-5 and (t["Z"] == Z).all(), e["id"]
        if not (Z > 1).any():
            noheavy += 1
            continue
        fm = fold_map(Z, R)
        n = len(Z)
        hole = np.zeros((n, t["hole"].shape[0]))
        elec = np.zeros_like(hole)
        np.add.at(hole, fm, t["hole"].T)
        np.add.at(elec, fm, t["elec"].T)
        mask = root_mask(np.asarray(t["exc_eV"]))
        nmask.append(mask)
        labels[e["idx"]] = dict(pop_hole=hole.astype(np.float32), pop_elec=elec.astype(np.float32),
                                pop_fold=fm.astype(np.int64), pop_mask=mask)
    nm = np.array(nmask)
    print(f"labels {len(labels)} of {len(sel)} (missing population file {miss}, no heavy atom {noheavy})")
    print("fraction of entries with root n usable:", " ".join(f"S{i + 1} {x:.2f}" for i, x in enumerate(nm.mean(0))))
    pickle.dump(labels, open(a.out, "wb"))
    print("written", a.out)


if __name__ == "__main__":
    main()
