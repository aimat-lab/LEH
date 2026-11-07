"""LEH eigenvalues and eigenvectors (S1-S5) on the geometries of selection.json.

Default: the four production seeds. --groups all: also every LEH model of coupling_vs_transfer.py (seed groups trained
on identical data), for the question whether agreement with TD-DFT predicts transfer.

Usage: python experiments/eigvec/tddft/leh_vectors.py [--groups prod|all] [--threads 12]
Output: experiments/eigvec/tddft/leh_vectors_<groups>.pkl  {run name: [dict(lam, V[:, :5]) per selection entry]}
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/eigvec"))
from exmlip.data.batching import MolDataset, collate, iterate  # noqa: E402
from exmlip.training.train import build_model_from_dir  # noqa: E402

NV = 5


@torch.no_grad()
def vectors(run, mols):
    model, cfg = build_model_from_dir(run)
    res = []
    for batch in iterate(MolDataset(mols), 16, False):
        b = collate(batch, cfg["n_states"], 0)
        out = model(b["Z"], b["R"], b["batch"], b["nmol"])
        for i in range(len(batch)):
            Mi = int(out["M"][i])
            lam, V = np.linalg.eigh(out["H"][i, :Mi, :Mi].double().numpy())
            res.append(dict(lam=lam, V=V[:, :NV].copy()))
    return res


def main(groups, threads):
    torch.set_num_threads(threads)
    sel = json.load(open(ROOT / "experiments/eigvec/tddft/selection.json"))
    mols = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    sub = [mols[s["idx"]] for s in sel]
    assert all(m["id"] == s["id"] for m, s in zip(sub, sel))
    runs = [ROOT / f"experiments/al/runs/prodw_leh_v4_s{s}" for s in range(4)]
    if groups == "all":
        from coupling_vs_transfer import GROUPS
        runs = [r for g in GROUPS.values() for r in g if (r / "best.pt").exists()]
    out = {}
    for r in runs:
        out[r.name] = vectors(r, sub)
        print(r.name, flush=True)
    pickle.dump(out, open(ROOT / f"experiments/eigvec/tddft/leh_vectors_{groups}.pkl", "wb"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="prod", choices=("prod", "all"))
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    main(a.groups, a.threads)
