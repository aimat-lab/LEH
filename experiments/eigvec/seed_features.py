"""What distinguishes the latent Hamiltonian of the outlier seed (prodw_leh_v4_s1) from the other three production seeds?

Per seed and geometry: signed S1 error; localization of the S1 eigenvector (participation number 1/sum w^2 in atoms,
spatial radius sqrt(sum_i w_i |r_i - c|^2), weight on hydrogen); coupling statistics of H (mean |H_ij| for bonded pairs
r < 1.7 A and for 1.7 A <= r < 5 A, coupling range sqrt(sum H_ij^2 r_ij^2 / sum H_ij^2), spread of the onsite terms);
unsupervised spectrum (lambda_11 - lambda_10). Sets: 1500 random-test thermal geometries, sulfur-free ladder molecules
and dyes at equilibrium (the sets where s1 generalizes worse).

Usage: python experiments/eigvec/seed_features.py [--threads 12]
Output: experiments/eigvec/seed_features.json (+ seed_features_rows.pkl)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.batching import MolDataset, collate, iterate  # noqa: E402
from exmlip.training.train import build_model_from_dir  # noqa: E402

RUNS = [ROOT / f"experiments/al/runs/prodw_leh_v4_s{s}" for s in range(4)]
OUT = ROOT / "experiments/eigvec"
SETS = ("test_random_offeq", "test_lad_noS_eq", "test_dye_noS_eq")


@torch.no_grad()
def features(run, mols):
    model, cfg = build_model_from_dir(run)
    rows = []
    for batch in iterate(MolDataset(mols), 32, False):
        b = collate(batch, cfg["n_states"], 0)
        out = model(b["Z"], b["R"], b["batch"], b["nmol"])
        for i, m in enumerate(batch):
            Mi = int(out["M"][i])
            if Mi < 4:  # fewer than four atoms: fewer than three excited states
                continue
            H = out["H"][i, :Mi, :Mi].numpy()
            lam, V = np.linalg.eigh(H)
            Z, R = np.asarray(m["Z"]), np.asarray(m["R"])
            D = np.linalg.norm(R[:, None] - R[None], axis=-1)
            off = ~np.eye(Mi, dtype=bool)
            bond, far = off & (D < 1.7), off & (D >= 1.7) & (D < 5.0)
            w = V[:, 0] ** 2
            c = w @ R
            h2 = H[off] ** 2
            rows.append(dict(
                err1=float(lam[0] - m["gaps"][0]), err13=float(np.mean(np.abs(lam[:3] - m["gaps"][:3]))),
                pr=float(1 / (w ** 2).sum()), rad=float(np.sqrt(w @ ((R - c) ** 2).sum(1))), wH=float(w[Z == 1].sum()),
                cb=float(np.abs(H[bond]).mean()) if bond.any() else np.nan,
                cf=float(np.abs(H[far]).mean()) if far.any() else np.nan,
                crange=float(np.sqrt((h2 * D[off] ** 2).sum() / h2.sum())),
                onsite=float(np.diag(H).std()), gap1110=float(lam[10] - lam[9]) if Mi > 10 else np.nan,
                n_heavy=int((Z > 1).sum()), id=m["id"]))
    return rows


def main(threads):
    torch.set_num_threads(threads)
    mols = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    rng = np.random.default_rng(0)
    sel = {s: sorted(rng.choice(split[s], size=min(1500, len(split[s])), replace=False).tolist()) for s in SETS}
    res = {s: [features(r, [mols[i] for i in idx]) for r in RUNS] for s, idx in sel.items()}
    pickle.dump(res, open(OUT / "seed_features_rows.pkl", "wb"))
    keys = ["err1", "err13", "pr", "rad", "wH", "cb", "cf", "crange", "onsite", "gap1110"]
    summ = {}
    for s, per_seed in res.items():
        summ[s] = {f"s{k}": {q: (float(np.nanmean(np.abs([r[q] for r in rows]))) if q in ("err13",) else
                                 float(np.nanmedian([r[q] for r in rows]))) for q in keys}
                   for k, rows in enumerate(per_seed)}
        for k, rows in enumerate(per_seed):
            summ[s][f"s{k}"]["mae1"] = float(np.mean(np.abs([r["err1"] for r in rows])))
            summ[s][f"s{k}"]["mean_err1"] = float(np.mean([r["err1"] for r in rows]))
    json.dump(summ, open(OUT / "seed_features.json", "w"), indent=1)
    cols = ["mae1", "mean_err1", "err13", "pr", "rad", "wH", "cb", "cf", "crange", "onsite", "gap1110"]
    print(f"{'set':18s} seed " + " ".join(f"{c:>9s}" for c in cols))
    for s in summ:
        for k in summ[s]:
            print(f"{s:18s} {k:4s} " + " ".join(f"{summ[s][k][c]:9.3f}" for c in cols))
    # ladders: S1 localization and error versus size, per seed
    print("\nladders: median S1 participation (atoms) / mean signed S1 error (eV) per heavy-atom bin")
    lad = res["test_lad_noS_eq"]
    for lo, hi in ((0, 10), (10, 16), (16, 22), (22, 31)):
        cells = []
        for k, rows in enumerate(lad):
            rr = [r for r in rows if lo <= r["n_heavy"] < hi]
            cells.append(f"s{k} {np.median([r['pr'] for r in rr]):5.1f}/{np.mean([r['err1'] for r in rr]):+.2f}"
                         if rr else f"s{k} -")
        print(f"  heavy {lo:2d}-{hi - 1:2d} (n={len([r for r in lad[0] if lo <= r['n_heavy'] < hi]):4d}): " + "  ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    main(ap.parse_args().threads)
