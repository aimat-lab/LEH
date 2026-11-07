"""Are the eigenvectors of the latent Hamiltonian consistent across the four production LEH seeds?

H is identified by the eigenvalue loss only up to H -> Q H Q^T. Locality excludes Q that mix distant atoms, so the exact
remaining freedom is a sign per atom (k = 1). The per-atom weight w_{n,i} = v_{i,n}^2 is invariant under it. Mixing of
nearby atoms and different minima of different seeds change w. This script measures how similar w is across seeds.

Geometries: training and random-test geometries of the production split, matched by kind (QCDGE minima, thermal,
excited-state sampled), 1500 each (fixed seed). Model disagreement is separated from the eigenvector question by binning
geometries by the seed standard deviation of lambda_1..lambda_3 (max over the three states). Near-degenerate states are
excluded state-wise (gap to the neighbouring eigenvalues < 0.1 eV in any seed), because their eigenvectors mix under
arbitrarily small perturbations.

Per seed pair and state:
  shared weight   S = sum_i min(w_i^A, w_i^B)          (1: identical weight distributions, 1 - total variation)
  overlap         O = |sum_i v_i^A v_i^B|                (also sensitive to per-atom signs)
  sign-free       B = sum_i |v_i^A| |v_i^B|              (Bhattacharyya coefficient of w; B - O: sign-gauge differences)
  null            S after permuting the atoms of seed B within each element (10 permutations), i.e. the similarity
                  expected from elemental composition alone.
  S_d, null_d     S and null after smoothing w over all atoms within d = 1.6 / 2.6 Angstrom (bonded / second
                  neighbours): w~_i = sum_{j: r_ij < d} w_j / norm. Mixing of atoms closer than d changes w~ much less
                  than w, so S_d >> S points to seeds that differ by a local re-mixing of nearby atoms.

Usage: python experiments/eigvec/seed_consistency.py [--predict] [--threads 12]
Outputs: experiments/eigvec/eigvecs_prodw_leh.pkl (cache), experiments/eigvec/seed_consistency.json
"""
import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.data.batching import MolDataset, collate, iterate  # noqa: E402
from exmlip.training.train import build_model_from_dir  # noqa: E402

RUNS = [ROOT / f"experiments/al/runs/prodw_leh_v4_s{s}" for s in range(4)]
OUT = ROOT / "experiments/eigvec"
CACHE = OUT / "eigvecs_prodw_leh.pkl"
NV = 3          # states analysed (S1-S3)
N_PER = 1500    # geometries per set and kind
DEG = 0.1       # eV, minimum gap to neighbouring eigenvalues for state-wise comparison
BINS = [0.0, 0.02, 0.05, 0.1, np.inf]  # eV, seed std of lambda_1..3 (max over states)


def select(mols, split):
    rng = np.random.default_rng(0)
    pick = lambda idx: sorted(rng.choice(idx, size=min(N_PER, len(idx)), replace=False).tolist())
    tr = split["train"]
    sets = {}
    for kind in ("qcdge_eq", "offeq", "excmd"):
        sets[f"train|{kind}"] = pick([i for i in tr if mols[i]["kind"] == kind])
    sets["test|qcdge_eq"] = pick(split["test_random_eq"])
    sets["test|offeq"] = pick(split["test_random_offeq"])
    sets["test|excmd"] = pick([i for k in ("test_random_exc", "test_random_exc2", "test_random_exc3", "test_random_exc4")
                               for i in split[k]])
    return sets


@torch.no_grad()
def eig_one_model(run, mols):
    model, cfg = build_model_from_dir(run)
    res, maxdev = [], 0.0
    for batch in iterate(MolDataset(mols), 64, False):
        b = collate(batch, cfg["n_states"], 0)
        out = model(b["Z"], b["R"], b["batch"], b["nmol"])
        H, M = out["H"], out["M"]
        for i in range(len(batch)):
            Mi = int(M[i])
            lam, V = torch.linalg.eigh(H[i, :Mi, :Mi])
            maxdev = max(maxdev, float((lam - out["eigvals"][i, :Mi]).abs().max()))
            res.append(dict(lam=lam.numpy().copy(), V=V[:, :NV].numpy().copy()))
    print(f"  {run.name}: {len(res)} geometries, max |eigh - model eigenvalues| = {maxdev:.2e} eV", flush=True)
    return res


def predict(threads):
    torch.set_num_threads(threads)
    mols = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    sets = select(mols, split)
    idx = sorted({i for v in sets.values() for i in v})
    sub = [mols[i] for i in idx]
    seeds = [eig_one_model(r, sub) for r in RUNS]
    meta = [dict(idx=i, id=m["id"], mol=m["mol"], kind=m["kind"], Z=m["Z"], R=m["R"], gaps_ref=m["gaps"],
                 smiles=m.get("smiles")) for i, m in zip(idx, sub)]
    pickle.dump(dict(sets=sets, idx=idx, meta=meta, seeds=seeds), open(CACHE, "wb"))


DS = (1.6, 2.6)  # Angstrom, smoothing radii


def smooth(w, K):
    x = K @ w
    return x / x.sum()


def metrics(vA, vB, Z, Ks, rng, nperm=10):
    wA, wB = vA ** 2, vB ** 2
    S = float(np.minimum(wA, wB).sum())
    O = float(abs(vA @ vB))
    B = float(np.abs(vA) @ np.abs(vB))
    Sd = [float(np.minimum(smooth(wA, K), smooth(wB, K)).sum()) for K in Ks]
    null, nulld = [], []
    for _ in range(nperm):
        p = np.arange(len(Z))
        for z in np.unique(Z):
            k = np.where(Z == z)[0]
            p[k] = rng.permutation(k)
        null.append(np.minimum(wA, wB[p]).sum())
        nulld.append([np.minimum(smooth(wA, K), smooth(wB[p], K)).sum() for K in Ks])
    return [S, O, B, float(np.mean(null))] + Sd + list(np.mean(nulld, 0))


def analyse():
    C = pickle.load(open(CACHE, "rb"))
    pos = {i: k for k, i in enumerate(C["idx"])}
    rng = np.random.default_rng(1)
    pairs = list(combinations(range(len(RUNS)), 2))
    res = {}
    for name, idxs in C["sets"].items():
        rows = []
        for gi in idxs:
            k = pos[gi]
            Z = np.asarray(C["meta"][k]["Z"])
            if len(Z) < NV + 1:  # molecules with fewer than four atoms have fewer than four eigenvalues
                continue
            R = np.asarray(C["meta"][k]["R"])
            D = np.linalg.norm(R[:, None] - R[None], axis=-1)
            Ks = [(D < d).astype(float) for d in DS]
            lams = np.array([C["seeds"][s][k]["lam"][: NV + 1] for s in range(len(RUNS))])  # [seed, NV+1]
            dis = float(lams[:, :NV].std(0).max())
            for n in range(NV):
                lo = lams[:, n] - (lams[:, n - 1] if n > 0 else -np.inf)
                hi = lams[:, n + 1] - lams[:, n]
                gap = float(min(lo.min(), hi.min()))
                Vs = [C["seeds"][s][k]["V"][:, n] for s in range(len(RUNS))]
                m = np.array([metrics(Vs[a], Vs[b], Z, Ks, rng) for a, b in pairs]).mean(0)
                w = np.mean([v ** 2 for v in Vs], 0)
                rows.append(dict(state=n + 1, dis=dis, gap=gap, S=m[0], O=m[1], B=m[2], null=m[3],
                                 S16=m[4], S26=m[5], null16=m[6], null26=m[7],
                                 pr=float(1 / (w ** 2).sum() / len(Z)), wH=float(w[Z == 1].sum())))
            # S1-S3 subspace weight, invariant under rotations within the subspace
            gap4 = float((lams[:, NV] - lams[:, NV - 1]).min())
            Ws = [(C["seeds"][s][k]["V"] ** 2).sum(1) / NV for s in range(len(RUNS))]
            S = np.mean([np.minimum(Ws[a], Ws[b]).sum() for a, b in pairs])
            rows.append(dict(state=0, dis=dis, gap=gap4, S=float(S), O=np.nan, B=np.nan, null=np.nan, pr=np.nan,
                             wH=np.nan, S16=np.nan, S26=np.nan, null16=np.nan, null26=np.nan))
        res[name] = rows
    # summaries: per set, state (0 = S1-S3 subspace), disagreement bin, non-degenerate only
    summ = {}
    for name, rows in res.items():
        for st in range(NV + 1):
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                sel = [r for r in rows if r["state"] == st and lo <= r["dis"] < hi and r["gap"] >= DEG]
                nall = len([r for r in rows if r["state"] == st and lo <= r["dis"] < hi])
                if not sel:
                    continue
                agg = {q: float(np.nanmedian([r[q] for r in sel]))
                       for q in ("S", "O", "B", "null", "pr", "wH", "S16", "S26", "null16", "null26")}
                agg.update(n=len(sel), n_all=nall,
                           S_q10=float(np.quantile([r["S"] for r in sel], 0.1)),
                           frac_S_above_0p8=float(np.mean([r["S"] > 0.8 for r in sel])))
                summ[f"{name}|S{st if st else '1-3 sub'}|{lo}-{hi}"] = agg
    json.dump(dict(summary=summ, settings=dict(n_per=N_PER, deg_eV=DEG, bins=[str(b) for b in BINS],
                                               runs=[r.name for r in RUNS])),
              open(OUT / "seed_consistency.json", "w"), indent=1)
    pickle.dump(res, open(OUT / "seed_consistency_rows.pkl", "wb"))
    print(f"{'set|state|seed-std bin (eV)':42s} {'n':>5s} {'n_all':>5s} {'shared':>6s} {'q10':>5s} {'>0.8':>5s} "
          f"{'null':>5s} {'|v.v|':>5s} {'sgnfree':>7s} {'PR/M':>5s} {'w_H':>5s} {'S1.6':>5s} {'n1.6':>5s} "
          f"{'S2.6':>5s} {'n2.6':>5s}")
    for k, a in summ.items():
        print(f"{k:42s} {a['n']:5d} {a['n_all']:5d} {a['S']:6.2f} {a['S_q10']:5.2f} {a['frac_S_above_0p8']:5.2f} "
              f"{a['null']:5.2f} {a['O']:5.2f} {a['B']:7.2f} {a['pr']:5.2f} {a['wH']:5.2f} {a['S16']:5.2f} "
              f"{a['null16']:5.2f} {a['S26']:5.2f} {a['null26']:5.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.predict or not CACHE.exists():
        predict(a.threads)
    analyse()
