"""Does the coupling structure of the latent Hamiltonian predict transfer to ladder molecules across seeds?

Hypothesis from the four production seeds (seed_features.py): the seed with strong couplings between bonded atoms and
weak longer-range couplings (short coupling range) has delocalized S1 eigenvectors on long conjugated chains and the
smallest size-dependent S1 overestimate. Test: all existing LEH models (k = 1, 128 channels) in groups of seeds trained
on identical data. Within each group, compare coupling features with the S1 error on the sulfur-free ladder molecules
at equilibrium (no model saw ladder molecules) and with the S1-S3 error on 500 random-test thermal geometries.

Usage: python experiments/eigvec/coupling_vs_transfer.py [--threads 12]
Output: experiments/eigvec/coupling_vs_transfer.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/eigvec"))
from seed_features import features  # noqa: E402

R = ROOT / "experiments"
GROUPS = {
    "production, weighted": [R / f"al/runs/prodw_leh_v4_s{s}" for s in range(4)],
    "production, uniform": [R / f"al/runs/prod_leh_v4_s{s}" for s in range(2)],
    "sampler round 2": [R / f"al/runs/sampler_v2_s{s}" for s in range(2)],
    "sampler round 3": [R / f"al/runs/sampler_v3_s{s}" for s in range(2)],
    "sampler round 4": [R / f"al/runs/sampler_v4_s{s}" for s in range(2)],
    "benchmark, all": [R / f"wb97/runs_main/main_leh_all_s{s}" for s in range(2)],
    "benchmark, 3k": [R / f"wb97/runs_main/main_leh_3k_s{s}" for s in range(3)],
    "benchmark, 1k": [R / f"wb97/runs_main/main_leh_1k_s{s}" for s in range(3)],
    "ablation 10k": [R / f"wb97/runs_abl/abl_leh_s{s}" for s in range(3)],
    "ablation 10k, S1-S6": [R / f"wb97/runs_abl/abl_leh_sup6_s{s}" for s in range(3)],
    "ablation 10k, no S1 geom.": [R / f"wb97/runs_abl/abl_leh_noexc_s{s}" for s in range(3)],
}
Q = ("cb", "cf", "crange", "pr", "rad", "wH", "onsite")


def summarise(rows):
    d = {q: float(np.nanmedian([r[q] for r in rows])) for q in Q}
    d["mae1"] = float(np.mean(np.abs([r["err1"] for r in rows])))
    d["mean_err1"] = float(np.mean([r["err1"] for r in rows]))
    d["mae13"] = float(np.mean([r["err13"] for r in rows]))
    big = [r for r in rows if r["n_heavy"] >= 22]
    if big:
        d["mean_err1_big"] = float(np.mean([r["err1"] for r in big]))
        d["pr_big"] = float(np.median([r["pr"] for r in big]))
    return d


def main(threads):
    torch.set_num_threads(threads)
    mols = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    rng = np.random.default_rng(0)
    lad = [mols[i] for i in split["test_lad_noS_eq"]]
    rnd = [mols[i] for i in sorted(rng.choice(split["test_random_offeq"], 500, replace=False).tolist())]
    res = {}
    for g, runs in GROUPS.items():
        res[g] = {}
        for r in runs:
            if not (r / "best.pt").exists():
                continue
            res[g][r.name] = dict(ladders=summarise(features(r, lad)), random=summarise(features(r, rnd)))
            a = res[g][r.name]
            print(f"{g:28s} {r.name:22s} lad S1 MAE {a['ladders']['mae1']:.3f} signed {a['ladders']['mean_err1']:+.3f} "
                  f"(>=22 heavy {a['ladders'].get('mean_err1_big', np.nan):+.3f})  cb {a['ladders']['cb']:.3f} "
                  f"cf {a['ladders']['cf']:.3f} range {a['ladders']['crange']:.2f} PR>=22 {a['ladders'].get('pr_big', np.nan):5.1f}"
                  f" | rand S1-3 {a['random']['mae13']:.3f}", flush=True)
        json.dump(res, open(ROOT / "experiments/eigvec/coupling_vs_transfer.json", "w"), indent=1)
    # within-group rank agreement: does a higher bonded/longer-range coupling ratio go with a lower ladder S1 MAE?
    agree, total = 0, 0
    for g, d in res.items():
        names = list(d)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = d[names[i]]["ladders"], d[names[j]]["ladders"]
                ra, rb = a["cb"] / a["cf"], b["cb"] / b["cf"]
                agree += int((ra - rb) * (a["mae1"] - b["mae1"]) < 0); total += 1
    print(f"\nwithin-group seed pairs where the larger cb/cf ratio has the lower ladder S1 MAE: {agree} of {total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    main(ap.parse_args().threads)
