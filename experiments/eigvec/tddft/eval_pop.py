"""Population-loss experiment (experiments/wb97/spec_abl_pop.yaml): does pulling the LEH eigenvector weights of S1-S3
towards the TD-DFT excitation populations change the agreement with TD-DFT, the transfer to ladders, or accuracy?

Models (10k ablation setting, identical except for the population term): abl_leh_s0-2 (baseline), abl_leh_pop_s0-2
(w_pop 1, target (hole + electron)/2), abl_leh_pop01_s0 (w_pop 0.1), abl_leh_popmin_s0 (target: closer of hole and
electron).
1. Agreement with TD-DFT on the 3,312 validation geometries of the 10k split (pop_labels_abl10k.pkl, never trained
   on): shared weight S (heavy-atom groups, as in compare.py) of eigenvector n to (hole + electron)/2, to the hole and
   to the electron, ln(PR_LEH / PR_TD-DFT); roots separated from their neighbours by >= 0.1 eV in TD-DFT and in LEH.
2. The same on the 460-geometry set of compare.py (training-type, test, sulfur-free ladders and dyes).
3. Transfer and accuracy: S1 MAE, signed error and R2 on the 1,226 sulfur-free ladder minima (also >= 22 heavy atoms)
   and on the 373 sulfur-free dye minima of split_prod_v4 (no model saw either), S1-S3 MAE on the 500 random off-eq
   test geometries of coupling_vs_transfer.py, median S1 participation number on ladders >= 22 heavy atoms.
In-distribution test metrics (MAE, R2 of energies and forces): experiments/wb97/analyze.py on runs_abl_pop.

Usage: python experiments/eigvec/tddft/eval_pop.py [--threads 12]
Output: experiments/eigvec/tddft/eval_pop.json (per model; seed mean and std per variant)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "experiments/eigvec/tddft"
sys.path.insert(0, str(D))
from compare import gap_ok, null, rows_for, shared  # noqa: E402
from leh_vectors import vectors  # noqa: E402

RUNS = ROOT / "experiments/wb97/runs_abl"
VARIANTS = {"baseline": [f"abl_leh_s{s}" for s in range(3)], "pop": [f"abl_leh_pop_s{s}" for s in range(3)],
            "pop w0.1": ["abl_leh_pop01_s0"], "pop min": ["abl_leh_popmin_s0"]}
NS = 3


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else np.nan


def r2(p, r):
    p, r = np.asarray(p), np.asarray(r)
    return float(1 - ((p - r) ** 2).sum() / ((r - r.mean()) ** 2).sum())


def val_agreement(vec, mols, kinds):
    """Shared weights on the validation geometries, per state and geometry kind (null: compare.null, element-
    preserving permutation of the LEH weights over the heavy atoms)."""
    rng = np.random.default_rng(0)
    rows = []
    for v, m, kind in zip(vec, mols, kinds):
        if "pop_hole" not in m:
            continue
        for n in range(NS):
            if not (m["pop_mask"][n] and gap_ok(v["lam"], n)):
                continue
            w = np.bincount(m["pop_fold"], weights=v["V"][:, n] ** 2, minlength=len(m["Z"]))
            h, e = m["pop_hole"][:, n].astype(float), m["pop_elec"][:, n].astype(float)
            p = 0.5 * (h + e)
            heavy = m["Z"] > 1
            rows.append(dict(state=n + 1, kind=kind, S=shared(w, p), S_hole=shared(w, h), S_elec=shared(w, e),
                             td_he=shared(h, e), null=null(w[heavy], p[heavy], m["Z"][heavy], rng),
                             lnpr=float(np.log((p[heavy] ** 2).sum() / (w[heavy] ** 2).sum()))))
    out = {}
    for n in range(1, NS + 1):
        for kind in ("all", "qcdge_eq", "offeq", "excmd"):
            rr = [r for r in rows if r["state"] == n and kind in ("all", r["kind"])]
            out[f"S{n}|{kind}"] = dict(n=len(rr), **{q: med([r[q] for r in rr]) for q in ("S", "S_hole", "S_elec", "null", "lnpr")},
                                       S_mean=float(np.mean([r["S"] for r in rr])),
                                       frac_hole=float(np.mean([r["S_hole"] > r["S_elec"] for r in rr])))
    return out


def set460(rows, m):
    """Agreement on the 460-geometry set (compare.rows_for rows) for model m."""
    sets = {"training-type": lambda r: r["group"].startswith("train|"), "test": lambda r: r["group"].startswith("test|"),
            "S1-MD (train+test)": lambda r: r["group"].endswith("|excmd"), "ladder": lambda r: r["group"] == "ladder",
            "ladder >= 22 heavy": lambda r: r["group"] == "ladder" and r["n_heavy"] >= 22, "dye": lambda r: r["group"] == "dye"}
    out = {}
    for n in range(1, NS + 1):
        for k, f in sets.items():
            rr = [r for r in rows if r["state"] == n and f(r) and r["td_gap_ok"] and r[m]["gap_ok"]]
            out[f"S{n}|{k}"] = dict(n=len(rr), **{q: med([r[m][q] for r in rr]) for q in ("S", "S_hole", "S_elec", "null")},
                                   lnpr=med([np.log(r[m]["pr"] / r["td_pr"]) for r in rr]),
                                   frac_hole=float(np.mean([r[m]["S_hole"] > r[m]["S_elec"] for r in rr])) if rr else np.nan)
    return out


def transfer(vec, mols):
    keep = [(v, m) for v, m in zip(vec, mols) if len(v["lam"]) >= 4]
    p1 = np.array([v["lam"][0] for v, _ in keep]); r1 = np.array([m["gaps"][0] for _, m in keep])
    nh = np.array([(m["Z"] > 1).sum() for _, m in keep])
    e13 = np.mean([np.mean(np.abs(v["lam"][:3] - m["gaps"][:3])) for v, m in keep])
    pr = np.array([1 / (v["V"][:, 0] ** 4).sum() for v, _ in keep])
    big = nh >= 22
    d = dict(n=len(keep), mae1=float(np.mean(np.abs(p1 - r1))), signed1=float(np.mean(p1 - r1)), r2_1=r2(p1, r1),
             mae13=float(e13))
    if big.any():
        d.update(n_big=int(big.sum()), mae1_big=float(np.mean(np.abs(p1 - r1)[big])),
                 signed1_big=float(np.mean((p1 - r1)[big])), pr1_big=float(np.median(pr[big])))
    return d


def main(threads):
    torch.set_num_threads(threads)
    models = [m for ms in VARIANTS.values() for m in ms]
    # 1. validation geometries of the 10k split with their TD-DFT populations
    mols = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/wb97/split_100k_s1_sub10k.json"))
    labels = pickle.load(open(D / "pop_labels_abl10k.pkl", "rb"))
    kind = {e["idx"]: e["kind"] for e in json.load(open(D / "selection_abl10k.json"))}
    val = [{**mols[i], **labels.get(i, {})} for i in split["val"]]
    vkind = [kind[i] for i in split["val"]]
    del mols
    # 2./3. 460-geometry set, ladders, dyes and random test geometries (all from wb97_prod_v4)
    pm = pickle.load(open(ROOT / "data/wb97_prod_v4.pkl", "rb"))
    ps = json.load(open(ROOT / "experiments/al/split_prod_v4.json"))
    sel = json.load(open(D / "selection.json"))
    sub = [pm[s["idx"]] for s in sel]
    assert all(m["id"] == s["id"] for m, s in zip(sub, sel))
    rng = np.random.default_rng(0)
    tsets = {"ladders": [pm[i] for i in ps["test_lad_noS_eq"]], "dyes": [pm[i] for i in ps["test_dye_noS_eq"]],
             "random off-eq 500": [pm[i] for i in sorted(rng.choice(ps["test_random_offeq"], 500, replace=False).tolist())]}
    del pm
    res, vec460 = {}, {}
    for m in models:
        run = RUNS / m
        res[m] = dict(val=val_agreement(vectors(run, val), val, vkind),
                      transfer={k: transfer(vectors(run, ms), ms) for k, ms in tsets.items()})
        vec460[m] = vectors(run, sub)
        log = [json.loads(l) for l in open(run / "train.log") if l.startswith("{")]
        res[m]["val_loss_best"] = min(x["val_loss"] for x in log)
        t = res[m]["transfer"]
        print(f"{m:20s} val S1 S {res[m]['val']['S1|all']['S']:.3f} | ladders S1 MAE {t['ladders']['mae1']:.3f} "
              f"R2 {t['ladders']['r2_1']:.3f} | dyes {t['dyes']['mae1']:.3f} | rand S1-3 {t['random off-eq 500']['mae13']:.3f}",
              flush=True)
    rows = rows_for(sel, vec460, models)
    for m in models:
        res[m]["set460"] = set460(rows, m)
    # seed mean and std per variant
    summ = {}
    for v, ms in VARIANTS.items():
        flat = {}
        for m in ms:
            for sec in ("val", "set460", "transfer"):
                for k, d in res[m][sec].items():
                    for q, x in d.items():
                        flat.setdefault(f"{sec}|{k}|{q}", []).append(x)
        summ[v] = {k: dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if len(x) > 1 else np.nan)
                   for k, x in flat.items()}
    json.dump(dict(models=res, variants=summ), open(D / "eval_pop.json", "w"), indent=1)
    show = [("val|S1|all|S", "val S1 S"), ("val|S2|all|S", "val S2 S"), ("val|S3|all|S", "val S3 S"),
            ("val|S1|all|lnpr", "val lnPR1"), ("val|S1|excmd|frac_hole", "S1-MD hole"),
            ("set460|S1|test|S", "460 test S1"), ("set460|S1|ladder|S", "460 lad S1"),
            ("set460|S1|ladder >= 22 heavy|lnpr", "lad22 lnPR1"), ("set460|S1|dye|S", "460 dye S1"),
            ("transfer|ladders|mae1", "lad S1 MAE"), ("transfer|ladders|r2_1", "lad S1 R2"),
            ("transfer|ladders|signed1_big", "lad22 sign"), ("transfer|dyes|mae1", "dye S1 MAE"),
            ("transfer|dyes|r2_1", "dye S1 R2"), ("transfer|random off-eq 500|mae13", "rand S1-3")]
    print("\n" + "variant".ljust(10) + "".join(h.rjust(13) for _, h in show))
    for v in VARIANTS:
        print(v.ljust(10) + "".join((f"{summ[v][k]['mean']:.3f}" + (f"±{summ[v][k]['std']:.3f}" if np.isfinite(summ[v][k]['std']) else "")).rjust(13)
                                    for k, _ in show))
    print("\nper model")
    for m in models:
        print(m.ljust(20) + "".join(f"{x:13.3f}" for x in
                                    (res[m]["val"]["S1|all"]["S"], res[m]["set460"]["S1|ladder"]["S"],
                                     res[m]["transfer"]["ladders"]["mae1"], res[m]["transfer"]["ladders"]["r2_1"],
                                     res[m]["transfer"]["dyes"]["mae1"], res[m]["transfer"]["random off-eq 500"]["mae13"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    main(ap.parse_args().threads)
