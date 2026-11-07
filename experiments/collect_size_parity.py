"""Per-size errors and parity data of the full-set models (runs on HoreKa2 next to the predictions).

Models (largest training size available): LEH (2 seeds), LEH L=2, Direct-softmin (2 seeds), X-MACE (full set if
finished, else 30k; flagged), MS-ANI, OMNI-P2x arch. from scratch, OMNI-P2x fine-tuned (if finished).
Test sets: random (QCDGE-derived, <= 10 heavy atoms), larger (ZINC, 11-20) and the S-free conjugated sets (all dyes,
ladders), each split into equilibrium and off-equilibrium geometries.
Outputs: experiments/wb97/size_parity.json (S1-S3 and S1 MAE per heavy-atom count and set, n per bin) and
experiments/wb97/parity.npz (S1 predicted / reference per model and set, seed 0).
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M, L = ROOT / "experiments/wb97/runs_main", ROOT / "experiments/wb97/runs_lc"


def first(*cands):
    return next((c for c in cands if (c / "predictions.pkl").exists()), None)


xm = first(L / "xmace128_suball", L / "xmace128_sub30k")
MODELS = {
    "leh": [(M / "main_leh_all_s0", M / "main_leh_all_s0/predictions_conj.pkl"), (M / "main_leh_all_s1", M / "main_leh_all_s1/predictions_conj.pkl")],
    "leh_L2": [(M / "main_leh_L2_all_s0", M / "main_leh_L2_all_s0/predictions_conj.pkl")],
    "softmin": [(M / "main_softmin_all_s0", M / "main_softmin_all_s0/predictions_conj.pkl"), (M / "main_softmin_all_s1", M / "main_softmin_all_s1/predictions_conj.pkl")],
    "xmace": [(xm, Path(str(xm) + "_conj") / "predictions.pkl")] if xm else [],
    "msani": [(L / "suball_msani", L / "suball_msani_conj/predictions.pkl")],
    "vecmsani": [(L / "suball_vecmsani", L / "suball_vecmsani_conj/predictions.pkl")],
    "omnift": [(L / "suball_omnift", L / "suball_omnift_conj/predictions.pkl")],
    "omniftA": [(L / "suball_omniftA", L / "suball_omniftA_conj/predictions.pkl")],
}
SETS = {"random_offeq": ["test_random_offeq"], "random_eq": ["test_random_eq"], "larger_offeq": ["test_large_offeq"],
        "larger_eq": ["test_large_eq"], "dyes_offeq": ["test_dyeall_noS_offeq"], "dyes_eq": ["test_dyeall_noS_eq"],
        "ladders_offeq": ["test_lad_noS_offeq"], "ladders_eq": ["test_lad_noS_eq"]}


def arrays(P):
    gp = np.array([np.pad(np.asarray(x["gaps_pred"], float), (0, 3))[:3] for x in P]); gp[gp == 0] = np.nan
    gr = np.array([np.asarray(x["gaps_ref"], float)[:3] for x in P])
    nh = np.array([int(x["n_heavy"]) for x in P])
    return gp, gr, nh


out = dict(models={}, xmace_size=("all" if xm and xm.name.endswith("suball") else "30k"))
parity = {}
for name, runs in MODELS.items():
    per_seed = []
    for k, (rd, conj) in enumerate(runs):
        if rd is None or not (rd / "predictions.pkl").exists():
            continue
        P = pickle.load(open(rd / "predictions.pkl", "rb"))
        if conj.exists():
            P.update(pickle.load(open(conj, "rb")))
        res = {}
        for sname, keys in SETS.items():
            rows = [x for kk in keys for x in P.get(kk, [])]
            if not rows:
                continue
            gp, gr, nh = arrays(rows)
            bins = {}
            for h in np.unique(nh):
                m = nh == h
                e3, e1 = np.abs(gp[m] - gr[m]), np.abs(gp[m, 0] - gr[m, 0])
                bins[int(h)] = dict(n=int(m.sum()), s13=float(np.nanmean(e3)), s1=float(np.nanmean(e1)))
            res[sname] = dict(all=dict(n=len(rows), s13=float(np.nanmean(np.abs(gp - gr))), s1=float(np.nanmean(np.abs(gp[:, 0] - gr[:, 0])))),
                              bins=bins)
            if k == 0:
                parity[f"{name}|{sname}|pred"] = gp[:, 0]; parity[f"{name}|{sname}|ref"] = gr[:, 0]; parity[f"{name}|{sname}|nheavy"] = nh
        per_seed.append(res)
    if per_seed:
        out["models"][name] = per_seed
        print(name, len(per_seed), "seed(s):", {s: round(v["all"]["s13"], 3) for s, v in per_seed[0].items()}, flush=True)
json.dump(out, open(ROOT / "experiments/wb97/size_parity.json", "w"), indent=1)
np.savez_compressed(ROOT / "experiments/wb97/parity.npz", **parity)
print("written size_parity.json, parity.npz")
