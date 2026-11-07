"""Learning curves on the final dataset (100k + S1-sampled geometries): our models and the baselines on identical
nested training subsets, identical validation subset and identical test sets.

Runs (all with the final settings for our models; baselines with their authors' settings):
  LEH / softmin: 1k, 3k (runs_main, 3 seeds), 10k (runs_abl, 3 seeds), 30k (runs_hp/hp3_c128_we0_100 = LEH seed 0,
  runs_main), all (runs_main, 2 seeds); LEH L=2 all (runs_main).
  X-MACE 128 ch (S0-S3 only), MS-ANI and OMNI-P2x architecture (vecmsani, from scratch): runs_lc, 1 seed.
Metrics per test set from predictions.pkl: S1-S3 MAE/R2 (shared by all models), S1-S6 MAE (not X-MACE),
S0 meV/atom, total E_S1 MAE, S0 and S1 forces. Mean +- std over seeds.
Writes experiments/wb97/learning_curves.json and prints compact tables.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "experiments/wb97"
SIZES = ["1k", "3k", "10k", "30k", "all"]
SETS = ["test_random_offeq", "test_random_eq", "test_random_exc", "test_scaffold_offeq", "test_scaffold_eq",
        "test_scaffold_exc", "test_large_offeq", "test_large_eq", "test_large_exc"]


def runs_for(model, size):
    if model in ("leh", "softmin"):
        h = "leh" if model == "leh" else "softmin"
        if size == "10k":
            return [W / f"runs_abl/abl_{h}_s{s}" for s in range(3)]
        if size == "30k" and model == "leh":
            return [W / "runs_hp/hp3_c128_we0_100", W / "runs_main/main_leh_30k_s1"]
        seeds = {"1k": 3, "3k": 3, "30k": 2, "all": 2}[size]
        return [W / f"runs_main/main_{h}_{size}_s{s}" for s in range(seeds)]
    if model == "leh_L2":
        return [W / "runs_main/main_leh_L2_all_s0"] if size == "all" else []
    if model == "xmace":
        return [W / f"runs_lc/xmace128_sub{size}"]
    if model in ("omnift", "omniftA"):  # A: adapted fine-tuning (no frozen layers, gap weight 1)
        return [W / f"runs_lc/sub{size}_{model}"]
    return [W / f"runs_lc/sub{size}_{model}"]


def r2(p, r):
    return float(1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2))


def metrics(P):
    out = {}
    gp = np.array([np.pad(np.asarray(x["gaps_pred"], float), (0, 10))[:10] for x in P])
    gr = np.array([np.pad(np.asarray(x["gaps_ref"], float), (0, 10))[:10] for x in P])
    gp[gp == 0] = np.nan
    for n in (1, 3, 6):
        a, b = gp[:, :n], gr[:, :n]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 10 and m.mean() > 0.5:
            out[f"S1-{n} MAE"] = float(np.mean(np.abs(a[m] - b[m])))
            out[f"S1-{n} R2"] = r2(a[m], b[m])
    E = [x for x in P if x.get("E0_ref") is not None and x.get("E0_pred") is not None]
    if E:
        out["E0 meV/at"] = float(1000 * np.mean([abs(x["E0_pred"] - x["E0_ref"]) / x["n_atoms"] for x in E]))
        es = [abs(x["E0_pred"] + np.asarray(x["gaps_pred"], float)[0] - x["E0_ref"] - np.asarray(x["gaps_ref"], float)[0]) for x in E]
        out["E_S1 MAE"] = float(np.nanmean(es))
    F = [x for x in P if x.get("F0_ref") is not None and x.get("F0_pred") is not None]
    if F:
        out["F0 MAE"] = float(np.mean(np.concatenate([np.abs(x["F0_pred"] - x["F0_ref"]).ravel() for x in F])))
        X = [x for x in F if x.get("Fex_ref") is not None and x.get("Fex_pred") is not None]
        if X:
            out["F_S1 MAE"] = float(np.nanmean(np.concatenate([np.abs(x["Fex_pred"][:, 0] - x["Fex_ref"][:, 0]).ravel() for x in X])))
    return out


res = defaultdict(dict)
for model in ["leh", "softmin", "leh_L2", "xmace", "msani", "vecmsani", "omnift", "omniftA"]:
    for size in SIZES:
        dirs = [d for d in runs_for(model, size) if (d / "predictions.pkl").exists()]
        if not dirs:
            continue
        per = defaultdict(lambda: defaultdict(list))
        for d in dirs:
            P = pickle.load(open(d / "predictions.pkl", "rb"))
            for s in SETS:
                if s in P and P[s]:
                    for k, v in metrics(P[s]).items():
                        per[s][k].append(v)
        res[model][size] = {s: {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in ms.items()} for s, ms in per.items()}
json.dump(res, open(W / "learning_curves.json", "w"), indent=1)

labels = {"leh": "LEH", "softmin": "softmin", "leh_L2": "LEH L=2", "xmace": "X-MACE", "msani": "MS-ANI", "vecmsani": "OMNI-P2x arch",
          "omnift": "OMNI-P2x ft", "omniftA": "OMNI-P2x ft adapted"}
for s in ["test_random_offeq", "test_scaffold_offeq", "test_random_exc", "test_large_offeq", "test_large_eq", "test_large_exc"]:
    for metric in ["S1-3 MAE", "S1-6 MAE", "F_S1 MAE", "E0 meV/at"]:
        print(f"\n== {s}: {metric}" + "  (mean over seeds)")
        print(f"{'model':14s}" + "".join(f"{z:>9s}" for z in SIZES))
        for model, lab in labels.items():
            row = [res[model].get(z, {}).get(s, {}).get(metric) for z in SIZES]
            if any(r is not None for r in row):
                print(f"{lab:14s}" + "".join(f"{r[0]:9.3f}" if r else f"{'-':>9s}" for r in row))
