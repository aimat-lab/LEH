"""Test-set errors of the production models (dataset v4, experiments/al/runs/prod_*) next to the frozen benchmark
models (v1, runs_main/main_*_all): the frozen test sets, the conjugated test sets (held-out dye scaffolds, ladders;
production = model-B training data incl. the training dye scaffolds) and the active-learning test frames of rounds
2-4 (test_{random,scaffold,large}_exc{2,3,4}; geometries from model-driven S1 dynamics of test molecules).
metrics() is identical to experiments/wb97/learning_curves.metrics (checked against learning_curves.json on start).
Writes experiments/al/production_metrics.json (mean, std, n over seeds).
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = {"bench_leh": [ROOT / f"experiments/wb97/runs_main/main_leh_all_s{s}" for s in (0, 1)],
          "bench_softmin": [ROOT / f"experiments/wb97/runs_main/main_softmin_all_s{s}" for s in (0, 1)],
          "prod_leh": [ROOT / f"experiments/al/runs/prod_leh_v4_s{s}" for s in (0, 1)],
          "prod_softmin": [ROOT / f"experiments/al/runs/prod_softmin_v4_s{s}" for s in (0, 1)]}
EXTRA = {"bench_leh": ROOT / "experiments/conj/predictions_conj.pkl"}  # conj sets of the benchmark models: analyze_B


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


# consistency check with the benchmark learning curves
LC = json.load(open(ROOT / "experiments/wb97/learning_curves.json"))
P0 = pickle.load(open(MODELS["bench_leh"][0] / "predictions.pkl", "rb"))
m0 = metrics(P0["test_random_offeq"])["S1-3 MAE"]
P1 = pickle.load(open(MODELS["bench_leh"][1] / "predictions.pkl", "rb"))
m1 = metrics(P1["test_random_offeq"])["S1-3 MAE"]
ref = LC["leh"]["all"]["test_random_offeq"]["S1-3 MAE"][0]
assert abs((m0 + m1) / 2 - ref) < 1e-9, ((m0 + m1) / 2, ref)
del P0, P1

res = {}
for name, dirs in MODELS.items():
    per = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        f = d / "predictions.pkl"
        if not f.exists():
            print("missing", f); continue
        P = pickle.load(open(f, "rb"))
        for s, X in P.items():
            if s.startswith("test") and X:
                for k, v in metrics(X).items():
                    per[s][k].append(v)
    res[name] = {s: {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in ms.items()} for s, ms in per.items()}
json.dump(res, open(ROOT / "experiments/al/production_metrics.json", "w"), indent=1)

SETS = ["test_random_offeq", "test_scaffold_offeq", "test_random_exc", "test_large_offeq", "test_large_exc",
        "test_random_exc2", "test_random_exc3", "test_random_exc4", "test_large_exc4", "test_dye_noS_eq", "test_lad_noS_eq"]
print("S1-S3 MAE (eV) | F_S1 MAE (eV/A)")
print("set".ljust(22) + "".join(n.ljust(26) for n in res))
for s in SETS:
    row = s.ljust(22)
    for n in res:
        r = res[n].get(s, {})
        a, b = r.get("S1-3 MAE"), r.get("F_S1 MAE")
        row += (f"{a[0]:.3f}" if a else "  -  ") + (f" | {b[0]:.3f}" if b else " |  -   ") + f" (n={a[2] if a else 0})".ljust(14)
    print(row)
