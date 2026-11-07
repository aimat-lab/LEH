"""Decision on the S1/low-gap-weighted gap loss (option 2, user 2026-10-25), on VALIDATION data only.

Compared: production models on v4 with the uniform loss (prod_*_v4_s{0,1}) vs the weighted loss (prodw_*_v4_s{0,1}),
identical otherwise. The test sets and test trajectories are not looked at for this decision.
  (1) validation split (molecules never in training, incl. their active-learning frames): S1 MAE and signed error
      binned by the TDDFT S1 (<1, 1-2, 2-3, 3-4, >4 eV), and S1-S3 MAE overall (in-distribution accuracy);
  (2) the seven validation trajectories: mean S1-gap / S0-rise MAE (experiments/al/convergence_*.json).
Decision rule (fixed before the runs finished): adopt the weighted loss if the mean |signed error| on validation frames
with TDDFT S1 < 2 eV drops by at least 30 % and neither the overall validation S1-S3 MAE nor the mean S1-gap MAE along
the validation trajectories gets worse by more than 5 %. Writes experiments/al/compare_loss.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/al/runs"
BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 99)]
GROUPS = {"uniform": ["prod_leh_v4_s0", "prod_leh_v4_s1", "prod_softmin_v4_s0", "prod_softmin_v4_s1"],
          "weighted": ["prodw_leh_v4_s0", "prodw_leh_v4_s1", "prodw_softmin_v4_s0", "prodw_softmin_v4_s1"]}
CONV = {"uniform": ["convergence_prod_v4.json", "convergence_prod_v4_softmin_s1.json"],
        "weighted": [f"convergence_{r}.json" for r in GROUPS["weighted"]]}  # evaluated locally, one file per run


def val_stats(run):
    P = pickle.load(open(RUNS / run / "predictions.pkl", "rb"))["val"]
    gp = np.array([x["gaps_pred"][0] for x in P]); gr = np.array([x["gaps_ref"][0] for x in P]); e = gp - gr
    G = np.array([np.pad(np.asarray(x["gaps_pred"], float), (0, 3))[:3] for x in P])
    R = np.array([np.pad(np.asarray(x["gaps_ref"], float), (0, 3))[:3] for x in P])
    out = {"S1-3 MAE": float(np.nanmean(np.abs(G - R))), "n": len(P)}
    for a, b in BINS:
        m = (gr >= a) & (gr < b)
        out[f"S1 {a}-{b}"] = dict(n=int(m.sum()), mae=float(np.abs(e[m]).mean()), signed=float(e[m].mean()))
    m = gr < 2
    out["S1<2 |signed|"] = float(abs(e[m].mean())); out["S1<2 MAE"] = float(np.abs(e[m]).mean())
    return out


res = {"runs": {}, "trajectories": {}}
for g, runs in GROUPS.items():
    for r in runs:
        if (RUNS / r / "predictions.pkl").exists():
            res["runs"][r] = dict(group=g, **val_stats(r))
    per = {}
    for f in CONV[g]:
        p = ROOT / "experiments/al" / f
        if p.exists():
            for run, x in json.load(open(p))["runs"].items():
                per[Path(run).name] = {q: float(np.mean([v[q] for v in x["trajectories"].values()])) for q in ("gap", "rise")}
    res["trajectories"][g] = per


def gmean(g, key, head=None):
    v = [x[key] for r, x in res["runs"].items() if x["group"] == g and (head is None or head in r)]
    return float(np.mean(v)) if v else np.nan


summ = {}
for head in ("leh", "softmin", None):
    k = head or "all"
    u, w = (gmean("uniform", "S1<2 |signed|", head), gmean("weighted", "S1<2 |signed|", head))
    du = (gmean("weighted", "S1-3 MAE", head) / gmean("uniform", "S1-3 MAE", head) - 1)
    tu = [v["gap"] for r, v in res["trajectories"]["uniform"].items() if head is None or head in r]
    tw = [v["gap"] for r, v in res["trajectories"]["weighted"].items() if head is None or head in r]
    dt = (np.mean(tw) / np.mean(tu) - 1) if tu and tw else np.nan
    summ[k] = dict(lowgap_bias_uniform=u, lowgap_bias_weighted=w, lowgap_bias_change=(w / u - 1) if u else np.nan,
                   val_S13_change=du, traj_gap_change=float(dt))
s = summ["all"]
res["summary"] = summ
res["adopt_weighted"] = bool(s["lowgap_bias_change"] <= -0.30 and s["val_S13_change"] <= 0.05 and s["traj_gap_change"] <= 0.05)
json.dump(res, open(ROOT / "experiments/al/compare_loss.json", "w"), indent=1)
for r, x in res["runs"].items():
    print(r.ljust(22), f"S1-3 {x['S1-3 MAE']:.3f}", " ".join(f"[{k[3:]}] {v['mae']:.2f}/{v['signed']:+.2f}" for k, v in x.items() if k.startswith("S1 ")))
for g, per in res["trajectories"].items():
    print(g, {r: (round(v["gap"], 3), round(v["rise"], 3)) for r, v in per.items()})
print(json.dumps(summ, indent=1)); print("adopt weighted loss:", res["adopt_weighted"])
