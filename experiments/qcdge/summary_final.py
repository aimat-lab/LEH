"""Summarise the QCDGE rerun with the final settings (runs_final) -> summary_final.json (consumed by make_tables.table3).

Metrics per test set: S1-3 / S1-6 / S1-10 MAE, per-state R2 averaged over S1-3, S1 MAE and R2, oscillator-strength
metrics where predicted. Seeds are pooled as mean, std, n. Runs without predictions are skipped, so this can be
rerun when X-MACE finishes.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.training.metrics import osc_metrics, stack, r2  # noqa

RUNS = ROOT / "experiments/qcdge/runs_final"
N_STATES = 10
ORDER = [("leh", "LEH"), ("softmin", "Direct-softmin"), ("sum", "Direct-sum"), ("mean", "Direct-mean"),
         ("xmace", "X-MACE"), ("vecmsani", "OMNI-P2x arch."), ("msani", "MS-ANI"),
         ("leh_osc", "LEH + transition dipoles"), ("rand_leh", "LEH, random split"), ("rand_softmin", "Direct-softmin, random split")]
RUN_DIRS = {"leh": "q_leh_s*", "softmin": "q_softmin_s*", "sum": "q_sum_s*", "mean": "q_mean_s*", "leh_osc": "q_leh_osc_s*",
            "rand_leh": "q_rand_leh_s*", "rand_softmin": "q_rand_softmin_s*", "xmace": "xmace128", "vecmsani": "q_vecmsani",
            "msani": "q_msani"}


def metrics(preds):
    P, R = stack(preds, "gaps_pred", N_STATES), stack(preds, "gaps_ref", N_STATES)
    E = np.abs(P - R)
    out = {}
    for n in (1, 3, 6, 10):
        out[f"S1-{n} MAE"] = float(np.nanmean(E[:, :n]))
        out[f"S1-{n} R2"] = float(np.nanmean([r2(P[:, s], R[:, s]) for s in range(n)]))
    out["S1 MAE"], out["S1 R2"] = out.pop("S1-1 MAE"), out.pop("S1-1 R2")
    if "osc_pred" in preds[0]:
        o = osc_metrics(preds, N_STATES)
        out.update({"osc MAE": o["osc_mae"], "osc R2 S1": o["osc_r2_S1"], "bright recall": o.get("bright_recall", np.nan)})
    return out


def load(d):
    f = d / "predictions.pkl"
    if not f.exists():
        f = d / "results" / "predictions.pkl"
    return pickle.load(open(f, "rb")) if f.exists() else None


res = {"order": ORDER, "models": {}, "n_seeds": {}}
for key, pat in RUN_DIRS.items():
    per_seed = [m for m in (load(d) for d in sorted(RUNS.glob(pat))) if m is not None]
    if not per_seed:
        print("missing", key)
        continue
    sets = per_seed[0].keys()
    M = [{s: metrics(P[s]) for s in sets} for P in per_seed]
    res["models"][key] = {s: {q: [float(np.mean([m[s][q] for m in M])), float(np.std([m[s][q] for m in M])), len(M)]
                              for q in M[0][s]} for s in sets}
    res["n_seeds"][key] = len(M)
    r = res["models"][key]
    print(key.ljust(14), len(M), " ".join(f"{s}: {r[s]['S1-3 MAE'][0]:.3f}/{r[s]['S1-3 R2'][0]:.2f}" for s in
                                          ("val", "test_9", "test_10", "test_10_novelscaf") if s in r))
json.dump(res, open(ROOT / "experiments/qcdge/summary_final.json", "w"), indent=1)
