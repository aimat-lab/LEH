"""Collect metrics for the new-dataset runs (energies + forces, random / scaffold / size-extrapolation tests).

Writes experiments/wb97/results.json and prints a compact table.
Forces: F0 and S1-S3 MAE in eV/A. Gaps: per-root MAE in eV. Size extrapolation: errors vs heavy-atom count.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.training.metrics import energy_force_metrics, osc_metrics, spectral_metrics  # noqa

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/wb97/runs"


def run_metrics(d):
    c = yaml.safe_load(open(d / "config.yaml"))
    P = pickle.load(open(d / "predictions.pkl", "rb"))
    r = dict(summary=json.load(open(d / "summary.json")), head=c["head"], head_kwargs=c.get("head_kwargs", {}),
             seed=c.get("seed", 0))
    for k, preds in P.items():
        if not preds:
            continue
        m = spectral_metrics(preds, c["n_states"], c["n_sup"])
        m.update(energy_force_metrics(preds, 3, c.get("n_force", 3)))
        m.update(osc_metrics(preds, c["n_states"]))
        m["n"] = len(preds)
        r[k] = m
        sizes = np.array([p["n_heavy"] for p in preds])
        if k.startswith("test_large") or k.startswith("test_random"):
            bs = {}
            for s in np.unique(sizes):
                sub = [p for p, z in zip(preds, sizes) if z == s]
                mm = spectral_metrics(sub, c["n_states"], c["n_sup"])
                mm.update(energy_force_metrics(sub, 3, c.get("n_force", 3)))
                mm["n"] = len(sub)
                bs[int(s)] = mm
            r[k + "_by_size"] = bs
    log = [json.loads(l) for l in open(d / "train.log") if l.startswith("{")]
    r["skipped_batches"] = int(sum(x["skipped"] for x in log))
    return r


if __name__ == "__main__":
    res = {}
    for d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        if (d / "predictions.pkl").exists():
            res[d.name] = run_metrics(d)
    out = RUNS.parent / ("results.json" if RUNS.name == "runs" else f"results_{RUNS.name}.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    cols = [("mae_E0_per_atom_meV", "E0 meV/at"), ("r2_E0_referenced", "R2 E0ref"), ("mae_dE1", "dE1"),
            ("r2_dE1", "R2 dE1"), ("mae_sup", "dE1-10"), ("r2_sup", "R2 dE"), ("mae_ES1", "E_S1"),
            ("ES1_errvar_share_E0", "var%E0"), ("mae_F0", "F0"), ("r2_F0", "R2 F0"), ("mae_FS1", "F_S1"),
            ("r2_FS1", "R2 FS1"), ("mae_Fgap1", "Fgap1"), ("r2_Fgap1", "R2 Fgap")]
    for split in ["test_random_offeq", "test_random_eq", "test_random_exc", "test_scaffold", "test_scaffold_offeq",
                  "test_scaffold_eq", "test_scaffold_exc", "test_large_offeq", "test_large_exc", "test_large_eq"]:
        print(f"\n== {split} (eV, eV/A)")
        print("run".ljust(22) + "".join(h.rjust(10) for _, h in cols))
        if not any(split in r for r in res.values()):
            continue
        for n, r in res.items():
            v = r.get(split, {})
            print(n.ljust(22) + "".join(f"{v.get(k, np.nan):10.3f}" for k, _ in cols))
