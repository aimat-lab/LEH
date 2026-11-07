"""Collect metrics for all QCDGE runs (E1/E2) and run the fragment test (E3)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.training.metrics import osc_metrics, spectral_metrics, by_size  # noqa
from exmlip.training.train import build_model_from_dir  # noqa
from exmlip.training.fragments import fragment_test  # noqa

runs = ROOT / "experiments" / "qcdge" / "runs"
res = {}
old = json.load(open(ROOT / "experiments/qcdge/results.json")) if (ROOT / "experiments/qcdge/results.json").exists() else {}
for d in sorted(runs.iterdir()):
    if not (d / "predictions.pkl").exists():
        continue
    cfg = json.load(open(d / "summary.json"))
    import yaml
    c = yaml.safe_load(open(d / "config.yaml"))
    P = pickle.load(open(d / "predictions.pkl", "rb"))
    r = dict(summary=cfg, n_sup=c["n_sup"])
    for k, preds in P.items():
        r[k] = spectral_metrics(preds, c["n_states"], c["n_sup"])
        r[k].update(osc_metrics(preds, c["n_states"]))
        r[k + "_by_size"] = by_size(preds, c["n_states"], c["n_sup"])
    log = [json.loads(l) for l in open(d / "train.log") if l.startswith("{")]
    r["skipped_batches"] = int(sum(x["skipped"] for x in log))
    if "fragments" in old.get(d.name, {}) and "--fragments" not in sys.argv:
        r["fragments"] = old[d.name]["fragments"]
    res[d.name] = r
    print(d.name.ljust(22), " ".join(f"{k}: MAE {r[k]['mae_sup']:.3f} R2 {r[k]['r2_sup']:.3f} |" for k in P if k in
                                    ("val", "test_9", "test_10")))

# fragment test on held-out molecules (pool = 9-heavy-atom test set)
if "--fragments" in sys.argv:
    mols = pickle.load(open(ROOT / "data" / "qcdge_sub.pkl", "rb"))
    split = json.load(open(ROOT / "experiments" / "qcdge" / "split_size.json"))
    pool = [mols[i] for i in split["test_9"][:2000]]
    torch.set_num_threads(16)
    for name in res:
        if name.startswith("rand_") or not (runs / name / "best.pt").exists():  # external baselines: no exmlip model
            continue
        model, c = build_model_from_dir(runs / name)
        res[name]["fragments"] = fragment_test(model, pool, c["n_states"], n_eval=min(6, c["n_sup"]), n_pairs=200, sep=20.0, n_sup=c["n_sup"])
        print(name, {k: np.round(v["mae_vs_union_of_monomer_preds"], 3).tolist() for k, v in res[name]["fragments"].items()})
json.dump(res, open(ROOT / "experiments" / "qcdge" / "results.json", "w"), indent=1, default=float)
