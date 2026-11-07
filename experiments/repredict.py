"""Recompute predictions of finished runs from best.pt with the current exmlip.training.train.predict.

By default the run's own data/split are used and predictions.pkl is overwritten. With --data/--split/--out the model
predicts another dataset or split into a separate file, e.g. model A on the conjugated test sets:
  python experiments/repredict.py experiments/wb97/runs_main/main_leh_all_s0 --data data/wb97_all_100k_s1_conj.pkl \
      --split experiments/conj/split_A_conj.json --keys test_dye test_lad test_dyeall --out predictions_conj.pkl
(--keys: prefixes of split keys to predict; val is only predicted without --keys.)
Usage: python experiments/repredict.py RUN_DIR [RUN_DIR ...] [--data D --split S --keys ... --out FILE]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exmlip.data.batching import MolDataset  # noqa
from exmlip.training import train as T  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--data"); ap.add_argument("--split")
ap.add_argument("--keys", nargs="*", default=None)
ap.add_argument("--out", default="predictions.pkl")
a = ap.parse_args()
cache = {}
for rd in map(Path, a.runs):
    if not (rd / "best.pt").exists() or not (rd / "meta.pt").exists():
        print("skip", rd); continue
    model, cfg = T.build_model_from_dir(rd)
    cfg = {**T.DEFAULT, **cfg}
    if a.data: cfg["data"] = a.data
    if a.split: cfg["split"] = a.split
    T.DEV = T.get_device(cfg)
    torch.set_num_threads(cfg.get("threads", 4))
    model = model.to(T.DEV)
    if a.keys is None:
        tr, va, tests = T.load(cfg)
        preds = {"val": T.predict(model, va, cfg)}
    else:
        if cfg["data"] not in cache:
            cache[cfg["data"]] = MolDataset(pickle.load(open(ROOT / cfg["data"], "rb")))
        ds, split = cache[cfg["data"]], json.load(open(ROOT / cfg["split"]))
        tests = {k[5:]: ds.subset(v) for k, v in split.items() if any(k.startswith(p) for p in a.keys)}
        known = set(int(z) for z in torch.load(rd / "meta.pt", weights_only=False)["z_table"])
        for k, t in tests.items():
            bad = {int(z) for m in t.mols for z in m["Z"]} - known
            if bad:
                raise ValueError(f"{rd.name}: set {k} contains elements {bad} the model never saw")
        preds = {}
    for k, t in tests.items():
        preds["test_" + k] = T.predict(model, t, cfg, forces=cfg["w_fex"] > 0)
    with open(rd / a.out, "wb") as f:
        pickle.dump(preds, f)
    print("predicted", rd.name, "->", a.out, {k: len(v) for k, v in preds.items()}, flush=True)
