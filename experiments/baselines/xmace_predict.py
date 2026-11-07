"""Predict exported test files with a trained X-MACE model and write predictions.pkl in exmlip format
(run with .venv_xmace). Usage: xmace_predict.py MODEL EXPORT_DIR RUN_DIR [--device cuda]"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("export"); ap.add_argument("run")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
from mace.calculators import MACECalculator  # noqa: E402

meta = json.load(open(Path(a.export) / "index.json"))
ns, mode = meta["n_states"], meta["mode"]
ROOT = Path(__file__).resolve().parents[2]


def local_path(f):
    """index.json stores absolute paths of the machine that exported the data; map them into this checkout."""
    f = Path(f)
    if f.exists():
        return f
    parts = f.parts
    if "transferable_excited_MLIP" in parts:
        return ROOT.joinpath(*parts[parts.index("transferable_excited_MLIP") + 1:])
    return ROOT / f


mols = pickle.load(open(local_path(meta["data"]), "rb"))
calc = MACECalculator(model_paths=a.model, n_energies=ns, device=a.device)
preds = {}
for name, idx in meta["index"].items():
    if name == "train":
        continue
    frames = read(Path(a.export) / f"{name}.xyz", ":")
    res = []
    for at, i in zip(frames, idx):
        m = mols[i]
        calc.calculate(at)
        E = np.asarray(calc.results["energy"]).reshape(-1)[:ns]
        r = dict(id=m.get("id"), n_atoms=len(m["Z"]), n_heavy=int((m["Z"] > 1).sum()), Z=np.asarray(m["Z"]),
                 gaps_ref=np.asarray(m["gaps"], float))
        if mode == "wb97":
            # X-MACE's decoder returns sorted eigenvalues (S0 < S1 < ...), so `order` is the identity; it is applied
            # anyway so that energies and forces always belong to the same adiabatic state
            order = np.argsort(E[1:])
            r["gaps_pred"] = (E[1:] - E[0])[order]
            r["state_order_permuted"] = bool(np.any(order != np.arange(len(order))))
            r["E0_pred"], r["E0_ref"] = float(E[0]), m.get("E0")
            F = np.asarray(calc.results["forces"]).reshape(len(m["Z"]), ns, 3)  # (atoms, states, 3)
            if m.get("F0") is not None:
                r["F0_pred"], r["F0_ref"] = F[:, 0], m["F0"]
                r["Fex_pred"], r["Fex_ref"] = F[:, 1:][:, order], m.get("Fex")
        else:
            r["gaps_pred"] = np.sort(E)
        res.append(r)
    preds["val" if name == "valid" else name] = res
    print(name, len(res), flush=True)
run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
pickle.dump(preds, open(run / "predictions.pkl", "wb"))
yaml.safe_dump(dict(head="xmace", head_kwargs={}, n_states=10 if mode == "wb97" else ns,
                    n_sup=(ns - 1) if mode == "wb97" else ns, n_force=ns - 1 if mode == "wb97" else 0,
                    data=meta["data"], split=meta["split"]), open(run / "config.yaml", "w"))
json.dump(dict(model=a.model), open(run / "summary.json", "w"))
(run / "train.log").write_text('{"skipped": 0}\n')
