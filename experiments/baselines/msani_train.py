"""Train MS-ANI (MLatom; the OMNI-P2x architecture, optionally its vecmsani variant) on our splits and write
predictions in exmlip's format so experiments/*/analyze.py score it like our models.

Two modes:
  --mode wb97    : off-equilibrium entries with E0 (S0 + 10 singlets as state energies, forces for S0-S3, NaN elsewhere).
                   Entries without E0 (QCDGE equilibrium spectra) cannot be used: MS-ANI learns absolute state energies.
  --mode spectra : QCDGE spectral benchmark; the n_sup lowest excitation energies are the state energies, no forces.
Units inside MLatom: Hartree and Hartree/A (its defaults); predictions are written in eV and eV/A.
Run with the separate environment .venv_mlatom.
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

H = 27.211386245988
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["wb97", "spectra"], required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="msani", choices=["msani", "vecmsani"])
ap.add_argument("--n_states", type=int, default=10, help="excited states (wb97) or supervised roots (spectra)")
ap.add_argument("--n_force", type=int, default=3, help="excited states with force labels (wb97)")
ap.add_argument("--max_epochs", type=int, default=2000)
ap.add_argument("--max_hours", type=float, default=7.0)
ap.add_argument("--limit", type=int, default=None, help="debug: limit molecules per split")
ap.add_argument("--device", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--predict_only", action="store_true", help="load out/model.pt (e.g. after a time-limited run) and predict")
a = ap.parse_args()

import torch  # noqa: E402
import mlatom as ml  # noqa: E402

np.random.seed(a.seed); torch.manual_seed(a.seed)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mols = pickle.load(open(a.data, "rb"))
split = json.load(open(a.split))
ns = a.n_states + (1 if a.mode == "wb97" else 0)  # model states


def usable(m):
    return ("E0" in m and np.isfinite(m.get("E0", np.nan))) if a.mode == "wb97" else True


def to_ml(m):
    mol = ml.data.molecule()
    for z, r in zip(m["Z"], m["R"]):
        mol.atoms.append(ml.data.atom(element_symbol=SYM[int(z)], xyz_coordinates=np.array(r, float)))
    if a.mode == "wb97":
        e = np.concatenate([[m["E0"]], m["E0"] + np.asarray(m["gaps"][: a.n_states], float)]) / H
        nanf = np.full((len(m["Z"]), 3), np.nan)
        g = [(-m["F0"] / H) if "F0" in m else nanf] + \
            [(-m["Fex"][:, s] / H) if ("Fex" in m and s < a.n_force) else nanf for s in range(a.n_states)]
    else:
        e = np.asarray(m["gaps"][: a.n_states], float) / H
        g = None
    for k in range(ns):
        st = mol.copy(atomic_labels=["xyz_coordinates"], molecular_labels=[])
        st.energy = float(e[k])
        if g is not None:
            st.add_xyz_derivative_property(np.asarray(g[k], float), "energy", "energy_gradients")
        mol.electronic_states.append(st)
    return mol


def db(idx):
    idx = [i for i in idx if usable(mols[i])][: a.limit]
    d = ml.data.molecular_database()
    for i in idx:
        d.append(to_ml(mols[i]))
    return d, idx


Model = ml.models.msani if a.model == "msani" else ml.models.vecmsani
t0 = time.time()
if a.predict_only:
    model = Model(model_file=str(out / "model.pt"), nstates=ns, device=a.device, verbose=1)
    tr, va = [], []
else:
    tr, tr_idx = db(split["train"])
    va, va_idx = db(split["val"])
    print(f"train {len(tr)} val {len(va)} states {ns}", flush=True)
    model = Model(model_file=str(out / "model.pt"), nstates=ns, device=a.device, verbose=1)
    hp = {"max_epochs": a.max_epochs}
    model.train(molecular_database=tr, validation_molecular_database=va, property_to_learn="energy",
                xyz_derivative_property_to_learn="energy_gradients" if a.mode == "wb97" else None, hyperparameters=hp)
hours = (time.time() - t0) / 3600


def predict(idx):
    idx = [i for i in idx if usable(mols[i])][: a.limit]
    if not idx:
        return []
    d = ml.data.molecular_database()
    for i in idx:
        d.append(to_ml(mols[i]))
    grads = a.mode == "wb97"
    model.predict(molecular_database=d, nstates=ns, calculate_energy=True, calculate_energy_gradients=grads)
    res = []
    for i, mol in zip(idx, d):
        m = mols[i]
        E = np.array([s.energy for s in mol.electronic_states[:ns]]) * H
        r = dict(id=m.get("id"), n_atoms=len(m["Z"]), n_heavy=int((m["Z"] > 1).sum()), Z=np.asarray(m["Z"]),
                 gaps_ref=np.asarray(m["gaps"], float))
        if a.mode == "wb97":
            # adiabatic ordering of the excited states (as for our direct heads and the OMNI-P2x predictors); the
            # gradients are permuted with the energies (before 2026-10-16 only the gaps were sorted, the forces were
            # taken in output order -> S1-S3 force errors of swapped geometries were wrong)
            order = np.argsort(E[1:])
            r["gaps_pred"] = (E[1:] - E[0])[order]
            r["state_order_permuted"] = bool(np.any(order != np.arange(len(order))))
            r["E0_pred"], r["E0_ref"] = float(E[0]), m.get("E0")
            G = np.array([s.get_energy_gradients() for s in mol.electronic_states[:ns]]) * H  # eV/A
            Gex = G[1:][order]
            r["F0_pred"], r["F0_ref"] = -G[0], m.get("F0")
            r["Fex_pred"], r["Fex_ref"] = np.transpose(-Gex[: a.n_force], (1, 0, 2)), m.get("Fex")
            if r["F0_ref"] is None:
                r.pop("F0_pred"); r.pop("F0_ref"); r.pop("Fex_pred"); r.pop("Fex_ref")
        else:
            r["gaps_pred"] = np.sort(E)
        res.append(r)
    return res


preds = {"val": predict(split["val"])}
for k, v in split.items():
    if k.startswith("test_"):
        preds[k] = predict(v)
pickle.dump(preds, open(out / "predictions.pkl", "wb"))
cfg = dict(head=a.model, head_kwargs={}, n_states=10 if a.mode == "wb97" else a.n_states, n_sup=a.n_states,
           n_force=a.n_force if a.mode == "wb97" else 0, data=a.data, split=a.split, seed=a.seed,
           max_epochs=a.max_epochs)
yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
json.dump(dict(epochs=None, hours=hours, n_train=len(tr), n_val=len(va)), open(out / "summary.json", "w"))
(out / "train.log").write_text('{"skipped": 0}\n')
print("done", hours, flush=True)
