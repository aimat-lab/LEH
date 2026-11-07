"""Fine-tuned OMNI-P2x on our learning-curve subsets (second OMNI-P2x comparison, next to the zero-shot one).

Protocol (OMNI-P2x's own fine-tuning settings from mlatom.addons.omnip2x.omnip2x.train, but with our splits):
each of the three pretrained ensemble members (OMNIP2x_CV_1..3) is fine-tuned separately on the same training
subset and validated on the same shared validation subset as all other models: 100 epochs, gap_coefficient 0.01,
fixed_layers [[0, 4]], energy shifter re-fitted to our level of theory (reset_energy_shifter), optimiser reset.
Labels: energies of S0-S10 and forces of S0-S3 (NaN = missing), exactly as for the from-scratch MS-ANI baselines.
MLatom's omni.train() is not used directly because it re-splits the given data 80/20 internally.

Prediction averages the three fine-tuned members (OMNI-P2x's ensemble mean; members called directly because the
MLatom 3.25.5 model tree drops per-state gradients) and uses the pretrained oscillator-strength model unchanged.
Run with .venv_mlatom. Usage:
  python experiments/baselines/omnip2x_finetune.py --split S --out DIR --member 1     (fine-tune one member)
  python experiments/baselines/omnip2x_finetune.py --split S --out DIR --predict     (after all three)
  ... --predict --pred_out DIR2: members from DIR, predictions/config/summary written to DIR2 (other data sets)
Adapted variant (2026-10-17, user approved): --fixed_layers none --gap_coefficient 1.0 (the state index enters only
the first layer, which the default protocol freezes, so per-state offsets from the pretraining level of theory could
not adapt; and the 0.01 gap weight leaves excitation energies practically unweighted).
"""
import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

H = 27.211386245988
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}
NS = 11          # S0 + S1-S10
N_FORCE = 3      # excited states with force labels

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/wb97_all_100k_s1.pkl")
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--member", type=int, choices=[1, 2, 3])
ap.add_argument("--predict", action="store_true")
ap.add_argument("--pred_out", default=None, help="write predictions here instead of --out (members are read from --out)")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--gap_coefficient", type=float, default=0.01, help="weight of the adjacent-state gap loss (OMNI-P2x: 0.01)")
ap.add_argument("--fixed_layers", default="0,4", help="frozen layers of every element network (OMNI-P2x: 0,4); 'none'")
ap.add_argument("--limit", type=int, default=None, help="debug: limit entries per split")
ap.add_argument("--chunk", type=int, default=256)
a = ap.parse_args()

import mlatom as ml  # noqa: E402

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
mols = pickle.load(open(a.data, "rb"))
split = json.load(open(a.split))
t0 = time.time()


def pretrained_dir():
    omni = ml.models.methods(method="OMNI-P2x")  # resolves $MODELSPATH / ~/.mlatom/models (downloads if missing)
    return Path(omni.mlatom_model_dir), omni


def to_ml(m, labels=True):
    mol = ml.data.molecule()
    for z, r in zip(m["Z"], m["R"]):
        mol.atoms.append(ml.data.atom(element_symbol=SYM[int(z)], xyz_coordinates=np.array(r, float)))
    if not labels:
        return mol
    e = np.concatenate([[m["E0"]], m["E0"] + np.asarray(m["gaps"][: NS - 1], float)]) / H
    nanf = np.full((len(m["Z"]), 3), np.nan)
    g = [(-m["F0"] / H) if m.get("F0") is not None else nanf] + \
        [(-m["Fex"][:, s] / H) if (m.get("Fex") is not None and s < N_FORCE) else nanf for s in range(NS - 1)]
    for k in range(NS):
        st = mol.copy(atomic_labels=["xyz_coordinates"], molecular_labels=[])
        st.energy = float(e[k])
        st.add_xyz_derivative_property(np.asarray(g[k], float), "energy", "energy_gradients")
        mol.electronic_states.append(st)
    return mol


def labelled_db(idx):
    idx = [i for i in idx if np.isfinite(mols[i].get("E0", np.nan))][: a.limit]
    return ml.data.molecular_database([to_ml(mols[i]) for i in idx])


if a.member:
    pdir, _ = pretrained_dir()
    f = out / f"member{a.member}.pt"
    shutil.copy(pdir / f"OMNIP2x_CV_{a.member}.pt", f)
    tr, va = labelled_db(split["train"]), labelled_db(split["val"])
    print(f"member {a.member}: train {len(tr)} val {len(va)} states {NS}", flush=True)
    model = ml.models.vecmsani(model_file=str(f), nstates=NS, verbose=1)
    fl = [] if a.fixed_layers == "none" else [[int(x) for x in a.fixed_layers.split(",")]]
    hp = {"max_epochs": a.epochs, "gap_coefficient": a.gap_coefficient, "fixed_layers": fl}
    print(f"member {a.member}: hyperparameters {hp}", flush=True)
    model.train(molecular_database=tr, validation_molecular_database=va, property_to_learn="energy",
                xyz_derivative_property_to_learn="energy_gradients", hyperparameters=hp,
                reset_energy_shifter=True, reset_optimizer=True)
    (out / f"member{a.member}.done").write_text(f"{(time.time() - t0) / 3600:.3f} h\n")
    print(f"member {a.member} done in {(time.time() - t0) / 3600:.2f} h", flush=True)

if a.predict:
    po = Path(a.pred_out) if a.pred_out else out
    po.mkdir(parents=True, exist_ok=True)
    _, omni = pretrained_dir()
    members = [ml.models.vecmsani(model_file=str(out / f"member{k}.pt"), nstates=NS, verbose=0) for k in (1, 2, 3)]

    def predict(idx):
        res = []
        for s in range(0, len(idx), a.chunk):
            part = idx[s : s + a.chunk]
            E = np.zeros((3, len(part), NS)); G = [[None] * len(part) for _ in range(3)]
            for k, mdl in enumerate(members):
                db = ml.data.molecular_database([to_ml(mols[i], labels=False) for i in part])
                mdl.predict(molecular_database=db, nstates=NS, calculate_energy=True, calculate_energy_gradients=True)
                for n, mol in enumerate(db.molecules):
                    st = mol.electronic_states[:NS]
                    E[k, n] = [x.energy for x in st]
                    G[k][n] = np.array([x.get_energy_gradients() for x in st])
            dbf = ml.data.molecular_database([to_ml(mols[i], labels=False) for i in part])
            omni.osc_model.predict(molecular_database=dbf, calculate_energy=False, calculate_energy_gradients=False,
                                   nstates=NS, property_to_predict="f")
            for n, (i, molf) in enumerate(zip(part, dbf.molecules)):
                m = mols[i]
                Em, Es = E[:, n].mean(0) * H, E[:, n].std(0) * H
                Gm = np.mean([G[k][n] for k in range(3)], axis=0) * H
                osc = np.array([max(0.0, float(getattr(molf, f"f_state{q}", np.nan))) for q in range(1, NS)])
                order = np.argsort(Em[1:])
                Gex = Gm[1:][order]
                res.append(dict(id=m.get("id"), n_atoms=len(m["Z"]), n_heavy=int((np.asarray(m["Z"]) > 1).sum()),
                                Z=np.asarray(m["Z"]), source=m.get("source"), kind=m.get("kind"),
                                gaps_pred=(Em[1:] - Em[0])[order], gaps_ref=np.asarray(m["gaps"], float),
                                E0_pred=float(Em[0]), E0_ref=m.get("E0"), F0_pred=-Gm[0], F0_ref=m.get("F0"),
                                Fex_pred=np.stack([-Gex[q] for q in range(N_FORCE)], axis=1), Fex_ref=m.get("Fex"),
                                osc_pred=osc[order], osc_ref=m.get("osc"),
                                energy_std=np.concatenate([Es[:1], Es[1:][order]])))
            print(f"  {len(res)}/{len(idx)} ({time.time() - t0:.0f} s)", flush=True)
        return res

    preds = {}
    for k in ["val"] + [k for k in split if k.startswith("test_")]:
        idx = split[k][: a.limit] if a.limit else split[k]
        if idx:
            preds[k] = predict(idx)
            pickle.dump(preds, open(po / "predictions.pkl", "wb"))
    yaml.safe_dump(dict(head="omnip2x_finetuned", head_kwargs={}, n_states=NS - 1, n_sup=NS - 1, n_force=N_FORCE,
                        data=a.data, split=a.split, seed=0, epochs=a.epochs, gap_coefficient=a.gap_coefficient,
                        fixed_layers=a.fixed_layers), open(po / "config.yaml", "w"))
    json.dump(dict(model="OMNI-P2x fine-tuned (3 members)", hours_predict=(time.time() - t0) / 3600,
                   n=dict((k, len(v)) for k, v in preds.items())), open(po / "summary.json", "w"))
    (po / "train.log").write_text('{"skipped": 0}\n')
    print("predictions written", flush=True)
