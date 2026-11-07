"""Zero-shot evaluation of the pretrained OMNI-P2x model (Martyka, Tong, Jankowska, Dral; MLatom) on our test sets.

OMNI-P2x is used exactly as released (ensemble of three vecmsani models + oscillator-strength model, 11 states:
S0 and S1-S10), without retraining. Its reference data differ from ours (excited states from PubChemQC TD-DFT,
ground state from ANI-1ccx-type CCSD(T) data), so absolute S0 energies are not comparable; the analysis
(omnip2x_analyze.py) uses excitation energies, within-molecule energy differences, forces and oscillator strengths,
raw and after a per-state linear recalibration fitted on the validation set.

Writes <out>/predictions.pkl in the exmlip format (eV, eV/A) with the ensemble standard deviation per state, plus
config.yaml / summary.json / train.log so that experiments/wb97/analyze.py can read the run.
Run with .venv_mlatom. Usage:
  python experiments/baselines/omnip2x_zeroshot.py --data data/wb97_all_100k_s1_conj.pkl \
      --split experiments/baselines/split_zeroshot.json --out experiments/wb97/runs_zeroshot/omnip2x [--limit N]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

H = 27.211386245988  # eV per Hartree
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nstates", type=int, default=11, help="S0 + excited states (OMNI-P2x provides 11)")
ap.add_argument("--n_force", type=int, default=3, help="excited states whose forces are compared (we have S1-S3)")
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--limit", type=int, default=None, help="debug: first N entries per set")
ap.add_argument("--sets", nargs="*", default=None, help="split keys to predict (default: val + all test_*)")
a = ap.parse_args()

import mlatom as ml  # noqa: E402

mols = pickle.load(open(a.data, "rb"))
split = json.load(open(a.split))
keys = a.sets or (["val"] + [k for k in split if k.startswith("test_")])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
omni = ml.models.methods(method="OMNI-P2x")
omni.warnings = False
print("OMNI-P2x loaded from", getattr(omni, "mlatom_model_dir", "?"), flush=True)


def to_ml(m):
    mol = ml.data.molecule()
    for z, r in zip(m["Z"], m["R"]):
        mol.atoms.append(ml.data.atom(element_symbol=SYM[int(z)], xyz_coordinates=np.array(r, float)))
    return mol


def predict(idx):
    """OMNI-P2x energies/gradients = mean over its three ensemble members (exactly OMNI-P2x's 'average' operator; the
    members are called directly because the MLatom 3.25.5 model tree drops per-state gradients, see STATUS);
    states are put in adiabatic (energy) order, gradients/oscillator strengths/std permuted accordingly."""
    members = [c.model for c in omni.emodel.children]
    res = []
    for s in range(0, len(idx), a.chunk):
        part = idx[s : s + a.chunk]
        E = np.zeros((len(members), len(part), a.nstates)); G = [[None] * len(part) for _ in members]
        for k, mdl in enumerate(members):
            db = ml.data.molecular_database([to_ml(mols[i]) for i in part])
            mdl.predict(molecular_database=db, nstates=a.nstates, calculate_energy=True, calculate_energy_gradients=True)
            for n, mol in enumerate(db.molecules):
                st = mol.electronic_states[: a.nstates]
                E[k, n] = [x.energy for x in st]
                G[k][n] = np.array([x.get_energy_gradients() for x in st])
        dbf = ml.data.molecular_database([to_ml(mols[i]) for i in part])
        omni.osc_model.predict(molecular_database=dbf, calculate_energy=False, calculate_energy_gradients=False,
                               nstates=a.nstates, property_to_predict="f")
        for n, (i, molf) in enumerate(zip(part, dbf.molecules)):
            m = mols[i]
            Em = E[:, n].mean(0) * H; Es = E[:, n].std(0) * H                            # eV
            Gm = np.mean([G[k][n] for k in range(len(members))], axis=0) * H             # eV/A, [state, atom, 3]
            osc = np.array([max(0.0, float(getattr(molf, f"f_state{q}", np.nan))) for q in range(1, a.nstates)])
            order = np.argsort(Em[1:])
            gaps = (Em[1:] - Em[0])[order]
            Gex = Gm[1:][order]
            r = dict(id=m.get("id"), n_atoms=len(m["Z"]), n_heavy=int((np.asarray(m["Z"]) > 1).sum()),
                     Z=np.asarray(m["Z"]), source=m.get("source"), kind=m.get("kind"),
                     gaps_pred=gaps, gaps_ref=np.asarray(m["gaps"], float), E0_pred=float(Em[0]), E0_ref=m.get("E0"),
                     F0_pred=-Gm[0], F0_ref=m.get("F0"),
                     Fex_pred=np.stack([-Gex[q] for q in range(a.n_force)], axis=1), Fex_ref=m.get("Fex"),
                     osc_pred=osc[order], osc_ref=m.get("osc"), energy_std=np.concatenate([Es[:1], Es[1:][order]]))
            res.append(r)
        print(f"  {len(res)}/{len(idx)} ({time.time() - t0:.0f} s)", flush=True)
    return res


preds = {}
for k in keys:
    idx = split.get(k, [])[: a.limit] if a.limit else split.get(k, [])
    if not idx:
        continue
    print(f"{k}: {len(idx)} geometries", flush=True)
    preds[k] = predict(idx)
    pickle.dump(preds, open(out / "predictions.pkl", "wb"))  # incremental (restart by rerunning)
yaml.safe_dump(dict(head="omnip2x_zeroshot", head_kwargs={}, n_states=a.nstates - 1, n_sup=a.nstates - 1,
                    n_force=a.n_force, data=a.data, split=a.split, seed=0), open(out / "config.yaml", "w"))
json.dump(dict(epochs=0, hours=(time.time() - t0) / 3600, model="OMNI-P2x (pretrained, zero-shot)",
               n_geometries={k: len(v) for k, v in preds.items()}), open(out / "summary.json", "w"))
(out / "train.log").write_text('{"skipped": 0}\n')
print("done", (time.time() - t0) / 3600, "h")
