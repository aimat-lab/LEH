"""Export our splits to X-MACE extxyz (REF_energy (1, n_states) and REF_forces (N, n_states, 3) in atoms.info).

--mode wb97    : states S0..S(n_states-1) as total energies (eV), forces for all of them where available. Frames
                 without complete force labels for these states (equilibrium entries: no excited forces) keep their
                 energies but get config_forces_weight = 0, because X-MACE masks forces per frame, not per state.
--mode spectra : the n_states lowest excitation energies as 'energies', no forces (as in the X-MACE spectra tutorial).
Writes OUT/{train,valid,test_<name>}.xyz and OUT/index.json (entry indices per file, to map predictions back).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["wb97", "spectra"], required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_states", type=int, default=4)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
mols = pickle.load(open(a.data, "rb"))
split = json.load(open(a.split))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ns = a.n_states


def frame(m):
    at = Atoms(numbers=m["Z"], positions=m["R"])
    if a.mode == "wb97":
        e = np.concatenate([[m["E0"]], m["E0"] + np.asarray(m["gaps"][: ns - 1], float)])
        at.info["REF_energy"] = e.reshape(1, ns)
        F = np.zeros((len(m["Z"]), ns, 3))
        full = "F0" in m and "Fex" in m and m["Fex"].shape[1] >= ns - 1 and np.all(np.isfinite(m["Fex"][:, : ns - 1]))
        if full:
            F[:, 0] = m["F0"]
            F[:, 1:] = m["Fex"][:, : ns - 1]
        at.info["REF_forces"] = F
        at.info["config_forces_weight"] = 1.0 if full else 0.0
    else:
        at.info["REF_energy"] = np.asarray(m["gaps"][:ns], float).reshape(1, ns)
    at.info["entry"] = int(m.get("_idx", -1))
    return at


def usable(m):
    return ("E0" in m and np.isfinite(m["E0"])) if a.mode == "wb97" else np.all(np.isfinite(m["gaps"][:ns]))


index = {}
for name, idx in [("train", split["train"]), ("valid", split["val"])] + \
        [(k, v) for k, v in split.items() if k.startswith("test_")]:
    idx = [i for i in idx if usable(mols[i])][: a.limit]
    if not idx:
        continue
    frames = []
    for i in idx:
        mols[i]["_idx"] = i
        frames.append(frame(mols[i]))
    write(out / f"{name}.xyz", frames, format="extxyz")
    index[name] = idx
    print(name, len(idx), "frames", "| forces off:", sum(f.info.get("config_forces_weight", 0) == 0 for f in frames))
json.dump(dict(index=index, mode=a.mode, n_states=ns, data=a.data, split=a.split), open(out / "index.json", "w"))
