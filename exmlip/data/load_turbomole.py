"""Collect TURBOMOLE single points (data/tm/<mol>/*.npz) into the common schema (eV, Angstrom)."""
import json
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
GRAD_CONV = HARTREE_EV / BOHR_A  # Ha/bohr -> eV/A


def load_tm(root, root_tol=2e-4, grad_e_tol=1e-5):
    mols, rejected = [], {}
    for d in sorted(Path(root).iterdir()):
        if not (d / "geoms.npz").exists():
            continue
        g = np.load(d / "geoms.npz")
        meta = dict(name=d.name, smiles=str(g["smiles"]), split=str(g["split"]), family=str(g["family"]))
        for f in sorted(d.glob("[0-9][0-9][0-9][0-9].npz")):
            r = np.load(f)
            fl = json.loads(str(r["flags"]))
            i = int(f.stem)
            bad = [k for k, v in fl.items() if (k.startswith("root") and v > root_tol) or
                   (k.endswith("energy_dev") and v > grad_e_tol) or (k.endswith("_ok") and not v) or
                   (k == "scf_converged" and not v) or (k == "e0_consistent" and not v)]
            Fex = -r["gradex"].transpose(1, 0, 2) * GRAD_CONV  # [n, q, 3]
            if bad:
                rejected[f"{d.name}/{i}"] = bad
                # keep energies, mask the forces of the problematic roots
                for k in bad:
                    if k.startswith("root") or k.startswith("egrad"):
                        s = int("".join(c for c in k.split("_")[0] if c.isdigit())) - 1
                        Fex[:, s] = np.nan
            mols.append(dict(Z=r["Z"], R=r["R"], E0=float(r["E0"]) * HARTREE_EV,
                             F0=-r["grad0"] * GRAD_CONV, gaps=r["exc"] * HARTREE_EV, Fex=Fex,
                             id=f"{d.name}_{i}", T=int(g["T"][i]), n_heavy=int((r["Z"] > 1).sum()), **meta))
    return mols, rejected
