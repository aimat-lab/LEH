"""Load the new TD-wB97X-D/6-31G* dataset (eV, Angstrom).

Every finished npz gives one off-equilibrium entry with E0, F0, 10 gaps and S1-S3 forces.
For QCDGE-derived molecules an equilibrium entry (QCDGE geometry + QCDGE gaps, no E0/forces) is added.
"""
import json
import re
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
GRAD = HARTREE_EV / BOHR_A
EXC_RE = re.compile(r"_s\d+md\d+$")
AL_RE = re.compile(r"_r(\d+)s\d+md\d+$")  # active-learning rounds >= 2 (experiments/dataset/s1_sampling.py --round s1rN)  # geometries from model-driven excited-state MD (experiments/dataset/s1_sampling.py)


def eq_labels(eq_dir, mid, R_eq):
    """E0 (eV) and F0 (eV/A) at the QCDGE minimum from compute_e0.py output, or None."""
    if eq_dir is None:
        return None
    f = Path(eq_dir) / f"{mid}.npz"
    if not f.exists():
        return None
    e = np.load(f)
    if not np.allclose(e["R"], R_eq, atol=1e-6):
        raise ValueError(f"geometry mismatch for {mid} in {f}")
    return dict(E0=float(e["E0"]) * HARTREE_EV, F0=-e["grad0"] * GRAD)


def load_dirs(dirs, root_tol=2e-4, grad_e_tol=1e-5, add_eq=True, eq_dir=None, eq_osc=None):
    """eq_osc: optional dict molecule id -> QCDGE oscillator strengths of the 10 singlets (equilibrium entries)."""
    mols, stats = [], dict(files=0, masked_roots=0, conn_changed=0, failed=0)
    eq_done = set()
    for d in dirs:
        d = Path(d)
        stats["failed"] += len(list(d.glob("*.failed.json")))
        for f in sorted(d.glob("*.npz")):
            r = np.load(f)
            fl = json.loads(str(r["flags"]))
            stats["files"] += 1
            mid = str(r["id"])
            Fex = -r["gradex"].transpose(1, 0, 2) * GRAD
            for s in range(Fex.shape[1]):
                bad = (fl.get(f"root{s+1}_dev_hartree", 0) > root_tol or fl.get(f"egrad{s+1}_energy_dev", 0) > grad_e_tol
                       or not fl.get(f"egrad{s+1}_ok", True))
                if bad:
                    Fex[:, s] = np.nan
                    stats["masked_roots"] += 1
            stats["conn_changed"] += int(fl.get("connectivity_changed", False))
            src = str(r["source"])
            if "mol" in r.files:  # several geometries of one molecule share the molecule id
                base = str(r["mol"])
            else:
                base = mid.rsplit("_", 1)[0] if src == "ZINC" else mid
            Z = r["Z"].astype(np.int64)
            mols.append(dict(Z=Z, R=r["R"], E0=float(r["E0"]) * HARTREE_EV, F0=-r["grad0"] * GRAD,
                             gaps=r["exc"] * HARTREE_EV, Fex=Fex, id=mid + ("" if src == "ZINC" else "_md"),
                             mol=base, source=src, kind="eq" if mid.endswith("_eq") else ("excmd" if (EXC_RE.search(mid) or AL_RE.search(mid)) else "offeq"),
                             al_round=int(AL_RE.search(mid).group(1)) if AL_RE.search(mid) else (1 if EXC_RE.search(mid) else 0),
                             T=float(fl.get("T", 0)), conn_changed=bool(fl.get("connectivity_changed", False)),
                             n_heavy=int((Z > 1).sum()), smiles=str(r["smiles"]) if "smiles" in r else ""))
            if "osc_strength" in r.files:  # stored from 2026-10-10 on
                mols[-1]["osc"] = np.asarray(r["osc_strength"], float)
            if add_eq and src != "ZINC" and np.all(np.isfinite(r["gaps_eq"])) and base not in eq_done:
                eq_done.add(base)
                e = dict(Z=Z, R=r["R_eq"], gaps=np.asarray(r["gaps_eq"], float), id=base + "_eq", mol=base,
                         source=src, kind="qcdge_eq", T=0.0, conn_changed=False, n_heavy=int((Z > 1).sum()))
                lab = eq_labels(eq_dir, base, r["R_eq"])
                if eq_osc is not None and base in eq_osc:
                    e["osc"] = np.asarray(eq_osc[base], float)
                if lab is not None:  # our S0 at the QCDGE minimum; excitation energies stay QCDGE's (agree to 5 meV)
                    e.update(lab)
                    stats["eq_with_E0"] = stats.get("eq_with_E0", 0) + 1
                mols.append(e)
    return mols, stats
