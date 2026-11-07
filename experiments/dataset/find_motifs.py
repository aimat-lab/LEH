"""Photoisomerisation-relevant motifs of all dataset molecules (bond orders perceived from the equilibrium geometry).

Motifs: acyclic X=Y (X, Y in C, N) conjugated with an aromatic ring (styrene/stilbene, aryl imines/azo) or with
another double bond (dienes/polyenes, enones C=C-C=O, enimines). These are the coordinates (double-bond twist,
S1/S0 approach) that S0-thermal and round-1 S1 sampling barely covered (round 1: 3 frames with an aryl-C=C twist
> 45 deg). Output: experiments/dataset/motifs.json {mol: [motif names]} (only molecules with >= 1 motif) + counts.
"""
import json
import pickle
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

ROOT = Path(__file__).resolve().parents[2]
SM = {"aryl_db": "[a]-[#6,#7]=!@[#6,#7]", "conj_db": "[#6,#7,#8]=[#6,#7]-[#6,#7]=!@[#6,#7]"}
PAT = {k: Chem.MolFromSmarts(v) for k, v in SM.items()}


def work(a):
    Z, R = a
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromXYZBlock(f"{len(Z)}\n\n" + "".join(f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {r[0]:.6f} "
                                                        f"{r[1]:.6f} {r[2]:.6f}\n" for z, r in zip(Z, R)))
        rdDetermineBonds.DetermineBonds(m, charge=0)
    except Exception:
        return None
    return [k for k, p in PAT.items() if m.HasSubstructMatch(p)]


if __name__ == "__main__":
    D = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
    eq = {}
    for m in D:
        if m["kind"] in ("qcdge_eq", "eq"):
            eq.setdefault(m["mol"], (np.asarray(m["Z"]), np.asarray(m["R"]), m["source"]))
    mols = list(eq)
    with Pool(12) as p:
        res = p.map(work, [eq[k][:2] for k in mols], chunksize=64)
    out = {k: r for k, r in zip(mols, res) if r}
    cnt = Counter((eq[k][2] == "ZINC", tuple(r)) for k, r in out.items())
    json.dump(out, open(ROOT / "experiments/dataset/motifs.json", "w"))
    print("molecules", len(mols), "perceived", sum(r is not None for r in res), "with motif", len(out))
    for k, n in sorted(cnt.items()):
        print(("ZINC " if k[0] else "QCDGE"), k[1], n)
