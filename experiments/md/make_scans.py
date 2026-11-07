"""PES-scan geometries for E5 (computed with experiments/dataset/compute.py --mode given).

stilbene: relaxed C=C torsion scan (GFN2-xTB, dihedral constrained), 180 -> 0 deg.
biphenyl: relaxed inter-ring torsion scan, 0 -> 90 deg.
acrolein: rigid C=O stretch from the xTB minimum, 1.10 -> 1.60 A.
Output: data/wb97_aux/tasks/scans.pkl (kept outside data/wb97 so that it never enters training).
"""
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc.sampling import read_xyz_frames, write_xyz  # noqa

WD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/scans")


def mol3d(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def xtb_opt(wd, Z, R, constraint=None):
    wd.mkdir(parents=True, exist_ok=True)
    write_xyz(wd / "start.xyz", Z, R)
    inp = ""
    if constraint:
        inp = f"$constrain\n   force constant=2.0\n   {constraint}\n$end\n"
    (wd / "c.inp").write_text(inp)
    subprocess.run("xtb start.xyz --opt tight --input c.inp > opt.out 2>&1", cwd=wd, shell=True,
                   env={**__import__("os").environ, "OMP_NUM_THREADS": "1"}, check=True)
    return read_xyz_frames(wd / "xtbopt.xyz")[0][1]


def torsion_scan(name, smi, find, angles):
    m = mol3d(smi)
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    i, j, k, l = find(m)
    conf = m.GetConformer()
    R = xtb_opt(WD / name / "min", Z, conf.GetPositions())
    Req = R.copy()
    tasks = []
    for ang in angles:
        for a_, r in enumerate(R):
            conf.SetAtomPosition(a_, r.tolist())
        rdMolTransforms.SetDihedralDeg(conf, i, j, k, l, float(ang))
        R = xtb_opt(WD / name / f"{ang:03d}", Z, conf.GetPositions(),
                    f"dihedral: {i + 1},{j + 1},{k + 1},{l + 1},{float(ang):.1f}")
        conf2 = Chem.Conformer(conf)
        for a_, r in enumerate(R):
            conf2.SetAtomPosition(a_, r.tolist())
        got = rdMolTransforms.GetDihedralDeg(conf2, i, j, k, l)
        tasks.append(dict(id=f"SCAN_{name}_{ang:03d}", Z=Z, R_eq=Req, R=R, mode="given", source="SCAN",
                          n_heavy=int((Z > 1).sum()), smiles=smi, gaps_eq=np.full(10, np.nan), coord=float(ang),
                          coord_actual=float(got), T=0))
        print(name, ang, round(got, 1), flush=True)
    return tasks


def stilbene_atoms(m):
    for b in m.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if b.GetBondType() == Chem.BondType.DOUBLE and not a.GetIsAromatic() and a.GetSymbol() == "C":
            ia = [n.GetIdx() for n in a.GetNeighbors() if n.GetIsAromatic()][0]
            ic = [n.GetIdx() for n in c.GetNeighbors() if n.GetIsAromatic()][0]
            return ia, a.GetIdx(), c.GetIdx(), ic


def biphenyl_atoms(m):
    for b in m.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if a.GetIsAromatic() and c.GetIsAromatic() and not b.IsInRing():
            ia = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != c.GetIdx() and n.GetSymbol() == "C"][0]
            ic = [n.GetIdx() for n in c.GetNeighbors() if n.GetIdx() != a.GetIdx() and n.GetSymbol() == "C"][0]
            return ia, a.GetIdx(), c.GetIdx(), ic


tasks = []
tasks += torsion_scan("stilbene", "c1ccc(cc1)/C=C/c1ccccc1", stilbene_atoms, list(range(180, -1, -15)))
tasks += torsion_scan("biphenyl", "c1ccc(cc1)-c1ccccc1", biphenyl_atoms, list(range(0, 91, 10)))
# acrolein rigid C=O stretch
m = mol3d("C=CC=O")
Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
Req = xtb_opt(WD / "acrolein" / "min", Z, m.GetConformer().GetPositions())
iO = int(np.where(Z == 8)[0][0])
iC = int(np.argsort(np.linalg.norm(Req - Req[iO], axis=1))[1])
u = (Req[iO] - Req[iC]) / np.linalg.norm(Req[iO] - Req[iC])
for d in np.round(np.arange(1.10, 1.601, 0.05), 2):
    R = Req.copy()
    R[iO] = Req[iC] + d * u
    tasks.append(dict(id=f"SCAN_acroleinCO_{int(round(d * 100)):03d}", Z=Z, R_eq=Req, R=R, mode="given", source="SCAN",
                      n_heavy=4, smiles="C=CC=O", gaps_eq=np.full(10, np.nan), coord=float(d), coord_actual=float(d), T=0))
print("acrolein r(C=O) at minimum", np.linalg.norm(Req[iO] - Req[iC]))
pickle.dump(tasks, open(ROOT / "data/wb97_aux/tasks/scans.pkl", "wb"))
print("tasks", len(tasks))
