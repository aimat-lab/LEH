"""Geometry generation: RDKit embedding, xtb optimization and xtb (GFN2) MD sampling."""
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16}
Z2SYM = {v: k for k, v in SYM2Z.items()}


def write_xyz(path, Z, R, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(Z)}\n{comment}\n")
        for z, r in zip(Z, R):
            f.write(f"{Z2SYM[int(z)]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")


def read_xyz_frames(path):
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i]); blk = lines[i + 2 : i + 2 + n]
        Z = np.array([SYM2Z[l.split()[0].capitalize()] for l in blk])
        R = np.array([[float(x) for x in l.split()[1:4]] for l in blk])
        frames.append((Z, R)); i += n + 2
    return frames


def embed(smiles, seed=1):
    """RDKit ETKDG embedding + MMFF cleanup (the xTB optimisation follows). Kept deterministic and identical for all
    datasets; a few strained/helical systems (helicenes, bridged cages: 7 of 5025 dyes) cannot be embedded and are
    recorded as failures (random-coordinate retries were tested and do not help)."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=seed) < 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(m)
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    return Z, m.GetConformer().GetPositions()


def xtb_opt(wd, Z, R):
    wd = Path(wd); wd.mkdir(parents=True, exist_ok=True)
    write_xyz(wd / "start.xyz", Z, R)
    subprocess.run("xtb start.xyz --opt tight > opt.out 2>&1", cwd=wd, shell=True,
                   env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    return read_xyz_frames(wd / "xtbopt.xyz")[0]


def xtb_md(wd, Z, R, temp, time_ps, dump_fs, seed=0):
    wd = Path(wd); wd.mkdir(parents=True, exist_ok=True)
    write_xyz(wd / "start.xyz", Z, R)
    (wd / "md.inp").write_text(
        f"$md\n   temp={temp}\n   time={time_ps}\n   dump={dump_fs}\n   step=0.5\n   hmass=1\n"
        f"   shake=0\n   nvt=true\n   skip=1\n$end\n")
    subprocess.run("xtb start.xyz --input md.inp --md > md.out 2>&1", cwd=wd, shell=True,
                   env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    return read_xyz_frames(wd / "xtb.trj")
