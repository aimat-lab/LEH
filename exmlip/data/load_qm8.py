"""QM8 (Ramakrishnan et al. 2015): equilibrium geometries + 2 lowest singlet excitations."""
import numpy as np

HARTREE_EV = 27.211386245988
SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
COLS = {"cc2": (0, 1), "pbe0_svp": (4, 5), "pbe0_tzvp": (8, 9), "cam_tzvp": (12, 13)}


def load_qm8(spec_path, xyz_path, method="pbe0_tzvp"):
    spec = {}
    for l in open(spec_path):
        if l.strip().startswith("#") or not l.strip():
            continue
        v = l.split()
        spec[int(v[0])] = np.array([float(x) for x in v[1:]])
    lines = open(xyz_path).read().splitlines()
    mols, i = [], 0
    while i < len(lines):
        n = int(lines[i]); idx = int(lines[i + 1].split()[1])
        blk = lines[i + 2 : i + 2 + n]
        Z = np.array([SYM2Z[l.split()[0]] for l in blk])
        R = np.array([[float(x) for x in l.split()[1:4]] for l in blk])
        c = COLS[method]
        mols.append(dict(Z=Z, R=R, gaps=spec[idx][list(c)] * HARTREE_EV, id=f"qm8_{idx}"))
        i += n + 2
    return mols
