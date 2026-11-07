"""QCDGE (Zhu et al. 2024): B3LYP-D3BJ/6-31G* minima, 10 singlet TD-wB97X-D/6-31G* excitations (eV)."""
import json

import h5py
import numpy as np


def scan_heavy_counts(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        nh = np.array([int((f[k]["excited_state/labels"][0] > 1).sum()) for k in keys])
    return keys, nh


def read_molecules(path, keys, prefix):
    out = []
    with h5py.File(path, "r") as f:
        for k in keys:
            g = f[k]["excited_state"]
            info = json.loads(g["Info_of_AllExcitedStates"][0])
            s = sorted(float(v["excitation_e_eV"].split()[0]) for v in info.values()
                       if v["state_type"].startswith("Singlet"))
            Z = g["labels"][0].astype(np.int64)
            out.append(dict(Z=Z, R=g["coords"][()].astype(float), gaps=np.array(s), id=f"{prefix}_{k}",
                            n_heavy=int((Z > 1).sum())))
    return out
