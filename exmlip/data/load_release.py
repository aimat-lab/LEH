"""Rebuild the training data files of this repository from the Zenodo dataset (leh_wb97xd_v4.h5).

The training configurations refer to pickled lists of entries (data/*.pkl). All of them are prefixes of the v4 entry
order, so they are written from the HDF5 file:
  data/wb97_prod_v4.pkl            all 303,061 entries (production models, splits experiments/al/split_prod_v4.json)
  data/wb97_all_100k_s1_conj.pkl   first 234,942 (benchmark data incl. dyes and ladders; experiments/conj/*.json)
  data/wb97_all_100k_s1.pkl        first 222,172 (benchmark, learning curves, ablations; experiments/wb97/split_100k_s1*.json)
  data/wb97_all_100k.pkl           first 204,341 (hyperparameter runs without excited-state sampling)
With --populations, tddft_populations.h5 is also unpacked into the per-geometry files that
experiments/eigvec/tddft/pop_labels.py reads (data/eigvec_tddft_abl10k/pop/, experiments/eigvec/tddft/pop/).

Usage: python -m exmlip.data.load_release /path/to/zenodo [--out data] [--populations]
"""
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np

PREFIXES = {"wb97_prod_v4.pkl": None, "wb97_all_100k_s1_conj.pkl": 234942, "wb97_all_100k_s1.pkl": 222172,
            "wb97_all_100k.pkl": 204341}
STR_KEYS = {"id": "id", "molecule": "mol", "source": "source", "kind": "kind", "smiles": "smiles",
            "conj_class": "conj", "dye_database": "db", "ladder_family": "family"}


def entries(h5_path):
    with h5py.File(h5_path, "r") as f:
        e, a = f["entries"], f["atoms"]
        s = {k: [x.decode() for x in e[k][:]] for k in STR_KEYS}
        n_at, off = e["n_atoms"][:], e["atom_offset"][:]
        gaps, osc, E0, T = e["excitation_energies"][:], e["oscillator_strengths"][:], e["E0"][:], e["md_temperature"][:]
        n_heavy, conn, alr = e["n_heavy"][:], e["connectivity_changed"][:], e["al_round"][:]
        Z, R, F0, Fex = a["Z"][:].astype(np.int64), a["R"][:], a["F0"][:], a["Fex"][:]
    out = []
    for i in range(len(E0)):
        sl = slice(off[i], off[i] + n_at[i])
        g = gaps[i][np.isfinite(gaps[i])]
        m = dict(Z=Z[sl], R=R[sl], E0=float(E0[i]), F0=F0[sl], gaps=g, n_heavy=int(n_heavy[i]),
                 conn_changed=bool(conn[i]), T=float(T[i]))
        fx = Fex[sl]
        m["Fex"] = fx if np.isfinite(fx).any() else None
        if np.isfinite(osc[i]).any():
            m["osc"] = osc[i][: len(g)]
        for k, key in STR_KEYS.items():
            if s[k][i] or key in ("id", "mol", "source", "kind", "smiles"):
                m[key] = s[k][i]
        if alr[i] >= 2:
            m["al_round"] = int(alr[i])
        out.append(m)
    return out


def unpack_populations(h5_path, D, out_dirs):
    with h5py.File(h5_path, "r") as f:
        ids, sets, ent = [x.decode() for x in f["id"][:]], [x.decode() for x in f["set"][:]], f["entry"][:]
        n_at, off, om_off = f["n_atoms"][:], f["atom_offset"][:], f["omega_offset"][:]
        cols = {k: f[k][:] for k in ("hole", "electron", "pop", "pop_X", "qtr")}
        exc, nxy, Z, om = f["exc_eV"][:], f["norm_xy"][:], f["Z"][:], f["omega"]
        for k, name in enumerate(ids):
            n, sl = n_at[k], slice(off[k], off[k] + n_at[k])
            d = Path(out_dirs[sets[k]]); d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{name}.npz", exc_eV=exc[k], norm_xy=nxy[k], Z=Z[sl].astype(np.int64), R=D[ent[k]]["R"],
                     hole=cols["hole"][sl].T, elec=cols["electron"][sl].T, pop=cols["pop"][sl].T,
                     pop_X=cols["pop_X"][sl].T, qtr=cols["qtr"][sl].T,
                     omega=om[om_off[k]: om_off[k] + 10 * n * n].astype(np.float64).reshape(10, n, n))
    print("populations unpacked:", len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zenodo_dir")
    ap.add_argument("--out", default="data")
    ap.add_argument("--populations", action="store_true")
    a = ap.parse_args()
    z, out = Path(a.zenodo_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = entries(z / "leh_wb97xd_v4.h5")
    for name, n in PREFIXES.items():
        pickle.dump(D[:n] if n else D, open(out / name, "wb"))
        print("written", out / name, n or len(D))
    if a.populations:
        unpack_populations(z / "tddft_populations.h5", D, {"training_subset": out / "eigvec_tddft_abl10k/pop",
                                                           "comparison_set": "experiments/eigvec/tddft/pop"})


if __name__ == "__main__":
    main()
