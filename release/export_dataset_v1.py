"""Export the frozen benchmark dataset (v1) for Zenodo and ChemMatData.

Outputs (default release/out_v1/):
  zenodo/leh_wb97xd_v1.h5        all geometries with labels (ragged atoms stored flat with offsets), splits as index arrays
  zenodo/fc_test_eq_forces.h5    S1-S3 forces at equilibrium geometries of test molecules (extra test set)
  zenodo/splits/*.json           the same splits as JSON (indices into the entry order of the HDF5 file)
  zenodo/README.md               dataset card (fields, units, level of theory, splits, license)
  chemmatdata/leh_wb97xd_eq.csv  one row per molecule at its equilibrium geometry: SMILES, S1-S10 (eV), f1-f10, split
  MANIFEST.json                  sha256 and bytes of every exported file
Optional: --extxyz writes one extended-XYZ file per split (MACE/ASE-compatible keys) into zenodo/extxyz/.

SMILES (column/dataset smiles_source): stored for dyes, ladders and ZINC molecules ("input"); for QCDGE-derived
molecules QCDGE's SMILES_RDKIT_CAN from the HDF5 attributes (--qcdge_h5_dir, "qcdge_hdf5") or final_all.csv
(--qcdge_csv, "qcdge_csv") if it is closed-shell, otherwise perceived from the QCDGE minimum with RDKit
(rdDetermineBonds, charge 0, closed-shell only; "perceived_from_geometry"); empty if neither works.
Run locally or in a job (not on a login node):
  python -I release/export_dataset.py --out release/out_v1_qcdge_smiles --qcdge_h5_dir raw_data/qcdge \
      --qcdge_csv external_data/qcdge_final_all/final_all.csv [--extxyz]
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data/wb97_all_100k_s1_conj.pkl"))
ap.add_argument("--fc", default=str(ROOT / "data/wb97/fc_test_labels.pkl"))
ap.add_argument("--out", default=str(ROOT / "release/out_v1"))
ap.add_argument("--qcdge_csv", default=None, help="QCDGE final_all.csv (SMILES; incomplete)")
ap.add_argument("--qcdge_h5_dir", default=None, help="directory with QCDGE A_9/A_10/B_9/B_10.hdf5 (SMILES attributes)")
ap.add_argument("--extxyz", action="store_true")
ap.add_argument("--limit", type=int, default=None, help="export only the first N entries (testing)")
a = ap.parse_args()

SPLITS = {"benchmark_A": ROOT / "experiments/conj/split_A_conj.json",   # main models (model A) incl. conjugated tests
          "model_B": ROOT / "experiments/conj/split_B.json"}            # + training dye scaffolds
VERSION = "v1"
OUT = Path(a.out); Z_DIR = OUT / "zenodo"; C_DIR = OUT / "chemmatdata"
for d in (Z_DIR / "splits", C_DIR):
    d.mkdir(parents=True, exist_ok=True)

D = pickle.load(open(a.data, "rb"))
if a.limit:
    D = D[:a.limit]
N = len(D)
print("entries", N)

# ---------------------------------------------------------------- SMILES
# QCDGE SMILES (SMILES_RDKIT_CAN, as in ChemMatData's qcdge dataset): from the HDF5 group attributes (complete) or
# final_all.csv (lacks ~11 % of our molecules). QCDGE writes '1' where its RDKit conversion failed. Our molecule id
# '<subset>_<key>' maps to the CSV Index '<Aa|Ab|Ba|Bb><number>' and to HDF5 group '<key>'.
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdDetermineBonds  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402
RDLogger.DisableLog("rdApp.*")
CSV_PREFIX = {"A9": "Aa", "A10": "Ab", "B9": "Ba", "B10": "Bb"}
H5_FILE = {"A9": "A_9.hdf5", "A10": "A_10.hdf5", "B9": "B_9.hdf5", "B10": "B_10.hdf5"}


def valid(s):
    """Usable QCDGE SMILES: parses and is closed-shell. ~6 % of QCDGE's SMILES_RDKIT_CAN carry radical atoms
    (e.g. [C][N] for a C=N double bond) although QCDGE computed closed-shell states; for those the SMILES
    perceived from our geometry is used instead."""
    m = Chem.MolFromSmiles(s) if s and s != "1" else None
    return m is not None and Descriptors.NumRadicalElectrons(m) == 0


smiles, smiles_src = {}, {}
qmols = sorted({m["mol"] for m in D if m["source"] in CSV_PREFIX})
if a.qcdge_h5_dir:
    for sub, fn in H5_FILE.items():
        p = Path(a.qcdge_h5_dir) / fn
        if not p.exists():
            print("missing", p); continue
        with h5py.File(p, "r") as f:
            for mol in (x for x in qmols if x.split("_")[0] == sub):
                key = mol.split("_", 1)[1]
                key = key if key in f else str(int(key))
                if key in f:
                    s = str(f[key].attrs.get("SMILES_RDKIT_CAN", ""))
                    if valid(s):
                        smiles[mol], smiles_src[mol] = s, "qcdge_hdf5"
if a.qcdge_csv:
    idx = {}
    with open(a.qcdge_csv) as f:
        for r in csv.DictReader(f):
            idx[(r["Index"][:2], int(r["Index"][2:]))] = r.get("Smiles_rdkit_can", "")
    for mol in qmols:
        sub, key = mol.split("_", 1); s = idx.get((CSV_PREFIX[sub], int(key)), "")
        if mol not in smiles and valid(s):
            smiles[mol], smiles_src[mol] = s, "qcdge_csv"
print("QCDGE SMILES:", len(smiles), "of", len(qmols), "QCDGE molecules")


def perceive(Z, R):
    try:
        m = Chem.RWMol()
        for z in Z:
            m.AddAtom(Chem.Atom(int(z)))
        conf = Chem.Conformer(len(Z))
        for i, r in enumerate(R):
            conf.SetAtomPosition(i, [float(x) for x in r])
        m.AddConformer(conf)
        rdDetermineBonds.DetermineBonds(m, charge=0)
        m = Chem.RemoveHs(m.GetMol())
        return Chem.MolToSmiles(m) if Descriptors.NumRadicalElectrons(m) == 0 else ""
    except Exception:
        return ""


def stereo_free(s):
    m = Chem.MolFromSmiles(s) if s else None
    if m is None:
        return ""
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


mol_smiles, n_perceived, agree = {}, 0, [0, 0]
for m in D:
    if m.get("smiles"):
        mol_smiles.setdefault(m["mol"], m["smiles"]); smiles_src.setdefault(m["mol"], "input")
for m in D:
    if m["kind"] in ("qcdge_eq", "eq") and m["mol"] not in mol_smiles:
        p = perceive(m["Z"], m["R"]); n_perceived += 1
        if m["mol"] in smiles:  # cross-check of the geometry against the QCDGE SMILES
            agree[0] += stereo_free(p) == stereo_free(smiles[m["mol"]]); agree[1] += 1
            mol_smiles[m["mol"]] = smiles[m["mol"]]
        else:
            mol_smiles[m["mol"]] = p; smiles_src[m["mol"]] = "perceived_from_geometry" if p else ""
print("SMILES:", len(mol_smiles), "molecules,", n_perceived, "perceived from geometry,",
      sum(not s for s in mol_smiles.values()), "without SMILES; perceived == QCDGE (stereo-free):", agree)
SMILES_STATS = dict(n_molecules=len(mol_smiles), sources=dict(__import__("collections").Counter(smiles_src.get(k, "")
                    for k in mol_smiles)), perceived_vs_qcdge_agree=agree[0], perceived_vs_qcdge_n=agree[1])

# ---------------------------------------------------------------- HDF5
n_at = np.array([len(m["Z"]) for m in D]); off = np.concatenate([[0], np.cumsum(n_at)[:-1]])
A = int(n_at.sum())
Zc = np.zeros(A, np.int8); Rc = np.zeros((A, 3)); F0c = np.zeros((A, 3)); Fexc = np.full((A, 3, 3), np.nan)
gaps = np.full((N, 10), np.nan); osc = np.full((N, 10), np.nan); E0 = np.zeros(N); T = np.full(N, np.nan)
for i, m in enumerate(D):
    s = slice(off[i], off[i] + n_at[i])
    Zc[s] = m["Z"]; Rc[s] = m["R"]; F0c[s] = m["F0"]; E0[i] = m["E0"]
    if m.get("Fex") is not None:
        Fexc[s] = m["Fex"]
    g = np.asarray(m["gaps"], float)[:10]; gaps[i, :len(g)] = g
    if m.get("osc") is not None:
        o = np.asarray(m["osc"], float)[:10]; osc[i, :len(o)] = o
    T[i] = m.get("T", np.nan) if m["kind"] in ("offeq", "excmd") else np.nan
has_fex = np.array([[m.get("Fex") is not None and np.all(np.isfinite(np.asarray(m["Fex"])[:, s])) for s in range(3)] for m in D])
strs = lambda key, f=lambda m, k: m.get(k, ""): np.array([str(f(m, key) or "") for m in D], dtype=h5py.string_dtype())
conj = np.array([m.get("conj", "") or "" for m in D], dtype=h5py.string_dtype())

h5 = Z_DIR / f"leh_wb97xd_{VERSION}.h5"
with h5py.File(h5, "w") as f:
    f.attrs.update(dict(
        title="TD-wB97X-D/6-31G* ground- and excited-state energies and forces of QCDGE-derived and conjugated molecules",
        version=VERSION, level_of_theory="TD-DFT wB97X-D/6-31G*, full linear response (RPA), TURBOMOLE 7.5.1",
        units="energies eV; excitation energies eV; forces eV/Angstrom; coordinates Angstrom; temperature K",
        n_entries=N, n_atoms_total=A,
        note="Fex[:, n] are forces of S_{n+1} (n=0,1,2); NaN = not available (masked or not computed). osc NaN = not available. "
             "At QCDGE minima (kind qcdge_eq) excitation energies and oscillator strengths are taken from QCDGE; E0/F0 recomputed."))
    e = f.create_group("entries")
    e["id"] = strs("id"); e["molecule"] = strs("mol"); e["source"] = strs("source"); e["kind"] = strs("kind")
    e["conj_class"] = conj; e["dye_database"] = strs("db"); e["ladder_family"] = strs("family")
    e["smiles"] = np.array([mol_smiles.get(m["mol"], "") for m in D], dtype=h5py.string_dtype())
    e["smiles_source"] = np.array([smiles_src.get(m["mol"], "") for m in D], dtype=h5py.string_dtype())
    e["n_atoms"] = n_at; e["n_heavy"] = np.array([m["n_heavy"] for m in D]); e["atom_offset"] = off
    e["E0"] = E0; e["excitation_energies"] = gaps; e["oscillator_strengths"] = osc
    e["has_excited_forces"] = has_fex; e["md_temperature"] = T
    e["connectivity_changed"] = np.array([bool(m.get("conn_changed", False)) for m in D])
    g = f.create_group("atoms")
    for k, v in (("Z", Zc), ("R", Rc), ("F0", F0c), ("Fex", Fexc)):
        g.create_dataset(k, data=v, compression="gzip", compression_opts=4, shuffle=True)
    sg = f.create_group("splits")
    for name, p in SPLITS.items():
        S = json.load(open(p)); g2 = sg.create_group(name)
        for k, idx in S.items():
            idx = np.asarray([i for i in idx if i < N], np.int64); g2[k] = idx
        json.dump({k: [int(i) for i in v if i < N] for k, v in S.items()}, open(Z_DIR / "splits" / f"{name}.json", "w"))
print("written", h5)

# ---------------------------------------------------------------- extra test set: S1-S3 forces at equilibrium
FC = pickle.load(open(a.fc, "rb"))
with h5py.File(Z_DIR / "fc_test_eq_forces.h5", "w") as f:
    f.attrs.update(dict(note="S1-S3 forces (eV/Angstrom) at equilibrium geometries of test molecules; 'entry' indexes "
                             "entries of leh_wb97xd_v1.h5 (same geometry)", level_of_theory="TD-wB97X-D/6-31G*"))
    keys = sorted(FC)
    f["id"] = np.array(keys, dtype=h5py.string_dtype()); f["entry"] = np.array([FC[k]["entry"] for k in keys])
    for k in ("Fex",):
        if k in FC[keys[0]]:
            f.create_dataset(k, data=np.concatenate([np.asarray(FC[x][k]) for x in keys]), compression="gzip")
            f["atom_offset"] = np.concatenate([[0], np.cumsum([len(FC[x][k]) for x in keys])[:-1]])
    print("fc keys", sorted(FC[keys[0]]))

# ---------------------------------------------------------------- ChemMatData: equilibrium, one row per molecule
role = {}
S = json.load(open(SPLITS["benchmark_A"]))
for k, idx in S.items():
    for i in idx:
        if i < N:
            role[i] = k
with open(C_DIR / f"leh_wb97xd_eq.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "smiles", "smiles_source", "source", "conj_class", "n_heavy", "split"] + [f"S{n}_eV" for n in range(1, 11)]
               + [f"f{n}" for n in range(1, 11)])
    for i, m in enumerate(D):
        if m["kind"] not in ("qcdge_eq", "eq") or not mol_smiles.get(m["mol"]):
            continue
        sp = role.get(i, "unused").replace("_eq", "")
        w.writerow([m["mol"], mol_smiles[m["mol"]], smiles_src.get(m["mol"], ""), m["source"], m.get("conj", "") or "", m["n_heavy"], sp]
                   + [f"{x:.4f}" for x in gaps[i]] + [("" if not np.isfinite(x) else f"{x:.4f}") for x in osc[i]])
print("written", C_DIR / "leh_wb97xd_eq.csv")
json.dump(SMILES_STATS, open(C_DIR / "smiles_stats.json", "w"), indent=1)

# ---------------------------------------------------------------- optional extended XYZ per split
if a.extxyz:
    xd = Z_DIR / "extxyz"; xd.mkdir(exist_ok=True)
    SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I", 15: "P", 14: "Si", 5: "B"}
    for k, idx in S.items():
        with open(xd / f"{k}.xyz", "w") as f:
            for i in idx:
                if i >= N:
                    continue
                m = D[i]; n = len(m["Z"])
                props = " ".join([f'id={m["id"]}', f'kind={m["kind"]}', f"energy={m['E0']:.8f}"]
                                 + [f"S{s+1}_gap={gaps[i, s]:.6f}" for s in range(10)])
                f.write(f"{n}\nProperties=species:S:1:pos:R:3:forces:R:3:forces_S1:R:3:forces_S2:R:3:forces_S3:R:3 {props} pbc=\"F F F\"\n")
                fx = Fexc[off[i]:off[i] + n]
                for j in range(n):
                    f.write(f"{SYM[int(m['Z'][j])]} " + " ".join(f"{x:.8f}" for x in m["R"][j]) + " "
                            + " ".join(f"{x:.8f}" for x in m["F0"][j]) + " "
                            + " ".join(f"{x:.8f}" for s in range(3) for x in fx[j, s]) + "\n")
    print("written", xd)

(Z_DIR / "README.md").write_text((ROOT / "release/README_dataset.md").read_text())  # dataset card, edited there
man = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.json":
        man[str(p.relative_to(OUT))] = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
json.dump(man, open(OUT / "MANIFEST.json", "w"), indent=1)
print("files", len(man), "total MB", round(sum(v["bytes"] for v in man.values()) / 1e6, 1))
