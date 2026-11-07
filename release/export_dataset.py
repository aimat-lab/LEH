"""Export the dataset of the paper (v4: v1 benchmark entries in their original order + all active-learning geometries)
for Zenodo. The v1 export script is kept as release/export_dataset_v1.py (output release/out_v1*).

Outputs (default release/out_v4/):
  zenodo/leh_wb97xd_v4.h5             all 303,061 geometries with labels (ragged atoms stored flat with offsets), splits
                                      of the benchmark, dye-augmented and production models as index arrays
  zenodo/splits/*.json                the same splits as JSON (indices into the entry order of the HDF5 file)
  zenodo/fc_test_eq_forces.h5         extra test set: S1-S3 forces at equilibrium geometries of 2,499 test molecules
  zenodo/molecule_pairs.h5            separability test: 120 monomers, 1,680 pair geometries, TD-DFT spectra of 180 pairs
  zenodo/reference_trajectories.h5    the 12 TD-wB97X-D S1 NVE trajectories (7 validation, 5 test) with initial
                                      conditions and the validity flag
  zenodo/tddft_populations.h5         Loewdin hole/electron populations and charge-transfer numbers of the 10 roots for
                                      the population-loss training subset and the 460-geometry comparison set
  zenodo/README.md                    dataset card (copied from release/README_dataset.md)
  MANIFEST.json                       sha256 and bytes of every exported file
  aux/raw_map.tsv                     archive path on bwUni -> raw group, folder name (input of release/repack_raw.py;
                                      not uploaded)
Optional: --extxyz writes one extended-XYZ file per production split (MACE/ASE-compatible keys) into zenodo/extxyz/.

SMILES (dataset smiles_source): stored for dyes, ladders and ZINC molecules ("input"); for QCDGE-derived molecules
QCDGE's SMILES_RDKIT_CAN from the HDF5 attributes (--qcdge_h5_dir, "qcdge_hdf5") or final_all.csv (--qcdge_csv,
"qcdge_csv") if it is closed-shell, otherwise perceived from the QCDGE minimum with RDKit (rdDetermineBonds, charge 0,
closed-shell only; "perceived_from_geometry"); empty if neither works.
Raw TURBOMOLE files exist for the calculations made after archiving started (2026-10-10): raw_file = "<group>/<entry id>"
is the folder of the entry in raw_turbomole_<group>.tar.gz, empty where no files were archived.
Run locally or in a job (not on a login node):
  python -I release/export_dataset.py --qcdge_h5_dir raw_data/qcdge --qcdge_csv external_data/qcdge_final_all/final_all.csv
"""
import argparse
import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data/wb97_prod_v4.pkl"))
ap.add_argument("--fc", default=str(ROOT / "data/wb97/fc_test_labels.pkl"))
ap.add_argument("--out", default=str(ROOT / "release/out_v4"))
ap.add_argument("--qcdge_csv", default=None, help="QCDGE final_all.csv (SMILES; incomplete)")
ap.add_argument("--qcdge_h5_dir", default=None, help="directory with QCDGE A_9/A_10/B_9/B_10.hdf5 (SMILES attributes)")
ap.add_argument("--raw_index", default=str(ROOT / "experiments/eigvec/tddft/raw_index.txt"),
                help="listing of the raw archives on bwUni (paths relative to the workspace data/ directory)")
ap.add_argument("--extxyz", action="store_true")
a = ap.parse_args()

VERSION = "v4"
N_BASE = 234942  # entries of the frozen v1 benchmark dataset (same order); later entries are active-learning geometries
SPLITS = {"benchmark": ROOT / "experiments/conj/split_A_conj.json",     # benchmark models (training on small molecules)
          "dye_augmented": ROOT / "experiments/conj/split_B.json",      # + training dye scaffolds
          "production": ROOT / "experiments/al/split_prod_v4.json"}     # + all active-learning geometries
TRAJ_ROLE = {"acrolein_s1": "test", "uracil_s1": "test", "stilbene_s1": "test", "hbdi_s1": "test", "dmabn_s1": "test",
             "styrene_s1": "validation", "azobenzene_s1": "validation", "cyclohexenone_s1": "validation",
             "hexatriene_s1": "validation", "cytosine_s1": "validation", "benzaldehyde_s1": "validation",
             "cinnamaldehyde_s1": "validation"}
# raw archive directories on bwUni -> public group names
RAW_GROUP = {"wb97_eq": "qcdge_minima", "t1": "thermal", "t2": "thermal", "dyes_eq": "conjugated_minima",
             "ladders_eq": "conjugated_minima", "dyes_md": "conjugated_thermal", "ladders_md": "conjugated_thermal",
             "s1r1": "excited_round1", "s1r2": "excited_round2", "s1r3": "excited_round3", "s1r4": "excited_round4",
             "fc_test": "fc_test_minima"}
RAW_DIRS = {"qcdge_eq": ["wb97_eq"], "offeq": ["t1", "t2", "dyes_md", "ladders_md"], "eq": ["dyes_eq", "ladders_eq"],
            "excmd": ["s1r1", "s1r2", "s1r3", "s1r4"]}
OUT = Path(a.out); Z_DIR = OUT / "zenodo"; AUX = OUT / "aux"
for d in (Z_DIR / "splits", AUX):
    d.mkdir(parents=True, exist_ok=True)
H5 = dict(compression="gzip", compression_opts=4, shuffle=True)
STR = h5py.string_dtype()

D = pickle.load(open(a.data, "rb"))
N = len(D)
print("entries", N)

# ---------------------------------------------------------------- SMILES
# QCDGE SMILES (SMILES_RDKIT_CAN): from the HDF5 group attributes (complete) or final_all.csv (lacks ~11 % of our
# molecules). QCDGE writes '1' where its RDKit conversion failed. Our molecule id '<subset>_<key>' maps to the CSV Index
# '<Aa|Ab|Ba|Bb><number>' and to HDF5 group '<key>'.
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors, rdDetermineBonds  # noqa: E402
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
SMILES_STATS = dict(n_molecules=len(mol_smiles), sources=dict(Counter(smiles_src.get(k, "") for k in mol_smiles)),
                    perceived_vs_qcdge_agree=agree[0], perceived_vs_qcdge_n=agree[1])

# ---------------------------------------------------------------- raw archives -> entries
by_dir = defaultdict(dict)  # archive dir -> {archive name: path relative to the workspace data/ directory}
for line in open(a.raw_index):
    p = line.strip()
    if p:
        by_dir[p.split("/")[-3]][p.rsplit("/", 1)[1][: -len(".tar.gz")]] = p
raw_file, raw_rows = [""] * N, []
for i, m in enumerate(D):
    base = re.sub(r"_(md|eq)$", "", m["id"])
    for d in RAW_DIRS[m["kind"]]:
        p = by_dir[d].get(m["id"]) or by_dir[d].get(base)
        if p:
            raw_file[i] = f"{RAW_GROUP[d]}/{m['id']}"; raw_rows.append((p, RAW_GROUP[d], m["id"])); break
print("raw archives for", sum(bool(r) for r in raw_file), "of", N, "entries")

# ---------------------------------------------------------------- main HDF5
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
# active-learning round of excited-state-sampled geometries: 1 = first round (part of the v1 benchmark data), 2-4 later
al_round = np.array([m.get("al_round", 1 if m["kind"] == "excmd" else 0) for m in D], np.int8)
assert set(al_round[N_BASE:]) <= {2, 3, 4} and set(al_round[:N_BASE]) <= {0, 1}
strs = lambda key: np.array([str(m.get(key, "") or "") for m in D], dtype=STR)

h5 = Z_DIR / f"leh_wb97xd_{VERSION}.h5"
with h5py.File(h5, "w") as f:
    f.attrs.update(dict(
        title="TD-wB97X-D/6-31G* ground- and excited-state energies and forces of 106,933 molecules at equilibrium, "
              "thermal, and excited-state geometries",
        version=VERSION, level_of_theory="TD-DFT wB97X-D/6-31G*, full linear response (RPA), TURBOMOLE 7.5.1",
        units="energies eV; excitation energies eV; forces eV/Angstrom; coordinates Angstrom; temperature K",
        n_entries=N, n_atoms_total=A, n_benchmark_entries=N_BASE,
        note="Fex[:, n] are forces of S_{n+1} (n=0,1,2); NaN = not available (masked or not computed). osc NaN = not "
             "available. At QCDGE minima (kind qcdge_eq) excitation energies and oscillator strengths are taken from "
             "QCDGE; E0/F0 recomputed. Entries 0..n_benchmark_entries-1 are the v1 benchmark data in their original order."))
    e = f.create_group("entries")
    e["id"] = strs("id"); e["molecule"] = strs("mol"); e["source"] = strs("source"); e["kind"] = strs("kind")
    e["conj_class"] = strs("conj"); e["dye_database"] = strs("db"); e["ladder_family"] = strs("family")
    e["smiles"] = np.array([mol_smiles.get(m["mol"], "") for m in D], dtype=STR)
    e["smiles_source"] = np.array([smiles_src.get(m["mol"], "") for m in D], dtype=STR)
    e["n_atoms"] = n_at; e["n_heavy"] = np.array([m["n_heavy"] for m in D]); e["atom_offset"] = off
    e["E0"] = E0; e["excitation_energies"] = gaps; e["oscillator_strengths"] = osc
    e["has_excited_forces"] = has_fex; e["md_temperature"] = T; e["al_round"] = al_round
    e["connectivity_changed"] = np.array([bool(m.get("conn_changed", False)) for m in D])
    e["raw_file"] = np.array(raw_file, dtype=STR)
    g = f.create_group("atoms")
    for k, v in (("Z", Zc), ("R", Rc), ("F0", F0c), ("Fex", Fexc)):
        g.create_dataset(k, data=v, **H5)
    sg = f.create_group("splits")
    for name, p in SPLITS.items():
        S = json.load(open(p)); g2 = sg.create_group(name)
        for k, idx in S.items():
            idx = np.asarray(idx, np.int64); assert idx.max() < N; g2[k] = idx
        json.dump({k: [int(i) for i in v] for k, v in S.items()}, open(Z_DIR / "splits" / f"{name}.json", "w"))
print("written", h5)

# ---------------------------------------------------------------- extra test set: S1-S3 forces at equilibrium
FC = pickle.load(open(a.fc, "rb"))
fc_raw = by_dir["fc_test"]
with h5py.File(Z_DIR / "fc_test_eq_forces.h5", "w") as f:
    f.attrs.update(dict(note=f"S1-S3 forces (eV/Angstrom) at equilibrium geometries of test molecules; 'entry' indexes "
                             f"entries of leh_wb97xd_{VERSION}.h5 (same geometry); Fex[:, n] = forces of S_(n+1)",
                        level_of_theory="TD-wB97X-D/6-31G*"))
    keys = sorted(FC)
    for k in keys:
        assert D[FC[k]["entry"]]["id"] == k
    f["id"] = np.array(keys, dtype=STR); f["entry"] = np.array([FC[k]["entry"] for k in keys])
    f.create_dataset("Fex", data=np.concatenate([np.asarray(FC[x]["Fex"]) for x in keys]), **H5)
    f["atom_offset"] = np.concatenate([[0], np.cumsum([len(FC[x]["Fex"]) for x in keys])[:-1]])
    rf = []
    for k in keys:
        p = fc_raw.get(f"{k}_fc")
        rf.append(f"fc_test_minima/{k}" if p else "")
        if p:
            raw_rows.append((p, "fc_test_minima", k))
    f["raw_file"] = np.array(rf, dtype=STR)
print("written fc_test_eq_forces.h5:", len(keys), "geometries,", sum(bool(r) for r in rf), "with raw files")

# ---------------------------------------------------------------- separability test: molecule pairs
P = pickle.load(open(ROOT / "data/frag_pairs.pkl", "rb"))
RP = {r["id"]: r for r in json.load(open(ROOT / "experiments/frag_tddft/results_pairs.json"))}
mono = [x for x in P if x["kind"] == "frag_mono"]; dim = [x for x in P if x["kind"] == "frag_dimer"]
mono_pos = {j: k for k, j in enumerate(i for i, x in enumerate(P) if x["kind"] == "frag_mono")}


def ragged(group, items, keys):
    n = np.array([len(x["Z"]) for x in items]); group["n_atoms"] = n
    group["atom_offset"] = np.concatenate([[0], np.cumsum(n)[:-1]])
    group.create_dataset("Z", data=np.concatenate([x["Z"] for x in items]).astype(np.int8), **H5)
    for k in keys:
        group.create_dataset(k, data=np.concatenate([np.asarray(x[k], float) for x in items]), **H5)


with h5py.File(Z_DIR / "molecule_pairs.h5", "w") as f:
    f.attrs.update(dict(note="Pairs of different (AB) and identical (AA) equilibrium test molecules at minimum interatomic "
                             "distances d (Angstrom), molecule B randomly rotated and shifted. TD-DFT dimer spectra "
                             "(TD-wB97X-D/6-31G*, 12 roots, dataset protocol) for 10 pairs of each kind at 2.5-15 Angstrom: "
                             "tddft_dimer = lowest 6 dimer roots, tddft_union = lowest 6 roots of the union of the two "
                             "monomer spectra (6 roots each), NaN where not computed. Energies eV, coordinates Angstrom."))
    gm = f.create_group("monomers")
    gm["id"] = np.array([x["id"] for x in mono], dtype=STR); gm["molecule"] = np.array([x["mol"] for x in mono], dtype=STR)
    gm["entry"] = np.array([x["src_index"] for x in mono]); gm["excitation_energies"] = np.array([x["gaps"] for x in mono])
    ragged(gm, mono, ["R"])
    gd = f.create_group("pairs")
    gd["id"] = np.array([x["id"] for x in dim], dtype=STR); gd["pair"] = np.array([x["pair"] for x in dim])
    gd["pair_kind"] = np.array([x["pair_kind"] for x in dim], dtype=STR); gd["d"] = np.array([x["d"] for x in dim])
    gd["monomers"] = np.array([[mono_pos[j] for j in x["mono"]] for x in dim])  # indices into monomers/
    ragged(gd, dim, ["R"])
    td = np.full((len(dim), 6), np.nan); un = np.full((len(dim), 6), np.nan)
    for k, x in enumerate(dim):
        if x["id"] in RP:
            td[k] = RP[x["id"]]["dimer"][:6]; un[k] = RP[x["id"]]["union"][:6]
    gd["tddft_dimer"] = td; gd["tddft_union"] = un
print("written molecule_pairs.h5:", len(mono), "monomers,", len(dim), "pairs,", int(np.isfinite(td[:, 0]).sum()), "with TD-DFT")

# ---------------------------------------------------------------- reference S1 trajectories


def valid_frames(t, s1_min=0.5, max_jump=0.1):  # as in experiments/md/eval_models.py
    """The first frame with S1 < s1_min eV or a total-energy jump > max_jump eV between consecutive steps ends the
    usable reference trajectory (TD-DFT breaks down near an S1/S0 intersection)."""
    et = t["Epot"] + t["Ekin"]
    bad = np.where((t["gaps"][:, 0] < s1_min) | np.r_[False, np.abs(np.diff(et)) > max_jump])[0]
    return int(bad[0]) if len(bad) else len(et)


with h5py.File(Z_DIR / "reference_trajectories.h5", "w") as f:
    f.attrs.update(dict(note="Adiabatic S1 NVE trajectories with TD-wB97X-D/6-31G* (RPA, 6 roots, TURBOMOLE 7.5.1), "
                             "velocity Verlet, dt 0.5 fs, 1000 steps (acrolein 2000; azobenzene and hexatriene end where TURBOMOLE "
                             "failed, after the last valid frame), initial velocities Maxwell-Boltzmann at 300 K (seed 0) at "
                             "the GFN2-xTB minimum. Units: Angstrom, fs, eV, eV/Angstrom, Angstrom/fs. valid: frames before "
                             "TD-DFT breaks down near an S1/S0 intersection (first frame with S1 < 0.5 eV or a total-energy "
                             "jump > 0.1 eV ends the usable part). Role: validation (model selection) or test."))
    for name, role in sorted(TRAJ_ROLE.items()):
        d = ROOT / "experiments/md/reference" / name
        t = np.load(d / "traj.npz"); c = pickle.load(open(d / "ckpt.pkl", "rb"))
        n = valid_frames(t); g = f.create_group(name.replace("_s1", ""))
        g.attrs.update(dict(role=role, smiles=c["args"]["smiles"], state=1, dt_fs=c["args"]["dt"], T_K=c["args"]["T"],
                            seed=c["args"]["seed"], steps_requested=c["args"]["steps"], n_frames=len(t["t_fs"]),
                            n_valid=n))
        for k in ("Z", "t_fs", "Epot", "Ekin", "E0"):
            g[k] = t[k]
        g.create_dataset("R", data=t["R"], **H5); g.create_dataset("F0", data=t["F0"], **H5)
        g["excitation_energies"] = t["gaps"]; g.create_dataset("F_S1", data=t["Fex"][:, 0], **H5)
        g["R0"] = c["R0"]; g["v0"] = c["v0"]; g["valid"] = np.arange(len(t["t_fs"])) < n
print("written reference_trajectories.h5:", len(TRAJ_ROLE), "trajectories")

# ---------------------------------------------------------------- TD-DFT excitation populations
ix = {m["id"]: i for i, m in enumerate(D)}
POP_SETS = [("training_subset", ROOT / "data/eigvec_tddft_abl10k/pop"), ("comparison_set", ROOT / "experiments/eigvec/tddft/pop")]
rows = []
for set_name, d in POP_SETS:
    for p in sorted(d.glob("*.npz")):
        rows.append((set_name, p.stem, p))
with h5py.File(Z_DIR / "tddft_populations.h5", "w") as f:
    f.attrs.update(dict(note="Per-atom analysis of the 10 TD-wB97X-D roots (escf restart, RPA, same settings as the dataset) "
                             "from the 1-particle transition density in the Loewdin basis, T = S^1/2 C_occ (X+Y) C_vir^T S^1/2 "
                             "with sum (X+Y)^2 = 1: charge-transfer numbers omega[n, A, B] = sum_{mu in A, nu in B} T^2 "
                             "(Plasser and Lischka 2012), hole_A = sum_B omega_AB, electron_B = sum_A omega_AB, population "
                             "p_A = (hole_A + electron_A)/2, pop_X = p_A from the amplitudes X, transition charges q_A = "
                             "sum_{mu in A} T_mu,mu. entry indexes leh_wb97xd_v4.h5. training_subset: geometries of the "
                             "10k-molecule training subset of the population-loss experiment; comparison_set: 460 "
                             "equilibrium, thermal, ladder and dye geometries. exc_eV: escf excitation energies."))
    ent, sets, ids, nat, exc, nxy, Zs, cols, oms = [], [], [], [], [], [], [], defaultdict(list), []
    dev = 0.0
    for set_name, name, p in rows:
        z = np.load(p); i = ix[name]
        assert np.array_equal(z["Z"], D[i]["Z"]); dev = max(dev, float(np.abs(z["R"] - D[i]["R"]).max()))
        ent.append(i); sets.append(set_name); ids.append(name); nat.append(len(z["Z"])); exc.append(z["exc_eV"])
        nxy.append(z["norm_xy"]); Zs.append(z["Z"]); oms.append(z["omega"].reshape(-1))
        for k in ("hole", "elec", "pop", "pop_X", "qtr"):
            cols[k].append(z[k].T)  # (n_atoms, 10)
    print("populations:", len(ent), "geometries, max |R - dataset R| =", dev)
    f["id"] = np.array(ids, dtype=STR); f["set"] = np.array(sets, dtype=STR); f["entry"] = np.array(ent)
    f["n_atoms"] = np.array(nat); f["atom_offset"] = np.concatenate([[0], np.cumsum(nat)[:-1]])
    om_n = np.array([len(o) for o in oms]); f["omega_offset"] = np.concatenate([[0], np.cumsum(om_n)[:-1]])
    f["exc_eV"] = np.array(exc); f["norm_xy"] = np.array(nxy)
    f.create_dataset("Z", data=np.concatenate(Zs).astype(np.int8), **H5)
    for k, v in cols.items():
        f.create_dataset(k if k != "elec" else "electron", data=np.concatenate(v), **H5)
    f.create_dataset("omega", data=np.concatenate(oms).astype(np.float32), **H5)  # entry i: omega[off:off+10*n*n].reshape(10, n, n)
print("written tddft_populations.h5")

# ---------------------------------------------------------------- optional extended XYZ per production split
if a.extxyz:
    xd = Z_DIR / "extxyz"; xd.mkdir(exist_ok=True)
    SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S"}
    for k, idx in json.load(open(SPLITS["production"])).items():
        with open(xd / f"{k}.xyz", "w") as f:
            for i in idx:
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

# ---------------------------------------------------------------- raw map, README, manifest
with open(AUX / "raw_map.tsv", "w") as f:
    for p, g, name in raw_rows:
        f.write(f"{p}\t{g}\t{name}\n")
json.dump(dict(SMILES_STATS, raw_per_group=dict(Counter(g for _, g, _ in raw_rows))), open(AUX / "stats.json", "w"), indent=1)
print("raw map:", len(raw_rows), "archives", dict(Counter(g for _, g, _ in raw_rows)))
(Z_DIR / "README.md").write_text((ROOT / "release/README_dataset.md").read_text())  # dataset card, edited there
man = {}
for p in sorted(Z_DIR.rglob("*")):
    if p.is_file():
        man[str(p.relative_to(Z_DIR))] = dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
json.dump(man, open(OUT / "MANIFEST.json", "w"), indent=1)
print("files", len(man), "total MB", round(sum(v["bytes"] for v in man.values()) / 1e6, 1))
