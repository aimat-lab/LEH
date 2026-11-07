# TD-ωB97X-D excited-state energies and forces of 106,933 molecules at equilibrium, thermal, and excited-state geometries (v4)

Dataset of the paper *Toward universal excited-state machine-learned potentials: a local excitation Hamiltonian for
size consistency and transferability* by Pascal Friederich (ORCID 0000-0003-4465-1465), Karlsruhe Institute of
Technology. Paper: TODO (journal, DOI; preprint arXiv TODO).

License: CC BY 4.0. Code and trained models: https://github.com/aimat-lab/LEH (MIT), archived on Zenodo
(TODO code DOI).

## Contents

| File | Content |
|---|---|
| `leh_wb97xd_v4.h5` | all 303,061 geometries with labels; splits of the benchmark, dye-augmented, and production models as index arrays |
| `split_{benchmark,dye_augmented,production}.json` | the same splits as JSON (indices into the entries) |
| `fc_test_eq_forces.h5` | extra test set: S1–S3 forces at the equilibrium geometries of 2,499 test molecules |
| `molecule_pairs.h5` | separability test: pairs of test molecules at 2.5–20 Å, TD-DFT dimer spectra of 180 pair geometries |
| `reference_trajectories.h5` | 12 TD-ωB97X-D S1 trajectories (7 validation, 5 test molecules) with initial conditions |
| `tddft_populations.h5` | per-atom hole and electron populations and charge-transfer numbers of the ten roots for 25,808 geometries |
| `raw_turbomole_<group>.tar.gz` | TURBOMOLE input and output files of 197,946 dataset geometries and the 2,499 extra-test geometries, without MO files |
| `MANIFEST.json` | sha256 checksums of all files |

## Level of theory

TD-DFT ωB97X-D/6-31G* (the level of QCDGE), full linear response (RPA), TURBOMOLE 7.5.1, RI-J, grid m4, SCF
convergence 1e-7 Eh, response convergence 1e-6. For every geometry: S0 energy and forces, the ten lowest singlet
excitation energies with oscillator strengths, and analytic S1, S2, and S3 forces (separate `egrad` runs). Force labels
of a state are masked (NaN) if its root deviates by more than 2e-4 Eh from the ten-root calculation or if the
gradient-file energy is inconsistent. At QCDGE minima (`kind == "qcdge_eq"`) the excitation energies and oscillator
strengths are those of QCDGE (reproduced to 0.005 eV by our setup); E0 and F0 were recomputed, and these entries have
no excited-state forces.

## Geometries (`kind`)

| kind | description |
|---|---|
| `qcdge_eq` | QCDGE minimum geometry |
| `offeq` | thermal geometry from GFN2-xTB NVT dynamics at 300–1000 K (`md_temperature`), one frame between 0.5 and 1.5 ps |
| `excmd` | geometry from model-driven S1/S2 dynamics (excited-state sampling); `al_round` 1–4 |
| `eq` | GFN2-xTB minimum of ZINC, dye, and ladder molecules |

`source`: QCDGE subset (`A9`, `A10`, `B9`, `B10`), `ZINC` (larger test molecules, 11–20 heavy atoms), `DYE`, `LAD`.
Dyes come from two experimental chromophore databases (`dye_database`: `beard` = Beard et al., Sci. Data 6, 307
(2019); `joung` = Joung et al., Sci. Data 7, 295 (2020)); ladder molecules are homologous series (`ladder_family`).

`al_round`: 0 = not from excited-state sampling, 1 = first sampling round (part of the benchmark data), 2–4 = the three
active-learning rounds for the production models. Entries 0–234,941 are the v1 benchmark data in their original order;
entries 234,942–303,060 are the 68,119 active-learning geometries.

## HDF5 layout of `leh_wb97xd_v4.h5`

```
entries/                      one row per geometry (N = 303,061)
  id, molecule, source, kind, conj_class, dye_database, ladder_family, smiles, smiles_source, raw_file  (strings)
  n_atoms, n_heavy, atom_offset (int)       atoms of entry i: atoms/*[atom_offset[i] : atom_offset[i] + n_atoms[i]]
  E0 (eV)                                   S0 total energy
  excitation_energies (N, 10) eV            S1..S10 vertical excitation energies
  oscillator_strengths (N, 10)              NaN where not available
  has_excited_forces (N, 3) bool            S1, S2, S3 forces available
  md_temperature (K), al_round (int), connectivity_changed (bool)
atoms/
  Z (A,), R (A, 3) Angstrom, F0 (A, 3) eV/Angstrom, Fex (A, 3, 3) eV/Angstrom   Fex[:, n] = forces of S_{n+1}
splits/<model>/<set>          int indices into entries; <model> = benchmark, dye_augmented, production
```

Total energy of state n: `E0 + excitation_energies[:, n-1]`. Molecules marked `connectivity_changed` are excluded
from all splits.

```python
import h5py, numpy as np
f = h5py.File("leh_wb97xd_v4.h5")
e, a = f["entries"], f["atoms"]
i = f["splits/production/test_random_offeq"][0]
o, n = e["atom_offset"][i], e["n_atoms"][i]
Z, R, F_S1 = a["Z"][o:o+n], a["R"][o:o+n], a["Fex"][o:o+n, 0]
gaps = e["excitation_energies"][i]
```

## Splits

By molecule (all geometries of a molecule in the same set): training 85 %, validation 5 %, random test 10 % by a hash
of the molecule identifier; an additional Bemis–Murcko scaffold hold-out (`test_scaffold_*`). Test sets:
`test_random_{eq,offeq,exc}`, `test_scaffold_{eq,offeq,exc}`, `test_large_{eq,offeq,exc}` (ZINC, 11–20 heavy atoms),
`test_dye_noS_*` (held-out dye scaffolds without sulfur), `test_lad_noS_*` (ladders without sulfur), and their
sulfur-containing counterparts `*_S_*`.

| split | training data | additional test sets |
|---|---|---|
| `benchmark` | v1 training molecules (QCDGE-derived, up to 10 heavy atoms) | `test_dyeall_noS_*` (all sulfur-free dyes) |
| `dye_augmented` | + 80 % of the dye scaffolds | `test_dye_S_*`, `test_lad_S_*` |
| `production` | + all active-learning geometries of training molecules | `test_{random,scaffold,large}_exc{2,3,4}` (active-learning geometries of test molecules by round) |

Molecules of the reference trajectories never enter training or validation of the production split.

## Raw TURBOMOLE files

`raw_turbomole_<group>.tar.gz` contains one folder per geometry, named by the entry `id` (for `fc_test_minima`, the
`id` in `fc_test_eq_forces.h5`); `entries/raw_file` = `<group>/<id>` (empty if no files exist). Files: `sp/` with
`control`, `coord`, `basis`, `auxbasis`, `ridft.out` (SCF with Mulliken populations), `rdgrad.out`, `egrad1.out`–
`egrad3.out` (ten roots with oscillator strengths, transition dipoles, dominant orbital contributions, and the S1–S3
gradients); for thermal geometries also `md/` with the GFN2-xTB MD input and trajectory, and for dyes and ladders `opt/`
with the GFN2-xTB optimization. The S0 recalculations at QCDGE minima (`qcdge_minima`) contain `control`, `coord`,
`basis`, `auxbasis`, `ridft.out`, `ridft_pop.out` (Mulliken populations), and `rdgrad.out`. MO files (`mos`) are not
included; they can be regenerated from `control`, `coord`, and `basis` with `ridft`.

| group | geometries |
|---|---|
| `qcdge_minima` | S0 recalculations at QCDGE minima |
| `thermal` | thermal geometries of QCDGE-derived molecules |
| `conjugated_minima`, `conjugated_thermal` | dyes and ladders |
| `excited_round1` … `excited_round4` | excited-state-sampled geometries by active-learning round |
| `fc_test_minima` | S1–S3 gradients at equilibrium geometries of test molecules |

Raw files were archived for all calculations from 2026-10-10 on: all excited-state-sampled, dye, ladder, and extra-test
geometries, 73,963 of the 103,322 thermal geometries of QCDGE-derived molecules, and 25,263 of the 100,021 S0 recalculations at QCDGE minima. The
earlier calculations and the ZINC geometries (998) have no raw files.

## `fc_test_eq_forces.h5`

`id`, `entry` (index into `leh_wb97xd_v4.h5`, same geometry), `Fex` (atoms × 3 states × 3, eV/Å, ragged with
`atom_offset`), `raw_file`.

## `molecule_pairs.h5`

`monomers/`: 120 equilibrium test molecules (`entry` into `leh_wb97xd_v4.h5`, `excitation_energies`, ragged `Z`, `R`).
`pairs/`: 1,680 geometries of 60 pairs of different (`pair_kind` AB) and 60 pairs of identical molecules (AA) at
minimum interatomic distances `d` = 2.5–20 Å, molecule B randomly rotated and shifted; `monomers` (indices into
`monomers/`), ragged `Z`, `R`. For 10 pairs of each kind at 2.5–15 Å: `tddft_dimer` (lowest six roots of a 12-root
TD-ωB97X-D calculation of the pair) and `tddft_union` (lowest six roots of the union of the two monomer spectra), NaN
elsewhere.

## `reference_trajectories.h5`

One group per molecule (attributes `role` = validation or test, `smiles`, `dt_fs`, `T_K`, `seed`, `steps_requested`,
`n_frames`, `n_valid`): `Z`, `R` (frames × atoms × 3, Å), `t_fs`, `Epot` (S1 total energy, eV), `Ekin`, `E0`, `F0`,
`excitation_energies` (six roots), `F_S1` (eV/Å), initial conditions `R0`, `v0` (Å/fs), and `valid` (frames before
TD-DFT breaks down near an S1/S0 intersection: the first frame with S1 < 0.5 eV or a total-energy jump > 0.1 eV ends
the usable part). Adiabatic S1 NVE, velocity Verlet, 0.5 fs, 1000 steps (acrolein 2000), Maxwell–Boltzmann velocities at 300 K at the
GFN2-xTB minimum. The azobenzene and hexatriene trajectories end at steps 785 and 568, where TURBOMOLE failed, after
their last valid frames (434 and 230). Validation: styrene, azobenzene, cyclohexenone, hexatriene, cytosine, benzaldehyde,
cinnamaldehyde. Test: acrolein, uracil, *trans*-stilbene, HBDI, DMABN.

## `tddft_populations.h5`

From an `escf` restart at the dataset settings (ten roots), the one-particle transition density in the Löwdin basis,
T = S^1/2 C_occ (X+Y) C_virᵀ S^1/2 with Σ(X+Y)² = 1, gives the charge-transfer numbers Ω_AB = Σ_{μ∈A, ν∈B} T²_μν
(Plasser and Lischka, J. Chem. Theory Comput. 8, 2777 (2012)), `hole` = Σ_B Ω_AB, `electron` = Σ_A Ω_AB, `pop` =
(hole + electron)/2, `pop_X` (the same from the amplitudes X), and transition charges `qtr` = Σ_{μ∈A} T_μμ. Per geometry:
`id`, `set` (`training_subset`: 25,348 geometries of the 10k-molecule training subset of the population-loss
experiment; `comparison_set`: 460 equilibrium, thermal, ladder, and dye geometries), `entry` (index into
`leh_wb97xd_v4.h5`), `exc_eV`, `norm_xy`; per atom (ragged with `atom_offset`, columns = roots): `Z`, `hole`,
`electron`, `pop`, `pop_X`, `qtr`; `omega` (float32) flat per geometry, `omega[omega_offset[i] : omega_offset[i] + 10 n²].reshape(10, n, n)`.

## SMILES

`entries/smiles_source`: `input` (dyes, ladders, ZINC: SMILES of the source database), `qcdge_hdf5` / `qcdge_csv`
(QCDGE's `SMILES_RDKIT_CAN`, used when it is closed-shell), `perceived_from_geometry` (RDKit `rdDetermineBonds` on the
QCDGE minimum, neutral, closed-shell only; used where QCDGE's SMILES is missing or contains radical atoms although the
molecule was computed closed-shell, about 6 % of QCDGE molecules), empty where neither is available.

## Sources and licenses

QCDGE (Zhu et al., Sci. Data 2024; figshare collection 10.6084/m9.figshare.c.7259125, CC0),
the dye databases above (Joung et al., figshare 12045567, CC BY 4.0; Beard et al., figshare 7619672, CC BY 4.0) and
the ZINC250k subset distributed with chemical_vae (Apache-2.0). All
permit redistribution with attribution, which this card provides.

## Citation

P. Friederich, Toward universal excited-state machine-learned potentials: a local excitation Hamiltonian for size
consistency and transferability, TODO (journal/preprint, DOI).
