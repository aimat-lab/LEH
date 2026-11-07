# LEH-MACE: a local excitation Hamiltonian for excited-state machine-learned potentials

Code, training configurations, and analysis scripts of the paper

> P. Friederich, *Toward universal excited-state machine-learned potentials: a local excitation Hamiltonian for size
> consistency and transferability* (2026), TODO: journal/preprint and DOI.

LEH-MACE predicts the singlet excitation energies of a molecule as the eigenvalues of a latent excitation Hamiltonian
assembled from atom-centred blocks: an onsite block for every atom and a coupling block for every atom pair within the
cutoff, both predicted by a MACE encoder. For noninteracting molecules the Hamiltonian is block diagonal, so the
predicted spectrum is exactly the union of the spectra of the molecules. The ground-state energy comes from a separate
extensive readout; energies and forces of S0 and all excited states follow by automatic differentiation.

- Dataset (TD-ωB97X-D/6-31G*, 303,061 geometries of 106,933 molecules, CC BY 4.0): Zenodo, TODO data DOI
- Trained models (checkpoints of all runs of the paper, all seeds): Zenodo, TODO code DOI

## Installation

Python 3.12, PyTorch 2.11 (CUDA 12.8 or CPU), MACE 0.3.16.

```bash
git clone https://github.com/aimat-lab/LEH.git && cd LEH
python -m venv .venv && source .venv/bin/activate
pip install -r envs/req_main.txt      # exact versions of the paper; or: pip install -e .
pytest tests
```

`envs/req_xmace.txt` and `envs/req_mlatom.txt` are the separate environments of the X-MACE and MS-ANI/OMNI-P2x
baselines (X-MACE with `external/xmace_exmlip_fix.patch`).

## Data

Download the dataset record from Zenodo into a folder, e.g. `zenodo/`, then write the training files of this
repository (all of them are prefixes of the released entry order):

```bash
python -m exmlip.data.load_release zenodo/ --out data [--populations]
```

The splits used by the training configurations are in `experiments/` (e.g. `experiments/al/split_prod_v4.json`); the
same splits of the benchmark, dye-augmented, and production models are also part of the dataset. The QCDGE size
extrapolation uses QCDGE itself (Zhu et al., Sci. Data 2024), read by `exmlip/data/load_qcdge.py`.

## Trained models

Unpack the checkpoint archive from Zenodo in the repository root. Every run directory then contains `config.yaml`,
`summary.json` (from this repository), `best.pt`, and `meta.pt`. The default production models (lowest validation
S1–S3 error, chosen before seeds 2 and 3 were trained) are `experiments/al/runs/prodw_leh_v4_s0` (LEH) and
`experiments/al/runs/prodw_softmin_v4_s3` (Direct-softmin).

```python
from ase.build import molecule
from exmlip.ase_calc import ExcitedStateCalculator

atoms = molecule("C6H6")
seeds = [f"experiments/al/runs/prodw_leh_v4_s{s}" for s in range(4)]
atoms.calc = ExcitedStateCalculator(seeds, state="col1")   # S1 (total energy E0 + lowest eigenvalue)
print(atoms.get_potential_energy(), atoms.get_forces())
print(atoms.calc.results["states"])        # E0 and all excited states, committee mean (eV)
print(atoms.calc.results["energy_std"])    # disagreement between the seeds
```

Energies are absolute TD-ωB97X-D/6-31G* energies in eV, forces in eV/Å. Use the calculator with any ASE dynamics;
`exmlip/md.py` contains the batched adiabatic NVE integrator used for the model-driven dynamics of the paper.

## Training

```bash
python -m exmlip.training.train experiments/al/runs/prodw_leh_v4_s0/config.yaml --set out=runs/my_leh
```

The configuration of every model of the paper is in its run directory:

| directory | models |
|---|---|
| `experiments/wb97/runs_main` | benchmark LEH and Direct-softmin models, learning curves (1k–all training molecules) |
| `experiments/wb97/runs_lc` | baselines of the learning curves: X-MACE (`xmace128_*`), MS-ANI architecture (`*_msani`, `*_vecmsani`), fine-tuned OMNI-P2x (`*_omnift*`, three ensemble members each) |
| `experiments/wb97/runs_abl` | readout ablations (10k training molecules) and the population loss (`abl_leh_pop*`) |
| `experiments/wb97/runs_hp`, `runs_hp3`, `runs_nk`, `runs_xlab` | hyperparameters, latent channels per atom, label ablation |
| `experiments/conj/runs_B` | dye-augmented models |
| `experiments/qcdge/runs_final` | QCDGE size extrapolation |
| `experiments/al/runs` | excited-state sampling models (`sampler_*`) and production models (`prod_*` uniform loss, `prodw_*` weighted loss) |

The X-MACE and MS-ANI/OMNI-P2x baselines were trained with their own codes (`envs/req_xmace.txt`,
`envs/req_mlatom.txt`, `experiments/baselines/`); their checkpoints are `xmace.model`, `model.pt`, and `member*.pt`.
`hpc/` contains the Slurm scripts we used on
HoreKa, bwUniCluster, and a GPU cluster at KIT, with cluster paths and accounts replaced by placeholders.

## Reproducing the paper

Results of the evaluations are stored as JSON next to the scripts, so numbers, tables, and figures can be regenerated
without retraining:

| output | script |
|---|---|
| numbers in the text (`paper/numbers.tex`) | `experiments/make_numbers.py` |
| Tables 1–3 and Supplementary Tables | `experiments/make_tables.py` |
| Fig. 1 | `experiments/fig_schematic.py` |
| Fig. 2 | `experiments/fig_dataset.py` |
| Fig. 3, Supplementary learning curves | `experiments/fig_learning_curves.py` |
| Fig. 4, Supplementary size and ratio figures | `experiments/fig_transfer.py`, `experiments/fig_size_parity.py` |
| Fig. 5, Supplementary fragment figure | `experiments/fig_separability.py`, `experiments/fig_fragments.py` |
| Fig. 6 | `experiments/fig_dynamics.py` |
| Supplementary molecule grids, ablations, channels | `experiments/fig_molecules.py`, `experiments/fig_ablations.py`, `experiments/fig_nk.py` |

Dataset generation (TURBOMOLE 7.5.1, GFN2-xTB sampling): `exmlip/qc/`, `experiments/dataset/`; active learning:
`experiments/al/` (see `experiments/al/README.md`); dynamics and reference trajectories: `experiments/md/`;
TD-DFT excitation populations and the eigenvector comparison: `experiments/eigvec/tddft/`; Zenodo export:
`release/export_dataset.py`.

## Citation

```bibtex
@misc{friederich2026leh,
  title  = {Toward universal excited-state machine-learned potentials: a local excitation Hamiltonian for size
            consistency and transferability},
  author = {Friederich, Pascal},
  year   = {2026},
  note   = {TODO}
}
```

## License

MIT (code). The dataset is licensed separately under CC BY 4.0.
