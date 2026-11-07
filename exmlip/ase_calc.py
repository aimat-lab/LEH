"""ASE calculator for trained excited-state models (one run directory or a committee of several).

The calculator follows one state: "E0" (the ground-state energy head, S0 for multistate models) or a gap column by
name (config key "columns", e.g. "T1" or "S1"; total energy E0 + gap) or by position ("col1", "col2", ...). For
latent (LEH) models the columns are the eigenvalues in ascending order. Energies in eV, forces in eV/A.

With several run directories (seeds), energy and forces are committee means, and results["energy_std"] /
results["forces_std_max"] report the disagreement, which flags geometries outside the training data.
results["states"] holds the committee-mean total energies of E0 and all columns, results["states_std"] their std.

Energy offset: prepare_data.py subtracts a constant from all absolute energies and stores it in <data>.meta.json.
The calculator adds it back if it finds that file (or if energy_offset is given), otherwise energies are relative.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from exmlip.training.train import build_model_from_dir


def _find_meta(run_dir, cfg):
    d = cfg.get("data")
    if not d:
        return None
    cands = [Path(d), Path.cwd() / d] + [p / d for p in Path(run_dir).resolve().parents]
    for c in cands:
        m = Path(str(c).rsplit(".", 1)[0] + ".meta.json")
        if m.exists():
            return json.load(open(m))
    return None


class ExcitedStateCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, run_dirs, state="E0", device=None, energy_offset=None, **kwargs):
        super().__init__(**kwargs)
        run_dirs = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models, cfgs = [], []
        for r in run_dirs:
            m, cfg = build_model_from_dir(r)
            self.models.append(m.to(self.device))
            cfgs.append(cfg)
        self.dtype = torch.get_default_dtype()
        cfg = cfgs[0]
        self.n_sup = cfg["n_sup"]
        self.columns = list(cfg.get("columns") or [f"col{i + 1}" for i in range(self.n_sup)])
        for c in cfgs[1:]:
            assert list(c.get("columns") or self.columns) == self.columns, "committee members have different columns"
        self.has_e0 = all(m.predict_e0 for m in self.models)
        self.state = state
        if state in ("E0", "S0"):
            self.col = None
        elif state in self.columns:
            self.col = self.columns.index(state)
        elif state.startswith("col"):
            self.col = int(state[3:]) - 1
        else:
            raise ValueError(f"state {state!r} not in E0/S0 or columns {self.columns}")
        if not self.has_e0:
            raise ValueError("model without an energy head: forces of total energies are not defined")
        if energy_offset is None:
            meta = _find_meta(run_dirs[0], cfg)
            if meta is None:
                warnings.warn("no <data>.meta.json found: energies are relative to the offset of prepare_data.py")
            energy_offset = (meta or {}).get("energy_offset_eV", 0.0)
        self.energy_offset = float(energy_offset)

    def _one(self, model, Z, R):
        Rt = torch.tensor(R, dtype=self.dtype, device=self.device, requires_grad=True)
        Zt = torch.tensor(Z, dtype=torch.long, device=self.device)
        b = torch.zeros(len(Z), dtype=torch.long, device=self.device)
        out = model(Zt, Rt, b, 1)
        gaps = out["gaps"][0, : self.n_sup]
        E = out["E0"][0] + (gaps[self.col] if self.col is not None else 0.0)
        F = -torch.autograd.grad(E, Rt)[0]
        states = torch.cat([out["E0"][:1], out["E0"][:1] + gaps]).detach().cpu().numpy().astype(float)
        return float(E.detach()), F.detach().cpu().numpy().astype(float), states

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        Z, R = self.atoms.get_atomic_numbers(), self.atoms.get_positions()
        res = [self._one(m, Z, R) for m in self.models]
        E = np.array([r[0] for r in res]) + self.energy_offset
        F = np.stack([r[1] for r in res])
        S = np.stack([r[2] for r in res]) + self.energy_offset
        self.results["energy"] = float(E.mean())
        self.results["forces"] = F.mean(0)
        self.results["energy_std"] = float(E.std())
        self.results["forces_std_max"] = float(np.linalg.norm(F.std(0), axis=-1).max())
        self.results["states"] = dict(zip(["E0"] + self.columns, S.mean(0)))
        self.results["states_std"] = dict(zip(["E0"] + self.columns, S.std(0)))
