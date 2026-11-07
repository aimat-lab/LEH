"""Convergence criterion of the active-learning loop (user, 2026-10-18: the S1 trajectories must look good and reliable).

Validation trajectories: TD-wB97X-D/6-31G* adiabatic S1 dynamics of molecules that never enter training
(experiments/al/validation_trajectories.json; computed with experiments/md/aimd_reference.py into
experiments/md/reference/<name>). The test trajectories of the paper (acrolein, uracil, trans-stilbene and
experiments/al/test_trajectories_extra.json) are NOT used here; they are evaluated once with the production model.

Per model and validation trajectory (frames before TDDFT breaks down, eval_models.valid_frames):
  gap   MAE of the S1 excitation energy along the trajectory (eV)
  rise  MAE of the S0 energy relative to t = 0 (eV) -- the quantity that failed for twisted geometries
  es1   MAE of the S1 energy relative to t = 0 (eV) -- the surface the dynamics actually runs on
  fs1   MAE of the S1 forces (eV/A)
and a stability test: model S1 NVE from the validation molecules' minima (5 seeds x 1 ps, 300 K): finite energies, no
bond change, total-energy drift.

Criterion (stop when all hold, or after round 4 at the latest):
  1. stability: all model trajectories finite, no bond change, drift < 0.1 eV;
  2. accuracy: mean gap MAE and mean rise MAE over the validation trajectories <= 0.2 eV
     (the typical TD-DFT error vs. high-level references is 0.2-0.3 eV), or
  3. plateau: mean gap and rise MAE improve by < 10 % relative to the previous round (--prev).
Usage: python experiments/al/convergence.py --tag r3 RUN_DIR [RUN_DIR ...] [--prev experiments/al/convergence_r2.json]
Writes experiments/al/convergence_<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/md"))
import eval_models as E  # noqa: E402
from exmlip.md import BatchedModelPES, maxwell_boltzmann, run_nve_batched  # noqa: E402
from exmlip.training.train import build_model_from_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--tag", required=True)
ap.add_argument("--prev", default=None)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--val", default="experiments/al/validation_trajectories.json")
a = ap.parse_args()
torch.set_default_dtype(torch.float64)
VAL = json.load(open(ROOT / a.val))


def traj_metrics(model, name):
    d = ROOT / "experiments/md/reference" / name
    if not (d / "traj.npz").exists():
        return None
    t = np.load(d / "traj.npz")
    n = E.valid_frames(t)
    E0p, gp, Fp = E.eval_geoms(model, t["Z"], t["R"][:n])
    e0r = t["E0"][:n] - t["E0"][0]; e0p = E0p - E0p[0]
    s1r = e0r + t["gaps"][:n, 0]; s1p = e0p + gp[:, 0]
    return dict(n=int(n), t_fs=float(t["t_fs"][n - 1]), gap=float(np.mean(np.abs(gp[:, 0] - t["gaps"][:n, 0]))),
                rise=float(np.mean(np.abs(e0p - e0r))), es1=float(np.mean(np.abs((s1p - s1p[0]) - (s1r - s1r[0])))),
                fs1=float(np.mean(np.abs(Fp - t["Fex"][:n, 0]))))


def stability(model):
    Zs, R0 = [], []
    for v in VAL:  # start from the reference trajectory's first frame (its GFN2-xTB minimum)
        f = ROOT / "experiments/md/reference" / v["name"] / "traj.npz"
        if not f.exists():
            continue
        t = np.load(f)
        Zs += [t["Z"]] * a.seeds; R0 += [t["R"][0]] * a.seeds
    if not Zs:
        return dict(n=0, finite=0, bond_change=0, max_drift=0.0)
    v0 = [maxwell_boltzmann(Z, 300.0, np.random.default_rng(100 + k)) for k, Z in enumerate(Zs)]
    tr = run_nve_batched(BatchedModelPES(model, Zs, state=1), Zs, R0, v0, dt_fs=0.5, n_steps=a.steps, every=10)
    Et = np.array(tr["Epot"]) + np.array(tr["Ekin"])
    fin = np.isfinite(Et).all(0)
    bc = [any(bool((E.bonds(Zs[k], fr[k]) != E.bonds(Zs[k], R0[k])).any()) for fr in tr["R"]) for k in range(len(Zs))]
    drift = np.nanmax(np.abs(Et - Et[0]), axis=0)
    return dict(n=len(Zs), finite=int(fin.sum()), bond_change=int(sum(bc)), max_drift=float(np.nanmax(drift)))


res = {}
for rd in a.runs:
    model, cfg = build_model_from_dir(ROOT / rd)
    E.N_SUP = cfg.get("n_sup")
    tm = {v["name"]: traj_metrics(model, v["name"]) for v in VAL}
    tm = {k: v for k, v in tm.items() if v is not None}
    res[rd] = dict(trajectories=tm, mean={q: float(np.mean([v[q] for v in tm.values()])) for q in ("gap", "rise", "es1", "fs1")}
                   if tm else {}, stability=stability(model))
mean = {q: float(np.mean([r["mean"][q] for r in res.values() if r["mean"]])) for q in ("gap", "rise", "es1", "fs1")}
stab_ok = all(r["stability"]["finite"] == r["stability"]["n"] and r["stability"]["bond_change"] == 0
              and r["stability"]["max_drift"] < 0.1 for r in res.values())
acc_ok = mean["gap"] <= 0.2 and mean["rise"] <= 0.2
plateau = None
if a.prev:
    pm = json.load(open(a.prev))["mean"]
    plateau = all((pm[q] - mean[q]) / pm[q] < 0.10 for q in ("gap", "rise"))
out = dict(tag=a.tag, runs=res, mean=mean, stable=stab_ok, accurate=acc_ok, plateau=plateau,
           converged=bool(stab_ok and (acc_ok or bool(plateau))))
json.dump(out, open(ROOT / f"experiments/al/convergence_{a.tag}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("mean", "stable", "accurate", "plateau", "converged")}, indent=1))
