"""Reference adiabatic S_n NVE trajectory with TURBOMOLE (TD-wB97X-D/6-31G*, RPA) for one molecule.

Initial conditions: minimum geometry (task geometry, or GFN2-xTB minimum for SMILES input) and Maxwell-Boltzmann
velocities (seeded). Velocity Verlet, labels (E0, F0, roots, S_n force) stored for every frame.
Restartable: the state is checkpointed after every step; rerunning the same command continues the trajectory.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.md import AMU, CONV, TurbomolePES, kinetic, maxwell_boltzmann  # noqa
from exmlip.qc import sampling, turbomole as tm  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--smiles")
ap.add_argument("--task_pkl")
ap.add_argument("--task_id")
ap.add_argument("--state", type=int, default=1)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--dt", type=float, default=0.5, help="fs")
ap.add_argument("--T", type=float, default=300.0)
ap.add_argument("--nroots", type=int, default=6)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_hours", type=float, default=1e9)
a = ap.parse_args()
t_start = time.time()
out = ROOT / "experiments/md/reference" / a.name
out.mkdir(parents=True, exist_ok=True)
scr = Path(os.environ.get("TM_SCRATCH", "/tmp/aimd")) / a.name
ck = out / "ckpt.pkl"


def lab(pes):
    return {k: v for k, v in pes.records[-1].items() if k != "R"}


def save(st):
    tmp = out / "ckpt.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(st, f)
    os.replace(tmp, ck)


if ck.exists():
    st = pickle.load(open(ck, "rb"))
    print(f"restart at step {st['step']}", flush=True)
else:
    if a.task_pkl:
        t = [x for x in pickle.load(open(a.task_pkl, "rb")) if x["id"] == a.task_id][0]
        Z, R0 = t["Z"], t["R_eq"]
    else:
        Z, R = sampling.embed(a.smiles)
        Z, R0 = sampling.xtb_opt(scr / "opt", Z, R)
    v0 = maxwell_boltzmann(Z, a.T, np.random.default_rng(a.seed))
    st = dict(Z=Z, R0=R0, v0=v0, step=0, R=R0.copy(), v=v0.copy(), frames=[], args=vars(a))
Z = st["Z"]
m = np.array([AMU[int(z)] for z in Z])[:, None]
tm.make_template(scr / "tpl", Z, st["R"], nroots=a.nroots, nthreads=a.threads, level="wb97xd_631gs_rpa")
pes = TurbomolePES(scr / "tpl", scr / "run", state=a.state, nroots=a.nroots, nthreads=a.threads)
E, F, gaps = pes(Z, st["R"])
if st["step"] == 0 and not st["frames"]:
    st["frames"].append(dict(t_fs=0.0, R=st["R"].copy(), v=st["v"].copy(), Epot=E, Ekin=kinetic(Z, st["v"]),
                             **lab(pes)))
    save(st)
status = "running"
while st["step"] < a.steps:
    if (time.time() - t_start) / 3600 > a.max_hours:
        status = "time_limit"
        break
    R, v = st["R"], st["v"]
    v = v + 0.5 * a.dt * F / m * CONV
    R = R + a.dt * v
    try:
        E, F, gaps = pes(Z, R)
    except RuntimeError as e:
        status = f"failed at step {st['step'] + 1}: {e}"
        break
    v = v + 0.5 * a.dt * F / m * CONV
    st["step"] += 1
    st["R"], st["v"] = R, v
    st["frames"].append(dict(t_fs=st["step"] * a.dt, R=R.copy(), v=v.copy(), Epot=E, Ekin=kinetic(Z, v),
                             **lab(pes)))
    save(st)
    if st["step"] % 20 == 0:
        fr = st["frames"]
        print(f"step {st['step']} Etot {fr[-1]['Epot'] + fr[-1]['Ekin']:.5f} drift "
              f"{(fr[-1]['Epot'] + fr[-1]['Ekin']) - (fr[0]['Epot'] + fr[0]['Ekin']):.2e} gaps {np.round(gaps[:3], 3)}",
              flush=True)
if st["step"] >= a.steps:
    status = "done"
fr = st["frames"]
np.savez(out / "traj.npz", Z=Z, R=np.array([f["R"] for f in fr]), t_fs=np.array([f["t_fs"] for f in fr]),
         Epot=np.array([f["Epot"] for f in fr]), Ekin=np.array([f["Ekin"] for f in fr]),
         E0=np.array([f["E0"] for f in fr]), F0=np.array([f["F0"] for f in fr]),
         gaps=np.array([f["gaps"] for f in fr]), Fex=np.array([f["Fex"] for f in fr]))
(out / "status.json").write_text(json.dumps(dict(status=status, step=st["step"], n_frames=len(fr))))
print(status, st["step"], flush=True)
