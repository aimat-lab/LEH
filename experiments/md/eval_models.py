"""E5: excited-state PES quality and dynamics of trained models (new-dataset runs).

For each run directory:
  frames    model S0/S1-S3 energies and S1 forces on every frame of the TURBOMOLE S1 reference trajectories
  scans     model curves along the PES scans (stilbene / biphenyl torsion, acrolein C=O stretch)
  nve_ref   model S1 NVE from the reference initial conditions (same R0, v0, dt, length)
  stability batched model S1 NVE (300 K MB velocities at the minimum, 1 ps) for held-out test molecules
Output: experiments/md/results/<run>.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.md import BatchedModelPES, maxwell_boltzmann, run_nve_batched  # noqa
from exmlip.training.train import build_model_from_dir  # noqa

H2EV = 27.211386245988
G2EVA = H2EV / 0.529177210903
N_SUP = None  # set per run from its config
COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05}


def bonds(Z, R):
    r = np.array([COV[int(z)] for z in Z])
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)
    b = d < 1.25 * (r[:, None] + r[None])
    np.fill_diagonal(b, False)
    return b


def eval_geoms(model, Z, Rs, chunk=64, state=1):
    E0, gaps, F = [], [], []
    for s in range(0, len(Rs), chunk):
        part = list(Rs[s : s + chunk])
        pes = BatchedModelPES(model, [Z] * len(part), state=state, n_valid=N_SUP)
        E, Fs, gs = pes(part)
        E0.append(E - gs[:, state - 1]); gaps.append(gs[:, :6]); F += Fs
    return np.concatenate(E0), np.concatenate(gaps), np.array(F)


def valid_frames(t, s1_min=0.5, max_jump=0.1):
    """Frames before TDDFT breaks down near an S1/S0 intersection: the first frame with S1 < s1_min eV or a total-energy
    jump > max_jump eV between consecutive steps ends the usable reference trajectory."""
    et = t["Epot"] + t["Ekin"]
    bad = np.where((t["gaps"][:, 0] < s1_min) | np.r_[False, np.abs(np.diff(et)) > max_jump])[0]
    return int(bad[0]) if len(bad) else len(et)


def do_frames(model):
    res = {}
    for d in sorted((ROOT / "experiments/md/reference").glob("*_s1")):
        if not (d / "traj.npz").exists():
            continue
        t = np.load(d / "traj.npz")
        n = valid_frames(t)
        E0p, gp, Fp = eval_geoms(model, t["Z"], t["R"][:n])
        res[d.name] = dict(t_fs=t["t_fs"][:n], E0_ref=t["E0"][:n], gaps_ref=t["gaps"][:n], FS1_ref=t["Fex"][:n, 0],
                           E0_pred=E0p, gaps_pred=gp, FS1_pred=Fp, n_valid=n, n_total=len(t["t_fs"]))
    return res


def do_scans(model):
    d = ROOT / "data/wb97_aux/scans"
    tasks = {t["id"]: t for t in pickle.load(open(ROOT / "data/wb97_aux/tasks/scans.pkl", "rb"))}
    series = {}
    for f in sorted(d.glob("SCAN_*.npz")):
        r = np.load(f)
        mid = str(r["id"])
        name = mid.split("_")[1]
        series.setdefault(name, []).append((tasks[mid]["coord"], r))
    res = {}
    for name, pts in series.items():
        pts.sort(key=lambda x: x[0])
        Z = pts[0][1]["Z"]
        Rs = np.array([p[1]["R"] for p in pts])
        E0p, gp, Fp = eval_geoms(model, Z, Rs)
        res[name] = dict(coord=np.array([p[0] for p in pts]), E0_ref=np.array([float(p[1]["E0"]) * H2EV for p in pts]),
                         gaps_ref=np.array([p[1]["exc"] * H2EV for p in pts]),
                         FS1_ref=np.array([-p[1]["gradex"][0] * G2EVA for p in pts]), E0_pred=E0p, gaps_pred=gp,
                         FS1_pred=Fp)
    return res


def do_nve_ref(model):
    res = {}
    for d in sorted((ROOT / "experiments/md/reference").glob("*_s1")):
        if not (d / "ckpt.pkl").exists():
            continue
        st = pickle.load(open(d / "ckpt.pkl", "rb"))
        n = st["args"]["steps"]
        if (d / "traj.npz").exists():
            n = min(n, valid_frames(np.load(d / "traj.npz")) - 1)
        pes = BatchedModelPES(model, [st["Z"]], state=1, n_valid=N_SUP)
        tr = run_nve_batched(pes, [st["Z"]], [st["R0"]], [st["v0"]], dt_fs=st["args"]["dt"], n_steps=n, every=2)
        res[d.name] = dict(t=np.array(tr["t"]), Epot=np.array(tr["Epot"])[:, 0], Ekin=np.array(tr["Ekin"])[:, 0],
                           gaps=np.array(tr["gaps"])[:, 0], R=np.array([x[0] for x in tr["R"]]))
    return res


def do_stability(model, n_per=50, steps=2000, seed=0):
    mols = pickle.load(open(ROOT / "data/wb97_all.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/wb97/split.json"))
    rng = np.random.default_rng(seed)
    sets = {"random": [i for i in split["test_random"] if mols[i]["kind"] == "qcdge_eq"],
            "large": [i for i in split["test_large"] if mols[i]["kind"] == "eq"]}
    res = {}
    for name, idx in sets.items():
        idx = list(rng.permutation(idx)[:n_per])
        if not idx:
            continue
        Zs = [mols[i]["Z"] for i in idx]
        R0 = [mols[i]["R"] for i in idx]
        v0 = [maxwell_boltzmann(Z, 300.0, np.random.default_rng(1000 + k)) for k, Z in enumerate(Zs)]
        pes = BatchedModelPES(model, Zs, state=1, n_valid=N_SUP)
        tr = run_nve_batched(pes, Zs, R0, v0, dt_fs=0.5, n_steps=steps, every=10)
        Etot = np.array(tr["Epot"]) + np.array(tr["Ekin"])  # [T, nmol]
        gaps = np.array(tr["gaps"])  # [T, nmol, 4]
        out = []
        for k, i in enumerate(idx):
            b0 = bonds(Zs[k], R0[k])
            changed = [bool((bonds(Zs[k], fr[k]) != b0).any()) for fr in tr["R"]]
            first = next((j for j, c in enumerate(changed) if c), None)
            fin = np.isfinite(Etot[:, k]).all()
            out.append(dict(id=mols[i]["id"], n_heavy=int(mols[i]["n_heavy"]), finite=bool(fin),
                            bond_change_t=None if first is None else float(tr["t"][first]),
                            max_drift=float(np.nanmax(np.abs(Etot[:, k] - Etot[0, k]))),
                            min_s1=float(np.nanmin(gaps[:, k, 0])), min_s2s1=float(np.nanmin(gaps[:, k, 1] - gaps[:, k, 0]))))
        res[name] = out
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--what", nargs="+", default=["frames", "scans", "nve_ref", "stability"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_per", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    outdir = ROOT / "experiments/md/results"
    outdir.mkdir(parents=True, exist_ok=True)
    for rd in a.runs:
        rd = Path(rd)
        model, cfg = build_model_from_dir(rd)
        N_SUP = cfg["n_sup"]
        f = outdir / f"{rd.name}.pkl"
        res = pickle.load(open(f, "rb")) if f.exists() else {}
        for w in a.what:
            if w == "stability":
                res[w] = do_stability(model, a.n_per, a.steps)
            else:
                res[w] = globals()[f"do_{w}"](model)
            pickle.dump(res, open(f, "wb"))
            print(rd.name, w, "done", flush=True)
