"""Compute one shard of a tranche: xtb MD distortion + TD-wB97X-D/6-31G* (RPA) E0/F0, 10 roots, S1-S3 forces.

Restartable: finished molecules (npz) and recorded failures are skipped.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import tarfile
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc import sampling, turbomole as tm  # noqa

COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05}
LEVEL = "wb97xd_631gs_rpa"


def bonds(Z, R):
    r = np.array([COV[int(z)] for z in Z])
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)
    b = d < 1.25 * (r[:, None] + r[None])
    np.fill_diagonal(b, False)
    return b


ARCHIVE = {"sp": ["control", "coord", "basis", "auxbasis", "mos", "ridft.out", "rdgrad.out", "egrad1.out",
                   "egrad2.out", "egrad3.out", "egrad1_noconv.out", "egrad2_noconv.out", "egrad3_noconv.out",
                   "dscf.out", "escf.out"],
           "md": ["md.inp", "xtb.trj"], "opt": ["xtbopt.xyz", "opt.out"]}


def archive(wd, dest):
    """Keep the useful raw outputs of one calculation (TURBOMOLE input/output, MOs, xTB trajectory) as tar.gz."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for sub, names in ARCHIVE.items():
            for name in names:
                f = wd / sub / name
                if f.exists():
                    tar.add(f, arcname=f"{sub}/{name}")
    os.replace(tmp, dest)


def one(task, outdir, scratch, threads, mode):
    mid = task["id"]
    f = outdir / f"{mid}.npz"
    if f.exists() or (outdir / f"{mid}.failed.json").exists():
        return mid, "skip", 0.0
    t0 = time.time()
    h = zlib.crc32(mid.encode())
    rng = np.random.default_rng(h)
    wd = scratch / mid
    flags = {}
    if "sampling" in task:  # provenance of model-sampled geometries (experiments/dataset/s1_sampling.py)
        flags["sampling"] = task["sampling"]
    try:
        mode = task.get("mode", mode)
        if "R_eq" in task:
            Z, Req = task["Z"], task["R_eq"]
        else:  # start from SMILES: RDKit embedding + GFN2-xTB minimum
            Z, R0 = sampling.embed(task["smiles"])
            Z, Req = sampling.xtb_opt(wd / "opt", Z, R0)
            flags["xtb_minimum"] = True
        if mode == "md":
            T = float(rng.uniform(300, 1000))
            fr = sampling.xtb_md(wd / "md", Z, Req, int(T), time_ps=1.6, dump_fs=100.0)
            k = int(rng.integers(5, min(16, len(fr))))
            R = fr[k][1]
            flags["T"] = T
            flags["md_frame"] = k
        elif mode == "eq":
            R = Req
            flags["T"] = 0
        else:  # given geometry
            R, T = task["R"], task.get("T", 0)
            flags["T"] = T
        flags["connectivity_changed"] = bool((bonds(Z, R) != bonds(Z, Req)).any())
        tm.make_template(wd / "tpl", Z, R, nroots=10, nthreads=threads, level=LEVEL)
        r = tm.single_point(wd / "tpl", wd / "sp", Z, R, nroots=10, nexgrad=3, nthreads=threads, keep=True)
        flags.update(r["flags"])
        if not r["ok"]:
            raise RuntimeError("single point failed")
        np.savez(f, Z=Z, R=R, E0=r["E0"], grad0=r["grad0"], exc=r["exc"], gradex=r["gradex"], R_eq=Req,
                 gaps_eq=task["gaps_eq"], id=mid, mol=task.get("mol", mid), source=task["source"], smiles=task.get("smiles", ""), flags=json.dumps(flags), level=LEVEL,
                 time=time.time() - t0, mulliken=r["mulliken"], mo_energies=r["mo_energies"], n_occ=r["n_occ"],
                 osc_strength=r["osc_strength"], tdip=r["tdip"])
        archive(wd, outdir / "raw" / f"{mid}.tar.gz")
        status = "ok"
    except Exception:
        flags["exception"] = traceback.format_exc()[-2000:]
        try:  # keep the outputs of failed calculations too (diagnosis)
            archive(wd, outdir / "raw" / f"{mid}.failed.tar.gz")
        except Exception:
            pass
        (outdir / f"{mid}.failed.json").write_text(json.dumps(flags, default=str))
        status = "failed"
    if not os.environ.get("KEEP_SCRATCH"):
        shutil.rmtree(wd, ignore_errors=True)
    return mid, status, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mode", default="md", choices=["md", "given", "eq"])
    ap.add_argument("--time_limit_h", type=float, default=1e9, help="stop submitting new work after this")
    a = ap.parse_args()
    tasks = [t for t in pickle.load(open(a.tasks, "rb")) if zlib.crc32(t["id"].encode()) % a.nshards == a.shard]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("TM_SCRATCH", "/tmp/tm_scratch")); scratch.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in tasks if not (out / f"{t['id']}.npz").exists() and not (out / f"{t['id']}.failed.json").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(tasks)} to do", flush=True)
    t_start = time.time()
    n = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        it = iter(tasks)
        def submit():
            if (time.time() - t_start) / 3600 > a.time_limit_h:
                return False
            t = next(it, None)
            if t is None:
                return False
            futs[ex.submit(one, t, out, scratch, a.threads, a.mode)] = t["id"]
            return True
        for _ in range(a.workers * 2):
            if not submit():
                break
        while futs:
            done = next(as_completed(futs))
            futs.pop(done)
            mid, st, dt = done.result()
            n += 1
            print(f"{n} {mid} {st} {dt:.0f}s", flush=True)
            submit()
