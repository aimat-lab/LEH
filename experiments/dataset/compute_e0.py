"""S0 energy and gradient at the QCDGE equilibrium geometries (R_eq of task files), with exactly the settings of the
dataset single points (TD-wB97X-D/6-31G* level, define via exmlip.qc.turbomole.make_template: RI-J, m4, scfconv 7,
denconv 1e-7, weight derivatives), but without the response part.

Output: OUTDIR/<task id>.npz with Z, R (= R_eq, Angstrom), E0 (Hartree), grad0 (Hartree/bohr), flags.
Restartable and sharded like compute.py (crc32 of the id).
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
from exmlip.qc import turbomole as tm  # noqa

LEVEL = "wb97xd_631gs_rpa"


def one(task, outdir, scratch, threads):
    mid = task["id"]
    f = outdir / f"{mid}.npz"
    if f.exists() or (outdir / f"{mid}.failed.json").exists():
        return mid, "skip", 0.0
    t0 = time.time()
    wd = scratch / mid
    flags = {}
    try:
        Z, R = np.asarray(task["Z"]), np.asarray(task["R_eq"], float)
        tm.make_template(wd, Z, R, nroots=10, nthreads=threads, level=LEVEL)  # define + converged ridft
        txt = (wd / "ridft.out").read_text()
        E0 = float(__import__("re").findall(r"total energy\s*=\s*(-?[\d.]+)", txt)[-1])
        # converged SCF again with population analysis (restarts from the converged MOs, one or two cycles)
        (wd / "control").write_text((wd / "control").read_text().replace("$end", "$pop\n$end"))
        tm.run("ridft", wd, tm.tm_env(threads), "ridft_pop.out")
        pop_txt = (wd / "ridft_pop.out").read_text()
        mull = tm.parse_mulliken(pop_txt, len(Z))
        mo_e = tm.parse_mo_energies(wd / "mos")
        n_occ = tm.parse_n_occ((wd / "control").read_text())
        tm.run("rdgrad", wd, tm.tm_env(threads), "rdgrad.out")
        eg, g0 = tm.parse_gradient(wd / "gradient", len(Z))
        flags["scf_converged"] = True
        flags["e0_consistent"] = abs(eg - E0) < 1e-6
        if not flags["e0_consistent"]:
            raise RuntimeError(f"gradient energy {eg} differs from SCF energy {E0}")
        np.savez(f, Z=Z, R=R, E0=E0, grad0=g0, id=mid, source=task.get("source", ""), flags=json.dumps(flags),
                 level=LEVEL, time=time.time() - t0, mulliken=mull, mo_energies=mo_e, n_occ=n_occ)
        (outdir / "raw").mkdir(exist_ok=True)
        with tarfile.open(outdir / "raw" / f"{mid}.tar.gz", "w:gz") as tar:  # keep the raw outputs
            for name in ["control", "coord", "basis", "auxbasis", "mos", "ridft.out", "ridft_pop.out", "rdgrad.out"]:
                if (wd / name).exists():
                    tar.add(wd / name, arcname=name)
        status = "ok"
    except Exception:
        flags["exception"] = traceback.format_exc()[-2000:]
        (outdir / f"{mid}.failed.json").write_text(json.dumps(flags, default=str))
        status = "failed"
    if not os.environ.get("KEEP_SCRATCH"):
        shutil.rmtree(wd, ignore_errors=True)
    return mid, status, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--time_limit_h", type=float, default=1e9)
    a = ap.parse_args()
    seen, tasks = set(), []
    for tf in a.tasks:
        for t in pickle.load(open(tf, "rb")):
            if "R_eq" in t and t["id"] not in seen and zlib.crc32(t["id"].encode()) % a.nshards == a.shard:
                seen.add(t["id"]); tasks.append(t)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get("TM_SCRATCH", "/tmp/tm_scratch")); scratch.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in tasks if not (out / f"{t['id']}.npz").exists() and not (out / f"{t['id']}.failed.json").exists()]
    print(f"shard {a.shard}/{a.nshards}: {len(tasks)} to do", flush=True)
    t_start, n = time.time(), 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs, it = {}, iter(tasks)

        def submit():
            if (time.time() - t_start) / 3600 > a.time_limit_h:
                return False
            t = next(it, None)
            if t is None:
                return False
            futs[ex.submit(one, t, out, scratch, a.threads)] = t["id"]
            return True
        for _ in range(a.workers * 2):
            if not submit():
                break
        while futs:
            done = next(as_completed(futs))
            futs.pop(done)
            mid, st, dt = done.result()
            n += 1
            if n % 50 == 0 or st == "failed":
                print(f"{n} {mid} {st} {dt:.0f}s", flush=True)
            submit()
    print("finished", n, flush=True)
