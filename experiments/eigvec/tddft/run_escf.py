"""TD-wB97X-D/6-31G* (RPA, 10 singlet roots) excitation vectors for the geometries of a selection file.

Restarts from the converged ground state of the raw archive of each geometry (control, basis, auxbasis, mos, coord of
the dataset calculation; same TURBOMOLE settings), runs escf and keeps sing_a, which the dataset runs discarded.
Entries without an archive (archive = null; dataset tranches stored as npz only) are recomputed from scratch with the
dataset settings (exmlip.qc.turbomole.make_template: define + ridft, then ridft with $pop for the Mulliken check).
Per geometry: <out>/raw/<id>.tar.gz (control, coord, basis, auxbasis, mos, escf.out, sing_a, ridft_ref.out = ridft
output with Mulliken charges, plus the ridft/dscf outputs of recomputed ground states) and <out>/pop/<id>.npz
(per-atom populations, populations.py). The log records the deviations of E0 and of the excitation energies from the
dataset labels (E0_ref, gaps_ref).

Variants (sensitivity of the excitation energies to TURBOMOLE settings, archived geometries only):
  nori  exact Coulomb instead of RI-J (dscf, then escf)
  m5    integration grid m5 instead of m4 (ridft, then escf)

Usage: python experiments/eigvec/tddft/run_escf.py <out> [--selection selection.json] [--workers 32] [--threads 4]
       [--pick 0,40,...] [--shard k/n] [--variant nori|m5]
"""
import argparse
import json
import re
import shutil
import sys
import tarfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exmlip.qc.turbomole import make_template, parse_exc, run, tm_env  # noqa: E402
from populations import analyse, mulliken_check, read_coord  # noqa: E402

DATA = ROOT / "data"
KEEP = ("control", "coord", "basis", "auxbasis", "mos", "escf.out", "sing_a", "ridft_ref.out", "ridft.out", "dscf.out",
        "define.out")
LEVEL = "wb97xd_631gs_rpa"
HARTREE_EV = 27.211386245988


def total_energy(txt):
    return float(re.findall(r"total energy\s*=\s*(-?[\d.]+)", txt)[-1])


def edit_control(wd, variant=""):
    ctl = (wd / "control").read_text()
    assert "$scfinstab rpas" in ctl, "not an RPA control file"
    ctl = re.sub(r"\$exopt.*\n", "", ctl)
    ctl = re.sub(r"\$pop\s*\n", "", ctl)
    ctl = re.sub(r"\$actual step.*\n", "", ctl)  # left over from an aborted egrad run; escf refuses to start
    ctl = re.sub(r"(\$soes\n\s*a\s+)\d+", lambda m: m.group(1) + "10", ctl)
    if variant == "nori":
        ctl = re.sub(r"\$rij\s*\n", "", ctl)
        ctl = re.sub(r"\$ricore.*\n", "", ctl)
        ctl = re.sub(r"\$jbas.*\n", "", ctl)
    elif variant == "m5":
        ctl, n = re.subn(r"gridsize\s+m4", "gridsize   m5", ctl)
        assert n == 1, "grid line not found"
    (wd / "control").write_text(ctl)


def prepare(entry, wd, threads, variant=""):
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    if entry.get("archive") is None:
        assert not variant, "variants need an archived ground state"
        make_template(wd, np.asarray(entry["Z"]), np.asarray(entry["R"]), nroots=10, nthreads=threads, level=LEVEL)
        (wd / "control").write_text((wd / "control").read_text().replace("$end", "$pop\n$end"))
        run("ridft", wd, tm_env(threads), "ridft_ref.out")
        edit_control(wd)
        return 0.0, "fresh"
    with tarfile.open(DATA / entry["archive"]) as t:
        members = {Path(m.name).name: m for m in t.getmembers() if m.isfile()
                   and (Path(m.name).parent.name in ("sp", "") or len(Path(m.name).parts) == 1)}
        for f in ("control", "coord", "basis", "auxbasis", "mos"):
            (wd / f).write_bytes(t.extractfile(members[f]).read())
        ref = members.get("ridft_pop.out") or members.get("ridft.out")
        (wd / "ridft_ref.out").write_bytes(t.extractfile(ref).read())
    R, _ = read_coord(wd / "coord")
    dev = float(np.abs(R - np.asarray(entry["R"])).max())
    assert dev < 1e-5, f"archive geometry deviates by {dev} A"
    edit_control(wd, variant)
    if variant == "nori":
        run("dscf", wd, tm_env(threads), "dscf.out")
        assert "convergence criteria satisfied" in (wd / "dscf.out").read_text(), "dscf not converged"
    elif variant == "m5":
        run("ridft", wd, tm_env(threads), "ridft.out")
        assert "convergence criteria satisfied" in (wd / "ridft.out").read_text(), "ridft not converged"
    return dev, "archive"


def one(args):
    entry, out, threads, variant = args
    wd = out / "work" / entry["id"]
    res = dict(id=entry["id"], ok=False)
    t0 = time.time()
    try:
        res["coord_dev_A"], res["mode"] = prepare(entry, wd, threads, variant)
        scf = {"nori": "dscf.out", "m5": "ridft.out"}.get(variant, "ridft_ref.out")
        if "E0_ref" in entry:
            res["e0_dev_eV"] = total_energy((wd / scf).read_text()) * HARTREE_EV - entry["E0_ref"]
        res["t_scf"] = time.time() - t0
        env = tm_env(threads)
        run("escf", wd, env, "escf.out")
        txt = (wd / "escf.out").read_text()
        if "escf : all done" not in txt and "No convergence within" in txt:
            (wd / "escf.out").rename(wd / "escf_noconv.out")
            ctl = (wd / "control").read_text()
            ctl = re.sub(r"\$escfiterlimit\s+\d+\s*\n", "", ctl).replace("$end", "$escfiterlimit 200\n$end")
            (wd / "control").write_text(ctl)
            run("escf", wd, env, "escf.out")
            txt = (wd / "escf.out").read_text()
            res["iterlimit_retry"] = True
        res["ok"] = "escf : all done" in txt and (wd / "sing_a").exists()
        res["t_total"] = time.time() - t0
        if res["ok"]:
            exc = np.array(parse_exc(txt)[:10]) * HARTREE_EV
            res["exc_eV"] = exc.tolist()
            res["exc_dev_eV"] = float(np.abs(exc - np.asarray(entry["gaps_ref"])[: len(exc)]).max())
            try:
                p = analyse(wd)
                res["mulliken_dev"] = mulliken_check(wd, wd / "ridft_ref.out")
                res["xy_overlap"] = p["xy_overlap"]
                np.savez_compressed(out / "pop" / f"{entry['id']}.npz", exc_eV=exc, pop=p["pop"], hole=p["hole"],
                                    elec=p["elec"], omega=p["omega"], qtr=p["qtr"], R=p["R"], pop_X=p["pop_X"],
                                    Z=np.asarray(entry["Z"]), norm_xy=p["norm_xy"])
            except Exception:
                res["analysis_error"] = traceback.format_exc()[-800:]
    except Exception:
        res["error"] = traceback.format_exc()[-800:]
    finally:
        if wd.exists():
            with tarfile.open(out / "raw" / f"{entry['id']}.tar.gz", "w:gz") as t:
                for f in KEEP + ("escf_noconv.out",):
                    if (wd / f).exists():
                        t.add(wd / f, arcname=f)
            shutil.rmtree(wd)
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--selection", default=str(ROOT / "experiments/eigvec/tddft/selection.json"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pick", default="", help="comma-separated positions in the selection file (test runs)")
    ap.add_argument("--shard", default="0/1", help="k/n: every n-th geometry starting at k")
    ap.add_argument("--variant", default="", choices=("", "nori", "m5"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    for s in ("work", "raw", "pop"):
        (out / s).mkdir(parents=True, exist_ok=True)
    sel = json.load(open(a.selection))
    if a.pick:
        sel = [sel[int(i)] for i in a.pick.split(",")]
    k, n = map(int, a.shard.split("/"))
    sel = sel[k::n]
    todo = [s for s in sel if not (out / "pop" / f"{s['id']}.npz").exists()]
    # largest first, for load balance
    todo.sort(key=lambda s: -len(s["Z"]))
    with Pool(a.workers) as pool, open(out / f"log_{k}of{n}.jsonl", "a") as f:
        res = []
        for r in pool.imap_unordered(one, [(s, out, a.threads, a.variant) for s in todo]):
            f.write(json.dumps(r) + "\n")
            f.flush()
            res.append(r)
    print(f"done: {sum(r['ok'] for r in res)} of {len(res)} ok")


if __name__ == "__main__":
    main()
