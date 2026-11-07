"""TURBOMOLE thread scaling for the production single point (define+ridft, rdgrad, escf/egrad S1 with 10 roots).

Runs all (molecule, threads) combinations concurrently on one node, as production does, and writes wall times.
Usage: python thread_bench.py OUT_JSON  (inside a job; TM_SCRATCH set)
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc import sampling, turbomole as tm  # noqa

MOLS = {  # (label, SMILES)
    "ha14": "CN(C)c1ccc2ccc(=O)oc2c1",
    "ha25": "CCN(CC)c1ccc2cc(-c3nc4ccccc4o3)c(=O)oc2c1",
    "ha23_S": "CN(C)c1ccc(/C=C/c2ccc(-c3ccc(C=O)s3)s2)cc1",
}
THREADS = [2, 4, 8, 16]


def run(args):
    label, smi, nt = args
    scr = Path(os.environ.get("TM_SCRATCH", "/tmp")) / f"{label}_{nt}"
    Z, R = sampling.embed(smi)
    Z, R = sampling.xtb_opt(scr / "opt", Z, R)
    t0 = time.time()
    tm.make_template(scr / "tpl", Z, R, nroots=10, nthreads=nt, level="wb97xd_631gs_rpa")
    t1 = time.time()
    r = tm.single_point(scr / "tpl", scr / "sp", Z, R, nroots=10, nexgrad=1, nthreads=nt)
    t2 = time.time()
    return dict(label=label, threads=nt, n_atoms=len(Z), template_s=t1 - t0, sp_s=t2 - t1, ok=r["ok"],
                core_s=(t2 - t0) * nt)


if __name__ == "__main__":
    jobs = [(l, s, nt) for l, s in MOLS.items() for nt in THREADS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(run, jobs))
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    for r in res:
        print(r, flush=True)
