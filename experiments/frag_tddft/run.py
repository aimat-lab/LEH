"""Does TD-wB97X-D/6-31G* give the union of monomer spectra for separated fragments (premise of the E3 reference)?

Pairs of held-out QCDGE molecules (9 heavy atoms, test_9) at minimum interatomic distance d; monomers and dimers
with the same settings; dimer with 2*nroots roots. Output experiments/frag_tddft/results.json.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc import turbomole as tm  # noqa
from exmlip.training.fragments import rand_rot  # noqa

SCR = Path(sys.argv[1])
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
DISTS = [5.0, 10.0]
NR = 6


def spectrum(name, Z, R, nroots):
    d = SCR / name
    if not (d / "escf.out").exists():
        tm.make_template(d, Z, R, nroots=nroots, nthreads=THREADS, level="wb97xd_631gs_rpa")
        subprocess.run("escf > escf.out 2>&1", cwd=d, shell=True, env=tm.tm_env(THREADS))
    return np.array(tm.parse_exc((d / "escf.out").read_text())[:nroots]) * tm.HARTREE_EV


def place(RA, RB, dmin, rng):
    RA = RA - RA.mean(0)
    RB = (RB - RB.mean(0)) @ rand_rot(rng).T
    lo, hi = 0.0, 50.0  # shift B along x until the minimum interatomic distance equals dmin (bisection)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        dd = np.linalg.norm(RA[:, None] - (RB + [mid, 0, 0])[None], axis=-1).min()
        lo, hi = (mid, hi) if dd < dmin else (lo, mid)
    return RA, RB + [hi, 0, 0]


mols = pickle.load(open(ROOT / "data/qcdge_sub.pkl", "rb"))
split = json.load(open(ROOT / "experiments/qcdge/split_size.json"))
rng = np.random.default_rng(3)
pool = [mols[i] for i in rng.choice(split["test_9"], 2 * NP, replace=False)]
out = []
for p in range(NP):
    A, B = pool[2 * p], pool[2 * p + 1]
    for kind, (X, Y) in [("AB", (A, B)), ("AA", (A, A))]:
        sA = spectrum(f"mono_{p}_{'A'}", A["Z"], A["R"], NR)
        sB = sA if kind == "AA" else spectrum(f"mono_{p}_B", B["Z"], B["R"], NR)
        union = np.sort(np.concatenate([sA, sB]))[:NR]
        for d in DISTS:
            RA, RB = place(X["R"], Y["R"], d, np.random.default_rng(p))
            sD = spectrum(f"dimer_{p}_{kind}_{d:.0f}", np.concatenate([X["Z"], Y["Z"]]), np.concatenate([RA, RB]), 2 * NR)
            rec = dict(pair=p, kind=kind, d=d, union=union.tolist(), dimer=sD[:NR].tolist(),
                       dev=(sD[:NR] - union).tolist(), maxdev=float(np.abs(sD[:NR] - union).max()))
            out.append(rec)
            print(kind, p, d, "max |dimer - union| (eV)", round(rec["maxdev"], 4), flush=True)
        json.dump(out, open(ROOT / "experiments/frag_tddft/results.json", "w"), indent=1)
