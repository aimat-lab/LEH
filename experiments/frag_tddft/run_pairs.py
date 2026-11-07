"""TD-wB97X-D/6-31G* reference for the fragment figure, on the same pair geometries the models see
(data/frag_pairs.pkl from experiments/fragments/make_pairs.py), including short distances where the two molecules
really interact (there, dimer != union is physics, not an error; beyond ~5 A it must vanish).

Monomers with NR roots, dimers with 2*NR roots (same level and settings as the dataset: RI-J, RPA, $rpaconv 6).
Restartable (finished escf.out are reused). Output experiments/frag_tddft/results_pairs.json: per dimer id the
lowest NR dimer roots, the union of the monomer roots and dev = dimer - union.
Usage: python experiments/frag_tddft/run_pairs.py SCRATCH [--pairs 10] [--dists 2.5 3 ...] [--workers 4] [--threads 4]
       [--only ID ...]   (e.g. a single dimer for timing)
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc import turbomole as tm  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("scratch")
ap.add_argument("--pairs", type=int, default=10)
ap.add_argument("--dists", type=float, nargs="+", default=[2.5, 3, 3.5, 4, 5, 6, 8, 10, 15])
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--only", nargs="*", default=None)
a = ap.parse_args()
SCR, NR = Path(a.scratch), 6
D = pickle.load(open(ROOT / "data/frag_pairs.pkl", "rb"))
byid = {m["id"]: k for k, m in enumerate(D)}


def spectrum(k, nroots):
    m = D[k]
    d = SCR / m["id"]
    if not (d / "escf.out").exists() or "Excitation energy" not in (d / "escf.out").read_text():
        t0 = time.time()
        tm.make_template(d, m["Z"], m["R"], nroots=nroots, nthreads=a.threads, level="wb97xd_631gs_rpa")
        subprocess.run("escf > escf.out 2>&1", cwd=d, shell=True, env=tm.tm_env(a.threads))
        print(f"{m['id']}: {len(m['Z'])} atoms, {time.time() - t0:.0f} s", flush=True)
    return np.array(tm.parse_exc((d / "escf.out").read_text())[:nroots]) * tm.HARTREE_EV


dimers = [k for k, m in enumerate(D) if m["kind"] == "frag_dimer" and m["pair"] < a.pairs
          and any(abs(m["d"] - x) < 1e-6 for x in a.dists)]
if a.only:
    dimers = [byid[i] for i in a.only]
monos = sorted({j for k in dimers for j in D[k]["mono"]})
with ThreadPoolExecutor(a.workers) as ex:
    ms = dict(zip(monos, ex.map(lambda j: spectrum(j, NR), monos)))
    ds = dict(zip(dimers, ex.map(lambda k: spectrum(k, 2 * NR), dimers)))
out = []
for k in dimers:
    m = D[k]
    i, j = m["mono"]
    union = np.sort(np.concatenate([ms[i], ms[j]]))[:NR]
    sd = ds[k][:NR]
    out.append(dict(id=m["id"], pair=m["pair"], kind=m["pair_kind"], d=m["d"], union=union.tolist(), dimer=sd.tolist(),
                    dev=(sd - union).tolist(), n_atoms=len(m["Z"])))
if not a.only:
    json.dump(out, open(ROOT / "experiments/frag_tddft/results_pairs.json", "w"), indent=1)
for r in out[:5]:
    print(r["id"], "max |dimer - union| (eV)", round(float(np.max(np.abs(r["dev"]))), 4))
print(len(out), "dimers done")
