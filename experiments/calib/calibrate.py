"""Can TURBOMOLE reproduce QCDGE (Gaussian, TD-wB97X-D/6-31G*)? Recompute 10 singlets for sample molecules."""
import json
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.qc import turbomole as tm  # noqa

OUT = ROOT / "experiments/calib"
SCR = Path("/tmp/claude-1000/calib")


def run(args):
    m, level = args
    d = SCR / f"{m['id']}_{level}"
    t0 = time.time()
    tm.make_template(d, m["Z"], m["R"], nroots=10, nthreads=4, level=level)
    t1 = time.time()
    subprocess.run("escf > escf.out 2>&1", cwd=d, shell=True, env=tm.tm_env(4))
    exc = np.array(tm.parse_exc((d / "escf.out").read_text())) * tm.HARTREE_EV
    E0 = float(__import__("re").findall(r"total energy\s*=\s*(-?[\d.]+)", (d / "ridft.out").read_text())[-1])
    return dict(id=m["id"], level=level, exc=exc.tolist(), ref=list(m["gaps"]), E0=E0, t_scf=t1 - t0, t_escf=time.time() - t1,
                natoms=len(m["Z"]))


if __name__ == "__main__":
    mols = pickle.load(open(ROOT / "data/qcdge_sub.pkl", "rb"))
    split = json.load(open(ROOT / "experiments/qcdge/split_size.json"))
    rng = np.random.default_rng(1)
    sel = [mols[i] for i in rng.choice(split["train"], 10, replace=False)] + [mols[i] for i in rng.choice(split["test_10"], 10, replace=False)]
    jobs = [(m, lv) for m in sel for lv in ["wb97xd_631gs_rpa", "wb97xd_631gs_tda"]]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, jobs))
    # reference ground-state energies (Gaussian, kcal/mol) from the excited-state calculation
    for r in res:
        f, key = ("A_9.hdf5", r["id"][3:]) if r["id"].startswith("A9_") else ("A_10.hdf5", r["id"][4:])
        with h5py.File(ROOT / "raw_data/qcdge" / f, "r") as h:
            r["E0_ref_hartree"] = float(h[key]["excited_state/Etot"][0, 0]) / 627.509474
    json.dump(res, open(OUT / "calib.json", "w"), indent=1)
    for lv in ["wb97xd_631gs_rpa", "wb97xd_631gs_tda"]:
        rr = [r for r in res if r["level"] == lv]
        d = np.array([np.array(r["exc"][:10]) - np.array(r["ref"][:10]) for r in rr])
        dE0 = np.array([r["E0"] - r["E0_ref_hartree"] for r in rr]) * tm.HARTREE_EV
        print(lv, "MAE per state (eV)", np.round(np.abs(d).mean(0), 3), "max", np.abs(d).max().round(3),
              "mean signed", d.mean().round(3), "| E0 diff eV mean/std", dE0.mean().round(3), dE0.std().round(3),
              "| t_scf+escf mean s", np.mean([r["t_scf"] + r["t_escf"] for r in rr]).round(1))
