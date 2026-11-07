"""QCDGE oscillator strengths of the ten lowest singlets (sorted by excitation energy, matching our gap labels).

Usage:
  extract_qcdge_osc.py QCDGE_DIR --ids-from TASKS.pkl [TASKS2.pkl ...] --out OSC.pkl   (dict id -> f[10])
  extract_qcdge_osc.py QCDGE_DIR --add-to data/qcdge_sub.pkl                            (adds 'osc' in place)
Ids are '<A9|A10|B9|B10>_<hdf5 key>' as used throughout this project.
"""
import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

FILES = {"A9": "A_9", "A10": "A_10", "B9": "B_9", "B10": "B_10"}


def singlet_osc(info_json, n=10):
    info = json.loads(info_json)
    s = sorted((float(v["excitation_e_eV"].split()[0]), float(v.get("oscillator_trength", v.get("oscillator_strength", np.nan))))
               for v in info.values() if v["state_type"].startswith("Singlet"))
    f = np.full(n, np.nan)
    f[: min(n, len(s))] = [x[1] for x in s[:n]]
    return f


def extract(qdir, ids):
    out, files = {}, {}
    for mid in ids:
        src, key = mid.split("_", 1)
        if src not in FILES:
            continue
        if src not in files:
            files[src] = h5py.File(Path(qdir) / f"{FILES[src]}.hdf5", "r")
        raw = files[src][key]["excited_state"]["Info_of_AllExcitedStates"][0]
        out[mid] = singlet_osc(raw.decode() if isinstance(raw, bytes) else str(raw))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("qcdge_dir")
    ap.add_argument("--ids-from", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--add-to")
    a = ap.parse_args()
    if a.add_to:
        mols = pickle.load(open(a.add_to, "rb"))
        osc = extract(a.qcdge_dir, [m["id"] for m in mols])
        for m in mols:
            m["osc"] = osc.get(m["id"], np.full(10, np.nan))
        pickle.dump(mols, open(a.add_to, "wb"))
        print("added osc to", len(mols), "entries; missing", sum(m["id"] not in osc for m in mols))
    else:
        ids = sorted({t["id"] for f in a.ids_from for t in pickle.load(open(f, "rb"))})
        osc = extract(a.qcdge_dir, ids)
        pickle.dump(osc, open(a.out, "wb"))
        print("extracted", len(osc), "of", len(ids))
