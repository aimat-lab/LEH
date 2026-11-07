"""Labels of the test-only Franck-Condon force set (experiments/dataset/make_fc_tasks.py -> bwUni compute.py output in
data/wb97/fc_test/*.npz): S0 energy/forces, excitation energies and S1-S3 forces at the equilibrium geometries of 2,000
random and 500 larger test molecules, with the same units and per-root quality masks as exmlip.data.load_wb97.
Reproducibility check against the frozen dataset entry (same geometry): E0 and F0 (our level; for QCDGE minima the
dataset's excitation energies are QCDGE's own, which agree with ours to a few meV). Writes data/wb97/fc_test_labels.pkl
({entry id: dict(entry, E0, F0, gaps, Fex)}) and prints the check.
"""
import glob
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H, GRAD = 27.211386245988, 27.211386245988 / 0.529177210903
index = json.load(open(ROOT / "data/wb97/tasks/fc_test_index.json"))
D = pickle.load(open(ROOT / "data/wb97_all_100k_s1.pkl", "rb"))
out, masked, dE0, dF0, dgap = {}, 0, [], [], []
for f in sorted(glob.glob(str(ROOT / "data/wb97/fc_test/*.npz"))):
    r = np.load(f)
    fl = json.loads(str(r["flags"])); tid = str(r["id"])
    Fex = -r["gradex"].transpose(1, 0, 2) * GRAD
    for s in range(Fex.shape[1]):
        if fl.get(f"root{s+1}_dev_hartree", 0) > 2e-4 or fl.get(f"egrad{s+1}_energy_dev", 0) > 1e-5 or not fl.get(f"egrad{s+1}_ok", True):
            Fex[:, s] = np.nan; masked += 1
    i = index[tid]; m = D[i]
    lab = dict(entry=i, kind=m["kind"], E0=float(r["E0"]) * H, F0=-r["grad0"] * GRAD, gaps=np.asarray(r["exc"], float) * H, Fex=Fex)
    out[m["id"]] = lab
    if m.get("E0") is not None:
        dE0.append(lab["E0"] - m["E0"]); dF0.append(np.abs(lab["F0"] - m["F0"]).max())
    dgap.append(np.abs(lab["gaps"][:3] - np.asarray(m["gaps"][:3])).max())
pickle.dump(out, open(ROOT / "data/wb97/fc_test_labels.pkl", "wb"))
dE0, dF0, dgap = map(np.array, (dE0, dF0, dgap))
print(f"{len(out)} labelled FC geometries, {masked} masked roots")
print(f"reproducibility vs frozen dataset: |dE0| max {np.abs(dE0).max()*1000:.3f} meV (n={len(dE0)}), max |dF0| {dF0.max():.2e} eV/A, "
      f"S1-S3 gap max dev: median {np.median(dgap)*1000:.1f} meV, 95th pct {np.percentile(dgap, 95)*1000:.1f} meV, max {dgap.max()*1000:.1f} meV")
