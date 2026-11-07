"""S1 trajectories: model energies along TD-wB97X-D/6-31G* S1 ab initio MD (acrolein, uracil, stilbene).

Top: S1 excitation energy (S1 - S0 gap) along the reference trajectory; bottom: S0 energy relative to t = 0
(the quantity the early models underestimated along S1-relaxation coordinates). Reference TDDFT in black; LEH
before the S1-sampling round (5k development model, trained on thermal S0 geometries only) dashed; final LEH
(full training set incl. S1-sampled geometries) solid; final LEH with L=2 dotted. Stilbene is larger than every
training molecule (14 heavy atoms) and its reference trajectory is truncated where TDDFT breaks down near the
S1/S0 crossing. Acrolein and uracil are shown for the first 300 fs; full-trajectory MAEs (1000 and 500 fs) are in
the CSV table view (figure caption). Inputs: experiments/md/results/{latent_k1, main_leh_all_s0, main_leh_L2_all_s0}.pkl.
Output: paper/figs/trajectories.{pdf,png,csv}.
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa

fs.setup()
RES = {k: pickle.load(open(ROOT / f"experiments/md/results/{f}.pkl", "rb"))["frames"]
       for k, f in [("old", "latent_k1"), ("final", "main_leh_all_s0"), ("L2", "main_leh_L2_all_s0")]}
LINES = [("old", "LEH, before S1 sampling (5k)", dict(linestyle=(0, (4, 2)), lw=1.1, alpha=0.75)),
         ("final", "LEH, final", dict(linestyle="-", lw=1.4)),
         ("L2", "LEH, final, L=2", dict(linestyle=(0, (1, 1.5)), lw=1.2))]
MOLS = [("acrolein_s1", "Acrolein"), ("uracil_s1", "Uracil"), ("stilbene_s1", "trans-Stilbene (larger than training)")]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex="col")
rows = []
for c, (m, title) in enumerate(MOLS):
    ref = RES["final"][m]
    t = ref["t_fs"]
    gr = ref["gaps_ref"][:, 0]; e0r = ref["E0_ref"] - ref["E0_ref"][0]
    axes[0, c].plot(t, gr, color=fs.INK, lw=1.6, label="TD-ωB97X-D (reference)")
    axes[1, c].plot(t, e0r, color=fs.INK, lw=1.6)
    for key, lab, st in LINES:
        v = RES[key][m]
        gp = v["gaps_pred"][:, 0]; e0p = v["E0_pred"] - v["E0_pred"][0]
        axes[0, c].plot(v["t_fs"], gp, color=fs.MODEL_COLOR["leh"], label=lab, **st)
        axes[1, c].plot(v["t_fs"], e0p, color=fs.MODEL_COLOR["leh"], **st)
        rows.append(dict(molecule=m, model=lab, n_frames=len(gp), mae_S1_gap_eV=float(np.mean(np.abs(gp - v["gaps_ref"][:, 0]))),
                         mae_S0_rise_eV=float(np.mean(np.abs(e0p - (v["E0_ref"] - v["E0_ref"][0])))),
                         mae_FS1_eVA=float(np.mean(np.abs(v["FS1_pred"] - v["FS1_ref"])))))
    axes[0, c].set_title(title, loc="left", color=fs.INK)
    if m != "stilbene_s1":  # fast oscillations: show the first 300 fs, MAE over the full trajectory in the panel
        for r in (0, 1):
            axes[r, c].set_xlim(0, 300)
    axes[1, c].set_xlabel("Time on S$_1$ (fs)")
    axes[1, c].axhline(0, color=fs.AXIS, lw=0.6)
axes[0, 0].set_ylabel("S$_1$ excitation energy (eV)")
axes[1, 0].set_ylabel("S$_0$ energy rel. to t = 0 (eV)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
fig.tight_layout()
out = ROOT / "paper/figs/trajectories"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
with open(out.with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written", out)
