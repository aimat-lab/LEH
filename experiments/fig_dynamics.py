"""Excited-state dynamics figure (main Fig. 6): active-learning convergence and the test trajectories.

(a, b) Convergence criterion along the seven validation trajectories (TD-wB97X-D S1 dynamics of molecules never used
    for training; experiments/al/convergence_*.json): mean S1-gap MAE (a) and mean S0-rise MAE (b) of the sampling
    models of each round (round 1 = frozen benchmark LEH; small dots: seeds, line: mean) and of the production models
    trained from scratch on all rounds (weighted loss, four seeds each; small: seeds, large: seed mean). Dashed line:
    accuracy threshold of the criterion (0.2 eV).
(c) S1-gap MAE per test trajectory (acrolein, uracil, trans-stilbene, HBDI, DMABN; never used for any decision) for
    the benchmark LEH (before active learning) and the production LEH and Direct-softmin models (small: seeds, large
    outlined: committee = seed-averaged prediction).
Structures above the columns: perceived from the reference geometries with RDKit.
(d-m) Along the TD-wB97X-D S1 test trajectories (frames before TDDFT breaks down): S1 excitation energy (d-h) and S0
    energy relative to t = 0 (i-m). Reference black; production LEH and Direct-softmin: committee line and the range over
    the four seeds (band); benchmark LEH before active learning grey dotted. Acrolein and uracil: first 300 fs.
Inputs: experiments/md/results/<run>.pkl (eval_models.py --what frames), experiments/al/convergence_*.json,
experiments/md/reference/<traj>/traj.npz (structures).
Output: paper/figs/dynamics.{pdf,png,csv}.
"""
import csv
import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa
from matplotlib.lines import Line2D  # noqa
from matplotlib.patches import Patch  # noqa
from rdkit import Chem  # noqa
from rdkit.Chem import AllChem, rdDetermineBonds  # noqa
from rdkit.Chem.Draw import rdMolDraw2D  # noqa

fs.setup()
RES = ROOT / "experiments/md/results"
AL = ROOT / "experiments/al"
BLUE, GREEN, GREY = fs.MODEL_COLOR["leh"], fs.MODEL_COLOR["softmin"], fs.MUTED
MODELS = {"bench": ["main_leh_all_s0"], "leh": [f"prodw_leh_v4_s{s}" for s in range(4)],
          "softmin": [f"prodw_softmin_v4_s{s}" for s in range(4)]}
STY = {"bench": dict(color=GREY, linestyle=(0, (1, 1.2)), lw=1.0, label="LEH, before active learning"),
       "leh": dict(color=BLUE, linestyle="-", lw=1.2, label="LEH (production)"),
       "softmin": dict(color=GREEN, linestyle=(0, (4, 1.6)), lw=1.2, label="Direct-softmin (production)")}
MARK = {"bench": "s", "leh": "o", "softmin": "D"}
TEST = [("acrolein_s1", "Acrolein"), ("uracil_s1", "Uracil"), ("stilbene_s1", "trans-Stilbene"), ("hbdi_s1", "HBDI"),
        ("dmabn_s1", "DMABN")]
F = {k: [pickle.load(open(RES / f"{f}.pkl", "rb"))["frames"] for f in fl] for k, fl in MODELS.items()}
rows = []


def committee(k, m, key):
    return np.mean([r[m][key] for r in F[k]], 0)


def structure_png(traj, size=(420, 300)):
    d = np.load(ROOT / f"experiments/md/reference/{traj}/traj.npz")
    pt = Chem.GetPeriodicTable()
    xyz = f"{len(d['Z'])}\n\n" + "\n".join(f"{pt.GetElementSymbol(int(z))} {x:.6f} {y:.6f} {w:.6f}"
                                          for z, (x, y, w) in zip(d["Z"], d["R"][0]))
    m = Chem.MolFromXYZBlock(xyz); rdDetermineBonds.DetermineBonds(m, charge=0)
    Chem.AssignStereochemistryFrom3D(m)
    m = Chem.RemoveHs(m); smi = Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(smi); AllChem.Compute2DCoords(m)
    dr = rdMolDraw2D.MolDraw2DCairo(*size)
    o = dr.drawOptions(); o.bondLineWidth = 2; o.clearBackground = True; o.padding = 0.06
    o.baseFontSize = 0.9; o.additionalAtomLabelPadding = 0.1
    dr.DrawMolecule(m); dr.FinishDrawing()
    img = plt.imread(io.BytesIO(dr.GetDrawingText()), format="png")
    ys, xs = np.where(img[..., :3].min(-1) < 0.95)  # crop the white margin
    return img[max(ys.min() - 6, 0):ys.max() + 7, max(xs.min() - 6, 0):xs.max() + 7], smi


fig = plt.figure(figsize=(7.2, 6.5))  # Nature: <= 183 x 170 mm
LABELS = []
gs = GridSpec(4, 5, figure=fig, height_ratios=[1.0, 0.5, 1.0, 0.85], hspace=0.45, wspace=0.42)
top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :], width_ratios=[1.0, 1.0, 0.95], wspace=0.75)

# (a, b) convergence on the validation trajectories
ROUNDS = [("1", "convergence_r1_benchmark.json"), ("2", "convergence_r2.json"), ("3", "convergence_r3.json"),
          ("4", "convergence_r4.json")]


def per_run(f, q):
    d = json.load(open(AL / f))
    return {r: float(np.mean([v[q] for v in x["trajectories"].values()])) for r, x in d["runs"].items()}


for j, (q, lab) in enumerate([("gap", "Mean absolute S$_1$\nerror (eV)"), ("rise", "Mean absolute error of\nS$_0$ energy change (eV)")]):
    ax = fig.add_subplot(top[0, j])
    xs, means = [], []
    for k, (rd, f) in enumerate(ROUNDS):
        v = list(per_run(f, q).values())
        ax.scatter([k] * len(v), v, s=10, color=BLUE, alpha=0.45, linewidths=0, zorder=3)
        xs.append(k); means.append(np.mean(v))
        rows += [dict(panel="a" if j == 0 else "b", metric=q, round=rd, model="sampling LEH", value=x) for x in v]
    ax.plot(xs, means, color=BLUE, lw=1.4, marker="o", ms=4, markeredgecolor="white", zorder=4)
    xp = len(ROUNDS) + 0.6
    for arch, dx in (("leh", -0.25), ("softmin", 0.25)):
        v = [list(per_run(f"convergence_{r}.json", q).values())[0] for r in MODELS[arch]]
        ax.scatter([xp + dx] * len(v), v, s=10, marker=MARK[arch], color=STY[arch]["color"], alpha=0.45, linewidths=0,
                   zorder=4)
        ax.scatter(xp + dx, np.mean(v), s=26, marker=MARK[arch], color=STY[arch]["color"], edgecolors="white",
                   linewidths=0.6, zorder=5)
        rows += [dict(panel="a" if j == 0 else "b", metric=q, round="production", model=r, value=x)
                 for r, x in zip(MODELS[arch], v)]
    if q == "gap":
        ax.axhline(0.2, color=fs.INK2, lw=0.7, linestyle=(0, (3, 2)))
        ax.text(-0.3, 0.205, "criterion", fontsize=6, color=fs.INK2, va="bottom")
    ax.set_xticks(list(range(len(ROUNDS))) + [xp]); ax.set_xticklabels([r for r, _ in ROUNDS] + ["prod."])
    ax.set_xlim(-0.4, xp + 0.6); ax.set_ylim(0, None)
    ax.set_xlabel("Active-learning round"); ax.set_ylabel(lab)
    LABELS.append((ax, "ab"[j]))

# (c) S1-gap MAE per test trajectory: seeds (small) and committee (large)
ax = fig.add_subplot(top[0, 2])
for i, (m, name) in enumerate(TEST):
    ref = F["leh"][0][m]["gaps_ref"][:, 0]
    for k, off in (("bench", -0.25), ("leh", 0.0), ("softmin", 0.25)):
        for s, r in enumerate(F[k]):
            v = float(np.mean(np.abs(r[m]["gaps_pred"][:, 0] - ref)))
            ax.scatter(v, i + off, s=9 if k != "bench" else 16, color=STY[k]["color"], marker=MARK[k],
                       alpha=0.5 if k != "bench" else 1, linewidths=0, zorder=3)
            rows.append(dict(panel="c", molecule=m, model=MODELS[k][s], metric="S1-gap MAE along trajectory", value=v))
        if k != "bench":
            v = float(np.mean(np.abs(committee(k, m, "gaps_pred")[:, 0] - ref)))
            ax.scatter(v, i + off, s=26, color=STY[k]["color"], marker=MARK[k], edgecolors=fs.INK, linewidths=0.6, zorder=4)
            rows.append(dict(panel="c", molecule=m, model=f"{k} committee", metric="S1-gap MAE along trajectory", value=v))
ax.set_yticks(range(len(TEST))); ax.set_yticklabels([n for _, n in TEST]); ax.invert_yaxis()
ax.set_xlim(0, None); ax.set_xlabel("Mean absolute S$_1$ error (eV)")
LABELS.append((ax, "c"))

# structures and (d-m) test trajectories
for c, (m, name) in enumerate(TEST):
    a0 = fig.add_subplot(gs[1, c])
    img, smi = structure_png(m)
    a0.imshow(img); a0.set_axis_off(); a0.set_anchor("S")
    a0.set_position(gs[1, c].get_position(fig).shrunk(1.0, 0.72))  # leave the top of the cell for the name
    cell = gs[1, c].get_position(fig)
    fig.text((cell.x0 + cell.x1) / 2, cell.y1 - 0.004, name, ha="center", va="top", fontsize=7, color=fs.INK)
    rows.append(dict(panel="structure", molecule=m, smiles=smi))
    a1, a2 = fig.add_subplot(gs[2, c]), fig.add_subplot(gs[3, c])
    ref = F["leh"][0][m]
    t = ref["t_fs"]
    a1.plot(t, ref["gaps_ref"][:, 0], color=fs.INK, lw=1.4, zorder=5)
    a2.plot(t, ref["E0_ref"] - ref["E0_ref"][0], color=fs.INK, lw=1.4, zorder=5)
    v = F["bench"][0][m]; st = {k: x for k, x in STY["bench"].items() if k != "label"}
    a1.plot(t, v["gaps_pred"][:, 0], zorder=2, **st)
    a2.plot(t, v["E0_pred"] - v["E0_pred"][0], zorder=2, **st)
    for k, z in (("softmin", 3), ("leh", 4)):
        st = {kk: x for kk, x in STY[k].items() if kk != "label"}
        g = np.array([r[m]["gaps_pred"][:, 0] for r in F[k]])
        e = np.array([r[m]["E0_pred"] - r[m]["E0_pred"][0] for r in F[k]])
        a1.fill_between(t, g.min(0), g.max(0), color=st["color"], alpha=0.18, lw=0, zorder=z - 0.5)
        a1.plot(t, g.mean(0), zorder=z, **st)
        a2.fill_between(t, e.min(0), e.max(0), color=st["color"], alpha=0.18, lw=0, zorder=z - 0.5)
        a2.plot(t, e.mean(0), zorder=z, **st)
    LABELS += [(a1, "defgh"[c]), (a2, "ijklm"[c])]
    tmax = 300 if m in ("acrolein_s1", "uracil_s1") else t[-1]
    for a in (a1, a2):
        a.set_xlim(0, tmax)
    a2.axhline(0, color=fs.AXIS, lw=0.6)
    a2.set_xlabel("Time (fs)")
    if c == 0:
        a1.set_ylabel("S$_1$ excitation\nenergy (eV)"); a2.set_ylabel("S$_0$ energy relative\nto t = 0 (eV)")
    rows += [dict(panel="defgh"[c], molecule=m, model="reference", n_frames=len(t), t_max_fs=float(t[-1]))]
h = [Line2D([], [], color=fs.INK, lw=1.4),
     Line2D([], [], marker=MARK["bench"], ms=4.5, markeredgecolor="white", **{k: v for k, v in STY["bench"].items() if k != "label"})]
for k in ("softmin", "leh"):
    h.append((Patch(color=STY[k]["color"], alpha=0.18, lw=0),
              Line2D([], [], marker=MARK[k], ms=4.5, markeredgecolor="white", **{kk: v for kk, v in STY[k].items() if kk != "label"})))
l = ["TD-$\\omega$B97X-D"] + [STY[k]["label"] for k in ("bench", "softmin", "leh")]
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.045), frameon=False)
fs.panel_labels(fig, LABELS)
out = ROOT / "paper/figs/dynamics"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
keys = sorted({k for r in rows for k in r})
with open(out.with_suffix(".csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("written", out, "| seeds:", {k: len(v) for k, v in F.items()})
