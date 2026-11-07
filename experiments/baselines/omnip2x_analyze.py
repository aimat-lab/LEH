"""Analysis of the zero-shot OMNI-P2x evaluation (omnip2x_zeroshot.py), optionally side by side with our runs.

OMNI-P2x's reference data differ from ours (TD-DFT excited states from PubChemQC, CCSD(T)-type ground state), so:
- excitation energies are reported raw and after a per-state recalibration fitted on the shared validation subset
  (shift only: E_ref = E_pred + b_n; linear: E_ref = a_n E_pred + b_n), never fitted on test data;
- S0 is compared through within-molecule energy differences (off-equilibrium minus minimum of the same molecule) and
  forces, not absolute energies;
- QCDGE-derived molecules are split into A (QM9/GDB-11-derived) and B (PubChemQC-derived) because OMNI-P2x's
  training data come from PubChemQC (possible molecule overlap, at other geometries and level of theory).
Usage:
  python experiments/baselines/omnip2x_analyze.py experiments/wb97/runs_zeroshot/omnip2x \
      [--compare experiments/wb97/runs_main/main_leh_all_s0 ...]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--compare", nargs="*", default=[], help="our run dirs (predictions.pkl with the same test keys)")
ap.add_argument("--n", type=int, default=6, help="excitation energies S1..Sn compared")
a = ap.parse_args()
N = a.n


def r2(p, r):
    return float(1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2))


def gaps(P):
    gp = np.array([np.asarray(x["gaps_pred"], float)[:N] for x in P])
    gr = np.array([np.asarray(x["gaps_ref"], float)[:N] for x in P])
    return gp, gr


def gap_metrics(gp, gr):
    m = np.isfinite(gp) & np.isfinite(gr)
    if m.sum() < 3:
        return {}
    return dict(mae=float(np.mean(np.abs(gp[m] - gr[m]))), median=float(np.median(np.abs(gp[m] - gr[m]))),
                r2=r2(gp[m], gr[m]), mae_S1=float(np.nanmean(np.abs(gp[:, 0] - gr[:, 0]))))


def force_metrics(P):
    F = [x for x in P if x.get("F0_ref") is not None and x.get("F0_pred") is not None]
    out = {}
    if F:
        out["F0"] = float(np.mean(np.concatenate([np.abs(x["F0_pred"] - x["F0_ref"]).ravel() for x in F])))
    X = [x for x in F if x.get("Fex_ref") is not None]
    if X:
        d = np.concatenate([np.abs(x["Fex_pred"][:, 0] - x["Fex_ref"][:, 0]).ravel() for x in X])
        out["F_S1"] = float(np.nanmean(d))
    return out


def osc_metrics(P):
    fp = np.array([np.asarray(x["osc_pred"], float)[0] for x in P if x.get("osc_ref") is not None and x.get("osc_pred") is not None])
    fr = np.array([np.asarray(x["osc_ref"], float)[0] for x in P if x.get("osc_ref") is not None and x.get("osc_pred") is not None])
    m = np.isfinite(fp) & np.isfinite(fr)
    if m.sum() < 10:
        return {}
    return dict(f_S1_mae=float(np.mean(np.abs(fp[m] - fr[m]))), f_S1_pearson=float(np.corrcoef(fp[m], fr[m])[0, 1]))


def relative_energies(P_eq, P_off):
    """Within-molecule differences (off-eq minus eq of the same molecule): S0 and total S1 energy."""
    eq = {str(x["id"]).rsplit("_", 1)[0]: x for x in P_eq if x.get("E0_ref") is not None}
    d0, d1 = [], []
    for x in P_off:
        y = eq.get(str(x["id"]).rsplit("_", 1)[0])
        if y is None or x.get("E0_ref") is None:
            continue
        d0.append((x["E0_pred"] - y["E0_pred"]) - (x["E0_ref"] - y["E0_ref"]))
        s1 = lambda z, k: z[f"E0_{k}"] + np.asarray(z[f"gaps_{k}"], float)[0]
        d1.append((s1(x, "pred") - s1(y, "pred")) - (s1(x, "ref") - s1(y, "ref")))
    if not d0:
        return {}
    return dict(rel_S0_mae=float(np.nanmean(np.abs(d0))), rel_S1_mae=float(np.nanmean(np.abs(d1))), n_pairs=len(d0))


run = Path(a.run)
P = pickle.load(open(run / "predictions.pkl", "rb"))
# recalibration on the validation subset only
gpv, grv = gaps(P["val"])
shift = np.nanmean(grv - gpv, axis=0)
lin = []
for n in range(N):
    m = np.isfinite(gpv[:, n]) & np.isfinite(grv[:, n])
    lin.append(np.polyfit(gpv[m, n], grv[m, n], 1))
lin = np.array(lin)
res = dict(calibration=dict(shift=shift.tolist(), linear=lin.tolist(), n_val=len(P["val"])), sets={})
pairs = {"test_random_offeq": "test_random_eq", "test_scaffold_offeq": "test_scaffold_eq", "test_large_offeq": "test_large_eq",
         "test_random_exc": "test_random_eq", "test_scaffold_exc": "test_scaffold_eq", "test_large_exc": "test_large_eq"}
for k, v in P.items():
    if k == "val":
        continue
    gp, gr = gaps(v)
    r = dict(n=len(v), raw=gap_metrics(gp, gr), shift=gap_metrics(gp + shift, gr), linear=gap_metrics(gp * lin[:, 0] + lin[:, 1], gr))
    r.update(force_metrics(v)); r.update(osc_metrics(v))
    if k in pairs and pairs[k] in P:
        r.update(relative_energies(P[pairs[k]], v))
    for grp, srcs in [("A_QM9_GDB", ("A9", "A10")), ("B_PubChemQC", ("B9", "B10"))]:
        sub = [x for x in v if x.get("source") in srcs]
        if len(sub) > 20:
            g1, g2 = gaps(sub)
            r[f"linear_{grp}"] = gap_metrics(g1 * lin[:, 0] + lin[:, 1], g2)
    res["sets"][k] = r
# our runs on the same entries (matched by id)
for c in a.compare:
    cp = {}
    for f in sorted(Path(c).glob("predictions*.pkl")):  # predictions.pkl + e.g. predictions_conj.pkl (repredict.py --out)
        cp.update(pickle.load(open(f, "rb")))
    for k, v in cp.items():
        if k not in res["sets"]:
            continue
        ids = {str(x["id"]) for x in P[k]}
        sub = [x for x in v if str(x["id"]) in ids]
        gp, gr = gaps(sub)
        res["sets"][k][f"ours:{Path(c).name}"] = dict(n=len(sub), **gap_metrics(gp, gr), **force_metrics(sub))
out = run / "omnip2x_results.json"
json.dump(res, open(out, "w"), indent=1)
print(f"calibration on {len(P['val'])} validation geometries; per-state shift (eV):", np.round(shift, 2))
for k, r in res["sets"].items():  # generous to OMNI-P2x: best of raw / shift / linear per set
    best = min(("raw", "shift", "linear"), key=lambda q: r[q].get("mae", 9e9) if r[q] else 9e9)
    r["best"] = dict(r[best], variant=best)
hdr = f"{'set':22s}{'n':>6s}{'raw MAE':>9s}{'shift MAE':>10s}{'lin MAE':>9s}{'lin R2':>8s}{'F0':>7s}{'F_S1':>7s}{'rel S0':>8s}{'rel S1':>8s}{'fS1 r':>7s}"
print(f"S1-S{N} excitation energies (eV), forces (eV/A), within-molecule energies (eV)\n" + hdr)
for k, r in res["sets"].items():
    g = lambda d, q: d.get(q, np.nan) if d else np.nan
    print(f"{k:22s}{r['n']:6d}{g(r['raw'], 'mae'):9.3f}{g(r['shift'], 'mae'):10.3f}{g(r['linear'], 'mae'):9.3f}"
          f"{g(r['linear'], 'r2'):8.3f}{r.get('F0', np.nan):7.3f}{r.get('F_S1', np.nan):7.3f}{r.get('rel_S0_mae', np.nan):8.3f}"
          f"{r.get('rel_S1_mae', np.nan):8.3f}{r.get('f_S1_pearson', np.nan):7.3f}")
    for kk, vv in r.items():
        if kk.startswith("linear_") or kk.startswith("ours:"):
            print(f"   {kk:32s} MAE {vv.get('mae', np.nan):.3f} R2 {vv.get('r2', np.nan):.3f}" +
                  (f" F_S1 {vv['F_S1']:.3f}" if "F_S1" in vv else ""))
json.dump(res, open(out, "w"), indent=1)  # again, now including the "best" entries
print("written", out)
