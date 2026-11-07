"""Follow-up checks on compare_rows_all.pkl (compare.py --vectors all).

1. Chance-corrected agreement K = (S - null) / (1 - null) per seed and set (S1, heavy-atom groups).
2. Bootstrap (over geometries) of the S difference between production seeds on ladders and S1-MD training geometries.
3. Per model: S on each training/test subset; within-group pairs and "is the best-agreeing seed the best-transferring
   seed" for each subset.
4. Per geometry (ladders, production seeds): Spearman between S and |error| of the S1 eigenvalue.

Usage: python experiments/eigvec/tddft/followup.py
Output: printed tables and experiments/eigvec/tddft/followup.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "experiments/eigvec/tddft"
sys.path.insert(0, str(ROOT / "experiments/eigvec"))
from coupling_vs_transfer import GROUPS  # noqa: E402

rows = pickle.load(open(D / "compare_rows_all.pkl", "rb"))
ct = json.load(open(ROOT / "experiments/eigvec/coupling_vs_transfer.json"))
runs = [k for k in rows[0] if isinstance(rows[0][k], dict)]
prod = [r for r in runs if r.startswith("prodw_leh_v4_s")]
SETS = {"train|qcdge_eq": None, "train|offeq": None, "train|excmd": None, "test|qcdge_eq": None, "test|offeq": None,
        "test|excmd": None, "ladder": None, "dye": None}


def sel(m, g, n=1):
    return [r for r in rows if r["state"] == n and r["group"] == g and r["td_gap_ok"] and r[m]["gap_ok"]]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


out = {}
print("1. chance-corrected agreement K = (S - null)/(1 - null), S1, median over geometries")
print(f"{'set':16s} " + " ".join(f"{s[-2:]:>6s}" for s in prod) + "   seed-seed")
out["K"] = {}
for g in SETS:
    ks = []
    for s in prod:
        rr = sel(s, g)
        ks.append(float(np.median([(r[s]["S"] - r[s]["null"]) / (1 - r[s]["null"]) for r in rr])))
    ss = float(np.median([r["seed_seed"] for r in rows if r["state"] == 1 and r["group"] == g and r["td_gap_ok"]]))
    out["K"][g] = dict(zip(prod, ks), seed_seed_S=ss)
    print(f"{g:16s} " + " ".join(f"{k:6.2f}" for k in ks) + f"   {ss:.2f}")

print("\n2. bootstrap of S(s0) - S(other seed), S1, mean over common geometries, 95 % interval")
rng = np.random.default_rng(0)
out["boot"] = {}
for g in ("train|excmd", "test|excmd", "ladder", "dye"):
    for s in prod[1:]:
        rr = [r for r in rows if r["state"] == 1 and r["group"] == g and r["td_gap_ok"] and r[prod[0]]["gap_ok"]
              and r[s]["gap_ok"]]
        d = np.array([r[prod[0]]["S"] - r[s]["S"] for r in rr])
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        out["boot"][f"{g}|{s}"] = dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi))
        print(f"{g:12s} s0 - {s[-2:]}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")

print("\n3. per model S1 agreement S per subset vs ladder S1 MAE (eV)")
models = {}
for g, rs in GROUPS.items():
    for rdir in rs:
        m = rdir.name
        if m in runs and m in ct.get(g, {}):
            models[m] = dict(group=g, lad=ct[g][m]["ladders"]["mae1"],
                             **{k: float(np.median([r[m]["S"] for r in sel(m, k)])) for k in SETS})
print(f"{'model':22s} {'ladMAE':>6s} " + " ".join(f"{k[:10]:>10s}" for k in SETS))
for m, d in models.items():
    print(f"{m:22s} {d['lad']:6.3f} " + " ".join(f"{d[k]:10.2f}" for k in SETS))
out["models"] = models
out["pairs"] = {}
print("\nwithin-group pairs (samplers excluded): larger S -> lower ladder MAE; groups where the max-S seed is the "
      "min-MAE seed")
for k in SETS:
    agree = total = top = ng = 0
    for g in GROUPS:
        if g.startswith("sampler"):
            continue
        ms = [m for m in models if models[m]["group"] == g]
        if len(ms) < 2:
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = models[ms[i]], models[ms[j]]
                agree += int((a[k] - b[k]) * (a["lad"] - b["lad"]) < 0)
                total += 1
        ng += 1
        top += int(max(ms, key=lambda m: models[m][k]) == min(ms, key=lambda m: models[m]["lad"]))
    out["pairs"][k] = dict(agree=agree, total=total, top=top, groups=ng)
    print(f"{k:16s} pairs {agree:2d}/{total}  best seed picked {top}/{ng}")

print("\n4. per geometry, ladders S1: Spearman(S, |eigenvalue error|) and mean |error| for S below/above median")
out["per_geom"] = {}
for s in prod:
    rr = sel(s, "ladder")
    S = np.array([r[s]["S"] for r in rr])
    e = np.array([abs(r[s]["err"]) for r in rr])
    nh = np.array([r["n_heavy"] for r in rr])
    lo = S < np.median(S)
    # partial out size: rank residuals within heavy-atom bins
    res = []
    for a, b in ((0, 10), (10, 16), (16, 22), (22, 99)):
        k = (nh >= a) & (nh < b)
        if k.sum() > 5:
            res.append(spearman(S[k], e[k]))
    out["per_geom"][s] = dict(rho=spearman(S, e), rho_bins=res, mae_lowS=float(e[lo].mean()),
                              mae_highS=float(e[~lo].mean()))
    print(f"{s}: rho {spearman(S, e):+.2f} (within size bins {', '.join(f'{x:+.2f}' for x in res)}), "
          f"|err| low S {e[lo].mean():.3f} / high S {e[~lo].mean():.3f} eV")

json.dump(out, open(D / "followup.json", "w"), indent=1)

print("\n5. S1-MD geometries (train + test excmd, partial charge transfer): does the S1 eigenvector follow the TD-DFT "
      "hole or the electron?")
out["hole_elec"] = {}
print(f"{'model':22s} {'ladMAE':>6s} {'S_hole':>6s} {'S_elec':>6s} {'frac(hole>elec)':>15s}")
for m, d in models.items():
    rr = [r for g in ("train|excmd", "test|excmd") for r in sel(m, g)]
    h = np.array([r[m]["S_hole"] for r in rr])
    e = np.array([r[m]["S_elec"] for r in rr])
    out["hole_elec"][m] = dict(group=d["group"], lad=d["lad"], S_hole=float(np.median(h)), S_elec=float(np.median(e)),
                               frac_hole=float(np.mean(h > e)), n=len(rr))
    print(f"{m:22s} {d['lad']:6.3f} {np.median(h):6.2f} {np.median(e):6.2f} {np.mean(h > e):15.2f}")
td = [r for r in rows if r["state"] == 1 and r["group"] in ("train|excmd", "test|excmd") and r["td_gap_ok"]]
print(f"TD-DFT hole-electron overlap on these geometries: median {np.median([r['td_he'] for r in td]):.2f}")
hm = [m for m in models if out["hole_elec"][m]["frac_hole"] >= 0.5]
em = [m for m in models if out["hole_elec"][m]["frac_hole"] < 0.5]
print(f"hole followers: {len(hm)}, mean ladder MAE {np.mean([models[m]['lad'] for m in hm]):.3f} eV; "
      f"electron followers: {len(em)} ({', '.join(em)}), mean ladder MAE {np.mean([models[m]['lad'] for m in em]):.3f} eV")
agree = total = 0
for g in GROUPS:
    if g.startswith("sampler"):
        continue
    ms = [m for m in models if models[m]["group"] == g]
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = out["hole_elec"][ms[i]], out["hole_elec"][ms[j]]
            if (a["frac_hole"] >= 0.5) != (b["frac_hole"] >= 0.5):
                total += 1
                hole_better = (a["lad"] < b["lad"]) == (a["frac_hole"] >= 0.5)
                agree += int(hole_better)
print(f"within-group pairs with one hole and one electron follower: hole follower transfers better in {agree} of {total}")
json.dump(out, open(D / "followup.json", "w"), indent=1)
