"""Pre-label error of one active-learning round: the sampling model's predicted S1 gap at the selected frames (stored in
the task metadata) vs the new TD-wB97X-D labels, overall, by pick criterion and by double-bond twist; plus the share
of near-degenerate frames (TDDFT S1 < 1 eV). Model-selected frames: not an unbiased test error, but its decrease from
round to round shows whether the sampled region is being learned.
Usage: python experiments/al/prelabel_stats.py s1r2   -> experiments/al/prelabel_s1r2.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H = 27.211386245988
rnd = sys.argv[1]
rows = []
for f in glob.glob(str(ROOT / f"data/wb97/{rnd}/*.npz")):
    z = np.load(f)
    smp = json.loads(str(z["flags"])).get("sampling", {})
    rows.append(dict(kind=smp.get("kind"), src=str(z["source"]), twist=smp.get("twist_deg"),
                     gp=float(np.asarray(smp.get("gaps_pred", [np.nan]))[0]), gr=float(np.asarray(z["exc"], float)[0] * H)))
gp = np.array([r["gp"] for r in rows]); gr = np.array([r["gr"] for r in rows]); e = gp - gr
tw = np.array([r["twist"] if r["twist"] is not None else np.nan for r in rows])
st = lambda m: dict(n=int(m.sum()), mae=float(np.nanmean(np.abs(e[m]))), signed=float(np.nanmean(e[m])),
                    median_tddft_s1=float(np.nanmedian(gr[m]))) if m.any() else None
out = dict(round=rnd, n=len(rows), tddft_s1_below_1eV=int((gr < 1).sum()), tddft_s1_below_0p5eV=int((gr < 0.5).sum()),
           all=st(np.ones(len(rows), bool)),
           by_pick={k: st(np.array([r["kind"] == k for r in rows])) for k in ("twist", "committee", "relaxed", "rand")},
           by_twist={f"{lo}-{hi}": st((tw >= lo) & (tw < hi)) for lo, hi in ((0, 30), (30, 60), (60, 75), (75, 91))},
           zinc=st(np.array([r["src"] == "ZINC" for r in rows])))
json.dump(out, open(ROOT / f"experiments/al/prelabel_{rnd}.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n", "tddft_s1_below_1eV", "all", "by_twist")}, indent=1))
