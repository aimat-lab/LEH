"""Statistical check of the uniform vs weighted gap-loss comparison (production models on v4; uniform seeds 0-1,
weighted seeds 0-3 where finished; the two-seed version of 2026-10-27 is kept as significance_2seeds.{py,json,log}).

Question: are the differences between losses, architectures and seeds larger than the sampling uncertainty of the
evaluation sets? Frames are not independent, so all intervals come from a cluster bootstrap (2000 resamples) over
the independent units of each set:
  test sets / validation split: molecules (all frames of a molecule share its id prefix: eq, md and S1-dynamics frames);
  conjugation ladders: the 13 homologous families (all oligomer lengths and end groups of a family together), and,
      for comparison, the (family, end groups) series;
  trajectories: whole trajectories (errors are autocorrelated over 10-160 fs, a trajectory is one sample).
Reported per run: S1-S3 MAE (or S1-gap MAE for trajectories) with a 95 % interval; per pair of runs on identical
frames: the paired difference with a 95 % interval (the same resample for both runs); per model (architecture x
loss): mean, standard deviation (ddof 1) and range over its seeds (training variance).
Writes experiments/al/significance.json.
"""
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {f"{p}_{a}_s{s}": ROOT / f"experiments/al/runs/{p}_{a}_v4_s{s}"
        for a in ("leh", "softmin") for p in ("prod", "prodw") for s in range(4)
        if (ROOT / f"experiments/al/runs/{p}_{a}_v4_s{s}/predictions.pkl").exists()
        and (ROOT / f"experiments/md/results/{p}_{a}_v4_s{s}.pkl").exists()}
SETS = ["val", "test_random_offeq", "test_scaffold_offeq", "test_random_exc", "test_large_offeq", "test_large_exc",
        "test_random_exc4", "test_scaffold_exc4", "test_large_exc4", "test_dye_noS_eq", "test_lad_noS_eq"]
VAL_TRAJ = ["styrene_s1", "azobenzene_s1", "cyclohexenone_s1", "hexatriene_s1", "cytosine_s1", "benzaldehyde_s1",
            "cinnamaldehyde_s1"]
TEST_TRAJ = ["acrolein_s1", "uracil_s1", "stilbene_s1", "hbdi_s1", "dmabn_s1"]
LAD = list(csv.DictReader(open(ROOT / "experiments/conj/ladders.csv")))
NB = 2000
rng = np.random.default_rng(0)


def molecule(i):
    return re.match(r"([AB]\d+_\d+|Z\d+|DYE_\d+|LAD_\d+)", i).group(1)


def cluster_of(i, how):
    m = molecule(i)
    if how == "family":
        return LAD[int(m[4:])]["family"]
    if how == "series":
        r = LAD[int(m[4:])]; return r["family"] + "|" + r["ends"]
    return m


def s13_err(P):
    """|error| summed over the valid S1-S3 entries of each frame, and the number of entries (frames x states)."""
    gp = np.array([np.pad(np.asarray(x["gaps_pred"], float), (0, 3))[:3] for x in P])
    gr = np.array([np.pad(np.asarray(x["gaps_ref"], float), (0, 3))[:3] for x in P])
    gp[gp == 0] = np.nan
    m = np.isfinite(gp) & np.isfinite(gr)
    return np.where(m, np.abs(gp - gr), 0).sum(1), m.sum(1)


def boot(num, den, idx):
    """MAE = sum(num) / sum(den) over clusters; num/den: runs x clusters; idx: resamples x clusters."""
    return num[:, idx].sum(-1) / den[:, idx].sum(-1)


def summarise(names, num, den, pairs):
    idx = rng.integers(0, num.shape[1], (NB, num.shape[1]))
    full = num.sum(1) / den.sum(1); B = boot(num, den, idx)
    out = {"n_clusters": int(num.shape[1]), "runs": {}, "pairs": {}}
    for k, r in enumerate(names):
        out["runs"][r] = [float(full[k]), *map(float, np.percentile(B[k], [2.5, 97.5]))]
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b); d = B[ib] - B[ia]
        out["pairs"][f"{b} - {a}"] = [float(full[ib] - full[ia]), *map(float, np.percentile(d, [2.5, 97.5])),
                                      float(np.mean(d > 0))]
    return out


names = list(RUNS)
PAIRS = ([(f"prod_{a}_s{s}", f"prodw_{a}_s{s}") for a in ("leh", "softmin") for s in (0, 1)]       # loss, per seed
         + [(f"{p}_{a}_s0", f"{p}_{a}_s1") for a in ("leh", "softmin") for p in ("prod", "prodw")]  # seed
         + [(f"{p}_softmin_s{s}", f"{p}_leh_s{s}") for p in ("prod", "prodw") for s in range(4)])   # architecture
PAIRS = [(a, b) for a, b in PAIRS if a in names and b in names]
res = {"sets": {}, "trajectories": {}}

# frame sets (predictions.pkl of the training runs: identical frames and order in all runs)
pred = {}
for r, d in RUNS.items():
    P = pickle.load(open(d / "predictions.pkl", "rb"))
    pred[r] = {s: (s13_err(P[s]), [x["id"] for x in P[s]], np.array([x["gaps_ref"][0] for x in P[s]]),
                   np.array([x["gaps_pred"][0] for x in P[s]])) for s in SETS}
    del P
for s in SETS:
    ids = pred[names[0]][s][1]
    assert all(pred[r][s][1] == ids for r in names), s
    for how in (["family", "series", "molecule"] if s == "test_lad_noS_eq" else ["molecule"]):
        cl = [cluster_of(i, how) for i in ids]; u = {c: k for k, c in enumerate(dict.fromkeys(cl))}
        ci = np.array([u[c] for c in cl])
        num = np.array([np.bincount(ci, pred[r][s][0][0], len(u)) for r in names])
        den = np.array([np.bincount(ci, pred[r][s][0][1], len(u)) for r in names]).astype(float)
        res["sets"][f"{s} [{how}]"] = dict(n_frames=len(ids), **summarise(names, num, den, PAIRS))
    if s == "test_lad_noS_eq":  # S1-S3 MAE per ladder family
        fam = np.array([cluster_of(i, "family") for i in ids])
        res["ladder_families"] = {f: dict(n=int((fam == f).sum()), **{r: float(pred[r][s][0][0][fam == f].sum()
                                  / pred[r][s][0][1][fam == f].sum()) for r in names}) for f in dict.fromkeys(fam)}
    if s == "val":  # signed S1 error of the low-gap frames (the quantity the weighted loss targets)
        gr = pred[names[0]][s][2]; m = gr < 2
        cl = [molecule(i) for i, k in zip(ids, m) if k]; u = {c: k for k, c in enumerate(dict.fromkeys(cl))}
        ci = np.array([u[c] for c in cl])
        num = np.array([np.bincount(ci, (pred[r][s][3] - gr)[m], len(u)) for r in names])
        den = np.array([np.bincount(ci, None, len(u)) for r in names]).astype(float)
        res["sets"]["val S1<2 eV signed S1 error [molecule]"] = dict(n_frames=int(m.sum()),
                                                                     **summarise(names, num, den, PAIRS))

# trajectories (eval_models.py --what frames; frames before TDDFT breaks down)
for key, T in (("validation", VAL_TRAJ), ("test", TEST_TRAJ)):
    num, per = [], {}
    for r, d in RUNS.items():
        F = pickle.load(open(ROOT / f"experiments/md/results/{d.name}.pkl", "rb"))["frames"]
        v = [float(np.mean(np.abs(F[t]["gaps_pred"][:, 0] - F[t]["gaps_ref"][:, 0]))) for t in T]
        num.append(v); per[r] = dict(zip(T, v))
    num = np.array(num)
    res["trajectories"][key] = dict(per_trajectory=per, **summarise(names, num, np.ones_like(num), PAIRS))
# training variance: spread over seeds of each model (architecture x loss)
MODELS = {m: [r for r in names if r.rsplit("_s", 1)[0] == m] for m in dict.fromkeys(r.rsplit("_s", 1)[0] for r in names)}
res["seeds"] = {}
for k, x in [(k, x) for k, x in res["sets"].items() if "[molecule]" in k] + [(k + " trajectories", x) for k, x in res["trajectories"].items()]:
    res["seeds"][k] = {m: dict(n=len(rs), mean=float(np.mean([x["runs"][r][0] for r in rs])),
                               std=float(np.std([x["runs"][r][0] for r in rs], ddof=1)) if len(rs) > 1 else np.nan,
                               min=float(min(x["runs"][r][0] for r in rs)), max=float(max(x["runs"][r][0] for r in rs)))
                       for m, rs in MODELS.items()}
json.dump(res, open(ROOT / "experiments/al/significance.json", "w"), indent=1)


def show(title, x):
    print(f"\n{title}  (clusters: {x['n_clusters']}{', frames: ' + str(x['n_frames']) if 'n_frames' in x else ''})")
    print("   " + "  ".join(f"{r} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for r, v in x["runs"].items()))
    for p, v in x["pairs"].items():
        sig = "*" if v[1] > 0 or v[2] < 0 else " "
        print(f"   {sig} {p.ljust(28)} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")


for k, x in res["sets"].items():
    show(k, x)
for k, x in res["trajectories"].items():
    show(f"{k} trajectories, S1-gap MAE", x)
print("\nladder families, S1-S3 MAE")
print("family".ljust(20) + "n".rjust(5) + "".join(r.rjust(16) for r in names))
for f, x in res["ladder_families"].items():
    print(f.ljust(20) + str(x["n"]).rjust(5) + "".join(f"{x[r]:16.3f}" for r in names))
print("\nseed spread: mean +- std [min, max] (n seeds)")
print("set".ljust(44) + "".join(m.ljust(30) for m in MODELS))
for k, x in res["seeds"].items():
    print(k.replace(" [molecule]", "").ljust(44) + "".join(
        f"{v['mean']:.3f}+-{v['std']:.3f} [{v['min']:.3f},{v['max']:.3f}] ({v['n']})".ljust(30) for v in x.values()))
