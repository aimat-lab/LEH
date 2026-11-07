"""Model B (100k + training-dye scaffolds, incl. S) vs model A (100k only) on the conjugated test sets.

Same test entries for both: held-out dye scaffolds and all ladders (S-free: test_{dye,lad}_noS_{eq,offeq}; model A's
predictions on them come from repredict.py on split_conj_only.json, which has identical keys), S-containing dyes and
ladders (model B only; model A never saw S), and the 100k test sets (does adding dyes cost accuracy on small molecules?).
Metrics per run and set: S1-S3 / S1 MAE (eV) and R2 (S1), S1 force MAE (eV/A, off-eq sets), S0 energy MAE (meV/atom);
mean +- std over seeds per model. Writes experiments/conj/model_B_vs_A.json and prints a table.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B, M, L = ROOT / "experiments/conj/runs_B", ROOT / "experiments/wb97/runs_main", ROOT / "experiments/wb97/runs_lc"
# (run dir, file with the conjugated-set predictions, relative to the run dir or absolute)
MODELS = {"A LEH": [(M / f"main_leh_all_s{s}", "predictions_conj.pkl") for s in (0, 1)],
          "A softmin": [(M / f"main_softmin_all_s{s}", "predictions_conj.pkl") for s in (0, 1)],
          "A X-MACE": [(L / "xmace128_suball", L / "xmace128_suball_conj/predictions.pkl")],
          "A OMNI arch": [(L / "suball_vecmsani", L / "suball_vecmsani_conj/predictions.pkl")],
          "B LEH": [(B / f"B_leh_s{s}", "predictions.pkl") for s in (0, 1)],
          "B softmin": [(B / f"B_softmin_s{s}", "predictions.pkl") for s in (0, 1)],
          "B noS LEH": [(B / "B_noS_leh_s0", "predictions.pkl")]}
CONJ = ["test_dye_noS_eq", "test_dye_noS_offeq", "test_lad_noS_eq", "test_lad_noS_offeq", "test_dye_S_eq", "test_dye_S_offeq",
        "test_lad_S_eq", "test_lad_S_offeq"]
SMALL = ["test_random_offeq", "test_scaffold_offeq", "test_large_offeq", "test_random_exc"]


def r2(p, r):
    return float(1 - np.sum((p - r) ** 2) / np.sum((r - r.mean()) ** 2))


def metrics(P):
    gp = np.array([np.pad(np.asarray(x["gaps_pred"], float), (0, 3))[:3] for x in P])
    gr = np.array([np.asarray(x["gaps_ref"], float)[:3] for x in P])
    m = np.isfinite(gp) & np.isfinite(gr)
    o = {"n": len(P), "S1-3 MAE": float(np.mean(np.abs(gp[m] - gr[m]))), "S1 MAE": float(np.nanmean(np.abs(gp[:, 0] - gr[:, 0]))),
         "S1 R2": r2(gp[m[:, 0], 0], gr[m[:, 0], 0])}
    E = [x for x in P if x.get("E0_ref") is not None and x.get("E0_pred") is not None]
    if E:
        o["E0 meV/at"] = float(1000 * np.mean([abs(x["E0_pred"] - x["E0_ref"]) / x["n_atoms"] for x in E]))
    X = [x for x in P if x.get("Fex_ref") is not None and x.get("Fex_pred") is not None]
    if X:
        o["F_S1 MAE"] = float(np.nanmean(np.concatenate([np.abs(x["Fex_pred"][:, 0] - x["Fex_ref"][:, 0]).ravel() for x in X])))
    return o



def main():
    res = {}
    for name, runs in MODELS.items():
        per = defaultdict(lambda: defaultdict(list))
        for rd, conj in runs:
            P = {}
            for f in (rd / "predictions.pkl", conj if Path(conj).is_absolute() else rd / conj):
                if Path(f).exists():
                    P.update(pickle.load(open(f, "rb")))
            for s in CONJ + SMALL:
                if P.get(s):
                    for k, v in metrics(P[s]).items():
                        per[s][k].append(v)
        res[name] = {s: {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in d.items()} for s, d in per.items()}
    json.dump(res, open(ROOT / "experiments/conj/model_B_vs_A.json", "w"), indent=1)
    for q in ("S1-3 MAE", "S1 R2", "F_S1 MAE"):
        print(f"\n== {q}")
        print(f"{'set':22s}" + "".join(f"{n:>12s}" for n in MODELS))
        for s in CONJ + SMALL:
            print(f"{s:22s}" + "".join(f"{res[n][s][q][0]:12.3f}" if s in res[n] and q in res[n][s] else f"{'-':>12s}" for n in MODELS))


if __name__ == "__main__":
    main()
