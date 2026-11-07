"""Production models (weighted loss, seeds 0-3): seed mean +- std and committee (seed-averaged prediction) per set.

Frame sets: S1-S3 MAE, S1 MAE, signed S1 error and S1 R^2 from predictions.pkl (identical frames and order in all runs).
Trajectories: S1-gap MAE and committee S0-rise MAE (S0 energy relative to t = 0) along the TD-wB97X-D S1 trajectories (experiments/md/results/<run>.pkl, frames before TDDFT
breaks down), per trajectory and as the mean over the validation (7) and test (5) trajectories. Per trajectory also the
committee MAE of the S1 potential energy relative to the S0 energy at t = 0 (S0 rise + S1 gap, the surface that drives S1
dynamics), the correlation between the committee S1-gap error and S0-rise error, and for the test trajectories the S1-gap
MAE of the benchmark LEH before active learning (main_leh_all_s0, Fig. 6c).
Writes experiments/al/committee.json.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARCH = {"leh": [f"prodw_leh_v4_s{s}" for s in range(4)], "softmin": [f"prodw_softmin_v4_s{s}" for s in range(4)]}
SETS = ["val", "test_random_offeq", "test_scaffold_offeq", "test_random_exc", "test_scaffold_exc", "test_large_offeq",
        "test_large_exc", "test_random_exc4", "test_scaffold_exc4", "test_large_exc4", "test_dye_noS_eq",
        "test_dye_noS_offeq", "test_lad_noS_eq", "test_lad_noS_offeq"]
VAL_TRAJ = ["styrene_s1", "azobenzene_s1", "cyclohexenone_s1", "hexatriene_s1", "cytosine_s1", "benzaldehyde_s1",
            "cinnamaldehyde_s1"]
TEST_TRAJ = ["acrolein_s1", "uracil_s1", "stilbene_s1", "hbdi_s1", "dmabn_s1"]


def gaps(P, key):
    g = np.array([np.pad(np.asarray(x[key], float), (0, 3))[:3] for x in P])
    g[g == 0] = np.nan
    return g


def scores(gp, gr):
    m = np.isfinite(gp) & np.isfinite(gr)
    s1 = m[:, 0]; d1 = gp[s1, 0] - gr[s1, 0]
    r2 = 1 - np.sum(d1 ** 2) / np.sum((gr[s1, 0] - gr[s1, 0].mean()) ** 2)
    return float(np.abs(gp - gr)[m].mean()), float(r2), float(np.abs(d1).mean()), float(d1.mean())


res = {"sets": {}, "trajectories": {}}
for s in SETS:
    res["sets"][s] = {}
    for a, runs in ARCH.items():
        G, ids = [], None
        for r in runs:
            P = pickle.load(open(ROOT / f"experiments/al/runs/{r}/predictions.pkl", "rb"))[s]
            i = [x["id"] for x in P]
            assert ids is None or i == ids, (r, s)
            ids = i; gr = gaps(P, "gaps_ref"); G.append(gaps(P, "gaps_pred"))
        per = np.array([scores(g, gr) for g in G])
        com = scores(np.nanmean(G, 0), gr)
        res["sets"][s][a] = dict(n=len(ids), mae=per[:, 0].tolist(), r2=per[:, 1].tolist(), s1_mae=per[:, 2].tolist(),
                                 mae_mean=float(per[:, 0].mean()), mae_std=float(per[:, 0].std(ddof=1)),
                                 r2_mean=float(per[:, 1].mean()), r2_std=float(per[:, 1].std(ddof=1)),
                                 s1_mae_mean=float(per[:, 2].mean()), s1_mae_std=float(per[:, 2].std(ddof=1)),
                                 s1_signed_mean=float(per[:, 3].mean()), committee_mae=com[0], committee_r2=com[1],
                                 committee_s1_mae=com[2], committee_s1_signed=com[3])
F = {r: pickle.load(open(ROOT / f"experiments/md/results/{r}.pkl", "rb"))["frames"] for rs in ARCH.values() for r in rs}
for key, T in (("validation", VAL_TRAJ), ("test", TEST_TRAJ)):
    res["trajectories"][key] = {}
    for a, runs in ARCH.items():
        ref = {t: F[runs[0]][t]["gaps_ref"][:, 0] for t in T}
        per = np.array([[np.mean(np.abs(F[r][t]["gaps_pred"][:, 0] - ref[t])) for t in T] for r in runs])
        com = [float(np.mean(np.abs(np.mean([F[r][t]["gaps_pred"][:, 0] for r in runs], 0) - ref[t]))) for t in T]
        sig = [float(np.mean(np.mean([F[r][t]["gaps_pred"][:, 0] for r in runs], 0) - ref[t])) for t in T]
        e0 = {t: F[runs[0]][t]["E0_ref"] - F[runs[0]][t]["E0_ref"][0] for t in T}  # S0 rise along the trajectory
        rise = [float(np.mean(np.abs(np.mean([F[r][t]["E0_pred"] - F[r][t]["E0_pred"][0] for r in runs], 0) - e0[t])))
                for t in T]
        pot, corr = [], []
        for t in T:
            dg = np.mean([F[r][t]["gaps_pred"][:, 0] for r in runs], 0) - ref[t]
            dr = np.mean([F[r][t]["E0_pred"] - F[r][t]["E0_pred"][0] for r in runs], 0) - e0[t]
            pot.append(float(np.mean(np.abs(dg + dr)))); corr.append(float(np.corrcoef(dg, dr)[0, 1]))
        res["trajectories"][key][a] = dict(per_trajectory_committee_s0rise=dict(zip(T, rise)),
                                           per_trajectory_committee_s1pot=dict(zip(T, pot)),
                                           per_trajectory_corr_gap_rise=dict(zip(T, corr)),
                                           per_trajectory_s0rise_ref_max=dict(zip(T, [float(e0[t].max()) for t in T])),
                                           per_trajectory_seeds=dict(zip(T, per.T.tolist())),
                                           per_trajectory_committee=dict(zip(T, com)),
                                           per_trajectory_committee_signed=dict(zip(T, sig)),
                                           mean_over_traj=per.mean(1).tolist(), mean=float(per.mean(1).mean()),
                                           std=float(per.mean(1).std(ddof=1)), committee=float(np.mean(com)))
B = pickle.load(open(ROOT / "experiments/md/results/main_leh_all_s0.pkl", "rb"))["frames"]
res["trajectories"]["test"]["benchmark_leh"] = dict(per_trajectory=dict(zip(TEST_TRAJ, [
    float(np.mean(np.abs(B[t]["gaps_pred"][:, 0] - F[ARCH["leh"][0]][t]["gaps_ref"][:, 0]))) for t in TEST_TRAJ])))
json.dump(res, open(ROOT / "experiments/al/committee.json", "w"), indent=1)

print(f"{'set':22s} {'n':>6s} | {'LEH seeds':>15s} {'comm':>6s} {'R2':>5s} {'S1':>5s} | {'softmin seeds':>15s} {'comm':>6s} {'R2':>5s} {'S1':>5s}")
for s, x in res["sets"].items():
    L, S = x["leh"], x["softmin"]
    print(f"{s:22s} {L['n']:6d} | {L['mae_mean']:.3f}+-{L['mae_std']:.3f}    {L['committee_mae']:.3f} {L['committee_r2']:.3f} "
          f"{L['committee_s1_mae']:.3f} | {S['mae_mean']:.3f}+-{S['mae_std']:.3f}    {S['committee_mae']:.3f} "
          f"{S['committee_r2']:.3f} {S['committee_s1_mae']:.3f}")
for key, x in res["trajectories"].items():
    L, S = x["leh"], x["softmin"]
    print(f"{key + ' traj.':22s} {'':6s} | {L['mean']:.3f}+-{L['std']:.3f}    {L['committee']:.3f} {'':5s} | "
          f"{S['mean']:.3f}+-{S['std']:.3f}    {S['committee']:.3f}")
    for t in L["per_trajectory_committee"]:
        print(f"   {t:19s} {'':6s} | {np.mean(L['per_trajectory_seeds'][t]):.3f} seeds   {L['per_trajectory_committee'][t]:.3f} "
              f"({L['per_trajectory_committee_signed'][t]:+.2f}) | {np.mean(S['per_trajectory_seeds'][t]):.3f} seeds   "
              f"{S['per_trajectory_committee'][t]:.3f} ({S['per_trajectory_committee_signed'][t]:+.2f})")
