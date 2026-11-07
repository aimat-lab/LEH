"""paper/numbers.tex: every result number quoted in the text, as LaTeX macros generated from the result files.

The prose uses the macros (e.g. \\vLEHrand), so rerunning this script updates all numbers without touching any sentence
the authors have edited. Never edit numbers.tex by hand; add new numbers here. Missing inputs give "\\todo{?}".
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "experiments/wb97"
M = {}


def load(p):
    p = ROOT / p
    return json.load(open(p)) if p.exists() else None


def put(name, val, fmt="{:.2f}"):
    M[name] = "\\todo{?}" if val is None or (isinstance(val, float) and not np.isfinite(val)) else fmt.format(val)


L = load("experiments/wb97/learning_curves.json") or {}
g = lambda m, s, q, z="all": L.get(m, {}).get(z, {}).get(s, {}).get(q, [None])[0]
for m, tag in [("leh", "LEH"), ("leh_L2", "LEHL"), ("softmin", "SM"), ("xmace", "XM"), ("vecmsani", "OMNI"), ("msani", "MSANI"),
               ("omnift", "OMNIFT")]:
    put(f"v{tag}rand", g(m, "test_random_offeq", "S1-3 MAE"))
    put(f"v{tag}scaf", g(m, "test_scaffold_offeq", "S1-3 MAE"))
    put(f"v{tag}exc", g(m, "test_random_exc", "S1-3 MAE"))
    put(f"v{tag}large", g(m, "test_large_offeq", "S1-3 MAE"))
    put(f"v{tag}largeexc", g(m, "test_large_exc", "S1-3 MAE"))
    put(f"f{tag}rand", g(m, "test_random_offeq", "F_S1 MAE"))
    put(f"f{tag}large", g(m, "test_large_offeq", "F_S1 MAE"))
    put(f"e{tag}rand", g(m, "test_random_offeq", "E0 meV/at"), "{:.1f}")
    put(f"e{tag}largeeq", g(m, "test_large_eq", "E0 meV/at"), "{:.1f}")
    put(f"fz{tag}large", g(m, "test_large_offeq", "F0 MAE"))
    put(f"v{tag}randtenk", g(m, "test_random_offeq", "S1-3 MAE", "10k"))
SP = load("experiments/wb97/size_parity.json") or {"models": {}}
for m, tag in [("leh", "LEH"), ("softmin", "SM"), ("xmace", "XM"), ("vecmsani", "OMNI"), ("leh_L2", "LEHL"), ("omnift", "OMNIFT")]:
    runs = SP["models"].get(m, [])
    for s, nm in [("dyes_eq", "dyes"), ("ladders_eq", "lad")]:
        put(f"v{tag}{nm}", float(np.mean([r[s]["all"]["s13"] for r in runs])) if runs else None)
B = load("experiments/conj/model_B_vs_A.json") or {}
gb = lambda k, s: B.get(k, {}).get(s, {}).get("S1-3 MAE", [None])[0]
put("vBLEHdyes", gb("B LEH", "test_dye_noS_eq")); put("vBSMdyes", gb("B softmin", "test_dye_noS_eq"))
put("vBLEHlad", gb("B LEH", "test_lad_noS_eq")); put("vBSMlad", gb("B softmin", "test_lad_noS_eq"))
put("vALEHdyesho", gb("A LEH", "test_dye_noS_eq")); put("vASMdyesho", gb("A softmin", "test_dye_noS_eq"))
put("vBLEHrand", gb("B LEH", "test_random_offeq")); put("vBLEHlarge", gb("B LEH", "test_large_offeq"))
FC = load("experiments/wb97/fc_eval.json") or {}
gf = lambda r, s: FC.get(r, {}).get(s, {}).get("F_S1 MAE")
put("fcLEHrand", gf("experiments/wb97/runs_main/main_leh_all_s0", "random_eq")); put("fcLEHlarge", gf("experiments/wb97/runs_main/main_leh_all_s0", "larger_eq"))
put("fcXMrand", gf("experiments/wb97/runs_lc/xmace128_suball", "random_eq")); put("fcXMlarge", gf("experiments/wb97/runs_lc/xmace128_suball", "larger_eq"))
put("fcSMrand", gf("experiments/wb97/runs_main/main_softmin_all_s0", "random_eq"))
r2 = FC.get("experiments/wb97/runs_main/main_leh_all_s0", {}).get("random_eq", {}).get("F_S1 R2"); put("fcLEHrtwo", r2)
D = load("experiments/wb97/diag_force_scaling.json") or {}
xs = D.get("experiments/wb97/runs_lc/xmace128_suball", D.get("xmace128_suball", {}))
put("slopeXMfifteen", xs.get("test_large_offeq|15|F0", {}).get("slope")); put("slopeXMtwenty", xs.get("test_large_offeq|20|F0", {}).get("slope"))
# dimers vs TDDFT (same pairs): full-set models (seed mean where two seeds exist); 10k heads as fallback
FRAG = {"leh": [W / f"runs_main/main_leh_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
        "softmin": [W / f"runs_main/main_softmin_all_s{i}/predictions_frag.pkl" for i in (0, 1)],
        "xmace": [ROOT / "experiments/fragments/full_xmace/predictions.pkl"],
        "vecmsani": [ROOT / "experiments/fragments/full_vecmsani/predictions.pkl"],
        "omnift": [ROOT / "experiments/fragments/full_omnift/predictions.pkl"]}


def frag_seeds(m):
    out = []
    for f in FRAG[m]:
        if f.exists():
            P = pickle.load(open(f, "rb"))
            out.append({x["id"]: np.asarray(x["gaps_pred"], float) for k in ("test_frag_mono", "test_frag_dimer") for x in P.get(k, [])})
    return out


try:
    R = json.load(open(ROOT / "experiments/frag_tddft/results_pairs.json"))
    H = pickle.load(open(ROOT / "experiments/fragments/heads_cache.pkl", "rb"))
    for m, tag in [("leh", "LEH"), ("softmin", "SM"), ("xmace", "XM"), ("vecmsani", "OMNI"), ("omnift", "OMNIFT")]:
        seeds = frag_seeds(m) or ([H[m]] if m in H else [])
        for kind in ("AB", "AA"):
            e = [np.mean([np.mean(np.abs(S[r["id"]][:3] - np.asarray(r["dimer"][:3]))) for r in R if r["kind"] == kind and r["id"] in S])
                 for S in seeds]
            put(f"dim{tag}{kind}", float(np.mean(e)) if e else None)
        e = [np.mean([np.mean(np.abs(S[r["id"]][:3] - np.asarray(r["dimer"][:3]))) for r in R if r["id"] in S]) for S in seeds]
        put(f"dim{tag}", float(np.mean(e)) if e else None)
        # the same molecules alone (monomer test set; nan-safe: k=1 gives only M eigenvalues for diatomics)
        e = [np.mean([np.nanmean(np.abs(np.asarray(x["gaps_pred"][:3], float) - np.asarray(x["gaps_ref"][:3], float)))
                      for x in pickle.load(open(f, "rb"))["test_frag_mono"]]) for f in FRAG[m] if f.exists()]
        put(f"mono{tag}", float(np.mean(e)) if e else None)
    for kind, d, w in (("AB", 2.5, "short"), ("AA", 2.5, "short"), ("AB", 5.0, "five"), ("AA", 5.0, "five"),
                       ("AB", 10.0, "ten"), ("AA", 10.0, "ten")):
        v = [np.mean(np.abs(r["dev"][:3])) for r in R if r["kind"] == kind and r["d"] == d]
        put(f"tdshift{kind}{w}", 1000 * float(np.mean(v)), "{:.0f}")
    put("nDimerPairs", len({r["id"] for r in R}), "{:,}")
except FileNotFoundError:
    pass
# QCDGE size extrapolation (runs_final): S1-S3 / S1-S10 MAE and S1-S3 R2 on 10 heavy atoms, oscillator strengths
Q = (load("experiments/qcdge/summary_final.json") or {}).get("models", {})
gq = lambda m, s, q: (Q.get(m, {}).get(s, {}).get(q) or [None])[0]
for m, tag in [("leh", "LEH"), ("softmin", "SM"), ("sum", "SUM"), ("mean", "MEAN"), ("xmace", "XM"), ("vecmsani", "OMNI"),
               ("msani", "MSANI"), ("leh_osc", "LEHosc"), ("rand_leh", "LEHrnd"), ("rand_softmin", "SMrnd")]:
    put(f"q{tag}val", gq(m, "val", "S1-3 MAE"), "{:.3f}"); put(f"q{tag}nine", gq(m, "test_9", "S1-3 MAE"), "{:.3f}")
    put(f"q{tag}ten", gq(m, "test_10", "S1-3 MAE"), "{:.3f}"); put(f"q{tag}tenall", gq(m, "test_10", "S1-10 MAE"), "{:.3f}")
    put(f"q{tag}tenrtwo", gq(m, "test_10", "S1-3 R2"))
for s, w in (("val", "val"), ("test_9", "nine"), ("test_10", "ten")):
    put(f"qOscRtwo{w}", gq("leh_osc", s, "osc R2 S1")); put(f"qOscRecall{w}", gq("leh_osc", s, "bright recall"))
NK = load("experiments/wb97/summary_runs_nk.json") or {}
for c, w in (("G10k_K1", "TenOne"), ("G10k_K2", "TenTwo"), ("G10k_K3", "TenThree"), ("G20k_K1", "TwentyOne"), ("G20k_K2", "TwentyTwo")):
    put("nk" + w, NK.get(f"nk_{c}", {}).get("test_random_offeq", {}).get("S1-6 MAE", [None])[0])
P2 = load("experiments/al/prelabel_s1r2.json") or {}
put("alRtwoN", P2.get("n"), "{:,}"); put("alRtwoMAE", (P2.get("all") or {}).get("mae"))
put("alRtwoTwistHigh", ((P2.get("by_twist") or {}).get("75-91") or {}).get("mae"))
put("alRtwoTwistLow", ((P2.get("by_twist") or {}).get("0-30") or {}).get("mae"))
DS = load("experiments/wb97/dataset_stats.json") or {}
for k, name in [("n_molecules_v1", "nMolTotal"), ("n_entries_v1", "nGeomTotal"), ("n_train_molecules", "nTrainMol"),
                ("n_train_geometries", "nTrainGeom"), ("n_train_excmd", "nTrainExc"), ("n_val_molecules", "nValMol"),
                ("n_zinc_molecules", "nLargeMol"), ("n_dyes", "nDyes"), ("n_ladders", "nLadders"), ("n_conj_entries", "nConjGeom")]:
    put(name, DS.get(k), "{:,}")
T = DS.get("tests", {})
put("nTestRandom", T.get("test_random_offeq"), "{:,}"); put("nTestScafMol", T.get("test_scaffold_eq"), "{:,}")
put("nTestExc", T.get("test_random_exc"), "{:,}")
put("nOffeqTotal", (DS.get("kinds") or {}).get("offeq"), "{:,}"); put("nExcTotal", (DS.get("kinds") or {}).get("excmd"), "{:,}")
OUT = load("experiments/wb97/outliers.json") or {}
put("nOutlier", OUT.get("s1_below_m5"), "{:d}"); put("nOutlierTotal", OUT.get("n"), "{:,}")
# final released dataset (v4 = v1 + conjugated test sets + active-learning rounds 2-4; experiments/al/dataset_stats_v4.py)
D4 = load("experiments/al/dataset_stats_v4.json") or {}
put("nGeomAll", D4.get("n_geometries"), "{:,}"); put("nMolAll", D4.get("n_molecules"), "{:,}")
put("nGeomFex", D4.get("n_with_Fex_S1"), "{:,}"); put("nGeomExcAll", D4.get("n_excited_sampled"), "{:,}")
for el in ("N", "O", "F", "S"):
    put(f"nMol{el}", (D4.get("elements") or {}).get(el), "{:,}")
T4 = D4.get("train", {})
put("nTrainGeomProd", T4.get("n"), "{:,}"); put("pctTrainLowGap", 100 * T4["share_below_1"] if T4 else None, "{:.0f}")
put("pctTrainLowGapTwo", 100 * T4["share_below_2"] if T4 else None, "{:.0f}")
# low-gap bias of the uniform-loss production models (validation frames with TDDFT S1 < 1 eV; compare_loss.json)
CL = load("experiments/al/compare_loss.json") or {}
v = [x["S1 0-1"]["signed"] for x in CL.get("runs", {}).values() if x["group"] == "uniform"]
put("lgUniformLow", float(np.mean(v)) if v else None)
A4 = D4.get("al", {})
put("nALGeom", A4.get("n"), "{:,}"); put("nALMol", A4.get("n_molecules"), "{:,}"); put("nALSoneLow", A4.get("s1_below_1"), "{:,}")
for r, w in (("2", "Two"), ("3", "Three"), ("4", "Four")):
    put(f"nALR{w.lower()}", (D4.get("by_type", {}).get(f"al_r{r}") or {}).get("n"), "{:,}")
    P = load(f"experiments/al/prelabel_s1r{r}.json") or {}
    put(f"alR{w.lower()}MAE", (P.get("all") or {}).get("mae")); put(f"alR{w.lower()}Signed", (P.get("all") or {}).get("signed"))
    put(f"alR{w.lower()}TwistHigh", ((P.get("by_twist") or {}).get("75-91") or {}).get("mae"))
# active-learning criterion on the validation trajectories: sampling models after each round (mean over the two seeds)
for f, w in (("convergence_r1_benchmark.json", "One"), ("convergence_r2.json", "Two"), ("convergence_r3.json", "Three"),
             ("convergence_r4.json", "Four")):
    C = (load(f"experiments/al/{f}") or {}).get("runs", {})
    for q in ("gap", "rise"):
        v = [float(np.mean([t[q] for t in x["trajectories"].values()])) for x in C.values()]
        put(f"al{q.capitalize()}R{w.lower()}", float(np.mean(v)) if v else None)
CP = {}
for f in ("convergence_prod_v4.json", "convergence_prod_v4_softmin_s1.json"):  # uniform-loss production models (stop decision)
    CP.update((load(f"experiments/al/{f}") or {}).get("runs", {}))
put("alGapProdUniform", float(np.mean([np.mean([t["gap"] for t in x["trajectories"].values()]) for x in CP.values()])) if CP else None)
ST = [x["stability"] for f in ("convergence_r2.json", "convergence_r3.json", "convergence_r4.json") for x in
      (load(f"experiments/al/{f}") or {}).get("runs", {}).values()] + [x["stability"] for x in CP.values()]
put("alStabN", ST[0]["n"] if ST else None, "{:d}")
put("alStabDrift", 1000 * max(s["max_drift"] for s in ST) if ST else None, "{:.0f}")
# production-model outliers (experiments/al/outliers_prod.py; bounded LEH blocks and Direct-softmin)
OP = load("experiments/al/outliers_prod.json") or {}
for a, tag in (("leh", "LEH"), ("softmin", "SM")):
    o = OP.get(a, {})
    put(f"oProd{tag}N", o.get("n"), "{:,}"); put(f"oProd{tag}below", o.get("s1_below_m5"), "{:d}")
    put(f"oProd{tag}min", -o["min_s1"] if o.get("min_s1") is not None else None, "{:.0f}")
# uniform-loss production models (two seeds) along the test trajectories (experiments/al/significance.json)
SGT = (load("experiments/al/significance.json") or {}).get("seeds", {}).get("test trajectories", {})
put("uLEHTestTraj", (SGT.get("prod_leh") or {}).get("mean")); put("uSMTestTraj", (SGT.get("prod_softmin") or {}).get("mean"))
# Hamiltonian blocks of the unbounded benchmark LEH models (experiments/wb97/outlier_blocks.py)
OB = load("experiments/wb97/outlier_blocks.json") or {}
if OB:
    pm = [o["pair_max"] for o in OB["outliers"]]
    put("oPairOutMin", min(pm), "{:.0f}"); put("oPairOutMax", max(pm), "{:.0f}")
    put("oPairOther", max(x["pair_max_other"] for x in OB["per_seed"].values()), "{:.1f}")
# QCDGE random-split control (experiments/qcdge/prepare.py): share of nine-heavy-atom training molecules (none has ten)
QI = load("experiments/qcdge/split_info.json") or {}
rn = QI.get("rand_train_n_heavy", {})
put("qRandNinePct", 100 * rn.get("9", 0) / sum(rn.values()) if rn else None, "{:.0f}")
# number of sulfur-free dyes in the transfer figure (all dyes, size_parity.json)
put("nDyesNoS", (SP["models"].get("leh") or [{}])[0].get("dyes_eq", {}).get("all", {}).get("n"), "{:,}")
# decision on the weighted loss (experiments/al/compare_loss.py, seeds 0-1, validation data): relative changes in %
CL = (load("experiments/al/compare_loss.json") or {}).get("summary", {})
for g, tag in (("leh", "LEH"), ("softmin", "SM"), ("all", "All")):
    x = CL.get(g, {})
    if x:
        put(f"loss{tag}lowgap", -100 * x["lowgap_bias_change"], "{:.0f}")   # decrease of the low-gap bias
        put(f"loss{tag}traj", 100 * x["traj_gap_change"], "{:+.0f}" if g != "all" else "{:+.1f}")
        put(f"loss{tag}val", 100 * x["val_S13_change"], "{:+.0f}")
# production models (weighted loss, four seeds; experiments/al/committee.py): seed mean, std and committee
PC = load("experiments/al/committee.json") or {}
PS, PT = PC.get("sets", {}), PC.get("trajectories", {})
for a, tag in (("leh", "LEH"), ("softmin", "SM")):
    for s, w in (("test_random_offeq", "rand"), ("test_scaffold_offeq", "scaf"), ("test_large_offeq", "large"),
                 ("test_large_exc", "largeexc"), ("test_random_exc4", "alfr"), ("test_dye_noS_eq", "dyes"),
                 ("test_lad_noS_eq", "lad")):
        x = PS.get(s, {}).get(a)
        if x:
            put(f"p{tag}{w}", x["mae_mean"], "{:.3f}"); put(f"p{tag}{w}std", x["mae_std"], "{:.3f}")
            put(f"p{tag}{w}comm", x["committee_mae"], "{:.3f}"); put(f"p{tag}{w}Sone", x["committee_s1_mae"])
            put(f"p{tag}{w}Rtwo", x["committee_r2"])
    for k, w in (("test", "TestTraj"), ("validation", "ValTraj")):
        x = PT.get(k, {}).get(a)
        if x:
            put(f"p{tag}{w}", x["mean"]); put(f"p{tag}{w}std", x["std"]); put(f"p{tag}{w}comm", x["committee"])
            put(f"p{tag}{w}max", max(x["mean_over_traj"]))
    x = PT.get("test", {}).get(a)
    if x:  # committee signed S1 error along the three larger chromophores (stilbene, HBDI, DMABN)
        sg = [x["per_trajectory_committee_signed"][t] for t in ("stilbene_s1", "hbdi_s1", "dmabn_s1")]
        put(f"p{tag}biasMin", min(sg)); put(f"p{tag}biasMax", max(sg))
        rs = [x["per_trajectory_committee_s0rise"][t] for t in ("stilbene_s1", "hbdi_s1", "dmabn_s1")]
        put(f"p{tag}riseMin", min(rs)); put(f"p{tag}riseMax", max(rs))
if PT.get("test"):
    put("pRiseRefMin", min(PT["test"]["leh"]["per_trajectory_s0rise_ref_max"][t] for t in ("stilbene_s1", "hbdi_s1", "dmabn_s1")), "{:.1f}")
    put("pRiseRefMax", max(PT["test"]["leh"]["per_trajectory_s0rise_ref_max"][t] for t in ("stilbene_s1", "hbdi_s1", "dmabn_s1")), "{:.1f}")
    rs = [PT["test"][a]["per_trajectory_committee_s0rise"][t] for a in ("leh", "softmin") for t in ("stilbene_s1", "hbdi_s1", "dmabn_s1")]
    put("pRiseErrMin", min(rs)); put("pRiseErrMax", max(rs))
    put("pSmallTrajMax", max(PT["test"][a]["per_trajectory_committee"][t] for a in ("leh", "softmin")
                             for t in ("acrolein_s1", "uracil_s1")))
if PS:  # committee gain over the seed mean (S1-S3 MAE), range over the frame test sets
    g = [1 - PS[s][a]["committee_mae"] / PS[s][a]["mae_mean"] for s in PS if s != "val" for a in ("leh", "softmin")]
    put("pCommGainMin", 100 * min(g), "{:.0f}"); put("pCommGainMax", 100 * max(g), "{:.0f}")
SG = (load("experiments/al/significance.json") or {}).get("seeds", {}).get("val S1<2 eV signed S1 error [molecule]", {})
for k, tag in (("prod_leh", "LEHuni"), ("prodw_leh", "LEHw"), ("prod_softmin", "SMuni"), ("prodw_softmin", "SMw")):
    put(f"pLowGap{tag}", (SG.get(k) or {}).get("mean"))  # mean signed S1 error, validation geometries with S1 < 2 eV
STW = [list((load(f"experiments/al/convergence_prodw_{a}_v4_s{s}.json") or {}).get("runs", {}).values())
       for a in ("leh", "softmin") for s in range(4)]
STW = [r[0]["stability"] for r in STW if r]
put("pStabDrift", 1000 * max(s["max_drift"] for s in STW) if STW else None, "{:.0f}")
put("pStabN", STW[0]["n"] if STW else None, "{:d}")
# S1 alone on the conjugated test sets (benchmark models A and B; experiments/conj/model_B_vs_A.json)
gbs = lambda k, s: B.get(k, {}).get(s, {}).get("S1 MAE", [None])[0]
for k, tag in (("A LEH", "ALEH"), ("A softmin", "ASM"), ("B LEH", "BLEH"), ("B softmin", "BSM")):
    put(f"v{tag}dyesSone", gbs(k, "test_dye_noS_eq")); put(f"v{tag}ladSone", gbs(k, "test_lad_noS_eq"))
    put(f"v{tag}ladSthree", gb(k, "test_lad_noS_eq")); put(f"v{tag}dyesSthree", gb(k, "test_dye_noS_eq"))
# S2-S3 alone, exact from S1-S3 and S1 MAE because every geometry of these sets has all three states:
# MAE(S2-S3) = (3 MAE(S1-S3) - MAE(S1)) / 2
s23 = lambda s13, s1: (3 * s13 - s1) / 2 if s13 is not None and s1 is not None else None
for k, tag in (("A LEH", "ALEH"), ("A softmin", "ASM"), ("B LEH", "BLEH"), ("B softmin", "BSM")):
    put(f"v{tag}dyesStwothree", s23(gb(k, "test_dye_noS_eq"), gbs(k, "test_dye_noS_eq")))
    put(f"v{tag}ladStwothree", s23(gb(k, "test_lad_noS_eq"), gbs(k, "test_lad_noS_eq")))
for a, tag in (("leh", "LEH"), ("softmin", "SM")):
    for s, w in (("test_lad_noS_eq", "lad"), ("test_dye_noS_eq", "dyes"), ("test_large_offeq", "large")):
        x = PS.get(s, {}).get(a)
        if x:
            put(f"p{tag}{w}Stwothree", s23(x["committee_mae"], x["committee_s1_mae"]))
# separability figure (experiments/fig_separability.py): dimer MAE vs TDDFT by state group, and of the monomer union
SN = load("experiments/separability_numbers.json") or {}
for m, tag in (("leh", "LEH"), ("softmin", "SM"), ("xmace", "XM"), ("vecmsani", "OMNI")):
    x = SN.get(m, {})
    put(f"sep{tag}one", x.get("dimer_1_1")); put(f"sep{tag}twothree", x.get("dimer_2_3"))
    put(f"sep{tag}foursix", x.get("dimer_4_6")); put(f"sep{tag}union", x.get("union_s13"))
# Fig. 6: decomposition of the S1 potential along the test trajectories of the three larger chromophores (committee.json)
BIG = ("stilbene_s1", "hbdi_s1", "dmabn_s1")
for a, tag in (("leh", "LEH"), ("softmin", "SM")):
    x = PT.get("test", {}).get(a, {})
    if "per_trajectory_committee_s1pot" in x:
        gg = [x["per_trajectory_committee"][t] for t in BIG]; q = [x["per_trajectory_committee_s1pot"][t] for t in BIG]
        put(f"p{tag}gapMin", min(gg)); put(f"p{tag}gapMax", max(gg)); put(f"p{tag}potMin", min(q)); put(f"p{tag}potMax", max(q))
    # validation trajectories, S0 change (Fig. 6b): seed mean of the mean over trajectories
    v = [list((load(f"experiments/al/convergence_prodw_{a}_v4_s{s_}.json") or {}).get("runs", {}).values()) for s_ in range(4)]
    v = [float(np.mean([y["rise"] for y in r[0]["trajectories"].values()])) for r in v if r]
    put(f"p{tag}ValRise", float(np.mean(v)) if v else None)
cc = [abs(PT["test"][a]["per_trajectory_corr_gap_rise"][t]) for a in ("leh", "softmin") for t in ("stilbene_s1", "hbdi_s1")
      if "per_trajectory_corr_gap_rise" in PT.get("test", {}).get(a, {})]
put("pCorrMin", min(cc) if cc else None); put("pCorrMax", max(cc) if cc else None)
bl = PT.get("test", {}).get("benchmark_leh", {}).get("per_trajectory")
put("bLEHTestTraj", float(np.mean(list(bl.values()))) if bl else None)
# Table 1: S0 energy error of the larger molecules at thermal geometries (the column of the table)
for m, tag in [("leh", "LEH"), ("softmin", "SM"), ("xmace", "XM"), ("vecmsani", "OMNI")]:
    put(f"e{tag}large", L.get(m, {}).get("all", {}).get("test_large_offeq", {}).get("E0 meV/at", [None])[0], "{:.1f}")
# Table 2: S1 R2 on the held-out dye scaffolds and ladder molecules (models A and B)
gr2 = lambda k, s: B.get(k, {}).get(s, {}).get("S1 R2", [None])[0]
for k, tag in (("A LEH", "ALEH"), ("A softmin", "ASM"), ("B LEH", "BLEH"), ("B softmin", "BSM")):
    put(f"v{tag}dyesRtwo", gr2(k, "test_dye_noS_eq")); put(f"v{tag}ladRtwo", gr2(k, "test_lad_noS_eq"))
# Table 3: ten-heavy-atom molecules with scaffolds absent from training
for m, tag in (("leh", "LEH"), ("softmin", "SM")):
    put(f"q{tag}tenscaf", gq(m, "test_10_novelscaf", "S1-3 MAE"), "{:.3f}")
# Discussion: power law MAE = a N^b fitted (log-log, all five subsets) to the S1-S3 MAE of random test molecules; number
# of training molecules at which the fit reaches 0.1 eV (millions) and its ratio to the full training set
NMOL = {"1k": 1000, "3k": 3000, "10k": 10000, "30k": 30000, "all": DS.get("n_train_molecules")}
nfit = {}
for m in ("leh", "softmin", "xmace"):
    pts = [(NMOL[z], L.get(m, {}).get(z, {}).get("test_random_offeq", {}).get("S1-3 MAE", [None])[0]) for z in NMOL]
    pts = [(n, y) for n, y in pts if n and y]
    if len(pts) == 5:
        b, a = np.polyfit(np.log([n for n, _ in pts]), np.log([y for _, y in pts]), 1)
        nfit[m] = float(np.exp((np.log(0.1) - a) / b))
put("lcMolMin", min(nfit.values()) / 1e6 if len(nfit) == 3 else None, "{:.1f}")
put("lcMolMax", max(nfit.values()) / 1e6 if len(nfit) == 3 else None, "{:.0f}")
put("lcFoldMin", min(nfit.values()) / NMOL["all"] if len(nfit) == 3 and NMOL["all"] else None, "{:.0f}")
put("lcFoldMax", max(nfit.values()) / NMOL["all"] if len(nfit) == 3 and NMOL["all"] else None, "{:.0f}")
# population loss on the LEH eigenvectors (10k setting; experiments/eigvec/tddft/eval_pop.py, results_runs_abl_pop.json):
# seed means of the baseline (abl_leh_s0-2) and of the population loss with weight 1 (abl_leh_pop_s0-2)
EP = (load("experiments/eigvec/tddft/eval_pop.json") or {}).get("variants", {})
ep = lambda v, k: (EP.get(v, {}).get(k) or {}).get("mean")
for v, tag in (("baseline", "Base"), ("pop", "Pop")):
    put(f"popSval{tag}", ep(v, "val|S1|all|S")); put(f"popSlad{tag}", ep(v, "set460|S1|ladder|S"))
    put(f"popLad{tag}", ep(v, "transfer|ladders|mae1"))
    put(f"popHole{tag}", ep(v, "val|S1|all|S_hole")); put(f"popElec{tag}", ep(v, "val|S1|all|S_elec"))
put("popSvalNull", ep("baseline", "val|S1|all|null"))
RP = load("experiments/wb97/results_runs_abl_pop.json") or {}
POPRUNS = {"Base": [f"abl_leh_s{s}" for s in range(3)], "Pop": [f"abl_leh_pop_s{s}" for s in range(3)]}
for tag, runs in POPRUNS.items():
    ok = all(r in RP for r in runs)
    put(f"popRand{tag}", float(np.mean([np.mean([RP[r]["test_random_offeq"][f"mae_S{n}"] for n in (1, 2, 3)])
                                        for r in runs])) if ok else None, "{:.3f}")
    put(f"popE{tag}", float(np.mean([RP[r]["test_random_offeq"]["mae_E0_per_atom_meV"] for r in runs])) if ok else None,
        "{:.1f}")
SELP = ROOT / "experiments/eigvec/tddft/selection_abl10k.json"
put("nPopGeom", len(json.load(open(SELP))) if SELP.exists() else None, "{:,}")
# Zenodo release: calculations with archived raw TURBOMOLE files (release/export_dataset.py)
RS = load("release/out_v4/aux/stats.json") or {}
put("nRawCalc", sum(RS["raw_per_group"].values()) if RS else None, "{:,}")
lines = ["% generated by experiments/make_numbers.py -- do not edit by hand"]
assert all(k.isalpha() for k in M), [k for k in M if not k.isalpha()]  # LaTeX macro names: letters only
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
(ROOT / "paper/numbers.tex").write_text("\n".join(lines) + "\n")
print(len(M), "macros;", sum("todo" in v for v in M.values()), "missing")
