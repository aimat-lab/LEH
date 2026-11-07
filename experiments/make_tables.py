"""LaTeX tables of the paper, generated from the result files (never edit the .tex outputs by hand; change this script).

paper/tables/table1_benchmark.tex  main benchmark, models trained on the full training set (76.6k molecules): S1-S3
    excitation-energy MAE on random, unseen-scaffold, S1-sampled and larger test molecules, S1 force MAE and S0
    energy MAE; mean +- std over seeds where available; best value per column in bold.
paper/tables/table2_conj.tex       conjugated dyes (held-out scaffolds) and ladders at equilibrium: model A (never saw
    conjugated molecules), model B (training dyes added) and the production models (B data + active learning, weighted
    loss; four seeds and committee), S1-S3 MAE, S1 MAE and S1 R2.
paper/tables/si_conj.tex           models A and B on the conjugated sets, equilibrium and thermal geometries (SI).
paper/tables/si_production.tex     production models on every test set and along the S1 trajectories (SI).
paper/tables/table3_qcdge.tex      QCDGE size extrapolation (if experiments/qcdge/summary_final.json exists).
paper/tables/si_benchmark_full.tex every test set and target with MAE and R2 (SI).
paper/tables/si_population.tex     population loss on the LEH eigenvectors, 10k setting (SI).
Inputs: experiments/wb97/learning_curves.json, experiments/conj/model_B_vs_A.json, experiments/qcdge/summary_final.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/tables"; OUT.mkdir(parents=True, exist_ok=True)
L = json.load(open(ROOT / "experiments/wb97/learning_curves.json"))
NAMES = {"leh": "LEH", "leh_L2": "LEH, $L=2$", "softmin": "Direct-softmin", "xmace": "X-MACE", "vecmsani": "OMNI-P2x arch.",
         "msani": "MS-ANI"}


def fmt(v, nd=3):
    if v is None or not np.isfinite(v[0]):
        return "--"
    m, s, n = v
    if n > 1 and round(s, nd) == 0:  # spread below the printed precision
        return f"{m:.{nd}f}" + "$\\pm{<}" + f"{10 ** -nd:g}" + "$"
    return f"{m:.{nd}f}" + (f"$\\pm${s:.{nd}f}" if n > 1 else "")


def fq(v, q, nd=3):
    """Errors in eV and eV/A are printed in meV and meV/A without decimals; other quantities with nd decimals."""
    if v is not None and ("MAE" in q or q in ("mae", "signed")) and "meV" not in q and np.isfinite(v[0]):
        return fmt([1000 * v[0], 1000 * v[1], v[2]], 0)
    return fmt(v, nd)


def table1():
    cols = [("test_random_offeq", "S1-3 MAE", 3), ("test_scaffold_offeq", "S1-3 MAE", 3), ("test_random_exc", "S1-3 MAE", 3),
            ("test_large_offeq", "S1-3 MAE", 3), ("test_random_offeq", "F_S1 MAE", 3), ("test_large_offeq", "F_S1 MAE", 3),
            ("test_random_offeq", "E0 meV/at", 1), ("test_large_offeq", "E0 meV/at", 1)]
    models = ["leh", "leh_L2", "softmin", "xmace", "vecmsani"]
    vals = {m: [L.get(m, {}).get("all", {}).get(s, {}).get(q) for s, q, _ in cols] for m in models}
    best = [min((vals[m][j][0] for m in models if vals[m][j]), default=None) for j in range(len(cols))]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{$S_1$--$S_3$ MAE (meV)} & \multicolumn{2}{c}{$S_1$ forces (meV/\AA)} & \multicolumn{2}{c}{$S_0$ (meV/atom)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"Model & random & scaffold & $S_1$-sampl. & larger & random & larger & random & larger \\", r"\midrule"]
    for m in models:
        cells = []
        for j, (_, _, nd) in enumerate(cols):
            v = vals[m][j]; c = fq(v, cols[j][1], nd)
            cells.append(f"\\textbf{{{c}}}" if v and best[j] is not None and abs(v[0] - best[j]) < 1e-12 else c)
        lines.append(NAMES[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table1_benchmark.tex").write_text("\n".join(lines) + "\n")


def table2():
    """Conjugated test sets at equilibrium. Bold: lowest MAE / highest R2 per column within each group of comparable
    rows (model A, model B, production seed means, production committees); ties at the printed precision are all bold."""
    f = ROOT / "experiments/conj/model_B_vs_A.json"
    if not f.exists():
        return
    R = json.load(open(f))
    C = json.load(open(ROOT / "experiments/al/committee.json"))["sets"]
    cols = [("test_dye_noS_eq", "S1-3 MAE"), ("test_dye_noS_eq", "S1 MAE"), ("test_dye_noS_eq", "S1 R2"),
            ("test_lad_noS_eq", "S1-3 MAE"), ("test_lad_noS_eq", "S1 MAE"), ("test_lad_noS_eq", "S1 R2")]
    # rows: (group, label in the table, [values per column as [mean, std, n]])
    rows = []
    for key, lab, grp in [("A LEH", "LEH", "A"), ("A softmin", "Direct-softmin", "A"), ("A X-MACE", "X-MACE", "A"),
                          ("A OMNI arch", "OMNI-P2x arch.", "A"), ("A OMNI-P2x ft", "OMNI-P2x, fine-tuned", "A"),
                          ("B LEH", "LEH", "B"), ("B softmin", "Direct-softmin", "B")]:
        rows.append((grp, lab, [R.get(key, {}).get(s_, {}).get(q) for s_, q in cols]))
    key = {"S1-3 MAE": ("mae", "committee_mae"), "S1 MAE": ("s1_mae", "committee_s1_mae"), "S1 R2": ("r2", "committee_r2")}
    for a, lab in [("leh", "LEH"), ("softmin", "Direct-softmin")]:
        seeds = [[float(np.mean(C[s_][a][key[q][0]])), float(np.std(C[s_][a][key[q][0]], ddof=1)), len(C[s_][a][key[q][0]])]
                 for s_, q in cols]
        comm = [[C[s_][a][key[q][1]], 0.0, 1] for s_, q in cols]
        rows += [("P", lab, seeds), ("Pc", f"{lab}, committee", comm)]
    rows = [rows[k] for k in (0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10)]  # production: LEH, softmin seeds, then the committees

    def shown(v, q):  # value at the printed precision (meV integers, R2 two decimals)
        return round(1000 * v[0]) if "MAE" in q else round(v[0], 2)
    best = {}
    for g in ("A", "B", "P", "Pc"):
        for c, (s_, q) in enumerate(cols):
            vals = [shown(r[2][c], q) for r in rows if r[0] == g and r[2][c] is not None]
            best[(g, c)] = (max if "R2" in q else min)(vals) if vals else None
    head = {"A": r"\multirow{5}{*}{benchmark}", "B": r"\multirow{2}{*}{+ training dyes}",
            "P": r"\multirow{4}{*}{\shortstack[l]{production: + training dyes\\+ active learning}}"}
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             r" & & \multicolumn{3}{c}{held-out dye scaffolds} & \multicolumn{3}{c}{ladder molecules} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Training data & Model & $S_1$--$S_3$ MAE & $S_1$ MAE & $S_1$ $R^2$ & $S_1$--$S_3$ MAE & $S_1$ MAE & $S_1$ $R^2$ \\",
             r" & & (meV) & (meV) & & (meV) & (meV) & \\"]
    prev = None
    for g, lab, vals in rows:
        if g in ("A", "B", "P") and g != prev:
            lines.append(r"\midrule")
        first = g in head and g != prev and not (g == "P" and prev == "Pc")
        cells = []
        for c, (s_, q) in enumerate(cols):
            v = vals[c]; t = fq(v, q, 2)
            cells.append(f"\\textbf{{{t}}}" if v is not None and shown(v, q) == best[(g, c)] else t)
        lines.append(f"{head[g] if first else ''} & {lab} & " + " & ".join(cells) + r" \\")
        prev = g
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table2_conj.tex").write_text("\n".join(lines) + "\n")


def si_conj():
    """Models A and B on the conjugated test sets at equilibrium and thermal geometries (SI; the old Table 2 + S1 MAE)."""
    R = json.load(open(ROOT / "experiments/conj/model_B_vs_A.json"))
    cols = [(s_, q) for q in ("S1-3 MAE", "S1 MAE", "S1 R2") for s_ in ("test_dye_noS_eq", "test_dye_noS_offeq",
                                                                        "test_lad_noS_eq", "test_lad_noS_offeq")]
    rows = [("A LEH", "LEH"), ("A softmin", "Direct-softmin"), ("A X-MACE", "X-MACE"), ("A OMNI arch", "OMNI-P2x arch."),
            ("A OMNI-P2x ft", "OMNI-P2x, fine-tuned"), ("B LEH", "LEH"), ("B softmin", "Direct-softmin")]
    hdr = " & ".join(["dyes eq.", "dyes therm.", "ladders eq.", "ladders therm."] * 3)
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             r" & & \multicolumn{4}{c}{$S_1$--$S_3$ MAE (meV)} & \multicolumn{4}{c}{$S_1$ MAE (meV)} & \multicolumn{4}{c}{$S_1$ $R^2$} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}", r"Training data & Model & " + hdr + r" \\", r"\midrule"]
    for k, (key, lab) in enumerate(rows):
        grp = r"\multirow{5}{*}{benchmark}" if k == 0 else r"\multirow{2}{*}{+ training dyes}" if k == 5 else ""
        if k == 5:
            lines.append(r"\midrule")
        r = R.get(key, {})
        lines.append(f"{grp} & {lab} & " + " & ".join(fq(r.get(s_, {}).get(q), q, 2) for s_, q in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_conj.tex").write_text("\n".join(lines) + "\n")


def si_loss():
    """Uniform vs weighted excitation-energy loss for the production models (experiments/al/significance.json):
    seed mean +- std (uniform: two seeds, weighted: four seeds) on the validation criteria and all test sets (SI)."""
    G = json.load(open(ROOT / "experiments/al/significance.json"))["seeds"]
    rows = [("val [molecule]", "validation, $S_1$--$S_3$ MAE"),
            ("val S1<2 eV signed S1 error [molecule]", "validation, $S_1<2$\\,eV, signed $S_1$ error"),
            ("validation trajectories", "validation trajectories, $S_1$ MAE"), None,
            ("test_random_offeq [molecule]", "random, thermal"), ("test_scaffold_offeq [molecule]", "unseen scaffolds, thermal"),
            ("test_random_exc [molecule]", "random, $S_1$-sampled"), ("test_large_offeq [molecule]", "larger, thermal"),
            ("test_large_exc [molecule]", "larger, $S_1$-sampled"),
            ("test_random_exc4 [molecule]", "random, active-learning geometries"),
            ("test_scaffold_exc4 [molecule]", "unseen scaffolds, active-learning geometries"),
            ("test_large_exc4 [molecule]", "larger, active-learning geometries"),
            ("test_dye_noS_eq [molecule]", "held-out dyes, equilibrium"), ("test_lad_noS_eq [molecule]", "ladder molecules, equilibrium"),
            ("test trajectories", "test trajectories, $S_1$ MAE")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \multicolumn{2}{c}{LEH} & \multicolumn{2}{c}{Direct-softmin} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Errors (meV) & uniform & weighted & uniform & weighted \\", r"\midrule",
             r"\multicolumn{5}{l}{\textit{decision criteria (validation data)}} \\"]
    for r in rows:
        if r is None:
            lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{test sets ($S_1$--$S_3$ MAE unless stated otherwise)}} \\"]
            continue
        k, lab = r; x = G[k]
        cells = [fq([x[m]["mean"], x[m]["std"], x[m]["n"]], "mae") for m in ("prod_leh", "prodw_leh", "prod_softmin", "prodw_softmin")]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_loss.tex").write_text("\n".join(lines) + "\n")


def si_production():
    """Production models (weighted loss, four seeds): every test set, seeds (mean +- std) and committee; trajectories."""
    D = json.load(open(ROOT / "experiments/al/committee.json"))
    C, T = D["sets"], D["trajectories"]
    sets = [("val", "validation split"), ("test_random_offeq", "random, thermal"), ("test_scaffold_offeq", "unseen scaffolds, thermal"),
            ("test_random_exc", "random, $S_1$-sampled"), ("test_scaffold_exc", "unseen scaffolds, $S_1$-sampled"),
            ("test_large_offeq", "larger, thermal"), ("test_large_exc", "larger, $S_1$-sampled"),
            ("test_random_exc4", "random, active-learning frames"), ("test_scaffold_exc4", "unseen scaffolds, active-learning frames"),
            ("test_large_exc4", "larger, active-learning frames"), ("test_dye_noS_eq", "held-out dyes, eq."),
            ("test_dye_noS_offeq", "held-out dyes, thermal"), ("test_lad_noS_eq", "ladders, eq."), ("test_lad_noS_offeq", "ladders, thermal")]
    lines = [r"\begin{tabular}{lrcccccccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{LEH} & \multicolumn{4}{c}{Direct-softmin} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r" & & \multicolumn{2}{c}{$S_1$--$S_3$ MAE (meV)} & $S_1$ MAE & $S_1$ $R^2$ & \multicolumn{2}{c}{$S_1$--$S_3$ MAE (meV)} & $S_1$ MAE & $S_1$ $R^2$ \\",
             r"Test set & $n$ & seeds & comm. & comm. & comm. & seeds & comm. & comm. & comm. \\", r"\midrule"]
    for k, lab in sets:
        cells = []
        for a in ("leh", "softmin"):
            x = C[k][a]
            cells += [fq([x["mae_mean"], x["mae_std"], 4], "mae"), fq([x["committee_mae"], 0, 1], "mae"),
                      fq([x["committee_s1_mae"], 0, 1], "mae"), fmt([x["committee_r2"], 0, 1], 2)]
        lines.append(f"{lab} & {C[k]['leh']['n']:,} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r" & & \multicolumn{8}{c}{$S_1$ MAE along the TD-$\omega$B97X-D $S_1$ trajectories (meV)} \\",
              r"\cmidrule(lr){3-10}"]
    names = {"acrolein_s1": "acrolein", "uracil_s1": "uracil", "stilbene_s1": "\\textit{trans}-stilbene", "hbdi_s1": "HBDI",
             "dmabn_s1": "DMABN"}
    for t, lab in names.items():
        cells = []
        for a in ("leh", "softmin"):
            x = T["test"][a]; v = x["per_trajectory_seeds"][t]
            cells += [fq([float(np.mean(v)), float(np.std(v, ddof=1)), 4], "mae"), fq([x["per_trajectory_committee"][t], 0, 1], "mae"),
                      r"\multicolumn{2}{c}{signed $" + f"{1000 * x['per_trajectory_committee_signed'][t]:+.0f}" + "$}"]
        lines.append(f"test: {lab} & & " + " & ".join(cells) + r" \\")
    for key, lab in (("test", "mean of the 5 test trajectories"), ("validation", "mean of the 7 validation trajectories")):
        cells = []
        for a in ("leh", "softmin"):
            x = T[key][a]
            cells += [fq([x["mean"], x["std"], 4], "mae"), fq([x["committee"], 0, 1], "mae"), r"\multicolumn{2}{c}{}"]
        lines.append(f"{lab} & & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_production.tex").write_text("\n".join(lines) + "\n")


def table3():
    """QCDGE size extrapolation: main table (architectures) and SI table (all variants incl. R2 and dipoles)."""
    f = ROOT / "experiments/qcdge/summary_final.json"
    if not f.exists():
        return
    R = json.load(open(f))
    lab = dict(R["order"])
    main = ["leh", "softmin", "sum", "mean", "xmace", "vecmsani"]
    cols = [("val", "S1-3 MAE", 3), ("test_9", "S1-3 MAE", 3), ("test_10", "S1-3 MAE", 3), ("test_10_novelscaf", "S1-3 MAE", 3),
            ("test_10", "S1-10 MAE", 3), ("test_10", "S1-3 R2", 2)]
    head = [r" & \multicolumn{4}{c}{$S_1$--$S_3$ MAE (meV)} & $S_1$--$S_{10}$ MAE (meV) & $S_1$--$S_3$ $R^2$ \\",
            r"\cmidrule(lr){2-5}", r"Heavy atoms & $\le 8$ (val.) & 9 & 10 & 10, new scaffolds & 10 & 10 \\", r"\midrule"]

    def write(keys, cols, head, fname, mark_best):
        best = [min((R["models"][k][s][q][0] for k in keys if k in R["models"] and s in R["models"][k]),
                    default=None, key=lambda v: -v if "R2" in q else v) for s, q, _ in cols] if mark_best else [None] * len(cols)
        lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule"] + head
        for k in keys:
            r = R["models"].get(k)
            cells = []
            for (s, q, nd), b in zip(cols, best):
                v = (r or {}).get(s, {}).get(q)
                c = fq(v, q, nd) if r else r"\todo{running}"
                cells.append(f"\\textbf{{{c}}}" if v and b is not None and abs(v[0] - b) < 1e-12 else c)
            lines.append(lab[k] + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (OUT / fname).write_text("\n".join(lines) + "\n")

    write(main, cols, head, "table3_qcdge.tex", True)
    allk = [k for k, _ in R["order"]]
    cols_si = cols + [("test_10", "S1 MAE", 3), ("test_10", "S1 R2", 2), ("val", "osc R2 S1", 2), ("test_10", "osc R2 S1", 2)]
    head_si = [r" & \multicolumn{4}{c}{$S_1$--$S_3$ MAE (meV)} & $S_1$--$S_{10}$ & $S_1$--$S_3$ $R^2$ & $S_1$ MAE & $S_1$ $R^2$ & \multicolumn{2}{c}{$f_{S_1}$ $R^2$} \\",
               r"\cmidrule(lr){2-5}\cmidrule(lr){10-11}", r"Heavy atoms & $\le 8$ (val.) & 9 & 10 & 10, new scaf. & 10 & 10 & 10 & 10 & $\le 8$ & 10 \\", r"\midrule"]
    write(allk, cols_si, head_si, "si_qcdge_full.tex", False)


def si_full():
    sets = [("test_random_offeq", "random, thermal"), ("test_random_eq", "random, eq."), ("test_scaffold_offeq", "scaffold, thermal"),
            ("test_random_exc", "random, $S_1$-sampled"), ("test_large_offeq", "larger, thermal"), ("test_large_eq", "larger, eq."),
            ("test_large_exc", "larger, $S_1$-sampled")]
    qs = [("S1-1 MAE", 3), ("S1-1 R2", 2), ("S1-3 MAE", 3), ("S1-3 R2", 2), ("E_S1 MAE", 3), ("E0 meV/at", 1), ("F0 MAE", 3), ("F_S1 MAE", 3)]
    models = ["leh", "leh_L2", "softmin", "xmace", "vecmsani", "msani"]
    lines = [r"\begin{tabular}{ll" + "c" * len(qs) + "}", r"\toprule",
             r"Test set & Model & $S_1$ MAE & $S_1$ $R^2$ & $S_1$--$S_3$ MAE & $S_1$--$S_3$ $R^2$ & $E_{S_1}$ MAE & $E_{S_0}$ & $F_{S_0}$ & $F_{S_1}$ \\",
             r" & & (meV) & & (meV) & & (meV) & (meV/atom) & (meV/\AA) & (meV/\AA) \\", r"\midrule"]
    for s, slab in sets:
        for k, m in enumerate(models):
            r = L.get(m, {}).get("all", {}).get(s, {})
            lines.append((slab if k == 0 else "") + " & " + NAMES[m] + " & " + " & ".join(fq(r.get(q), q, nd) for q, nd in qs) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "si_benchmark_full.tex").write_text("\n".join(lines) + "\n")


def si_dataset():
    """Composition of the released dataset by geometry type (experiments/al/dataset_stats_v4.json)."""
    f = ROOT / "experiments/al/dataset_stats_v4.json"
    if not f.exists():
        return
    D = json.load(open(f)); B = D["by_type"]
    rows = [("qcdge_eq", "QCDGE minima"), ("offeq", "thermal (GFN2-xTB, 300--1000\\,K)"),
            ("excmd", "excited-state sampling, round 1"), ("al_r2", "active learning, round 2"),
            ("al_r3", "active learning, round 3"), ("al_r4", "active learning, round 4"),
            ("eq", "minima of the larger test molecules"), ("conj_eq", "dyes and ladders, minima"),
            ("conj_offeq", "dyes and ladders, thermal")]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Geometry type & Geometries & Molecules & with $S_1$--$S_3$ forces \\", r"\midrule"]
    for k, lab in rows:
        b = B.get(k, {})
        lines.append(f"{lab} & {b.get('n', 0):,} & {b.get('n_molecules', 0):,} & {b.get('n_Fex_S1', 0):,} \\\\")
    lines += [r"\midrule", f"total & {D['n_geometries']:,} & {D['n_molecules']:,} & {D['n_with_Fex_S1']:,} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_dataset.tex").write_text("\n".join(lines) + "\n")


def si_al():
    """Active-learning rounds: labelled frames, pre-label error of the sampling models and the criterion afterwards."""
    AL = ROOT / "experiments/al"
    D = json.load(open(AL / "dataset_stats_v4.json"))["by_type"]

    def crit(f):
        C = json.load(open(AL / f))["runs"]
        return {q: [float(np.mean([t[q] for t in x["trajectories"].values()])) for x in C.values()] for q in ("gap", "rise")}

    rows = [("1", "excmd", None, "convergence_r1_benchmark.json", "preliminary LEH"),
            ("2", "al_r2", "prelabel_s1r2.json", "convergence_r2.json", "benchmark LEH"),
            ("3", "al_r3", "prelabel_s1r3.json", "convergence_r3.json", "warm-started from dye-augmented LEH"),
            ("4", "al_r4", "prelabel_s1r4.json", "convergence_r4.json", "warm-started from round 3")]
    lines = [r"\begin{tabular}{llrrrrrcc}", r"\toprule",
             r" & & & & \multicolumn{3}{c}{pre-label $S_1$ error (meV)} & \multicolumn{2}{c}{criterion after the round (meV)} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"Round & Sampling models & Geometries & Molecules & MAE & signed & twist $>75^\circ$ & $S_1$ MAE & $S_0$-change MAE \\",
             r"\midrule"]
    for rd, key, pre, cf, src in rows:
        b = D[key]; P = json.load(open(AL / pre)) if pre else None
        c = crit(cf)
        pm = (f"{1000 * P['all']['mae']:.0f} & ${1000 * P['all']['signed']:+.0f}$ & {1000 * P['by_twist']['75-91']['mae']:.0f}"
              if P else "-- & -- & --")
        cc = " & ".join(f"{1000 * np.mean(c[q]):.0f} ({', '.join(f'{1000 * v:.0f}' for v in c[q])})" for q in ("gap", "rise"))
        lines.append(f"{rd} & {src} & {b['n']:,} & {b['n_molecules']:,} & {pm} & {cc} \\\\")
    cp = {"gap": [], "rise": []}
    for f in ("convergence_prod_v4.json", "convergence_prod_v4_softmin_s1.json"):
        for q, v in crit(f).items():
            cp[q] += v
    lines += [r"\midrule", "production, uniform loss & trained from scratch & & & & & & " +
              " & ".join(f"{1000 * np.mean(cp[q]):.0f}" for q in ("gap", "rise")) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_al.tex").write_text("\n".join(lines) + "\n")


def si_force_scaling():
    """Least-squares slope (through the origin) and correlation of predicted vs reference forces per heavy-atom bin."""
    f = ROOT / "experiments/wb97/diag_force_scaling.json"
    if not f.exists():
        return
    D = json.load(open(f))
    models = [("main_leh_all_s0", "LEH"), ("main_softmin_all_s0", "Direct-softmin"), ("xmace128_suball", "X-MACE")]
    bins = [("test_random_offeq|9", "9"), ("test_large_offeq|10", "10--14"), ("test_large_offeq|15", "15--19"), ("test_large_offeq|20", "20")]
    lines = [r"\begin{tabular}{ll" + "c" * len(bins) + "}", r"\toprule",
             r" & & \multicolumn{" + str(len(bins)) + r"}{c}{heavy atoms} \\", r"\cmidrule(lr){3-" + str(2 + len(bins)) + "}",
             r"Forces & Model & " + " & ".join(b for _, b in bins) + r" \\", r"\midrule"]
    for q, qlab in (("F0", "$S_0$"), ("F_S1", "$S_1$")):
        for k, (m, lab) in enumerate(models):
            cells = []
            for b, _ in bins:
                v = D.get(m, {}).get(f"{b}|{q}")
                cells.append(f"{v['slope']:.2f} ({v['corr']:.2f})" if v else "--")
            lines.append((qlab if k == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
        if q == "F0":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_force_scaling.tex").write_text("\n".join(lines) + "\n")


def si_label_matched():
    """LEH trained on exactly the labels X-MACE uses vs LEH (all labels) and X-MACE, 30k training molecules."""
    f = ROOT / "experiments/wb97/results_runs_xlab.json"
    if not f.exists():
        return
    X = json.load(open(f))
    sets = [("test_random_offeq", "random"), ("test_large_offeq", "larger"), ("test_scaffold_offeq", "scaffold")]
    cols = [(s, "S1-3 MAE", "mae_sup") for s, _ in sets] + [(s, "F0 MAE", "mae_F0") for s, _ in sets[:2]] + \
           [(s, "F_S1 MAE", "mae_FS1") for s, _ in sets[:2]]
    xl = {(s, q): [float(np.mean([r[s][k] for r in X.values()])), float(np.std([r[s][k] for r in X.values()])), len(X)]
          for s, q, k in cols}
    rows = [("LEH, all labels", lambda s, q: L["leh"]["30k"].get(s, {}).get(q)),
            ("LEH, X-MACE's labels", lambda s, q: xl[(s, q)]),
            ("X-MACE", lambda s, q: L["xmace"]["30k"].get(s, {}).get(q))]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{3}{c}{$S_1$--$S_3$ MAE (meV)} & \multicolumn{2}{c}{$S_0$ forces (meV/\AA)} & \multicolumn{2}{c}{$S_1$ forces (meV/\AA)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Model (30k molecules) & random & larger & scaffold & random & larger & random & larger \\", r"\midrule"]
    for lab, get in rows:
        lines.append(lab + " & " + " & ".join(fq(get(s, q), q) for s, q, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_label_matched.tex").write_text("\n".join(lines) + "\n")


def si_fc():
    """Excited-state forces at equilibrium geometries (Franck-Condon point; experiments/wb97/fc_eval.json)."""
    f = ROOT / "experiments/wb97/fc_eval.json"
    if not f.exists():
        return
    F = json.load(open(f))
    W = "experiments/wb97/"
    models = [("LEH", [W + f"runs_main/main_leh_all_s{s}" for s in (0, 1)]), ("LEH, $L=2$", [W + "runs_main/main_leh_L2_all_s0"]),
              ("Direct-softmin", [W + f"runs_main/main_softmin_all_s{s}" for s in (0, 1)]), ("X-MACE", [W + "runs_lc/xmace128_suball"]),
              ("OMNI-P2x arch.", [W + "runs_lc/suball_vecmsani"]), ("MS-ANI", [W + "runs_lc/suball_msani"])]
    cols = [("random_eq", "F_S1 MAE", 3), ("random_eq", "F_S1 R2", 2), ("larger_eq", "F_S1 MAE", 3), ("larger_eq", "F_S1 R2", 2),
            ("random_eq", "F_S2 MAE", 3), ("random_eq", "F_S3 MAE", 3)]
    n = {s: F[models[0][1][0]][s]["n"] for s in ("random_eq", "larger_eq")}
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{2}{c}{random (" + f"{n['random_eq']:,}" + r")} & \multicolumn{2}{c}{larger (" + f"{n['larger_eq']:,}" +
             r")} & \multicolumn{2}{c}{random} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Model & $F_{S_1}$ MAE & $F_{S_1}$ $R^2$ & $F_{S_1}$ MAE & $F_{S_1}$ $R^2$ & $F_{S_2}$ MAE & $F_{S_3}$ MAE \\", r"\midrule"]
    for lab, runs in models:
        cells = []
        for s, q, nd in cols:
            v = [F[r][s][q] for r in runs if r in F and q in F[r].get(s, {})]
            cells.append(fq([float(np.mean(v)), float(np.std(v)), len(v)], q, nd) if v else "--")
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_fc.tex").write_text("\n".join(lines) + "\n")


def si_population():
    """Population loss on the LEH eigenvectors, 10k setting (SI): shared weight with the TD-DFT excitation populations
    (experiments/eigvec/tddft/eval_pop.json) and errors (experiments/wb97/results_runs_abl_pop.json, ladders and dyes
    from eval_pop.json); mean +- std over three seeds (baseline, weight 1), single seeds for the two variants."""
    E = json.load(open(ROOT / "experiments/eigvec/tddft/eval_pop.json"))["models"]
    R = json.load(open(ROOT / "experiments/wb97/results_runs_abl_pop.json"))
    cols = [[f"abl_leh_s{s}" for s in range(3)], [f"abl_leh_pop_s{s}" for s in range(3)], ["abl_leh_pop01_s0"],
            ["abl_leh_popmin_s0"]]
    s13 = lambda q, sp: (lambda r: float(np.mean([R[r][sp][f"{q}_S{n}"] for n in (1, 2, 3)])))
    rows = [("validation, $S_1$", lambda r: E[r]["val"]["S1|all"]["S"], "S"),
            ("validation, $S_1$, chance level", lambda r: E[r]["val"]["S1|all"]["null"], "S"),
            ("validation, $S_2$", lambda r: E[r]["val"]["S2|all"]["S"], "S"),
            ("validation, $S_3$", lambda r: E[r]["val"]["S3|all"]["S"], "S"),
            ("ladder molecules, $S_1$", lambda r: E[r]["set460"]["S1|ladder"]["S"], "S"),
            ("dyes, $S_1$", lambda r: E[r]["set460"]["S1|dye"]["S"], "S"), None,
            ("random, thermal, $S_1$--$S_3$ MAE", s13("mae", "test_random_offeq"), "mae"),
            ("random, thermal, $S_1$--$S_3$ $R^2$", s13("r2", "test_random_offeq"), "r2"),
            ("random, $S_1$-sampled, $S_1$--$S_3$ MAE", s13("mae", "test_random_exc"), "mae"),
            ("unseen scaffolds, thermal, $S_1$--$S_3$ MAE", s13("mae", "test_scaffold_offeq"), "mae"),
            ("larger, thermal, $S_1$--$S_3$ MAE", s13("mae", "test_large_offeq"), "mae"),
            ("random, thermal, $S_0$ energy MAE (meV per atom)",
             lambda r: R[r]["test_random_offeq"]["mae_E0_per_atom_meV"], "e0"),
            ("random, thermal, $S_1$ force MAE (meV/\\AA)", lambda r: R[r]["test_random_offeq"]["mae_FS1"], "mae"),
            ("ladder molecules, equilibrium, $S_1$ MAE", lambda r: E[r]["transfer"]["ladders"]["mae1"], "mae"),
            ("ladder molecules, equilibrium, $S_1$ $R^2$", lambda r: E[r]["transfer"]["ladders"]["r2_1"], "r2"),
            ("held-out dyes, equilibrium, $S_1$ MAE", lambda r: E[r]["transfer"]["dyes"]["mae1"], "mae"),
            ("held-out dyes, equilibrium, $S_1$ $R^2$", lambda r: E[r]["transfer"]["dyes"]["r2_1"], "r2")]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r" & Baseline & Population loss & Weight 0.1 & Hole or electron \\", r"\midrule",
             r"\multicolumn{5}{l}{\textit{shared weight with the TD-DFT excitation population}} \\"]
    for row in rows:
        if row is None:
            lines += [r"\midrule", r"\multicolumn{5}{l}{\textit{errors (meV unless stated otherwise)}} \\"]
            continue
        lab, f, kind = row
        cells = []
        for runs in cols:
            x = [f(r) for r in runs]
            v = [float(np.mean(x)), float(np.std(x)), len(x)]
            cells.append(fq(v, "mae") if kind == "mae" else fmt(v, 1) if kind == "e0" else fmt(v, 2 if kind == "S" else 3))
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "si_population.tex").write_text("\n".join(lines) + "\n")


table1(); table2(); table3(); si_conj(); si_loss(); si_production(); si_full(); si_dataset(); si_al(); si_force_scaling(); si_label_matched(); si_fc()
si_population()
print("written", sorted(p.name for p in OUT.glob("*.tex")))
