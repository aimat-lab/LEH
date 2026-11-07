"""Fragment (size-) consistency figure: two non-bonded molecules at minimum interatomic distance d.

For each model, the deviation between its spectrum of the dimer and the union of its own monomer spectra
(mean |dimer - union| over S1-S3, the states every model predicts) is plotted against d. Once the molecules no longer
interact the two must agree; at short distance a real interaction shifts/splits the states, which the TDDFT
reference on the same geometries quantifies (black; mean over the computed pairs).
Rows: (a, b) the four heads on the same MACE encoder (runs_abl, seed 0); (c, d) LEH vs the baselines. All models
trained on the same 10k molecules. Columns: two different molecules (A+B) / two identical molecules (A+A).
Geometries: data/frag_pairs.pkl (experiments/fragments/make_pairs.py, 60 pairs per kind); baseline predictions:
experiments/fragments/<model>/predictions.pkl (hpc/hk_frag_predict.sh); TDDFT: experiments/frag_tddft/results_pairs.json.
Per-state atomic sums (MS-ANI, OMNI-P2x) give E_k(A+B) = E_k(A) + E_k(B), i.e. dimer S_n ~ S_n(A) + S_n(B) (both
molecules excited): deviations of 6-7 eV at every distance. In (b) Direct-softmin and Direct-mean overlap (0.47 eV).
Output: paper/figs/fragments.{pdf,png,csv} (main: baselines X-MACE, OMNI-P2x arch. from scratch),
fragments_si.* (SI: + OMNI-P2x fine-tuned and pretrained); fragments_pairwise.csv: LEH vs TDDFT on the same pairs at 2.5-4 A. Head predictions are cached in experiments/fragments/heads_cache.pkl.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa
import matplotlib.pyplot as plt  # noqa

NE, FLOOR = 3, 1e-4  # display floor: LEH is exactly 0 (~1e-15 eV) beyond the cutoff; shown at 1e-4
G = pickle.load(open(ROOT / "data/frag_pairs.pkl", "rb"))
ids = [m["id"] for m in G]
MONO = [k for k, m in enumerate(G) if m["kind"] == "frag_mono"]
DIM = [k for k, m in enumerate(G) if m["kind"] == "frag_dimer"]
D = sorted({G[k]["d"] for k in DIM})
NPAIR = 1 + max(G[k]["pair"] for k in DIM)
HEADS = {"leh": "runs_abl/abl_leh_s0", "softmin": "runs_abl/abl_softmin_s0",
         "direct_mean": "runs_abl/abl_mean_s0", "direct_sum": "runs_abl/abl_sum_s0"}
BASE = {"xmace": "xmace128_sub10k", "msani": "sub10k_msani", "vecmsani": "sub10k_vecmsani",
        "omnift": "sub10k_omnift", "omni_zeroshot": "omni_zeroshot"}


def head_gaps():
    cache = ROOT / "experiments/fragments/heads_cache.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    import torch
    from exmlip.training.fragments import predict_mols
    from exmlip.training.train import build_model_from_dir
    torch.set_num_threads(12); torch.set_default_dtype(torch.float64)
    out = {}
    for name, rd in HEADS.items():
        model, cfg = build_model_from_dir(ROOT / "experiments/wb97" / rd)
        mols = [dict(Z=G[k]["Z"], R=G[k]["R"], gaps=np.zeros(cfg["n_states"])) for k in MONO + DIM]
        with torch.no_grad():  # chunks of 60 geometries (one batch of all 1800 runs out of memory)
            g = np.concatenate([predict_mols(model, mols[s:s + 60], cfg["n_states"], cfg["n_sup"])
                                for s in range(0, len(mols), 60)])
        out[name] = {ids[k]: np.asarray(v, float) for k, v in zip(MONO + DIM, g)}
        print(name, "done", flush=True)
    pickle.dump(out, open(cache, "wb"))
    return out


def base_gaps(d):
    f = ROOT / "experiments/fragments" / d / "predictions.pkl"
    if not f.exists():
        return None
    P = pickle.load(open(f, "rb"))
    return {x["id"]: np.asarray(x["gaps_pred"], float) for k in ("test_frag_mono", "test_frag_dimer") for x in P.get(k, [])}


def deviations(gaps):
    """dev[kind] = array (len(D), NPAIR) of mean |dimer - union| over S1-S3."""
    dev = {k: np.full((len(D), NPAIR), np.nan) for k in ("AB", "AA")}
    for k in DIM:
        m = G[k]
        a, b = m["mono"]
        ga, gb, gd = gaps.get(ids[a]), gaps.get(ids[b]), gaps.get(ids[k])
        if ga is None or gb is None or gd is None:
            continue
        u = np.sort(np.concatenate([ga[:NE], gb[:NE]]))[:NE]
        dev[m["pair_kind"]][D.index(m["d"]), m["pair"]] = np.mean(np.abs(gd[:NE] - u))
    return dev


res = {name: deviations(g) for name, g in head_gaps().items()}
for name, d in BASE.items():
    g = base_gaps(d)
    if g is not None:
        res[name] = deviations(g)
    else:
        print("no predictions yet:", name)
tdf = ROOT / "experiments/frag_tddft/results_pairs.json"
TD = json.load(open(tdf)) if tdf.exists() else []

fs.setup()
def make_frag(baselines, annot, fname):
    LAB = []
    ROWS = [("a", ["leh", "softmin", "direct_mean", "direct_sum"], "same encoder, different heads"),
            ("c", baselines, "LEH vs baselines")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), sharey=True, sharex=True)
    rows = []
    for r, (lt, models, rtitle) in enumerate(ROWS):
        for c, (kind, ktitle) in enumerate([("AB", "two different molecules"), ("AA", "two identical molecules")]):
            ax = axes[r, c]
            for name in models:
                if name not in res:
                    continue
                dev = res[name][kind]
                ok = np.isfinite(dev).all(0)
                if not ok.any():
                    continue
                dv = dev[:, ok]
                y = np.maximum(dv.mean(1), FLOOR)
                lo, hi = np.maximum(np.percentile(dv, 25, axis=1), FLOOR), np.maximum(np.percentile(dv, 75, axis=1), FLOOR)
                st = dict(fs.STYLE.get(name, {"marker": "o"}))
                st.setdefault("marker", "o")
                ax.plot(D, y, color=fs.MODEL_COLOR[name], label=fs.LABEL[name], lw=1.4, ms=3.5, markeredgecolor="white",
                        markeredgewidth=0.6, **st)
                ax.fill_between(D, lo, hi, color=fs.MODEL_COLOR[name], alpha=0.12, lw=0)
                rows += [dict(row=rtitle, kind=kind, model=fs.LABEL[name], d=d, n_pairs=int(ok.sum()), mean_dev=float(m),
                              p25=float(a), p75=float(b)) for d, m, a, b in
                         zip(D, dv.mean(1), np.percentile(dv, 25, axis=1), np.percentile(dv, 75, axis=1))]
            td = {}
            for t in TD:
                if t["kind"] == kind:
                    td.setdefault(t["d"], []).append(np.mean(np.abs(t["dev"][:NE])))
            if td:
                xs = sorted(td)
                ax.plot(xs, [max(np.mean(td[x]), FLOOR) for x in xs], color=fs.INK, lw=1.0, marker="*", ms=7,
                        label=f"TD-ωB97X-D ({len(td[xs[0]])} pairs)")
                rows += [dict(row=rtitle, kind=kind, model="TDDFT", d=x, n_pairs=len(td[x]), mean_dev=float(np.mean(td[x])),
                              p25=float(np.percentile(td[x], 25)), p75=float(np.percentile(td[x], 75))) for x in xs]
            ax.axvline(5, color=fs.MUTED, lw=0.8, linestyle=(0, (2, 2)))
            ax.axhline(FLOOR, color=fs.GRID, lw=0.8)
            ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.6, 30)
            LAB.append((ax, chr(ord(lt) + c)))
            if r == 1:
                ax.set_xlabel("Minimum interatomic distance (Å)")
        axes[r, 0].set_ylabel("Mean |dimer − union of\nmonomers|, S$_1$–S$_3$ (eV)")
    axes[0, 0].text(5.3, 2.5e-4, "cutoff r$_c$ = 5 Å", fontsize=6, color=fs.INK2)
    for ax in axes[1]:  # the four per-state atomic-sum models coincide
        ax.text(20, 13, annot, fontsize=6.2, color=fs.INK2,
                ha="right", va="bottom")
    axes[1, 0].text(20, FLOOR * 1.5, "values ≤ 10$^{-4}$ eV shown at 10$^{-4}$", fontsize=6, color=fs.INK2, ha="right")
    hs, ls = [], []
    for ax in axes.ravel():  # one legend for all panels (unique entries), below the panels so no data is covered
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls:
                hs.append(h); ls.append(l)
    fig.legend(hs, ls, loc="lower center", ncol=4, fontsize=6, handlelength=2.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.075, 1, 1)); fig.subplots_adjust(hspace=0.25)
    fs.panel_labels(fig, LAB)
    out = ROOT / f"paper/figs/{fname}"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


make_frag(["leh", "xmace", "vecmsani"], "OMNI-P2x arch.: dimer S$_n$ ≈ S$_n$(A) + S$_n$(B)", "fragments")
make_frag(["leh", "xmace", "msani", "vecmsani", "omnift", "omni_zeroshot"],
          "per-state atomic sums: dimer S$_n$ ≈ S$_n$(A) + S$_n$(B)", "fragments_si")
out = ROOT / "paper/figs/fragments"

# LEH vs TDDFT on the same pairs at 2.5-4 A (where the molecules really interact): magnitude, rank correlation, sign
G_ID = {m["id"]: m for m in G}
pw = []
hg = head_gaps()["leh"]
for t in TD:
    if t["d"] > 4.01:
        continue
    m = G_ID[t["id"]]; a, b = m["mono"]
    u = np.sort(np.concatenate([hg[ids[a]][:NE], hg[ids[b]][:NE]]))[:NE]
    dm = hg[t["id"]][:NE] - u
    pw.append(dict(id=t["id"], kind=t["kind"], d=t["d"], tddft_mean_abs=float(np.mean(np.abs(t["dev"][:NE]))),
                   leh_mean_abs=float(np.mean(np.abs(dm))), tddft_mean=float(np.mean(t["dev"][:NE])), leh_mean=float(np.mean(dm))))
if pw:
    with open(out.parent / "fragments_pairwise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pw[0])); w.writeheader(); w.writerows(pw)
    for k in ("AB", "AA"):
        x = np.array([q["tddft_mean_abs"] for q in pw if q["kind"] == k]); y = np.array([q["leh_mean_abs"] for q in pw if q["kind"] == k])
        rk = lambda v: np.argsort(np.argsort(v))
        print(f"{k}: TDDFT {1000 * x.mean():.1f} meV, LEH {1000 * y.mean():.1f} meV, Spearman {np.corrcoef(rk(x), rk(y))[0, 1]:.2f} (n={len(x)})")
print("written", out)
