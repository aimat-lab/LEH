"""Do the LEH eigenvectors locate the excitations where TD-DFT does?

TD-DFT: per-atom excitation population p_A = (hole_A + electron_A) / 2 of root n (populations.py, Loewdin basis).
LEH: per-atom weight w_A = v_A^2 of eigenvector n. Both sum to 1. Hydrogens are folded into their nearest heavy atom
(main comparison, "heavy-atom groups"); the atom-resolved comparison is reported as well.

Per geometry, state n = 1-3 and seed:
  shared   S = sum_A min(w_A, p_A)   (1: same distribution)
  null     S after permuting the LEH weights within each element (10 permutations)
  seeds    S between two LEH seeds on the same geometries (the ceiling set by the gauge freedom)
  PR       participation numbers 1 / sum w^2 and 1 / sum p^2 (heavy-atom groups)
  best     the LEH state m in 1-5 with the largest S to TD-DFT root n (order swaps)
States are compared only if the TD-DFT root and the LEH eigenvalue are separated from their neighbours by >= DEG.

Usage: python experiments/eigvec/tddft/compare.py [--vectors prod|all]
Output: experiments/eigvec/tddft/compare_<vectors>.json, compare_rows_<vectors>.pkl
"""
import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "experiments/eigvec/tddft"
DEG = 0.1  # eV
NS = 3
LAD_BINS = ((0, 10), (10, 16), (16, 22), (22, 99))


def fold_map(Z, R):
    heavy = np.where(Z > 1)[0]
    dist = np.linalg.norm(R[:, None] - R[None, heavy], axis=-1)
    return heavy[np.argmin(dist, 1)]  # atom -> heavy atom it belongs to (heavy atoms map to themselves)


def fold(w, fmap, heavy):
    f = np.bincount(fmap, weights=w, minlength=len(w))
    return f[heavy]


def shared(a, b):
    return float(np.minimum(a, b).sum())


def null(a, b, Zh, rng, n=10):
    out = []
    for _ in range(n):
        p = np.arange(len(Zh))
        for z in np.unique(Zh):
            k = np.where(Zh == z)[0]
            p[k] = rng.permutation(k)
        out.append(shared(a[p], b))
    return float(np.mean(out))


def gap_ok(lam, n):
    lo = lam[n] - lam[n - 1] if n > 0 else np.inf
    hi = lam[n + 1] - lam[n] if n + 1 < len(lam) else np.inf
    return min(lo, hi) >= DEG


def rows_for(sel, vec, runs):
    rng = np.random.default_rng(0)
    rows = []
    for k, e in enumerate(sel):
        f = D / "pop" / f"{e['id']}.npz"
        if not f.exists():
            continue
        t = np.load(f)
        Z, R = np.asarray(e["Z"]), np.asarray(e["R"])
        if len(Z) < 4:
            continue
        fmap = fold_map(Z, R)
        heavy = np.where(Z > 1)[0]
        Zh = Z[heavy]
        exc = t["exc_eV"]
        nh = len(heavy)
        base = dict(id=e["id"], group=e["group"], n_heavy=nh, n_atoms=len(Z))
        for n in range(NS):
            pt = t["pop"][n]
            ph = fold(pt, fmap, heavy)
            hh, eh = fold(t["hole"][n], fmap, heavy), fold(t["elec"][n], fmap, heavy)
            r = dict(base, state=n + 1, td_gap_ok=gap_ok(exc, n), td_pr=float(1 / (ph ** 2).sum()),
                     td_pr_atoms=float(1 / (pt ** 2).sum()), td_wH=float(pt[Z == 1].sum()),
                     td_ct=float(1 - np.trace(t["omega"][n]) if t["omega"].ndim == 3 else np.nan),
                     td_he=shared(hh, eh), exc=float(exc[n]), ref=float(e["gaps_ref"][n]))
            ws = {}
            for s in runs:
                lam, V = vec[s][k]["lam"], vec[s][k]["V"]
                w = V[:, n] ** 2
                wh = fold(w, fmap, heavy)
                ws[s] = wh
                best = [shared(fold(V[:, m] ** 2, fmap, heavy), ph) for m in range(V.shape[1])]
                r[s] = dict(S=shared(wh, ph), S_atoms=shared(w, pt), S_hole=shared(wh, hh), S_elec=shared(wh, eh),
                            null=null(wh, ph, Zh, rng), pr=float(1 / (wh ** 2).sum()), gap_ok=gap_ok(lam, n),
                            err=float(lam[n] - e["gaps_ref"][n]), best_m=int(np.argmax(best)) + 1,
                            S_best=float(max(best)))
            prod = [s for s in runs if s.startswith("prodw_leh_v4_s")]
            r["seed_seed"] = float(np.mean([shared(ws[a], ws[b]) for a, b in combinations(prod, 2)]))
            rows.append(r)
    return rows


def med(x):
    x = [v for v in x if np.isfinite(v)]
    return float(np.median(x)) if x else np.nan


def summarise(rows, runs, label, sel_fn):
    out = {}
    for n in range(1, NS + 1):
        rr = [r for r in rows if r["state"] == n and sel_fn(r) and r["td_gap_ok"]]
        if not rr:
            continue
        d = dict(n=len(rr), td_pr=med([r["td_pr"] for r in rr]), td_wH=med([r["td_wH"] for r in rr]),
                 td_ct=med([r["td_ct"] for r in rr]), td_he=med([r["td_he"] for r in rr]),
                 seed_seed=med([r["seed_seed"] for r in rr]))
        for s in runs:
            ok = [r for r in rr if r[s]["gap_ok"]]
            d[s] = dict(n=len(ok), **{q: med([r[s][q] for r in ok]) for q in
                                      ("S", "S_atoms", "S_hole", "S_elec", "null", "pr", "S_best")},
                        frac_best_same=float(np.mean([r[s]["best_m"] == n for r in ok])) if ok else np.nan,
                        mae=float(np.mean([abs(r[s]["err"]) for r in ok])) if ok else np.nan)
        out[f"{label}|S{n}"] = d
    return out


def models(rows, runs):
    """Per model: agreement with TD-DFT (S1, heavy-atom groups) on training geometries, random-test geometries and
    ladders, and its participation number relative to TD-DFT on ladders. Within seed groups trained on identical data
    (coupling_vs_transfer.py): does the model that agrees better with TD-DFT transfer better to ladders?"""
    import sys
    sys.path.insert(0, str(ROOT / "experiments/eigvec"))
    from coupling_vs_transfer import GROUPS
    ct = json.load(open(ROOT / "experiments/eigvec/coupling_vs_transfer.json"))
    sets = dict(train=lambda r: r["group"].startswith("train|"), test=lambda r: r["group"].startswith("test|"),
                ladder=lambda r: r["group"] == "ladder")
    out = {}
    for g, rs in GROUPS.items():
        for rdir in rs:
            m = rdir.name
            if m not in runs or m not in ct.get(g, {}):
                continue
            d = dict(group=g, lad_mae1=ct[g][m]["ladders"]["mae1"], rand_mae13=ct[g][m]["random"]["mae13"])
            for k, f in sets.items():
                rr = [r for r in rows if r["state"] == 1 and f(r) and r["td_gap_ok"] and r[m]["gap_ok"]]
                d[f"S_{k}"] = med([r[m]["S"] for r in rr])
                d[f"S13_{k}"] = med([r[m]["S"] for r in rows if r["state"] <= 3 and f(r) and r["td_gap_ok"]
                                     and r[m]["gap_ok"]])
                d[f"logPR_{k}"] = med([np.log(r[m]["pr"] / r["td_pr"]) for r in rr])
            out[m] = d
    print(f"\n{'model':22s} {'group':26s} {'ladMAE':>6s} {'S_tr':>5s} {'S_te':>5s} {'S_lad':>5s} {'S13tr':>5s} "
          f"{'lnPRtr':>6s} {'lnPRlad':>7s}")
    for m, d in out.items():
        print(f"{m:22s} {d['group'][:26]:26s} {d['lad_mae1']:6.3f} {d['S_train']:5.2f} {d['S_test']:5.2f} "
              f"{d['S_ladder']:5.2f} {d['S13_train']:5.2f} {d['logPR_train']:+6.2f} {d['logPR_ladder']:+7.2f}")
    # within-group pairs (samplers are lineages, not independent seeds)
    for q in ("S_train", "S13_train", "S_test", "S_ladder", "logPR_train", "logPR_ladder"):
        agree = total = 0
        for g in GROUPS:
            if g.startswith("sampler"):
                continue
            ms = [m for m in out if out[m]["group"] == g]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = out[ms[i]], out[ms[j]]
                    agree += int((a[q] - b[q]) * (a["lad_mae1"] - b["lad_mae1"]) < 0)
                    total += 1
        print(f"within-group pairs where the larger {q} has the lower ladder S1 MAE: {agree} of {total}")
    return out


def main(which):
    sel = json.load(open(D / "selection.json"))
    vec = pickle.load(open(D / f"leh_vectors_{which}.pkl", "rb"))
    runs = list(vec)
    prod = [r for r in runs if r.startswith("prodw_leh_v4_s")]
    rows = rows_for(sel, vec, runs)
    pickle.dump(rows, open(D / f"compare_rows_{which}.pkl", "wb"))
    groups = sorted({r["group"] for r in rows})
    summ = {}
    for g in groups:
        summ.update(summarise(rows, prod, g, lambda r, g=g: r["group"] == g))
    for lo, hi in LAD_BINS:
        summ.update(summarise(rows, prod, f"ladder heavy {lo}-{hi - 1}",
                              lambda r, lo=lo, hi=hi: r["group"] == "ladder" and lo <= r["n_heavy"] < hi))
    json.dump(summ, open(D / f"compare_{which}.json", "w"), indent=1)
    print(f"{'set|state':28s} {'n':>4s} {'tdPR':>5s} {'tdH':>5s} {'h~e':>5s} {'seeds':>5s} | per seed: S (null) S_atoms PR "
          f"best-same")
    for k, d in summ.items():
        cells = "  ".join(f"{d[s]['S']:.2f}({d[s]['null']:.2f}) {d[s]['S_atoms']:.2f} {d[s]['pr']:4.1f} "
                          f"{d[s]['frac_best_same']:.2f}" for s in prod)
        print(f"{k:28s} {d['n']:4d} {d['td_pr']:5.1f} {d['td_wH']:5.2f} {d['td_he']:5.2f} {d['seed_seed']:5.2f} | {cells}")
    if which == "all":
        summ["models"] = models(rows, runs)
        json.dump(summ, open(D / f"compare_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="prod", choices=("prod", "all"))
    main(ap.parse_args().vectors)
