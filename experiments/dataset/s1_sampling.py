"""Excited-state sampling round: model-driven S1/S2 dynamics -> geometries for new TDDFT single points.

Motivation (2026-10-12): along reference TDDFT S1 trajectories the models underestimate how far S0 rises along
S1-relaxation coordinates (C=O stretch, double-bond twist, ring puckering); the training geometries come from
thermal S0 (xTB) MD only. This script samples the regions that excited-state dynamics actually visits.

Per molecule (hash-selected subset of the dataset, all roles, so the split assignment of the new geometries follows
the molecule): Maxwell-Boltzmann velocities at T0 ~ U(300, 600) K at the equilibrium geometry, adiabatic NVE on S1
(80 %) or S2 (20 %) of model A for up to `--fs` fs. A trajectory is stopped at the first bond change (covalent-radius
criterion), when the predicted S1 gap drops below 0.5 eV (TDDFT is unreliable near S1/S0 crossings) or when the
energy becomes non-finite. Two frames per molecule are selected among the valid frames (recorded every 10 fs before the stop):
  a) a random frame (t >= 50 fs if available),
  b) the frame with the largest disagreement between model A and model B (independently trained seed):
     |dgap_S1| + |dE0_rel| (E0 relative to t=0, the quantity that failed along the reference trajectories).
Output: compute.py tasks ('given' geometries) with id <mol>_s<state>md<k>, plus the sampling metadata.

Round 2 (s1r2, 2026-10-16, user approved; iteration 1 of an active-learning loop), changes via options:
  --motifs experiments/dataset/motifs.json: all molecules (all roles) with an acyclic C=C/C=N/N=N conjugated with an
    aromatic ring or another double bond (find_motifs.py), the photoisomerisation coordinates round 1 missed (3 frames
    with an aryl-C=C twist > 45 deg; stilbene S1-AIMD fails along the twist);
  --models: the final full-set LEH seeds; --fs 1000; --gap_stop 0.2 (the stop uses the model's gap, which is
    overestimated exactly where S0 rises along a twist, so 0.5 eV hardly ever triggered there);
  --picks: 'twist' = frame with the largest double-bond twist of the motif bonds (geometric, model-independent),
    'relaxed' = lowest propagated-state energy relative to t=0 (most S1-relaxed), 'committee', 'rand'; duplicates
    are replaced by the next-best frame of the same criterion; criteria that do not apply (no motif) are skipped;
    --max_picks caps the frames per trajectory (criteria taken in the given order);
  ids <mol>_r<N>s<state>md<k> (N from --round s1rN) so that round-N frames get their own split keys; S1/S2 choice
    and initial velocities are drawn with round-specific seeds.

Usage (one shard; restartable per chunk):
  python experiments/dataset/s1_sampling.py --data data/wb97_all_100k.pkl --models RUN_A RUN_B --shard i --nshards n
  python experiments/dataset/s1_sampling.py --merge --out data/wb97/tasks/s1r1.pkl
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exmlip.md import AMU, CONV, BatchedModelPES, maxwell_boltzmann  # noqa
from exmlip.training.train import build_model_from_dir  # noqa

COV = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 16: 1.05}


def bonds(Z, R):
    r = np.array([COV[int(z)] for z in Z])
    d = np.linalg.norm(R[:, None] - R[None], axis=-1)
    b = d < 1.25 * (r[:, None] + r[None])
    np.fill_diagonal(b, False)
    return b


def h01(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 16**8


def select_molecules(mols, frac, zinc, motifs=None, salt="s1r1:", frac_extra=0.0):
    """Equilibrium entries of the selected molecules: QCDGE-derived (with our E0 at the minimum; hash-selected
    fraction, or all molecules listed in `motifs`) and, optionally, all ZINC size-extrapolation molecules (their
    xTB-minimum TDDFT entries)."""
    out = []
    for m in mols:
        if m["source"] == "ZINC":
            if zinc and m["kind"] == "eq":
                out.append(m)
        elif m["kind"] == "qcdge_eq" and "E0" in m and (
                (m["mol"] in motifs or h01(salt + m["mol"]) < frac_extra) if motifs is not None
                else h01("s1r1:" + m["mol"]) < frac):
            out.append(m)
    return out


MOTIF_SMARTS = {"aryl_db": "[a]-[#6,#7]=!@[#6,#7]", "conj_db": "[#6,#7,#8]=[#6,#7]-[#6,#7]=!@[#6,#7]"}


def motif_quads(Z, R):
    """Dihedral quadruples (a, x, y, b) of the acyclic motif double bonds x=y (bond orders from the eq geometry)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromXYZBlock(f"{len(Z)}\n\n" + "".join(f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {r[0]:.6f} "
                                                          f"{r[1]:.6f} {r[2]:.6f}\n" for z, r in zip(Z, R)))
        rdDetermineBonds.DetermineBonds(m, charge=0)
    except Exception:
        return []
    quads = set()
    for sm in MOTIF_SMARTS.values():
        for match in m.GetSubstructMatches(Chem.MolFromSmarts(sm)):
            a, x, y = match[-3], match[-2], match[-1]
            nb = sorted((n.GetIdx() for n in m.GetAtomWithIdx(y).GetNeighbors() if n.GetIdx() != x),
                        key=lambda k: -m.GetAtomWithIdx(k).GetAtomicNum())
            if nb:
                quads.add((a, x, y, nb[0]))
    return sorted(quads)


def max_twist(R, quads):
    """Largest deviation from planarity (deg; 0 planar, 90 perpendicular) over the motif double bonds."""
    best = 0.0
    for q in quads:
        p0, p1, p2, p3 = (R[k] for k in q)
        b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
        b1 = b1 / np.linalg.norm(b1); v = b0 - (b0 @ b1) * b1; w = b2 - (b2 @ b1) * b1
        phi = abs(np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w)))
        best = max(best, min(phi, 180.0 - phi))
    return best


def run_chunk(modelA, modelB, ms, state, fs, dt, every, seed, gap_stop=0.5, picks=("rand", "committee"), vsalt="v",
              max_picks=None):
    Zs = [m["Z"] for m in ms]
    R = [np.array(m["R"], float) for m in ms]
    rngs = [np.random.default_rng(zlib.crc32((vsalt + m["mol"]).encode()) + seed) for m in ms]
    quads = [motif_quads(m["Z"], np.asarray(m["R"], float)) if "twist" in picks else [] for m in ms]
    T0 = [float(300 + 300 * r.random()) for r in rngs]
    v = [maxwell_boltzmann(Z, T, r) for Z, T, r in zip(Zs, T0, rngs)]
    mass = [np.array([AMU[int(z)] for z in Z])[:, None] for Z in Zs]
    b0 = [bonds(Z, r) for Z, r in zip(Zs, R)]
    pes = BatchedModelPES(modelA, Zs, state=state)
    E, F, g = pes(R)
    E0_0 = E - g[:, state - 1]
    g0 = g[:, state - 1].copy()  # E_state(t) - E_state(0) = E - E0_0 - g0
    active = np.ones(len(ms), bool)
    reason = [None] * len(ms)
    t_stop = [None] * len(ms)
    frames = [[] for _ in ms]  # (t, R, gaps(4), E0 relative to t=0)
    n_steps = int(round(fs / dt))
    for step in range(1, n_steps + 1):
        v = [vi + 0.5 * dt * Fi / mi * CONV if a else vi for vi, Fi, mi, a in zip(v, F, mass, active)]
        R = [ri + dt * vi if a else ri for ri, vi, a in zip(R, v, active)]
        E, F, g = pes(R)
        v = [vi + 0.5 * dt * Fi / mi * CONV if a else vi for vi, Fi, mi, a in zip(v, F, mass, active)]
        if step % every == 0:
            t = step * dt
            for k in np.where(active)[0]:
                if not np.isfinite(E[k]) or not np.all(np.isfinite(R[k])):
                    reason[k] = "nonfinite"
                elif g[k, 0] < gap_stop:
                    reason[k] = "s1_gap"
                elif bool((bonds(Zs[k], R[k]) != b0[k]).any()):
                    reason[k] = "bond_change"
                if reason[k] is not None:
                    active[k] = False
                    t_stop[k] = t
                    v[k] = np.zeros_like(v[k])
                    continue
                frames[k].append((t, R[k].copy(), g[k, :4].copy(), float(E[k] - g[k, state - 1] - E0_0[k]),
                                  float(E[k] - E0_0[k] - g0[k]), max_twist(R[k], quads[k])))
        if not active.any():
            break
    # committee: model B on all recorded frames (energies only)
    res = []
    for k, m in enumerate(ms):
        fr = frames[k]
        if not fr:
            res.append(dict(mol=m["mol"], n_frames=0, stopped=not active[k], reason=reason[k], t_stop=t_stop[k],
                            state=state))
            continue
        Rs = [m["R"]] + [f[1] for f in fr]
        pb = BatchedModelPES(modelB, [m["Z"]] * len(Rs), state=1)
        Eb, _, gb = pb(Rs)
        e0b = Eb - gb[:, 0]
        e0b_rel = e0b[1:] - e0b[0]
        dis = np.abs(gb[1:, 0] - np.array([f[2][0] for f in fr])) + np.abs(e0b_rel - np.array([f[3] for f in fr]))
        rng = np.random.default_rng(zlib.crc32(("pick" + m["mol"]).encode()) + seed)
        late = [i for i, f in enumerate(fr) if f[0] >= 50.0] or list(range(len(fr)))
        rank = {"rand": [int(rng.choice(late))], "committee": [int(i) for i in np.argsort(-dis)],
                "relaxed": [int(i) for i in np.argsort([f[4] for f in fr])],
                "twist": [int(i) for i in np.argsort([-f[5] for f in fr])] if quads[k] else []}
        chosen = []
        for kind in picks:
            if max_picks and len(chosen) >= max_picks:
                break
            i = next((i for i in rank[kind] if i not in [c for _, c in chosen]), None)
            if i is not None:
                chosen.append((kind, i))
        sel = []
        for kind, i in chosen:
            t, Rk, gk, e0r, es_rel, tw = fr[i]
            sel.append(dict(kind=kind, t_fs=float(t), R=Rk, gaps_pred=gk, E0_rel_pred=e0r, disagreement=float(dis[i]),
                            E_state_rel_pred=es_rel, twist_deg=tw))
        res.append(dict(mol=m["mol"], n_frames=len(fr), stopped=not active[k], reason=reason[k], t_stop=t_stop[k],
                        t_end=float(fr[-1][0]),
                        T0=T0[k], state=state, picks=sel, dis_max=float(dis.max()), dis_median=float(np.median(dis)),
                        twist_max=float(max(f[5] for f in fr)), n_motif_bonds=len(quads[k])))
    return res


def shard_main(a):
    torch.set_num_threads(a.threads)
    mA, cA = build_model_from_dir(ROOT / a.models[0])
    mB, cB = build_model_from_dir(ROOT / a.models[1])
    mA, mB = mA.to(a.device), mB.to(a.device)
    for mdl in (mA, mB):
        for p in mdl.parameters():
            p.requires_grad_(False)
    mols = pickle.load(open(ROOT / a.data, "rb"))
    motifs = set(json.load(open(ROOT / a.motifs))) if a.motifs else None
    sel = select_molecules(mols, a.frac, a.zinc, motifs, salt=f"{a.round}:", frac_extra=a.frac_extra)
    sel = [m for m in sel if zlib.crc32(m["mol"].encode()) % a.nshards == a.shard]
    if a.max_mols:
        sel = sel[: a.max_mols]
    out = ROOT / a.workdir / f"shard{a.shard:03d}"
    out.mkdir(parents=True, exist_ok=True)
    by_state = {1: [], 2: []}
    for m in sel:
        salt = "state:" if a.round == "s1r1" else f"state:{a.round}:"
        by_state[2 if h01(salt + m["mol"]) < a.frac_s2 else 1].append(m)
    t_start = time.time()
    for state, ms in by_state.items():
        ms.sort(key=lambda m: len(m["Z"]))  # similar sizes per chunk
        for c in range(0, len(ms), a.chunk):
            f = out / f"s{state}_chunk{c // a.chunk:04d}.pkl"
            if f.exists():
                continue
            if (time.time() - t_start) / 3600 > a.time_limit_h:
                print("time limit reached", flush=True)
                return
            t0 = time.time()
            part = ms[c : c + a.chunk]
            res = run_chunk(mA, mB, part, state, a.fs, a.dt, a.every, a.seed, gap_stop=a.gap_stop,
                            picks=tuple(a.picks.split(",")), vsalt="v" if a.round == "s1r1" else f"v{a.round}",
                            max_picks=a.max_picks)
            for r, m in zip(res, part):
                r.update(Z=m["Z"], R_eq=m["R"], source=m["source"], gaps_eq=np.asarray(m["gaps"], float),
                         n_heavy=int(m["n_heavy"]))
            pickle.dump(res, open(f, "wb"))
            print(f"state {state} chunk {c // a.chunk} n {len(part)} frames/mol "
                  f"{np.mean([r['n_frames'] for r in res]):.1f} stopped {np.mean([r['stopped'] for r in res]):.2f} "
                  f"{time.time() - t0:.0f}s", flush=True)


def merge_main(a):
    tasks, info = [], dict(n_mols=0, n_no_frames=0, n_stopped=0, per_state={1: 0, 2: 0})
    for f in sorted((ROOT / a.workdir).glob("shard*/s*_chunk*.pkl")):
        for r in pickle.load(open(f, "rb")):
            info["n_mols"] += 1
            info["n_stopped"] += int(r.get("stopped", False))
            if not r.get("picks"):
                info["n_no_frames"] += 1
                continue
            info["per_state"][r["state"]] += 1
            tag = "" if a.round == "s1r1" else f"r{a.round.split('r')[-1]}"
            for k, p in enumerate(r["picks"]):
                tasks.append(dict(id=f"{r['mol']}_{tag}s{r['state']}md{k}", mol=r["mol"], Z=r["Z"], R_eq=r["R_eq"], R=p["R"],
                                  T=r["T0"], source=r["source"], gaps_eq=r["gaps_eq"], n_heavy=r["n_heavy"],
                                  mode="given", sampling=dict(round=a.round, state=r["state"], kind=p["kind"],
                                                              t_fs=p["t_fs"], T0=r["T0"],
                                                              disagreement=p["disagreement"],
                                                              gaps_pred=[float(x) for x in p["gaps_pred"]],
                                                              E0_rel_pred=p["E0_rel_pred"],
                                                              twist_deg=p.get("twist_deg"),
                                                              models=a.models)))
    info["n_tasks"] = len(tasks)
    rs = [r for f in sorted((ROOT / a.workdir).glob("shard*/s*_chunk*.pkl")) for r in pickle.load(open(f, "rb"))]
    for st in (1, 2):
        sub = [r for r in rs if r.get("state") == st]
        info[f"stop_reasons_S{st}"] = {str(k): sum(r.get("reason") == k for r in sub)
                                       for k in ("s1_gap", "bond_change", "nonfinite", None)}
        ts = [r["t_stop"] for r in sub if r.get("t_stop") is not None]
        info[f"t_stop_median_S{st}"] = float(np.median(ts)) if ts else None
    info["n_tasks_zinc"] = sum(t["source"] == "ZINC" for t in tasks)
    info["picks_by_kind"] = {k: sum(t["sampling"]["kind"] == k for t in tasks) for k in ("rand", "committee", "relaxed", "twist")}
    tw = [r["twist_max"] for r in rs if r.get("twist_max") is not None and r.get("n_motif_bonds")]
    if tw:
        info["twist_max_traj"] = {f">{x}": int(np.sum(np.array(tw) > x)) for x in (30, 45, 60, 75)}
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(tasks, open(ROOT / a.out, "wb"))
    json.dump(info, open(ROOT / (a.out.replace(".pkl", "_info.json")), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/wb97_all_100k.pkl")
    ap.add_argument("--models", nargs=2, default=["experiments/wb97/runs/latent_k1", "experiments/wb97/runs/latent_k1_s1"])
    ap.add_argument("--frac", type=float, default=0.10, help="fraction of QCDGE-derived molecules (hash-selected)")
    ap.add_argument("--frac_s2", type=float, default=0.20, help="fraction of molecules propagated on S2")
    ap.add_argument("--zinc", action="store_true", help="also all ZINC size-extrapolation molecules (test only)")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=20, help="record every n steps (10 fs)")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_mols", type=int, default=0)
    ap.add_argument("--time_limit_h", type=float, default=1e9)
    ap.add_argument("--workdir", default="data/wb97_aux/s1r1_sampling")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--round", default="s1r1", help="round tag s1rN (ids, seeds); s1r1 reproduces round 1")
    ap.add_argument("--motifs", default=None, help="json {mol: motifs}: select these QCDGE-derived molecules (all roles)")
    ap.add_argument("--gap_stop", type=float, default=0.5, help="stop when the model's S1 gap drops below this (eV)")
    ap.add_argument("--picks", default="rand,committee", help="frame criteria per trajectory: rand,committee,relaxed,twist")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frac_extra", type=float, default=0.0,
                    help="with --motifs: additionally a round-specific hash slice of all QCDGE-derived molecules")
    ap.add_argument("--max_picks", type=int, default=0, help="at most this many frames per trajectory (0: all criteria)")
    ap.add_argument("--out", default="data/wb97/tasks/s1r1.pkl")
    a = ap.parse_args()
    merge_main(a) if a.merge else shard_main(a)
