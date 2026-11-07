"""Noninteracting-fragment size-consistency test (E3).

Builds A+B (+C) supersystems with random orientations and large separations; the reference
spectrum is the multiset union of the monomer reference spectra.
"""
import numpy as np
import torch

from exmlip.data.batching import collate


def rand_rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def make_cluster(mols, sep, rng):
    Zs, Rs = [], []
    for k, m in enumerate(mols):
        R = (m["R"] - m["R"].mean(0)) @ rand_rot(rng).T + np.array([k * sep, 0, 0])
        Zs.append(m["Z"]); Rs.append(R)
    ref = np.sort(np.concatenate([np.asarray(m["gaps"], float) for m in mols]))
    return dict(Z=np.concatenate(Zs), R=np.concatenate(Rs), gaps=ref)


@torch.no_grad()
def predict_mols(model, mols, n_states, n_sup=None):
    b = collate(mols, n_states)
    out = model(b["Z"], b["R"], b["batch"], b["nmol"])
    val = out["valid"].clone()
    if model.head_type != "latent" and n_sup is not None:  # unsupervised direct outputs are not predictions
        val[:, n_sup:] = False
    g = torch.where(val, out["gaps"], torch.nan).numpy().astype(float)
    return np.sort(np.where(np.isnan(g), np.inf, g), axis=1)


def fragment_test(model, pool, n_states, n_eval, n_pairs=200, sep=20.0, seed=0, kinds=("AA", "AB", "AAB"), n_sup=None):
    rng = np.random.default_rng(seed)
    res = {}
    for kind in kinds:
        errs_ref, errs_cons = [], []
        for _ in range(n_pairs):
            a, b = rng.choice(len(pool), 2, replace=False)
            parts = [pool[a] if c == "A" else pool[b] for c in kind]
            cl = make_cluster(parts, sep, rng)
            p_cl = predict_mols(model, [cl], n_states, n_sup)[0][:n_eval]
            p_mono = predict_mols(model, parts, n_states, n_sup)
            union_pred = np.sort(np.concatenate([p[np.isfinite(p)] for p in p_mono]))[:n_eval]
            ref = cl["gaps"][:n_eval]
            k = min(len(p_cl), len(ref), len(union_pred))
            errs_ref.append(np.abs(p_cl[:k] - ref[:k]))
            errs_cons.append(np.abs(p_cl[:k] - union_pred[:k]))
        er, ec = np.array(errs_ref), np.array(errs_cons)
        res[kind] = dict(mae_vs_union_ref=er.mean(0).tolist(), mae_vs_union_of_monomer_preds=ec.mean(0).tolist())
    return res
