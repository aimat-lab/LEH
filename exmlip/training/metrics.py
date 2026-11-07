"""Spectral metrics: per-state MAE, size-resolved errors, outliers, mode intrusion."""
import numpy as np


def stack(preds, key, n):
    out = np.full((len(preds), n), np.nan)
    for i, p in enumerate(preds):
        v = np.asarray(p[key], float)[:n]
        out[i, : len(v)] = v
    return out


def r2(pred, ref):
    """Coefficient of determination over finite pairs (nan if fewer than 3)."""
    pred, ref = np.asarray(pred, float).ravel(), np.asarray(ref, float).ravel()
    m = np.isfinite(pred) & np.isfinite(ref)
    if m.sum() < 3:
        return np.nan
    p, r = pred[m], ref[m]
    sst = ((r - r.mean()) ** 2).sum()
    return float(1 - ((p - r) ** 2).sum() / sst) if sst > 0 else np.nan


def spectral_metrics(preds, n_states, n_sup, outlier=0.5):
    P = stack(preds, "gaps_pred", n_states)
    R = stack(preds, "gaps_ref", n_states)
    E = P - R
    res = {}
    for s in range(n_states):
        e = E[:, s][~np.isnan(E[:, s])]
        res[f"mae_S{s+1}"] = float(np.mean(np.abs(e))) if len(e) else np.nan
        res[f"r2_S{s+1}"] = r2(P[:, s], R[:, s])
    # per-state R2 averaged over supervised roots (pooling roots would inflate R2 by the spread between roots)
    res["r2_sup"] = float(np.nanmean([res[f"r2_S{s+1}"] for s in range(n_sup)]))
    sup = E[:, :n_sup]
    res["mae_sup"] = float(np.nanmean(np.abs(sup)))
    res["rmse_sup"] = float(np.sqrt(np.nanmean(sup**2)))
    res["outlier_frac_S1"] = float(np.nanmean(np.abs(E[:, 0]) > outlier))
    res["coverage_sup"] = float(np.mean(~np.isnan(P[:, :n_sup])))
    if n_states > n_sup:
        uns = E[:, n_sup:]
        res["mae_unsup"] = float(np.nanmean(np.abs(uns)))
        res["signed_unsup"] = float(np.nanmean(uns))
    # eigenvalue counting (latent models): number of model eigenvalues below the highest reference root
    if "eigvals" in preds[0]:
        cnt_err, neg = [], []
        for p in preds:
            ref = np.asarray(p["gaps_ref"], float)[:n_states]
            ev = np.asarray(p["eigvals"])
            ec = np.nanmax(ref) - 0.05
            n_ref = int(np.sum(ref < ec))
            cnt_err.append(int(np.sum(ev < ec)) - n_ref)
            neg.append(np.any(ev < 0.5))
        cnt_err = np.array(cnt_err)
        res["count_err_mean"] = float(cnt_err.mean())
        res["count_err_abs"] = float(np.abs(cnt_err).mean())
        res["frac_extra_modes"] = float(np.mean(cnt_err > 0))
        res["frac_missing_modes"] = float(np.mean(cnt_err < 0))
        res["frac_unphysical_low"] = float(np.mean(neg))
    return res


def by_size(preds, n_states, n_sup, key="n_heavy"):
    sizes = np.array([p[key] for p in preds])
    return {int(s): spectral_metrics([p for p, z in zip(preds, sizes) if z == s], n_states, n_sup)
            for s in np.unique(sizes)}


def element_referenced(E, Zs, elems=(1, 6, 7, 8, 9)):
    """E minus a least-squares per-element linear reference (fitted to E itself): removes the composition part,
    so that R2 of total energies across different molecules is not trivially ~1."""
    A = np.array([[np.sum(np.asarray(Z) == z) for z in elems] for Z in Zs], float)
    coef = np.linalg.lstsq(A, E, rcond=None)[0]
    return E - A @ coef, A, coef


def energy_force_metrics(preds, n_states_e=3, n_force=3):
    """S0, excitation energies, total excited-state energies E_Sn = E0 + dE_n, and forces: MAE and R2.

    Error decomposition for E_Sn: err_Sn = err_E0 + err_dEn; reports var shares of the two parts and their covariance.
    Units: eV, eV/A. Only entries with an E0 label are used.
    """
    P = [p for p in preds if p.get("E0_ref") is not None and np.isfinite(p.get("E0_ref", np.nan))]
    out = {"n_with_E0": len(P)}
    if not P:
        return out
    e0p = np.array([p["E0_pred"] for p in P]); e0r = np.array([p["E0_ref"] for p in P])
    nat = np.array([p["n_atoms"] for p in P], float)
    out["mae_E0"] = float(np.mean(np.abs(e0p - e0r)))
    out["mae_E0_per_atom_meV"] = float(1000 * np.mean(np.abs(e0p - e0r) / nat))
    if all("Z" in p for p in P) and len(P) > 10:
        rr, A, coef = element_referenced(e0r, [p["Z"] for p in P])
        out["r2_E0_referenced"] = r2(e0p - A @ coef, rr)
    gp = stack(P, "gaps_pred", n_states_e); gr = stack(P, "gaps_ref", n_states_e)
    for n in range(n_states_e):
        d0, dg = e0p - e0r, gp[:, n] - gr[:, n]
        m = np.isfinite(dg)
        tot = d0[m] + dg[m]
        out[f"mae_dE{n+1}"] = float(np.mean(np.abs(dg[m])))
        out[f"r2_dE{n+1}"] = r2(gp[m, n], gr[m, n])
        out[f"mae_ES{n+1}"] = float(np.mean(np.abs(tot)))
        if "r2_E0_referenced" in out:
            esr = (e0r + gr[:, n])[m]
            rr, A, coef = element_referenced(esr, [p["Z"] for p, k in zip(P, m) if k])
            out[f"r2_ES{n+1}_referenced"] = r2((e0p + gp[:, n])[m] - A @ coef, rr)
        v0, vg, c = np.var(d0[m]), np.var(dg[m]), np.cov(d0[m], dg[m])[0, 1]
        out[f"ES{n+1}_errvar_share_E0"] = float(v0 / np.var(tot))
        out[f"ES{n+1}_errvar_share_dE"] = float(vg / np.var(tot))
        out[f"ES{n+1}_errvar_share_cov"] = float(2 * c / np.var(tot))
    F = [p for p in P if p.get("F0_ref") is not None and "F0_pred" in p]
    if F:
        f0p = np.concatenate([p["F0_pred"].ravel() for p in F]); f0r = np.concatenate([p["F0_ref"].ravel() for p in F])
        out["mae_F0"] = float(np.mean(np.abs(f0p - f0r))); out["r2_F0"] = r2(f0p, f0r)
        for s in range(n_force if any(p.get("Fex_ref") is not None for p in F) else 0):  # eq-only sets: S0 forces only
            fp = np.concatenate([p["Fex_pred"][:, s].ravel() for p in F if p.get("Fex_ref") is not None])
            fr = np.concatenate([p["Fex_ref"][:, s].ravel() for p in F if p.get("Fex_ref") is not None])
            gp_ = np.concatenate([(p["Fex_pred"][:, s] - p["F0_pred"]).ravel() for p in F if p.get("Fex_ref") is not None])
            gr_ = np.concatenate([(p["Fex_ref"][:, s] - p["F0_ref"]).ravel() for p in F if p.get("Fex_ref") is not None])
            m = np.isfinite(fr)
            out[f"mae_FS{s+1}"] = float(np.mean(np.abs(fp[m] - fr[m]))); out[f"r2_FS{s+1}"] = r2(fp, fr)
            out[f"mae_Fgap{s+1}"] = float(np.mean(np.abs(gp_[m] - gr_[m]))); out[f"r2_Fgap{s+1}"] = r2(gp_, gr_)
    return out


def osc_metrics(preds, n_states):
    """Oscillator strengths (heads with dipoles): per-state MAE and R2 of f, plus MAE of sqrt(f) and the share of
    bright states (f > 0.1) predicted bright. Empty dict if no predictions/labels."""
    P = [p for p in preds if p.get("osc_pred") is not None and p.get("osc_ref") is not None]
    if not P:
        return {}
    fp = stack(P, "osc_pred", n_states); fr = stack(P, "osc_ref", n_states)
    out = {}
    for s in range(n_states):
        m = np.isfinite(fp[:, s]) & np.isfinite(fr[:, s])
        if m.sum() > 2:
            out[f"osc_mae_S{s+1}"] = float(np.mean(np.abs(fp[m, s] - fr[m, s])))
            out[f"osc_r2_S{s+1}"] = r2(fp[m, s], fr[m, s])
    m = np.isfinite(fp) & np.isfinite(fr)
    out["osc_mae"] = float(np.mean(np.abs(fp[m] - fr[m])))
    out["osc_sqrt_mae"] = float(np.mean(np.abs(np.sqrt(np.clip(fp[m], 0, None)) - np.sqrt(np.clip(fr[m], 0, None)))))
    bright = fr[m] > 0.1
    out["bright_recall"] = float(np.mean(fp[m][bright] > 0.1)) if bright.any() else np.nan
    return out
