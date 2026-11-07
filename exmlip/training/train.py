"""Generic trainer for spectra-only (QCDGE) and energy+force (TURBOMOLE) data.

Units: energies eV, forces eV/A, coordinates A.
"""
import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from exmlip.data.batching import MolDataset, collate, iterate
from exmlip.models.model import ExcitedStateModel

DEFAULT = dict(
    data=None, split=None, out=None, seed=0,
    n_states=10,            # number of reference roots kept for evaluation
    n_sup=6,                # supervised roots S1..S_n_sup
    head="latent", head_kwargs={}, enc_kwargs=dict(channels=32, max_L=1, num_interactions=2, r_max=5.0),
    alpha_offdiag=0.0,      # size-normalized off-diagonal penalty
    w_gap=1.0, w_e0=0.0, w_f0=0.0, w_fex=0.0, n_force=0,
    w_osc=0.0,              # oscillator-strength loss (on sqrt f) for heads with dipoles=True
    lr=2e-3, batch_size=32, epochs=100, patience=12, max_hours=4.0, threads=4,
    dtype="float32", eval_batch_size=128, max_train=None, device=None,
    ema_decay=0.0,          # exponential moving average of the weights (MACE uses 0.99); 0 = off
    max_updates=None,       # if set, epochs = ceil(max_updates * batch_size / n_train) (MACE heuristic ~200k)
    freeze_e_ref=False,     # keep the least-squares per-element E0 references fixed (MACE-style E0s)
    wall_hours=None,        # per-job budget: checkpoint to last.pt and exit (code 3) before exceeding it
    lr_schedule="plateau",  # "plateau": ReduceLROnPlateau on val loss (factor 0.5) + early stopping;
                            # "cosine": linear warm-up then cosine decay to lr_min_frac*lr over the whole update budget,
                            # stepped per update; no early stopping (best-on-val checkpoint is still kept)
    warmup_updates=1000,
    lr_min_frac=0.01,
    init_from=None,           # warm start (active-learning sampling models): run directory whose best.pt weights,
                              # element set and gap shift are taken over; the new run then trains as configured
    force_frames_only=False,  # control ablation: X-MACE's per-frame force mask, i.e. no S0 forces in train/val on
                              # frames without S1..S_n_force forces (X-MACE cannot mask forces per state)
    gap_state_w=None,         # per-state weights of the gap loss (list, >= n_sup entries); None = uniform
    w_lowgap=0.0,             # frame weight 1 + w_lowgap * exp(-max(S1_ref, lowgap_floor) / lowgap_tau) on all
    lowgap_tau=1.0,           # its gap terms (emphasises geometries near S1/S0 crossings); 0 = off
    lowgap_floor=0.5,         # eV; no further up-weighting below (TDDFT unreliable at the crossing itself)
    w_eex=0.0,                # loss on total excited-state energies E0 + gap_n (key "Eex", frames whose E0 is unknown)
    sort_pred=True,           # predict(): sort the gaps (adiabatic order). False for direct heads whose columns are
                              # fixed states (e.g. [T1, S1] of different multiplicity, which cross without coupling)
    columns=None,             # optional names of the gap columns (e.g. [T1, S1]); stored in config.yaml for inference
    w_pop=0.0,                # latent heads: loss sum_A (w_An - p_An)^2 between the per-atom eigenvector weights and the
                              # TD-DFT excitation populations (both folded onto heavy atoms), S1..S_n_pop; training loss
                              # only (checkpoint selection on the same validation loss as without it)
    n_pop=3,
    pop_target="avg",         # p = (hole + electron)/2 ("avg"), "hole", "elec", or "min" (closer of hole and electron)
    pop_labels=None,          # pickle {dataset index: dict(pop_hole, pop_elec, pop_fold, pop_mask)}
                              # (experiments/eigvec/tddft/pop_labels.py)
)

EXIT_RESUME = 3  # exit code: checkpoint written, training not finished (hpc/train_chain.sh resubmits)


def get_device(cfg):
    return torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))


DEV = torch.device("cpu")


def to_dev(b):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in b.items()}


def setup(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    dt = dict(float32=torch.float32, float64=torch.float64)[cfg["dtype"]]
    torch.set_default_dtype(dt)
    global DEV
    DEV = get_device(cfg)
    return dt


def load(cfg):
    with open(cfg["data"], "rb") as f:
        mols = pickle.load(f)
    split = json.load(open(cfg["split"]))
    if cfg.get("pop_labels"):
        with open(cfg["pop_labels"], "rb") as f:
            pops = pickle.load(f)
        for i, lab in pops.items():
            mols[i] = {**mols[i], **lab}
    ds = MolDataset(mols)
    tr = ds.subset(split["train"][: cfg["max_train"]] if cfg["max_train"] else split["train"])
    va = ds.subset(split["val"])
    tests = {k[5:]: ds.subset(v) for k, v in split.items() if k.startswith("test_")}
    if cfg.get("force_frames_only"):
        nf = cfg["n_force"]

        def strip(m):
            ok = "Fex" in m and np.asarray(m["Fex"]).shape[1] >= nf and np.all(np.isfinite(np.asarray(m["Fex"])[:, :nf]))
            return m if ok else {k: v for k, v in m.items() if k != "F0"}
        tr, va = MolDataset([strip(m) for m in tr.mols]), MolDataset([strip(m) for m in va.mols])
    return tr, va, tests


def z_table(ds):
    return sorted(set(int(z) for m in ds.mols for z in m["Z"]))


def has_e0(cfg):
    """The model has a ground-state energy head if E0 or total excited-state energies are trained."""
    return cfg["w_e0"] > 0 or cfg.get("w_eex", 0.0) > 0


def gap_weights(ref, mask, cfg):
    """Weights of the squared gap errors (frames x states); with the defaults the mask itself (uniform mean)."""
    W = mask.to(ref.dtype)
    if cfg.get("gap_state_w") is not None:
        W = W * torch.as_tensor(cfg["gap_state_w"], dtype=ref.dtype, device=ref.device)[None, :ref.shape[1]]
    if cfg.get("w_lowgap", 0.0) > 0:
        s1 = torch.nan_to_num(ref[:, 0], nan=1e9).clamp(min=cfg.get("lowgap_floor", 0.5))
        W = W * (1 + cfg["w_lowgap"] * torch.exp(-s1 / cfg.get("lowgap_tau", 1.0)))[:, None]
    return W


def batch_loss(model, b, cfg, train=True):
    nf = cfg["n_force"] if cfg["w_fex"] > 0 else 0
    need_f = cfg["w_f0"] > 0 or nf > 0
    npop = cfg["n_pop"] if cfg.get("w_pop", 0.0) > 0 and "pop_hole" in b else 0
    out = model(b["Z"], b["R"], b["batch"], b["nmol"], compute_forces=need_f, force_states=nf, training=train,
                pop_states=npop)
    ns = cfg["n_sup"]
    ref = b["gaps"][:, :ns]
    pred = out["gaps"][:, :ns]
    mask = (~torch.isnan(ref)) & out["valid"][:, :ns]
    diff = torch.where(mask, pred - torch.nan_to_num(ref), 0.0)
    W = gap_weights(ref, mask, cfg)
    loss = cfg["w_gap"] * (W * diff**2).sum() / W.sum().clamp(min=1e-12)
    logs = {"gap_mae": (diff.abs().sum() / mask.sum().clamp(min=1)).item()}
    if cfg["w_osc"] > 0 and "osc" in out and "osc" in b:
        ref_f = b["osc"][:, :ns]
        mf = (~torch.isnan(ref_f)) & out["valid"][:, :ns]
        pf = torch.sqrt(out["osc"][:, :ns].clamp(min=0) + 1e-10)
        df = torch.where(mf, pf - torch.sqrt(torch.nan_to_num(ref_f).clamp(min=0)), 0.0)
        loss = loss + cfg["w_osc"] * (df ** 2).sum() / mf.sum().clamp(min=1)
        logs["osc_mae"] = (torch.where(mf, out["osc"][:, :ns] - torch.nan_to_num(ref_f), 0.0).abs().sum()
                           / mf.sum().clamp(min=1)).item()
    if npop and "atom_w" in out:
        W = out["atom_w"]
        Wf = torch.zeros_like(W).index_add(0, b["pop_fold"], W)  # hydrogens folded onto their heavy atom
        h, e = b["pop_hole"][:, :npop], b["pop_elec"][:, :npop]
        mp = b["pop_mask"][:, :npop] & out["valid"][:, :npop]

        def per_mol(x):
            return torch.zeros(b["nmol"], npop, dtype=x.dtype, device=x.device).index_add(0, b["batch"], x)
        avg = 0.5 * (h + e)
        d2 = {"avg": lambda: per_mol((Wf - avg) ** 2), "hole": lambda: per_mol((Wf - h) ** 2),
              "elec": lambda: per_mol((Wf - e) ** 2),
              "min": lambda: torch.minimum(per_mol((Wf - h) ** 2), per_mol((Wf - e) ** 2))}[cfg["pop_target"]]()
        if train:
            loss = loss + cfg["w_pop"] * torch.where(mp, d2, 0.0).sum() / mp.sum().clamp(min=1)
        shared = 1.0 - 0.5 * per_mol((Wf - avg).abs())  # sum_A min(w, p) for normalized distributions
        logs["pop_shared"] = (torch.where(mp, shared, 0.0).sum() / mp.sum().clamp(min=1)).item()
    if cfg["alpha_offdiag"] > 0 and "H" in out:
        H = out["H"]
        off = H - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
        M = out["M"].to(H.dtype)
        loss = loss + cfg["alpha_offdiag"] * ((off**2).sum((1, 2)) / M).mean()
    if cfg["w_e0"] > 0 and "E0" in b:
        na = torch.bincount(b["batch"], minlength=b["nmol"]).to(pred.dtype)
        me = ~torch.isnan(b["E0"])
        d = torch.where(me, (out["E0"] - torch.nan_to_num(b["E0"])) / na, 0.0)
        loss = loss + cfg["w_e0"] * (d**2).sum() / me.sum().clamp(min=1)
        logs["e0_mae_per_atom"] = (d.abs().sum() / me.sum().clamp(min=1)).item()
    if cfg.get("w_eex", 0.0) > 0 and "Eex" in b:
        ref_e = b["Eex"][:, :ns]
        me = (~torch.isnan(ref_e)) & out["valid"][:, :ns]
        d = torch.where(me, out["E0"][:, None] + pred - torch.nan_to_num(ref_e), 0.0)
        loss = loss + cfg["w_eex"] * (d ** 2).sum() / me.sum().clamp(min=1)
        logs["eex_mae"] = (d.abs().sum() / me.sum().clamp(min=1)).item()
    if cfg["w_f0"] > 0 and "F0" in b:
        mf = ~torch.isnan(b["F0"])
        d = torch.where(mf, out["F0"] - torch.nan_to_num(b["F0"]), 0.0)
        loss = loss + cfg["w_f0"] * (d**2).sum() / mf.sum().clamp(min=1)
        logs["f0_mae"] = (d.abs().sum() / mf.sum().clamp(min=1)).item()
    if nf > 0 and "Fex" in b:
        fex_pred = out["F0"][:, None, :] + out["Fgap"]  # F_Sn = F0 - grad gap
        ref = b["Fex"]
        m = ~torch.isnan(ref)
        m = m & out["valid"][b["batch"], :nf, None]
        d = torch.where(m, fex_pred - torch.nan_to_num(ref), 0.0)
        loss = loss + cfg["w_fex"] * (d**2).sum() / m.sum().clamp(min=1)
        logs["fex_mae"] = (d.abs().sum() / m.sum().clamp(min=1)).item()
    return loss, logs, out


@torch.no_grad()
def predict(model, ds, cfg, forces=False):
    """Returns per-molecule predictions (sorted adiabatic gaps, all eigenvalues for latent heads)."""
    res = []
    model.eval()
    nf = cfg["n_force"] if forces else 0
    for mols in iterate(ds, cfg["eval_batch_size"] if not forces else 16, False):
        b = to_dev(collate(mols, cfg["n_states"], nf))
        with torch.enable_grad() if forces else torch.no_grad():
            out = model(b["Z"], b["R"], b["batch"], b["nmol"], compute_forces=forces, force_states=nf)
        g = out["gaps"].detach().cpu()
        val = out["valid"].cpu()
        if model.head_type != "latent":  # direct heads: outputs beyond n_sup are unsupervised, not predictions
            val = val.clone(); val[:, cfg["n_sup"]:] = False
        g = torch.where(val, g, torch.nan)
        if cfg.get("sort_pred", True):
            gs = torch.sort(torch.nan_to_num(g, nan=1e6), dim=1)[0]
            gs = torch.where(gs > 1e5, torch.nan, gs)
        else:
            gs = g
        ptr = np.cumsum([0] + [len(m["Z"]) for m in mols])
        for i, m in enumerate(mols):
            r = dict(gaps_pred=gs[i].numpy().astype(float), gaps_ref=np.asarray(m["gaps"], float), id=m.get("id"),
                     n_atoms=len(m["Z"]), n_heavy=int((m["Z"] > 1).sum()), Z=np.asarray(m["Z"]))
            if "osc" in out:
                r["osc_pred"] = out["osc"][i].detach().cpu().numpy().astype(float)
                r["osc_ref"] = m.get("osc")
            if "eigvals" in out:
                Mi = int(out["M"][i])
                r["eigvals"] = out["eigvals"][i, :Mi].detach().cpu().numpy().astype(float)
            if "E0" in out:
                r["E0_pred"] = float(out["E0"][i]); r["E0_ref"] = m.get("E0")
            if forces:
                sl = slice(ptr[i], ptr[i + 1])
                r["F0_pred"] = out["F0"][sl].detach().cpu().numpy()
                r["Fex_pred"] = (out["F0"][sl, None] + out["Fgap"][sl]).detach().cpu().numpy()
                r["F0_ref"], r["Fex_ref"] = m.get("F0"), m.get("Fex")
            res.append(r)
    return res


def evaluate(model, ds, cfg):
    model.eval()
    tot, n = 0.0, 0
    logs_acc = {}
    with torch.enable_grad() if (cfg["w_f0"] > 0 or cfg["w_fex"] > 0) else torch.no_grad():
        for mols in iterate(ds, cfg["eval_batch_size"] if cfg["w_fex"] == 0 else 16, False):
            b = to_dev(collate(mols, cfg["n_states"], cfg["n_force"]))
            loss, logs, _ = batch_loss(model, b, cfg, train=False)
            tot += loss.item() * len(mols); n += len(mols)
            for k, v in logs.items():
                logs_acc[k] = logs_acc.get(k, 0) + v * len(mols)
    if n == 0:
        return math.inf, {}
    return tot / n, {k: v / n for k, v in logs_acc.items()}


def train(cfg):
    dt = setup(cfg)
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    tr, va, tests = load(cfg)
    zt = z_table(MolDataset(tr.mols + va.mols + [m for t in tests.values() for m in t.mols]))
    # shift: mean of supervised gaps
    allg = np.concatenate([np.asarray(m["gaps"][: cfg["n_sup"]], float) for m in tr.mols])
    shift = float(np.nanmean(allg)) if np.isfinite(allg).any() else 0.0  # 0: no gap labels (ground-state-only model)
    src_meta = None
    if cfg.get("init_from"):
        src_meta = torch.load(Path(cfg["init_from"]) / "meta.pt")
        assert list(src_meta["z_table"]) == list(zt), f"element sets differ: {src_meta['z_table']} vs {zt}"
        shift = float(src_meta["shift"])
    model = ExcitedStateModel(zt, cfg["n_states"], head=cfg["head"], enc_kwargs=cfg["enc_kwargs"],
                              head_kwargs=cfg["head_kwargs"], predict_e0=has_e0(cfg), shift=shift).to(dt)
    if cfg["w_e0"] > 0:  # per-element reference energies by least squares
        elems = zt
        withE = [m for m in tr.mols if "E0" in m]
        A = np.array([[np.sum(m["Z"] == z) for z in elems] for m in withE], float)
        y = np.array([m["E0"] for m in withE])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        model.e0_head.e_ref.data = torch.tensor(coef, dtype=dt)
        if cfg.get("freeze_e_ref"):
            model.e0_head.e_ref.requires_grad_(False)
    if src_meta is not None:  # weights incl. element references of the source run (same architecture required)
        model.load_state_dict(torch.load(Path(cfg["init_from"]) / "best.pt", map_location="cpu"))
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    if cfg.get("max_updates"):
        cfg["epochs"] = int(math.ceil(cfg["max_updates"] * cfg["batch_size"] / max(len(tr), 1)))
    ema = None
    if cfg.get("ema_decay", 0) > 0:
        dec = cfg["ema_decay"]
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(dec), use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-6)
    cosine = cfg.get("lr_schedule", "plateau") == "cosine"
    if cosine:
        n_total = cfg["epochs"] * int(math.ceil(len(tr) / cfg["batch_size"]))
        wu, fmin = cfg.get("warmup_updates", 1000), cfg.get("lr_min_frac", 0.01)

        def lr_factor(step):
            if step < wu:
                return (step + 1) / wu
            x = min(1.0, (step - wu) / max(1, n_total - wu))
            return fmin + (1 - fmin) * 0.5 * (1 + math.cos(math.pi * x))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, cfg["patience"] // 3))
    rng = np.random.default_rng(cfg["seed"])
    best, bad, t0 = math.inf, 0, time.time()
    hist = []
    log = open(out / "train.log", "a")
    start_ep, h_prev, training_done, t_last = 0, 0.0, False, 0.0
    ck = out / "last.pt"
    if ck.exists():  # resume (chained jobs): restores weights, EMA, optimiser, scheduler and RNG states
        st = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None and st.get("ema") is not None:
            ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"])
        start_ep, best, bad, h_prev = st["epoch"], st["best"], st["bad"], st["hours"]
        training_done, hist = st.get("training_done", False), [None] * st["epoch"]
        print(f"resumed from {ck}: epoch {start_ep} best {best:.5f} hours {h_prev:.2f}", file=log, flush=True)

    def hours():
        return h_prev + (time.time() - t0) / 3600

    def checkpoint(ep, done=False):
        st = dict(model=model.state_dict(), ema=ema.state_dict() if ema is not None else None,
                  opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.bit_generator.state,
                  torch_rng=torch.get_rng_state(), epoch=ep, best=best, bad=bad, hours=hours(), training_done=done)
        torch.save(st, out / "last.tmp"); (out / "last.tmp").replace(ck)

    print(f"device {DEV} threads {torch.get_num_threads()}", file=log, flush=True)
    print(f"params {nparam} train {len(tr)} val {len(va)} shift {shift:.3f} epochs {cfg['epochs']} "
          f"batch {cfg['batch_size']} ema {cfg.get('ema_decay', 0)}", file=log, flush=True)
    for ep in range(start_ep, cfg["epochs"]):
        if training_done:
            break
        if cfg.get("wall_hours") and (time.time() - t0) / 3600 + 1.2 * t_last / 3600 > cfg["wall_hours"]:
            checkpoint(ep)
            print(f"wall budget reached before epoch {ep}: checkpointed, exit {EXIT_RESUME}", file=log, flush=True)
            raise SystemExit(EXIT_RESUME)
        model.train()
        te = time.time(); acc = {}; nb = 0; nskip = 0
        for mols in iterate(tr, cfg["batch_size"], True, rng):
            b = to_dev(collate(mols, cfg["n_states"], cfg["n_force"]))
            loss, logs, _ = batch_loss(model, b, cfg, train=True)
            opt.zero_grad()
            loss.backward()
            bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
            if not torch.isfinite(loss) or bad_grad:
                nskip += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if cosine:
                sched.step()
            if ema is not None:
                ema.update_parameters(model)
            nb += 1
            for k, v in logs.items():
                acc[k] = acc.get(k, 0) + v
        eval_model = ema.module if ema is not None else model
        vl, vlogs = evaluate(eval_model, va, cfg)
        if not cosine:
            sched.step(vl)
        rec = dict(epoch=ep, val_loss=vl, **{"val_" + k: v for k, v in vlogs.items()},
                   **{"train_" + k: v / max(nb, 1) for k, v in acc.items()}, lr=opt.param_groups[0]["lr"],
                   skipped=nskip, t_epoch=time.time() - te)
        hist.append(rec)
        print(json.dumps(rec), file=log, flush=True)
        if vl < best:
            best, bad = vl, 0
            torch.save(eval_model.state_dict(), out / "best.pt")
        else:
            bad += 1
        t_last = time.time() - te
        stop = hours() > cfg["max_hours"] or (not cosine and (bad >= cfg["patience"] or opt.param_groups[0]["lr"] < 1e-5))
        if cfg.get("wall_hours"):
            checkpoint(ep + 1, done=stop or ep + 1 == cfg["epochs"])
        if stop:
            break
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV))
    torch.save(dict(z_table=zt, shift=shift, n_params=nparam), out / "meta.pt")
    preds = {"val": predict(model, va, cfg)}
    for k, t in tests.items():
        preds["test_" + k] = predict(model, t, cfg, forces=cfg["w_fex"] > 0)
    with open(out / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    json.dump(dict(best_val=best, epochs=len(hist), hours=hours(), n_params=nparam),
              open(out / "summary.json", "w"))
    return model


def build_model_from_dir(d):
    cfg = yaml.safe_load(open(Path(d) / "config.yaml"))
    meta = torch.load(Path(d) / "meta.pt")
    dt = dict(float32=torch.float32, float64=torch.float64)[cfg["dtype"]]
    torch.set_default_dtype(dt)
    model = ExcitedStateModel(meta["z_table"], cfg["n_states"], head=cfg["head"], enc_kwargs=cfg["enc_kwargs"],
                              head_kwargs=cfg["head_kwargs"], predict_e0=has_e0(cfg), shift=meta["shift"]).to(dt)
    model.load_state_dict(torch.load(Path(d) / "best.pt", map_location="cpu"))
    return model.eval(), cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml values)")
    a = ap.parse_args()
    cfg = dict(DEFAULT)
    cfg.update(yaml.safe_load(open(a.config)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    train(cfg)
