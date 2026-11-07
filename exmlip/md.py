"""Adiabatic NVE molecular dynamics on an energy-ordered S_n surface (model or TURBOMOLE)."""
import numpy as np
import torch

AMU = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 16: 32.06}
# eV, A, amu, fs: a [A/fs^2] = F[eV/A] / m[amu] * CONV
CONV = 1.602176634e-19 / 1.66053906660e-27 * 1e-10  # (eV/A/amu) -> A/fs^2 = 0.009648533
KB = 8.617333262e-5  # eV/K


def maxwell_boltzmann(Z, T, rng):
    m = np.array([AMU[int(z)] for z in Z])
    v = rng.normal(size=(len(Z), 3)) * np.sqrt(KB * T / m * CONV)[:, None]
    v -= (m[:, None] * v).sum(0) / m.sum()
    return v


def kinetic(Z, v):
    m = np.array([AMU[int(z)] for z in Z])
    return 0.5 * (m[:, None] * v**2).sum() / CONV


class ModelPES:
    def __init__(self, model, state, n_states_out=4):
        self.model, self.state, self.nout = model, state, n_states_out

    def __call__(self, Z, R):
        dt = torch.get_default_dtype()
        Zt = torch.tensor(Z); Rt = torch.tensor(R, dtype=dt)
        b = torch.zeros(len(Z), dtype=torch.long)
        out = self.model(Zt, Rt, b, 1, compute_forces=True, force_states=max(self.state, 1))
        gaps = out["gaps"][0].detach().numpy().astype(float)
        E0 = float(out["E0"][0])
        if self.state == 0:
            return E0, out["F0"].detach().numpy().astype(float), gaps
        # adiabatic ordering: the latent eigenvalues are already sorted
        F = (out["F0"] + out["Fgap"][:, self.state - 1]).detach().numpy().astype(float)
        return E0 + gaps[self.state - 1], F, gaps


def run_nve(pes, Z, R0, v0, dt_fs=0.5, n_steps=400, every=1):
    m = np.array([AMU[int(z)] for z in Z])[:, None]
    R, v = R0.copy(), v0.copy()
    E, F, gaps = pes(Z, R)
    traj = dict(R=[R.copy()], Epot=[E], Ekin=[kinetic(Z, v)], gaps=[gaps])
    for step in range(1, n_steps + 1):
        v = v + 0.5 * dt_fs * F / m * CONV
        R = R + dt_fs * v
        E, F, gaps = pes(Z, R)
        v = v + 0.5 * dt_fs * F / m * CONV
        if step % every == 0:
            traj["R"].append(R.copy()); traj["Epot"].append(E); traj["Ekin"].append(kinetic(Z, v)); traj["gaps"].append(gaps)
    return {k: np.array(v) for k, v in traj.items()}


class TurbomolePES:
    """S_n (n>=1) or S0 surface from TURBOMOLE; restarts each step from the previous step's MOs."""

    def __init__(self, template, workroot, state=1, nroots=10, nthreads=8):
        from pathlib import Path
        self.tpl, self.root, self.state, self.nroots, self.nt = Path(template), Path(workroot), state, nroots, nthreads
        self.k = 0
        self.records = []

    def __call__(self, Z, R):
        import shutil
        from exmlip.qc import turbomole as tm
        wd = self.root / f"step{self.k % 2}"
        r = tm.single_point(self.tpl, wd, Z, R, nroots=self.nroots, nexgrad=max(self.state, 1), nthreads=self.nt, keep=True)
        if not r["ok"]:
            raise RuntimeError(f"TURBOMOLE failed at step {self.k}: {r['flags']}")
        # next step restarts from these MOs
        nxt = self.root / "tpl_next"
        nxt.mkdir(parents=True, exist_ok=True)
        if self.tpl != nxt:
            for f in ["control", "basis", "auxbasis"]:
                shutil.copy(self.tpl / f, nxt / f)
        shutil.copy(wd / "mos", nxt / "mos")
        self.tpl = nxt
        self.k += 1
        H, G = tm.HARTREE_EV, tm.HARTREE_EV / tm.BOHR_A
        gaps = r["exc"] * H
        self.records.append(dict(R=R.copy(), E0=r["E0"] * H, F0=-r["grad0"] * G, gaps=gaps, Fex=-r["gradex"] * G))
        if self.state == 0:
            return r["E0"] * H, -r["grad0"] * G, gaps
        return (r["E0"] + r["exc"][self.state - 1]) * H, -r["gradex"][self.state - 1] * G, gaps


class BatchedModelPES:
    """Adiabatic S_n surfaces of many molecules at once. S_n uses the n-th lowest predicted gap for every head type
    (latent eigenvalues are sorted anyway; for direct heads this is the adiabatic choice)."""

    def __init__(self, model, Zs, state=1, n_valid=None, device=None):
        self.model, self.state, self.n_valid = model, state, n_valid
        self.Zs = Zs
        self.device = device or next(model.parameters()).device  # the model's device (CPU unless moved)
        self.Z = torch.tensor(np.concatenate(Zs)).to(self.device)
        self.batch = torch.cat([torch.full((len(z),), i, dtype=torch.long) for i, z in enumerate(Zs)]).to(self.device)
        self.ptr = np.cumsum([0] + [len(z) for z in Zs])

    def __call__(self, R_list):
        dt = torch.get_default_dtype()
        R = torch.tensor(np.concatenate(R_list), dtype=dt, device=self.device, requires_grad=True)
        out = self.model(self.Z, R, self.batch, len(self.Zs))
        val = out["valid"]
        if self.n_valid is not None and self.model.head_type != "latent":  # unsupervised direct outputs
            val = val.clone(); val[:, self.n_valid:] = False
        g = torch.where(val, out["gaps"], torch.full_like(out["gaps"], 1e3))
        gs = torch.sort(g, dim=1)[0]
        E0 = out["E0"]
        E = E0 + (gs[:, self.state - 1] if self.state > 0 else 0.0)
        F = -torch.autograd.grad(E.sum(), R)[0].detach().cpu().numpy()
        E, gs = E.detach().cpu().numpy().astype(float), gs.detach().cpu().numpy().astype(float)
        return E, [F[self.ptr[i]:self.ptr[i + 1]].astype(float) for i in range(len(self.Zs))], gs


def run_nve_batched(pes, Zs, R0s, v0s, dt_fs=0.5, n_steps=2000, every=10):
    ms = [np.array([AMU[int(z)] for z in Z])[:, None] for Z in Zs]
    R = [r.copy() for r in R0s]
    v = [x.copy() for x in v0s]
    E, F, gaps = pes(R)
    ek = lambda: np.array([kinetic(Z, vi) for Z, vi in zip(Zs, v)])
    traj = dict(t=[0.0], R=[[r.copy() for r in R]], Epot=[E], Ekin=[ek()], gaps=[gaps[:, :4]])
    for step in range(1, n_steps + 1):
        v = [vi + 0.5 * dt_fs * Fi / m * CONV for vi, Fi, m in zip(v, F, ms)]
        R = [ri + dt_fs * vi for ri, vi in zip(R, v)]
        E, F, gaps = pes(R)
        v = [vi + 0.5 * dt_fs * Fi / m * CONV for vi, Fi, m in zip(v, F, ms)]
        if step % every == 0:
            traj["t"].append(step * dt_fs); traj["R"].append([r.copy() for r in R]); traj["Epot"].append(E)
            traj["Ekin"].append(ek()); traj["gaps"].append(gaps[:, :4])
        if not all(np.isfinite(e) for e in E):
            break
    return traj
