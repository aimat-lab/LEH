"""Readout heads: extensive E0, direct multistate gap heads, latent excitation Hamiltonian."""
import math

import torch
from torch import nn

from .eigh import safe_eigh

COULOMB_EV_A = 14.399645  # e^2 / (4 pi eps0) in eV * Angstrom
HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903


def mlp(din, dh, dout, n=2):
    layers, d = [], din
    for _ in range(n - 1):
        layers += [nn.Linear(d, dh), nn.SiLU()]
        d = dh
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


def scatter_sum(x, idx, n):
    out = torch.zeros((n,) + x.shape[1:], dtype=x.dtype, device=x.device)
    return out.index_add_(0, idx, x)


def smootherstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def poly_cutoff(r, rc, p=6):
    x = (r / rc).clamp(max=1.0)
    f = 1 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
    return f * (r < rc)


class BesselRBF(nn.Module):
    def __init__(self, rc, n=8):
        super().__init__()
        self.rc = rc
        self.register_buffer("freq", torch.arange(1, n + 1) * math.pi / rc)

    def forward(self, r):
        r = r.clamp(min=1e-6)
        return math.sqrt(2 / self.rc) * torch.sin(self.freq * r[:, None]) / r[:, None]


class EnergyHead(nn.Module):
    """Extensive ground-state energy: sum of atomic energies + per-element reference."""

    def __init__(self, dfeat, n_elem, dh=64):
        super().__init__()
        self.net = mlp(dfeat, dh, 1)
        self.e_ref = nn.Parameter(torch.zeros(n_elem))

    def forward(self, h, attrs, batch, nmol):
        e_at = self.net(h).squeeze(-1) + attrs @ self.e_ref
        return scatter_sum(e_at, batch, nmol)


class DirectHead(nn.Module):
    """Direct multistate head: per-atom m-vectors pooled by sum / mean / softmin."""

    def __init__(self, dfeat, m, pooling="sum", dh=64, shift=7.0, tau=0.3):
        super().__init__()
        self.m, self.pooling = m, pooling
        self.net = mlp(dfeat, dh, m)
        self.shift = shift
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        if pooling == "softmin":
            self.score = mlp(dfeat, dh, m)

    def forward(self, h, batch, nmol, natoms):
        e = self.net(h)
        if self.pooling == "sum":
            out = scatter_sum(e, batch, nmol) + self.shift
        elif self.pooling == "mean":
            out = scatter_sum(e, batch, nmol) / natoms[:, None] + self.shift
        elif self.pooling == "softmin":
            # per-state extremal (soft-min) pooling over atoms: non-additive and intensive
            e = e + self.shift
            tau = self.log_tau.exp()
            logits = -e / tau + self.score(h)
            mx = torch.full((nmol, self.m), -1e9, dtype=e.dtype, device=e.device)
            mx = mx.scatter_reduce(0, batch[:, None].expand_as(logits), logits, "amax")
            w = torch.exp(logits - mx[batch])
            out = scatter_sum(w * e, batch, nmol) / scatter_sum(w, batch, nmol)
        else:
            raise ValueError(self.pooling)
        return out  # [nmol, m] (not necessarily sorted)


class LatentHamiltonianHead(nn.Module):
    """Variable-size symmetric latent excitation Hamiltonian, assembled from local blocks.

    centers: 'all' or 'heavy'. k_heavy/k_h channels per center for heavy atoms / hydrogen.
    H_ii = sym(onsite(h_i)) + shift*1, H_ij = fc(r_ij) * pair(h_i,h_j,rbf), symmetrized.
    Returns ascending eigenvalues padded to a fixed number of rows and the matrix size per molecule.
    """

    def __init__(self, dfeat, k=1, centers="all", r_cut=5.0, dh=64, shift=7.0,
                 offdiag=True, n_rbf=8, k_h=None, pad_value=1e3, dipoles=False, long_range=False,
                 lr_on=4.0, lr_off=6.0, mu_cut=None, bound_on=None, bound_pair=None):
        """
        dipoles: predict a transition-dipole vector per channel (e*Angstrom) and give oscillator strengths of all
            eigenstates, f_n = 2/3 * dE_n * |sum_rows U_rn mu_r|^2 (atomic units).
        long_range: add H_long_ij = s(r_ij) * k_e * [mu_i.mu_j - 3 (mu_i.rhat)(mu_j.rhat)] / r_ij^3 between channels
            of different atoms, smootherstep switch s from 0 at lr_on to 1 at lr_off (implies dipoles=True).
        bound_on / bound_pair (eV): optional soft bounds c*tanh(x/c) on the onsite / pair block elements. Unbounded
            blocks can extrapolate to huge values for rare out-of-distribution environments, which drives eigenvalues
            far negative (frozen benchmark: 4 of 43,698 test geometries with predicted S1 < -5 eV). None = off.
        """
        super().__init__()
        self.dipoles = dipoles or long_range
        self.long_range = long_range
        self.lr_on, self.lr_off = lr_on, lr_off
        self.mu_cut = mu_cut or r_cut
        if self.dipoles:
            self.rbf_mu = BesselRBF(self.mu_cut, n_rbf)
            self.mu_net = mlp(2 * dfeat + n_rbf, dh, k)  # scalar weight per neighbour bond vector and channel
        self.k = k
        self.k_h = (k if centers == "all" else 0) if k_h is None else k_h
        self.centers = centers
        self.r_cut = r_cut
        self.offdiag = offdiag
        self.shift = shift
        self.pad_value = pad_value
        self.bound_on, self.bound_pair = bound_on, bound_pair
        kk = k * k
        self.onsite = mlp(dfeat, dh, kk)
        if self.k_h > 0 and self.k_h != k:
            self.onsite_h = mlp(dfeat, dh, self.k_h * self.k_h)
        if offdiag:
            self.rbf = BesselRBF(r_cut, n_rbf)
            self.pair = mlp(2 * dfeat + n_rbf, dh, kk)  # only same-k blocks; see _kmax note
        self.kmax = max(k, self.k_h)

    def channels_per_atom(self, Z):
        kc = torch.full_like(Z, self.k)
        kc[Z == 1] = self.k_h
        return kc

    def build(self, h, Z, pos, batch, nmol):
        """Return list-free padded H [nmol, Mmax, Mmax] and sizes M [nmol]."""
        dev, dt = h.device, h.dtype
        kc = self.channels_per_atom(Z)
        # global row offsets per atom, then per molecule
        M = scatter_sum(kc, batch, nmol)
        Mmax = int(M.max())
        cum_mol = torch.cumsum(M, 0) - M
        cum_atom = torch.cumsum(kc, 0) - kc  # global
        local_off = cum_atom - cum_mol[batch]
        self._kc, self._local_off = kc, local_off  # row layout, for atom_weights
        H = torch.zeros(nmol, Mmax, Mmax, dtype=dt, device=dev)
        # onsite blocks
        for kval, net in self._onsite_nets():
            sel = (kc == kval).nonzero().squeeze(-1)
            if sel.numel() == 0:
                continue
            A = net(h[sel]).view(-1, kval, kval)
            if self.bound_on:
                A = self.bound_on * torch.tanh(A / self.bound_on)
            A = 0.5 * (A + A.transpose(1, 2))
            a = torch.arange(kval, device=dev)
            rows = local_off[sel][:, None, None] + a[None, :, None]
            cols = local_off[sel][:, None, None] + a[None, None, :]
            mol = batch[sel][:, None, None].expand(-1, kval, kval)
            H = H.index_put((mol, rows.expand(-1, kval, kval), cols.expand(-1, kval, kval)), A, accumulate=True)
        # pair blocks between centers with identical k (heavy-heavy, or all when k_h==k)
        if self.offdiag:
            cen = (kc == self.k).nonzero().squeeze(-1)
            ii, jj = self._pairs(cen, pos, batch)
            if ii.numel() > 0:
                r = (pos[ii] - pos[jj]).norm(dim=-1)
                fc = poly_cutoff(r, self.r_cut)
                x = torch.cat([h[ii], h[jj], self.rbf(r)], dim=-1)
                B = self.pair(x).view(-1, self.k, self.k)
                if self.bound_pair:
                    B = self.bound_pair * torch.tanh(B / self.bound_pair)
                B = B * fc[:, None, None]
                a = torch.arange(self.k, device=dev)
                rows = (local_off[ii][:, None, None] + a[None, :, None]).expand(-1, self.k, self.k)
                cols = (local_off[jj][:, None, None] + a[None, None, :]).expand(-1, self.k, self.k)
                mol = batch[ii][:, None, None].expand(-1, self.k, self.k)
                H = H.index_put((mol, rows, cols), B, accumulate=True)
            if self.k_h > 0 and self.k_h != self.k:
                raise NotImplementedError("pair blocks for mixed k not implemented")
        mu_rows = None
        if self.dipoles:
            mu_rows, pos_rows, owner = self._dipoles(h, Z, pos, batch, nmol, kc, local_off, Mmax)
            if self.long_range:
                H = H + self._long_range(mu_rows, pos_rows, owner)
        self._mu_rows = mu_rows
        H = 0.5 * (H + H.transpose(1, 2))
        # diagonal shift for real rows, distinct large values for padding rows
        ar = torch.arange(Mmax, device=dev)
        real = ar[None, :] < M[:, None]
        diag = torch.where(real, torch.full_like(H[:, :, 0], self.shift),
                           self.pad_value + 10.0 * ar[None, :].to(dt))
        H = H + torch.diag_embed(diag)
        return H, M

    def _onsite_nets(self):
        out = [(self.k, self.onsite)]
        if self.k_h > 0 and self.k_h != self.k:
            out.append((self.k_h, self.onsite_h))
        return out

    def _pairs(self, cen, pos, batch):
        # all ordered pairs of centers within the same molecule and within r_cut
        b = batch[cen]
        same = b[:, None] == b[None, :]
        d = torch.cdist(pos[cen].detach(), pos[cen].detach())
        mask = same & (d < self.r_cut)
        mask.fill_diagonal_(False)
        i, j = mask.nonzero(as_tuple=True)
        return cen[i], cen[j]

    def _dipoles(self, h, Z, pos, batch, nmol, kc, local_off, Mmax):
        """Per-channel transition dipoles mu_{i,a} = sum_j w_a(h_i, h_j, r_ij) fc(r_ij) (r_j - r_i) / r_ij
        (rotation equivariant by construction), laid out like the rows of H.

        Returns mu [nmol, Mmax, 3] (e*Angstrom), row positions [nmol, Mmax, 3] and the owning atom index of every row
        [nmol, Mmax] (-1 for padding rows)."""
        dev, dt = h.device, h.dtype
        cen = (kc == self.k).nonzero().squeeze(-1)
        allat = torch.arange(len(Z), device=dev)
        same = batch[cen][:, None] == batch[None, :]
        d = torch.cdist(pos[cen].detach(), pos.detach())
        mask = same & (d < self.mu_cut) & (cen[:, None] != allat[None, :])
        ci, j = mask.nonzero(as_tuple=True)
        i = cen[ci]
        vec = pos[j] - pos[i]
        r = vec.norm(dim=-1).clamp(min=1e-6)
        w = self.mu_net(torch.cat([h[i], h[j], self.rbf_mu(r)], dim=-1)) * poly_cutoff(r, self.mu_cut)[:, None]
        mu_atom = torch.zeros(len(Z), self.k, 3, dtype=dt, device=dev)
        mu_atom = mu_atom.index_add(0, i, w[:, :, None] * (vec / r[:, None])[:, None, :])
        a = torch.arange(self.k, device=dev)
        rows = (local_off[cen][:, None] + a[None, :]).reshape(-1)
        mols = batch[cen][:, None].expand(-1, self.k).reshape(-1)
        mu = torch.zeros(nmol, Mmax, 3, dtype=dt, device=dev).index_put((mols, rows), mu_atom[cen].reshape(-1, 3))
        prow = torch.zeros(nmol, Mmax, 3, dtype=dt, device=dev).index_put(
            (mols, rows), pos[cen][:, None, :].expand(-1, self.k, 3).reshape(-1, 3))
        owner = torch.full((nmol, Mmax), -1, dtype=torch.long, device=dev).index_put(
            (mols, rows), cen[:, None].expand(-1, self.k).reshape(-1))
        return mu, prow, owner

    def _long_range(self, mu, prow, owner):
        """Switched point-dipole coupling (eV) between channels on different atoms, [nmol, Mmax, Mmax]."""
        rij = prow[:, None, :, :] - prow[:, :, None, :]
        valid = (owner[:, :, None] >= 0) & (owner[:, None, :] >= 0) & (owner[:, :, None] != owner[:, None, :])
        r = rij.norm(dim=-1)
        r_safe = torch.where(valid, r, torch.ones_like(r))
        rhat = rij / r_safe[..., None]
        mumu = (mu[:, :, None, :] * mu[:, None, :, :]).sum(-1)
        mui_r = (mu[:, :, None, :] * rhat).sum(-1)
        muj_r = (mu[:, None, :, :] * rhat).sum(-1)
        J = COULOMB_EV_A * (mumu - 3.0 * mui_r * muj_r) / r_safe ** 3
        s = smootherstep((r_safe - self.lr_on) / (self.lr_off - self.lr_on))
        return torch.where(valid, s * J, torch.zeros_like(J))

    def oscillator_strengths(self, H, lam):
        """f_n = 2/3 dE_n[Hartree] |mu_n[e bohr]|^2 from the eigenvectors (degeneracy-safe gradients)."""
        _, U = safe_eigh(H)
        mu_n = torch.einsum("bmn,bmx->bnx", U, self._mu_rows)
        return (2.0 / 3.0) * (lam / HARTREE_EV) * (mu_n / BOHR_A).pow(2).sum(-1)

    def atom_weights(self, H, n, batch):
        """Per-atom weights w_An = sum over the rows r of atom A of U_rn^2 for the n lowest eigenvectors, [natoms, n]
        (each column sums to 1 per molecule; invariant to the eigenvector signs; degeneracy-safe gradients).
        Uses the row layout of the last build() call."""
        _, U = safe_eigh(H)
        U2 = U[:, :, :n].pow(2)
        W = torch.zeros(len(batch), U2.shape[-1], dtype=H.dtype, device=H.device)
        for c in range(self.kmax):
            a = (self._kc > c).nonzero().squeeze(-1)
            W = W.index_add(0, a, U2[batch[a], self._local_off[a] + c])
        return W

    def forward(self, h, Z, pos, batch, nmol):
        H, M = self.build(h, Z, pos, batch, nmol)
        lam = torch.linalg.eigvalsh(H)  # exact, twice differentiable path for energies and forces
        return lam, M, H
