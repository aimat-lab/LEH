"""Excited-state model: shared MACE encoder + E0 head + excitation head (direct or latent H)."""
import torch
from torch import nn

from .encoder import MACEEncoder
from .heads import DirectHead, EnergyHead, LatentHamiltonianHead


def radius_graph(pos, batch, r_max):
    with torch.no_grad():
        same = batch[:, None] == batch[None, :]
        d = torch.cdist(pos, pos)
        mask = same & (d < r_max)
        mask.fill_diagonal_(False)
        i, j = mask.nonzero(as_tuple=True)
    return torch.stack([i, j])  # sender, receiver


class ExcitedStateModel(nn.Module):
    def __init__(self, z_table, n_states, head="latent", enc_kwargs=None, head_kwargs=None,
                 predict_e0=False, shift=7.0):
        super().__init__()
        enc_kwargs = enc_kwargs or {}
        head_kwargs = dict(head_kwargs or {})
        self.encoder = MACEEncoder(z_table, **enc_kwargs)
        d = self.encoder.out_dim
        self.n_states = n_states
        self.head_type = head
        if head == "latent":
            self.head = LatentHamiltonianHead(d, shift=shift, **head_kwargs)
        else:
            self.head = DirectHead(d, n_states, pooling=head, shift=shift, **head_kwargs)
        self.predict_e0 = predict_e0
        if predict_e0:
            self.e0_head = EnergyHead(d, len(z_table))

    def forward(self, Z, pos, batch, nmol, compute_forces=False, force_states=0, training=False, pop_states=0):
        """Return dict with 'gaps' [nmol, n_states] (eV), 'valid' mask, optional 'E0', forces.

        force_states: number of excited states (S1..Sq) for which forces are computed.
        pop_states: latent heads only, per-atom eigenvector weights 'atom_w' [natoms, pop_states] of S1..S_pop_states.
        """
        if compute_forces:
            pos = pos if pos.requires_grad else pos.requires_grad_(True)
        edge_index = radius_graph(pos, batch, self.encoder.r_max)
        h, attrs = self.encoder(Z, pos, edge_index)
        natoms = torch.bincount(batch, minlength=nmol).to(h.dtype)
        out = {}
        if self.head_type == "latent":
            lam, M, H = self.head(h, Z, pos, batch, nmol)
            n = min(self.n_states, lam.shape[1])
            gaps = lam[:, :n]
            if n < self.n_states:
                gaps = torch.cat([gaps, gaps.new_full((nmol, self.n_states - n), 1e3)], 1)
            valid = torch.arange(self.n_states, device=Z.device)[None, :] < M[:, None]
            out.update(eigvals=lam, M=M, H=H)
            if self.head.dipoles:
                osc = self.head.oscillator_strengths(H, lam)[:, :n]
                if n < self.n_states:
                    osc = torch.cat([osc, osc.new_zeros((nmol, self.n_states - n))], 1)
                out["osc"] = osc
            if pop_states:
                out["atom_w"] = self.head.atom_weights(H, pop_states, batch)
        else:
            gaps = self.head(h, batch, nmol, natoms)
            valid = torch.ones_like(gaps, dtype=torch.bool)
        out["gaps"], out["valid"] = gaps, valid
        if self.predict_e0:
            out["E0"] = self.e0_head(h, attrs, batch, nmol)
        if compute_forces:
            grads_needed = []
            if self.predict_e0:
                grads_needed.append(out["E0"].sum())
            for s in range(force_states):
                grads_needed.append(torch.where(valid[:, s], gaps[:, s], 0.0).sum())
            gs = [torch.autograd.grad(g, pos, create_graph=training, retain_graph=True)[0]
                  for g in grads_needed]
            k = 0
            if self.predict_e0:
                out["F0"] = -gs[0]
                k = 1
            if force_states:
                # gradient of the gaps; excited forces F_Sn = F0 - grad(gap_n)
                out["Fgap"] = torch.stack([-g for g in gs[k:]], dim=1)  # [natoms, q, 3]
        return out
