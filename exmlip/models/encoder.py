"""MACE backbone that returns invariant node features (no energy readout).

Reuses the embedding / interaction / product blocks of mace.modules.MACE.
"""
from typing import Dict, List

import numpy as np
import torch
from e3nn import o3
from mace import modules
from mace.modules.blocks import RealAgnosticResidualInteractionBlock, RealAgnosticInteractionBlock


class MACEEncoder(torch.nn.Module):
    def __init__(self, z_table: List[int], r_max: float = 5.0, channels: int = 32,
                 max_L: int = 1, max_ell: int = 2, correlation: int = 3,
                 num_interactions: int = 2, num_bessel: int = 8, avg_num_neighbors: float = 10.0):
        super().__init__()
        self.z_table = list(z_table)
        z2idx = torch.full((max(z_table) + 1,), -1, dtype=torch.long)
        for i, z in enumerate(z_table):
            z2idx[z] = i
        self.register_buffer("z2idx", z2idx)
        irreps = "+".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_L + 1))
        self.mace = modules.MACE(
            r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=5, max_ell=max_ell,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=num_interactions, num_elements=len(z_table),
            hidden_irreps=o3.Irreps(irreps), MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros(len(z_table)), avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=list(z_table), correlation=correlation, gate=torch.nn.functional.silu,
        )
        # drop the energy readouts, we only want features
        self.mace.readouts = torch.nn.ModuleList()
        self.channels = channels
        self.num_interactions = num_interactions
        # all layers' scalar channels are concatenated -> invariant features
        self.out_dim = channels * num_interactions
        self.r_max = r_max

    def node_attrs(self, Z):
        idx = self.z2idx[Z]
        return torch.nn.functional.one_hot(idx, len(self.z_table)).to(torch.get_default_dtype())

    def forward(self, Z, pos, edge_index):
        m = self.mace
        attrs = self.node_attrs(Z)
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        lengths = vec.norm(dim=-1, keepdim=True)
        node_feats = m.node_embedding(attrs)
        edge_attrs = m.spherical_harmonics(vec)
        edge_feats, cutoff = m.radial_embedding(lengths, attrs, edge_index, m.atomic_numbers)
        scalars = []
        for i, (inter, prod) in enumerate(zip(m.interactions, m.products)):
            node_feats, sc = inter(node_attrs=attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                                   edge_feats=edge_feats, edge_index=edge_index, cutoff=cutoff,
                                   first_layer=(i == 0))
            node_feats = prod(node_feats=node_feats, sc=sc, node_attrs=attrs)
            scalars.append(node_feats[:, : self.channels])  # l=0 block comes first
        return torch.cat(scalars, dim=-1), attrs
