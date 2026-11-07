"""Simple molecular batching (concatenated atoms + batch index)."""
import numpy as np
import torch


class MolDataset:
    """Holds a list of molecules as dicts of numpy arrays.

    keys: Z [n], R [n,3] Angstrom, gaps [m] eV (nan = missing), optional E0 eV, F0 [n,3] eV/A,
    Fex [n,q,3] eV/A (nan = missing), Eex [m] eV (total energies of the excited states, for frames without E0;
    nan = missing), optional TD-DFT populations pop_hole/pop_elec [n,P] (folded onto heavy atoms, zero on hydrogens),
    pop_fold [n] (index of the heavy atom each atom is folded into), pop_mask [P] (states used), plus metadata.
    """

    def __init__(self, mols):
        self.mols = mols

    def __len__(self):
        return len(self.mols)

    def subset(self, idx):
        return MolDataset([self.mols[i] for i in idx])


def collate(mols, n_states, n_force=0, dtype=None):
    dtype = dtype or torch.get_default_dtype()
    Z = torch.from_numpy(np.concatenate([m["Z"] for m in mols])).long()
    R = torch.from_numpy(np.concatenate([m["R"] for m in mols])).to(dtype)
    batch = torch.cat([torch.full((len(m["Z"]),), i, dtype=torch.long) for i, m in enumerate(mols)])
    gaps = np.full((len(mols), n_states), np.nan)
    for i, m in enumerate(mols):
        g = m["gaps"][:n_states]
        gaps[i, : len(g)] = g
    out = dict(Z=Z, R=R, batch=batch, nmol=len(mols), gaps=torch.from_numpy(gaps).to(dtype))
    if any("osc" in m for m in mols):  # oscillator strengths of the roots (NaN = missing)
        osc = np.full((len(mols), n_states), np.nan)
        for i, m in enumerate(mols):
            if "osc" in m:
                f = np.asarray(m["osc"], float)[:n_states]
                osc[i, : len(f)] = f
        out["osc"] = torch.from_numpy(osc).to(dtype)
    # missing labels are NaN so that mixed batches (equilibrium spectra only + full labels) can be masked
    if any("E0" in m for m in mols):
        out["E0"] = torch.tensor([m.get("E0", np.nan) for m in mols], dtype=dtype)
    if any("F0" in m for m in mols):
        out["F0"] = torch.from_numpy(np.concatenate(
            [m["F0"] if "F0" in m else np.full((len(m["Z"]), 3), np.nan) for m in mols])).to(dtype)
    if any("Eex" in m for m in mols):  # total energies of the excited-state columns (frames without an E0 label)
        eex = np.full((len(mols), n_states), np.nan)
        for i, m in enumerate(mols):
            if "Eex" in m:
                e = np.asarray(m["Eex"], float)[:n_states]
                eex[i, : len(e)] = e
        out["Eex"] = torch.from_numpy(eex).to(dtype)
    if any("pop_hole" in m for m in mols):  # TD-DFT hole/electron populations folded onto heavy atoms
        P = next(np.asarray(m["pop_hole"]).shape[1] for m in mols if "pop_hole" in m)
        off = np.cumsum([0] + [len(m["Z"]) for m in mols])[:-1]
        hole, elec, fold, pmask = [], [], [], np.zeros((len(mols), P), bool)
        for i, m in enumerate(mols):
            n = len(m["Z"])
            if "pop_hole" in m:
                hole.append(m["pop_hole"]); elec.append(m["pop_elec"])
                fold.append(np.asarray(m["pop_fold"]) + off[i]); pmask[i] = m["pop_mask"]
            else:
                hole.append(np.zeros((n, P))); elec.append(np.zeros((n, P))); fold.append(np.arange(n) + off[i])
        out["pop_hole"] = torch.from_numpy(np.concatenate(hole)).to(dtype)
        out["pop_elec"] = torch.from_numpy(np.concatenate(elec)).to(dtype)
        out["pop_fold"] = torch.from_numpy(np.concatenate(fold)).long()
        out["pop_mask"] = torch.from_numpy(pmask)
    if n_force and any("Fex" in m for m in mols):
        out["Fex"] = torch.from_numpy(np.concatenate(
            [m["Fex"][:, :n_force] if "Fex" in m else np.full((len(m["Z"]), n_force, 3), np.nan) for m in mols])).to(dtype)
    return out


def iterate(ds, batch_size, shuffle, rng=None, max_atoms=None):
    idx = np.arange(len(ds))
    if shuffle:
        (rng or np.random).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield [ds.mols[i] for i in idx[s : s + batch_size]]
