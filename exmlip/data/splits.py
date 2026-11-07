"""Scaffold- and size-separated splits."""
import hashlib

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdDetermineBonds


def mol_from_xyz(Z, R):
    from rdkit.Chem import rdchem
    xyz = f"{len(Z)}\n\n" + "".join(f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n" for z, r in zip(Z, R))
    m = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(m)
    return m


def scaffold_of(smiles=None, Z=None, R=None):
    try:
        m = Chem.MolFromSmiles(smiles) if smiles else mol_from_xyz(Z, R)
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        m = Chem.RemoveHs(m, sanitize=False)
        Chem.FastFindRings(m)
        # generic Murcko framework (atom/bond types removed) -> coarse scaffold families
        s = MurckoScaffold.GetScaffoldForMol(m)
        g = MurckoScaffold.MakeScaffoldGeneric(s)
        return Chem.MolToSmiles(g)
    except Exception:
        return "unknown"


def group_split(groups, fractions, seed=0):
    """Assign whole groups to splits (largest groups first, greedy fill)."""
    rng = np.random.default_rng(seed)
    uniq, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-cnt[order], kind="stable")]
    n = len(groups)
    target = np.array(fractions) * n
    fill = np.zeros(len(fractions))
    assign = np.zeros(len(uniq), int)
    for g in order:
        k = int(np.argmax(target - fill))
        assign[g] = k
        fill[k] += cnt[g]
    return [np.where(assign[inv] == k)[0] for k in range(len(fractions))]


def graph_key(Z, R):
    """Constitution key from a geometry: canonical SMILES of the covalent graph with all bonds single (stereo ignored)."""
    try:
        m = mol_from_xyz(Z, R)
        for b in m.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
        for a in m.GetAtoms():
            a.SetNoImplicit(True)
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return "unknown"


def graph_key_smiles(smiles, seed=1):
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    return graph_key(Z, m.GetConformer().GetPositions())


# ---------------------------------------------------------------------------------------------------------------
# Scaffold hold-out (balanced Bemis-Murcko scaffold split) for the new TDDFT dataset
# ---------------------------------------------------------------------------------------------------------------
def typed_scaffold_from_xyz(Z, R):
    """Bemis-Murcko scaffold SMILES (atom and bond types kept) from a closed-shell neutral equilibrium geometry.

    Bond orders are perceived with RDKit's DetermineBonds (charge 0). Returns '' for acyclic molecules and None if
    bond perception fails (such molecules are never held out)."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        xyz = f"{len(Z)}\n\n" + "".join(f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {r[0]:.6f} {r[1]:.6f} "
                                        f"{r[2]:.6f}\n" for z, r in zip(Z, R))
        m = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(m, charge=0)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.RemoveHs(m))
    except Exception:
        return None


def _scaf_worker(args):
    return typed_scaffold_from_xyz(*args)


def scaffolds_parallel(items, nproc=16):
    """items: list of (Z, R); returns list of typed scaffolds."""
    from multiprocessing import Pool
    with Pool(nproc) as p:
        return p.map(_scaf_worker, items, chunksize=64)


def select_scaffold_holdout(scaf_by_mol, frac=0.10, cap=0.002, salt="wb97scaf:"):
    """Balanced scaffold hold-out (chemprop 'scaffold_balanced' idea, made deterministic).

    Eligible families: cyclic scaffolds (not '' and not None) with at most cap * N molecules (large, common ring
    families such as cyclopropane or benzene stay in training). Eligible families are ordered by an md5 hash of the
    scaffold SMILES and taken in that order until frac * N molecules are held out. Returns (set of scaffolds, stats)."""
    from collections import Counter
    n = len(scaf_by_mol)
    cnt = Counter(scaf_by_mol.values())
    elig = [s for s, c in cnt.items() if s and c <= cap * n]
    elig.sort(key=lambda s: hashlib.md5((salt + s).encode()).hexdigest())
    held, nm = [], 0
    for s in elig:
        if nm >= frac * n:
            break
        held.append(s)
        nm += cnt[s]
    stats = dict(n_molecules=n, n_families=len(cnt), n_acyclic=cnt.get("", 0), n_failed=cnt.get(None, 0),
                 n_eligible_families=len(elig), eligible_molecule_share=sum(cnt[s] for s in elig) / max(n, 1),
                 n_heldout_families=len(held), n_heldout_molecules=nm, heldout_share=nm / max(n, 1),
                 largest_heldout_family=max((cnt[s] for s in held), default=0), frac=frac, cap=cap)
    return set(held), stats


def apply_scaffold_holdout(mols, split, heldout_mols):
    """Move every entry of a held-out molecule from train/val/test_random* to test_scaffold(_eq/_offeq).

    Entries are matched by m['mol']; ZINC size-extrapolation entries (test_large*) are not touched."""
    move = {i for i, m in enumerate(mols) if m.get("mol") in heldout_mols and m.get("source") != "ZINC"}
    out = {k: [i for i in v if i not in move] for k, v in split.items() if not k.startswith("test_scaffold")}
    out["test_scaffold"] = sorted(move)
    eq_kinds = ("qcdge_eq", "qcdge_eq_extra", "eq")
    out["test_scaffold_eq"] = [i for i in out["test_scaffold"] if mols[i].get("kind") in eq_kinds]
    out["test_scaffold_offeq"] = [i for i in out["test_scaffold"] if mols[i].get("kind") == "offeq"]
    out["test_scaffold_exc"] = [i for i in out["test_scaffold"] if mols[i].get("kind") == "excmd"]
    return out
