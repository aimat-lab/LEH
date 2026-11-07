"""GPU smoke test + timing of one training update (energies, gaps, S0/S1-S3 forces, double backward) for the
model sizes of hyperparameter stage 2, on 16 RDKit-embedded small molecules (float64, as in training)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exmlip.models.model import ExcitedStateModel  # noqa

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", "torch", torch.__version__)
torch.set_default_dtype(torch.float64)
smi = ["CC(=O)C=C", "O=C1NC=CC(=O)N1", "CCOC(=O)C1CC1", "c1ccncc1O", "CC1=CC(=O)CCC1", "N#CC1CCOC1", "OC1CCN(C)C1",
       "C=CC(=O)OC", "CC(C)C(=O)N", "c1cc[nH]c1C=O", "O=CC1CC2CC12", "CC#CC(O)C", "FC1=CC=CC=C1N", "C1CC2(CO2)C1",
       "CN1C=NC=C1C", "OCC(=O)C=C"]
Zs, Rs = [], []
for s in smi:
    m = Chem.AddHs(Chem.MolFromSmiles(s)); AllChem.EmbedMolecule(m, randomSeed=1)
    Zs.append(np.array([a.GetAtomicNum() for a in m.GetAtoms()])); Rs.append(m.GetConformer().GetPositions())
Z = torch.tensor(np.concatenate(Zs), device=dev)
R = torch.tensor(np.concatenate(Rs), device=dev)
batch = torch.cat([torch.full((len(z),), i, dtype=torch.long) for i, z in enumerate(Zs)]).to(dev)
for ch, L in [(32, 1), (64, 1), (128, 1), (64, 2), (128, 2)]:
    model = ExcitedStateModel([1, 6, 7, 8, 9], 10, head="latent", head_kwargs=dict(k=1), predict_e0=True, shift=7.0,
                              enc_kwargs=dict(channels=ch, max_L=L, num_interactions=2, r_max=5.0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    times = []
    for it in range(8):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        out = model(Z, R, batch, len(Zs), compute_forces=True, force_states=3)
        loss = out["E0"].pow(2).mean() + out["gaps"].pow(2).mean() + out["F0"].pow(2).mean() + out["Fgap"].pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t)
    print(f"channels {ch:3d} L {L}: {np.median(times[3:]) * 1000:.0f} ms per batch-16 update, "
          f"max memory {torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0:.1f} GB", flush=True)
