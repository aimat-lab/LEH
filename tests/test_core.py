import numpy as np
import pytest
import torch

from exmlip.models.model import ExcitedStateModel

torch.set_default_dtype(torch.float64)
ZT = [1, 6, 7, 8, 9]

# formaldehyde-like and ethylene-like geometries (Angstrom)
CH2O = (np.array([6, 8, 1, 1]), np.array([[0, 0, 0], [0, 0, 1.21], [0, 0.94, -0.59], [0, -0.94, -0.59]], float))
C2H4 = (np.array([6, 6, 1, 1, 1, 1]), np.array([[0, 0, 0.67], [0, 0, -0.67], [0, 0.92, 1.24], [0, -0.92, 1.24], [0, 0.92, -1.24], [0, -0.92, -1.24]], float))


def rand_rot(seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def make(head="latent", **hk):
    torch.manual_seed(0)
    m = ExcitedStateModel(ZT, n_states=4, head=head, predict_e0=True, enc_kwargs=dict(channels=8, r_max=4.0), head_kwargs=hk)
    # make couplings non-negligible
    return m.double().eval()


def run(m, mols, forces=False, q=0):
    Z = torch.tensor(np.concatenate([z for z, _ in mols]))
    R = torch.tensor(np.concatenate([r for _, r in mols]))
    b = torch.cat([torch.full((len(z),), i) for i, (z, _) in enumerate(mols)])
    return m(Z, R, b, len(mols), compute_forces=forces, force_states=q)


def combine(*frags):
    return (np.concatenate([f[0] for f in frags]), np.concatenate([f[1] for f in frags]))


@pytest.mark.parametrize("hk", [dict(k=1), dict(k=2), dict(k=1, centers="heavy"), dict(k=2, centers="heavy")])
def test_symmetry_and_invariance(hk):
    m = make(**hk)
    o = run(m, [C2H4])
    H = o["H"][0]
    assert torch.allclose(H, H.T)
    Q = rand_rot(1)
    o2 = run(m, [(C2H4[0], C2H4[1] @ Q.T + np.array([1.0, -2.0, 3.0]))])
    n = int(o["M"][0])
    assert torch.allclose(o["eigvals"][0, :n], o2["eigvals"][0, :n], atol=1e-8)
    assert torch.allclose(o["E0"], o2["E0"], atol=1e-8)


def test_permutation_covariance():
    m = make(k=1)
    perm = np.array([3, 1, 0, 2, 5, 4])
    o = run(m, [C2H4])
    o2 = run(m, [(C2H4[0][perm], C2H4[1][perm])])
    H, H2 = o["H"][0], o2["H"][0]
    P = torch.eye(6)[perm]
    assert torch.allclose(P @ H @ P.T, H2, atol=1e-10)
    assert torch.allclose(o["eigvals"], o2["eigvals"], atol=1e-10)


@pytest.mark.parametrize("hk", [dict(k=1), dict(k=2, centers="heavy")])
def test_fragment_union_and_degeneracy(hk):
    m = make(**hk)
    a = run(m, [CH2O])
    b = run(m, [C2H4])
    na, nb = int(a["M"][0]), int(b["M"][0])
    shift = np.array([0, 0, 30.0])
    ab = run(m, [combine(CH2O, (C2H4[0], C2H4[1] @ rand_rot(3).T + shift))])
    union = torch.sort(torch.cat([a["eigvals"][0, :na], b["eigvals"][0, :nb]]))[0]
    assert torch.allclose(ab["eigvals"][0, : na + nb], union, atol=1e-9)
    assert torch.allclose(ab["E0"], a["E0"] + b["E0"], atol=1e-9)
    # identical fragments -> exact duplicates
    aa = run(m, [combine(CH2O, (CH2O[0], CH2O[1] @ rand_rot(4).T + shift))])
    ev = aa["eigvals"][0, : 2 * na]
    assert torch.allclose(ev[0::2], ev[1::2], atol=1e-9)
    assert torch.allclose(ev[0::2], a["eigvals"][0, :na], atol=1e-9)


def test_direct_sum_is_not_size_consistent():
    m = make(head="sum")
    a = run(m, [CH2O])["gaps"]
    aa = run(m, [combine(CH2O, (CH2O[0], CH2O[1] + np.array([0, 0, 30.0])))])["gaps"]
    assert not torch.allclose(a, aa, atol=1e-3)


@pytest.mark.parametrize("head,hk", [("latent", dict(k=1)), ("latent", dict(k=2, centers="heavy")), ("softmin", {})])
def test_forces_vs_finite_difference(head, hk):
    m = make(head=head, **hk)
    Z, R = C2H4
    R = R + np.random.default_rng(0).normal(scale=0.05, size=R.shape)
    o = run(m, [(Z, R)], forces=True, q=2)
    h = 1e-5
    for (a, x) in [(0, 2), (2, 1), (5, 0)]:
        Rp, Rm = R.copy(), R.copy()
        Rp[a, x] += h
        Rm[a, x] -= h
        op, om = run(m, [(Z, Rp)]), run(m, [(Z, Rm)])
        fd0 = -(op["E0"] - om["E0"]) / (2 * h)
        assert abs(fd0.item() - o["F0"][a, x].item()) < 1e-6
        for s in range(2):
            fd = -(op["gaps"][0, s] - om["gaps"][0, s]) / (2 * h)
            assert abs(fd.item() - o["Fgap"][a, s, x].item()) < 1e-6


def test_two_state_toy_gradient():
    # H(x) = [[x, c], [c, -x]] -> lambda_1 = -sqrt(x^2+c^2), d lambda_1/dx = -x/sqrt(x^2+c^2)
    x = torch.tensor(0.3, requires_grad=True)
    c = 0.2
    H = torch.stack([torch.stack([x, torch.tensor(c)]), torch.stack([torch.tensor(c), -x])])
    lam = torch.linalg.eigvalsh(H)[0]
    (g,) = torch.autograd.grad(lam, x)
    assert abs(g.item() + 0.3 / np.sqrt(0.09 + 0.04)) < 1e-10


def test_padding_and_masking():
    m = make(k=1, centers="heavy")
    o = run(m, [CH2O, C2H4])  # 2 heavy atoms each -> only 2 valid states
    assert o["valid"].sum(1).tolist() == [2, 2]
    assert (o["eigvals"][:, :2] < 100).all() and (o["eigvals"][:, 2:] > 100).all()


@pytest.mark.parametrize("head", ["latent", "softmin"])
def test_batched_md_forces_and_energy_conservation(head):
    from exmlip.md import BatchedModelPES, maxwell_boltzmann, run_nve_batched
    m = make(head)
    Zs, Rs = [CH2O[0], C2H4[0]], [CH2O[1].copy(), C2H4[1].copy()]
    pes = BatchedModelPES(m, Zs, state=1)
    E, F, _ = pes(Rs)
    h = 1e-5
    for mol, atom, ax in [(0, 1, 2), (1, 0, 0)]:
        Rp = [r.copy() for r in Rs]; Rm = [r.copy() for r in Rs]
        Rp[mol][atom, ax] += h; Rm[mol][atom, ax] -= h
        fd = -(pes(Rp)[0][mol] - pes(Rm)[0][mol]) / (2 * h)
        assert abs(fd - F[mol][atom, ax]) < 1e-5
    v0 = [maxwell_boltzmann(Z, 300.0, np.random.default_rng(k)) for k, Z in enumerate(Zs)]
    tr = run_nve_batched(pes, Zs, Rs, v0, dt_fs=0.25, n_steps=40, every=5)
    Et = np.array(tr["Epot"]) + np.array(tr["Ekin"])
    assert np.abs(Et - Et[0]).max() < 5e-3


def test_graph_key_identity():
    from exmlip.data.splits import graph_key, graph_key_smiles
    Z, R = CH2O
    assert graph_key(Z, R) == graph_key(Z, R @ rand_rot(3).T + 1.0)
    perm = np.array([2, 0, 3, 1])
    assert graph_key(Z[perm], R[perm]) == graph_key(Z, R)
    assert graph_key(Z, R) == graph_key_smiles("C=O")
    assert graph_key_smiles("CCO") != graph_key_smiles("COC")


def test_direct_head_unsupervised_outputs_are_not_predictions():
    """Regression: unsupervised direct-head outputs (index >= n_sup) must not enter the sorted prediction."""
    from exmlip.data.batching import MolDataset
    from exmlip.training.train import predict
    m = make("sum")
    mols = [dict(Z=CH2O[0], R=CH2O[1], gaps=np.array([4.0, 8.0, 9.0, 10.0])),
            dict(Z=C2H4[0], R=C2H4[1], gaps=np.array([7.0, 8.0, 9.0, 10.0]))]
    cfg = dict(n_states=4, n_sup=2, eval_batch_size=8)
    p = predict(m, MolDataset(mols), cfg)
    for r in p:
        assert np.all(np.isfinite(r["gaps_pred"][:2])) and np.all(np.isnan(r["gaps_pred"][2:]))


def test_md_units():
    """Regression: eV/A/amu -> A/fs^2 and Maxwell-Boltzmann speeds (C at 300 K: rms 3D speed ~ 0.0079 A/fs)."""
    from exmlip.md import CONV, kinetic, maxwell_boltzmann
    assert abs(CONV - 0.009648533) < 1e-8
    Z = np.full(20000, 6)
    v = maxwell_boltzmann(Z, 300.0, np.random.default_rng(0))
    rms = np.sqrt((v**2).sum(1).mean())
    assert abs(rms - 0.0079) < 0.0002
    assert abs(kinetic(Z, v) / (1.5 * len(Z) * 8.617333262e-5 * 300.0) - 1) < 0.02


def _rows_mu(m):
    return m.head._mu_rows[0].detach().numpy()


def test_dipoles_equivariant_and_osc_invariant():
    m = make("latent", k=1, dipoles=True, long_range=True)
    Z, R = combine(CH2O, (C2H4[0], C2H4[1] + np.array([0.0, 0.0, 7.0])))
    Q = rand_rot(5)
    o1 = run(m, [(Z, R)]); mu1 = _rows_mu(m)
    o2 = run(m, [(Z, R @ Q.T)]); mu2 = _rows_mu(m)
    assert np.allclose(o1["eigvals"].detach().numpy(), o2["eigvals"].detach().numpy(), atol=1e-8)
    assert np.allclose(o1["osc"].detach().numpy(), o2["osc"].detach().numpy(), atol=1e-8)
    assert np.allclose(mu1 @ Q.T, mu2, atol=1e-8)
    assert (o1["osc"].detach().numpy() >= -1e-12).all()
    assert np.abs(mu1).max() > 0  # dipoles are non-trivial at initialisation


def test_long_range_coupling_limits():
    """Identical fragments: union/degeneracy at large separation; only the long-range term splits them at 6.5 A."""
    shift = lambda d: (CH2O[0], CH2O[1] + np.array([d, 0.0, 0.0]))
    for lr in (False, True):
        m = make("latent", k=1, long_range=lr)
        far = np.sort(run(m, [combine(CH2O, shift(60.0))])["eigvals"].detach().numpy()[0][:8])
        mono = np.sort(run(m, [CH2O])["eigvals"].detach().numpy()[0][:4])
        assert np.allclose(far, np.sort(np.concatenate([mono, mono])), atol=1e-5)
        near = np.sort(run(m, [combine(CH2O, shift(6.5))])["eigvals"].detach().numpy()[0][:8])
        split = np.abs(near[1::2] - near[0::2]).max()
        assert (split > 1e-6) if lr else (split < 1e-9)


def test_forces_fd_with_long_range():
    m = make("latent", k=1, long_range=True)
    Z, R = combine(CH2O, (C2H4[0], C2H4[1] + np.array([0.0, 0.0, 5.5])))
    out = run(m, [(Z, R)], forces=True, q=1)
    F = (out["F0"] + out["Fgap"][:, 0]).detach().numpy()
    h = 1e-5
    for atom, ax in [(1, 2), (5, 0)]:
        Rp, Rm = R.copy(), R.copy()
        Rp[atom, ax] += h; Rm[atom, ax] -= h
        ep = run(m, [(Z, Rp)]); em = run(m, [(Z, Rm)])
        Ep = ep["E0"][0] + ep["gaps"][0, 0]; Em = em["E0"][0] + em["gaps"][0, 0]
        assert abs(-(Ep - Em).item() / (2 * h) - F[atom, ax]) < 1e-5


def test_osc_loss_trains():
    from exmlip.data.batching import collate
    from exmlip.training.train import DEFAULT, batch_loss
    m = make("latent", k=1, dipoles=True).train()
    mols = [dict(Z=CH2O[0], R=CH2O[1], gaps=np.array([4.0, 8.0, 9.0, 10.0]), osc=np.array([0.0, 0.1, 0.02, 0.3])),
            dict(Z=C2H4[0], R=C2H4[1], gaps=np.array([7.0, 8.0, 9.0, 10.0]))]
    cfg = {**DEFAULT, "n_states": 4, "n_sup": 4, "w_osc": 1.0}
    b = collate(mols, 4)
    assert torch.isnan(b["osc"][1]).all()
    loss, logs, _ = batch_loss(m, b, cfg, train=True)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads) and "osc_mae" in logs


def test_gap_weights_default_and_lowgap():
    """Default gap weights are the mask (uniform mean, as before); state and low-gap weights act as documented."""
    import torch
    from exmlip.training.train import DEFAULT, gap_weights
    ref = torch.tensor([[0.2, 3.0, 4.0], [4.0, 5.0, float("nan")]], dtype=torch.float64)
    mask = ~torch.isnan(ref)
    W = gap_weights(ref, mask, dict(DEFAULT))
    assert torch.equal(W, mask.to(torch.float64))
    cfg = dict(DEFAULT, gap_state_w=[2.0, 1.0, 1.0], w_lowgap=4.0, lowgap_tau=1.0, lowgap_floor=0.5)
    W = gap_weights(ref, mask, cfg)
    f0 = 1 + 4.0 * torch.exp(torch.tensor(-0.5, dtype=torch.float64))  # S1 = 0.2 eV is floored at 0.5 eV
    f1 = 1 + 4.0 * torch.exp(torch.tensor(-4.0, dtype=torch.float64))
    assert torch.allclose(W[0], torch.stack([2 * f0, f0, f0]))
    assert torch.allclose(W[1, :2], torch.stack([2 * f1, f1])) and W[1, 2] == 0


@pytest.mark.parametrize("hk", [dict(k=1), dict(k=2), dict(k=2, centers="heavy")])
def test_atom_weights_match_eigenvectors(hk):
    """atom_w sums the squared eigenvector rows per atom: columns sum to 1 per molecule and equal U^2 summed by hand."""
    m = make("latent", **hk)
    mols = [CH2O, C2H4]
    Z = torch.tensor(np.concatenate([z for z, _ in mols]))
    R = torch.tensor(np.concatenate([r for _, r in mols]))
    b = torch.cat([torch.full((len(z),), i) for i, (z, _) in enumerate(mols)])
    out = m(Z, R, b, 2, pop_states=3)
    W = out["atom_w"]
    sums = torch.zeros(2, 3, dtype=W.dtype).index_add(0, b, W)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)
    _, U = torch.linalg.eigh(out["H"][1])
    kc = m.head.channels_per_atom(Z[b == 1])
    rows = torch.repeat_interleave(torch.arange(len(kc)), kc)
    ref = torch.zeros(len(kc), 3, dtype=W.dtype).index_add(0, rows, U[: len(rows), :3] ** 2)
    assert torch.allclose(W[b == 1], ref, atol=1e-10)


@pytest.mark.parametrize("target", ["avg", "hole", "elec", "min"])
def test_pop_loss_trains_and_is_train_only(target):
    from exmlip.data.batching import collate
    from exmlip.training.train import DEFAULT, batch_loss
    m = make("latent", k=1).train()
    h = np.zeros((4, 3)); h[0] = [0.6, 0.5, 0.2]; h[1] = [0.4, 0.5, 0.8]  # hydrogens folded onto C (index 0)
    e = np.zeros((4, 3)); e[0] = [0.2, 0.9, 0.5]; e[1] = [0.8, 0.1, 0.5]
    mols = [dict(Z=CH2O[0], R=CH2O[1], gaps=np.array([4.0, 8.0, 9.0, 10.0]), pop_hole=h, pop_elec=e,
                 pop_fold=np.array([0, 1, 0, 0]), pop_mask=np.array([True, True, False])),
            dict(Z=C2H4[0], R=C2H4[1], gaps=np.array([7.0, 8.0, 9.0, 10.0]))]
    cfg = {**DEFAULT, "n_states": 4, "n_sup": 4, "w_pop": 1.0, "n_pop": 3, "pop_target": target}
    b = collate(mols, 4)
    assert b["pop_mask"].tolist() == [[True, True, False], [False, False, False]]
    assert b["pop_fold"].tolist() == [0, 1, 0, 0, 4, 5, 6, 7, 8, 9]
    loss, logs, _ = batch_loss(m, b, cfg, train=True)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grads) and 0 <= logs["pop_shared"] <= 1
    l_val, logs_val, _ = batch_loss(m, b, cfg, train=False)
    l0, _, _ = batch_loss(m, b, {**cfg, "w_pop": 0.0}, train=False)
    assert torch.isclose(l_val, l0) and "pop_shared" in logs_val
