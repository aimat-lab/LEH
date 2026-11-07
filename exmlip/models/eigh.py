"""Symmetric eigendecomposition with a degeneracy-safe backward pass.

torch.linalg.eigh has gradients for eigenvectors that contain 1/(lambda_j - lambda_i), which diverge at (near-)
degenerate eigenvalues. Losses on eigenvector-derived quantities (oscillator strengths) need eigenvector gradients,
so the backward pass here uses the Lorentzian-broadened factor (lambda_j - lambda_i) / ((lambda_j - lambda_i)^2 + eps^2),
which equals the exact expression away from degeneracies and stays finite at them. Eigenvalue gradients are exact.
"""
import torch


class SafeEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, eps):
        lam, U = torch.linalg.eigh(A)
        ctx.save_for_backward(lam, U)
        ctx.eps = eps
        return lam, U

    @staticmethod
    def backward(ctx, g_lam, g_U):
        lam, U = ctx.saved_tensors
        Ut = U.transpose(-1, -2)
        inner = torch.zeros_like(U)
        if g_lam is not None:
            inner = inner + torch.diag_embed(g_lam)
        if g_U is not None:
            d = lam.unsqueeze(-2) - lam.unsqueeze(-1)  # d_ij = lam_j - lam_i
            F = d / (d * d + ctx.eps ** 2)
            F = F - torch.diag_embed(torch.diagonal(F, dim1=-2, dim2=-1))  # zero diagonal
            inner = inner + F * (Ut @ g_U)
        gA = U @ inner @ Ut
        return 0.5 * (gA + gA.transpose(-1, -2)), None


def safe_eigh(A, eps=1e-4):
    """Ascending eigenvalues and eigenvectors (columns) of a batch of symmetric matrices with a stable backward."""
    return SafeEigh.apply(A, eps)
