import math
from typing import Optional, Union

import torch


TensorLike = Union[torch.Tensor]


def _as_square_dense_tensor(
    adj: TensorLike,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if not torch.is_tensor(adj):
        adj = torch.as_tensor(adj, dtype=dtype, device=device)
    else:
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = adj.to(device=device, dtype=dtype)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square 2D, got shape={tuple(adj.shape)}")

    return adj


def _degenerate_block_bounds(eigenvalues: torch.Tensor, tol: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find contiguous degenerate-eigenvalue blocks from sorted eigenvalues.

    For eigh output (ascending), a new block starts where adjacent gap >= tol.
    Returns:
        starts: 1D long tensor of block starts (inclusive)
        ends:   1D long tensor of block ends   (exclusive)
    """
    n = eigenvalues.numel()
    device = eigenvalues.device
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    if n == 1:
        starts = torch.zeros(1, dtype=torch.long, device=device)
        ends = torch.ones(1, dtype=torch.long, device=device)
        return starts, ends

    gaps = (eigenvalues[1:] - eigenvalues[:-1]).abs()
    new_block = gaps >= tol
    starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            torch.nonzero(new_block, as_tuple=False).squeeze(1) + 1,
        ],
        dim=0,
    )
    ends = torch.cat(
        [
            starts[1:],
            torch.tensor([n], dtype=torch.long, device=device),
        ],
        dim=0,
    )
    return starts, ends


@torch.no_grad()
def perturbation_torch(
    adj_unperturbed: torch.Tensor,
    adj_full: torch.Tensor,
    *,
    tol: float = 1e-11,
) -> torch.Tensor:
    """Torch version of MATLAB `perturbation(AdjTraining, Adj)`."""
    adj_unperturbed = (adj_unperturbed + adj_unperturbed.T) * 0.5
    adj_full = (adj_full + adj_full.T) * 0.5

    n = adj_full.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(adj_unperturbed)

    v2 = eigenvectors.clone()
    adj_pertu = adj_full - adj_unperturbed

    starts, ends = _degenerate_block_bounds(eigenvalues, tol=tol)
    block_sizes = ends - starts
    multi_blocks = block_sizes > 1

    # Each degenerate eigenspace needs its own small eigendecomposition.
    # This is already matrix-form inside each block; only block iteration remains.
    for start, end in zip(starts[multi_blocks].tolist(), ends[multi_blocks].tolist()):
        idx = torch.arange(start, end, device=adj_full.device)
        v_redundant = eigenvectors[:, idx]
        m = v_redundant.T @ adj_pertu @ v_redundant
        m = (m + m.T) * 0.5

        _, v_r = torch.linalg.eigh(m)
        v_redundant = v_redundant @ v_r
        v_redundant = v_redundant / torch.linalg.norm(v_redundant, dim=0, keepdim=True).clamp_min(1e-30)
        v2[:, idx] = v_redundant

    diag_vals = torch.diagonal(v2.T @ adj_full @ v2)
    adj_anneal = (v2 * diag_vals.unsqueeze(0)) @ v2.T
    adj_anneal = (adj_anneal + adj_anneal.T) * 0.5
    return adj_anneal


@torch.no_grad()
def spm_carlo_edited_torch(
    adj_training: TensorLike,
    *,
    perturb_ratio: float = 0.1,
    perturbations: int = 10,
    tol: float = 1e-11,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Torch matrix implementation of MATLAB `SPM_carlo_edited`.

    Args:
        adj_training: NxN undirected adjacency matrix (0/1 or weighted).
        perturb_ratio: ratio of existing lower-triangle edges to remove each run.
        perturbations: number of independent perturbations.
        tol: eigenvalue equality tolerance for degeneracy handling.
        device: CPU/CUDA device.
        dtype: tensor dtype.
        seed: optional RNG seed for deterministic edge sampling.
    """
    if seed is not None:
        torch.manual_seed(seed)

    adj_training = _as_square_dense_tensor(adj_training, device=device, dtype=dtype)
    adj_training = (adj_training + adj_training.T) * 0.5

    n = adj_training.shape[0]
    prob_matrix = torch.zeros((n, n), dtype=dtype, device=adj_training.device)

    lower_edge_idx = torch.nonzero(torch.tril(adj_training != 0), as_tuple=False)
    num_lower_edges = lower_edge_idx.shape[0]
    pertu_size = int(math.ceil(perturb_ratio * num_lower_edges))

    if num_lower_edges == 0 or pertu_size == 0 or perturbations <= 0:
        return prob_matrix

    for _ in range(perturbations):
        pick = torch.randperm(num_lower_edges, device=adj_training.device)[:pertu_size]
        removed_edges = lower_edge_idx[pick]

        adj_unpertu = adj_training.clone()
        r, c = removed_edges[:, 0], removed_edges[:, 1]
        adj_unpertu[r, c] = 0
        adj_unpertu[c, r] = 0

        prob_matrix = prob_matrix + perturbation_torch(adj_unpertu, adj_training, tol=tol)

    return prob_matrix


if __name__ == "__main__":
    a = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=torch.float64,
    )
    out = spm_carlo_edited_torch(a, perturbations=3, seed=0)
    print(out)
