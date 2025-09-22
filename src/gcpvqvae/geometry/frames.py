"""Local frame utilities, geometric transformations, and alignment helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def centralize(
    coords: torch.Tensor,
    mask: torch.Tensor | None = None,
    use_ca_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Centralizes coordinates by subtracting the centroid.

    Args:
        coords: A tensor of coordinates, e.g., [L, 3, 3] for (N, CA, C).
        mask: A boolean mask [L] indicating which residues to use for
              calculating the centroid. If None, all are used.
        use_ca_only: If True, uses only the C-alpha atom (index 1) to
                     compute the centroid. Otherwise, uses all atoms.

    Returns:
        A tuple containing the centralized coordinates and the centroid translation.
    """
    if use_ca_only:
        points_for_centroid = coords[:, 1]  # C-alpha atoms
    else:
        points_for_centroid = coords.reshape(-1, 3)

    if mask is not None:
        if use_ca_only:
            points_for_centroid = points_for_centroid[mask]
        else:
            # Mask needs to be expanded for all atoms
            points_for_centroid = points_for_centroid[mask.repeat_interleave(3)]

    centroid = torch.mean(points_for_centroid, dim=0)
    return coords - centroid, centroid


def kabsch(
    P: torch.Tensor,
    Q: torch.Tensor,
    mask: torch.Tensor | None = None,
    allow_reflections: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the optimal rigid transformation (rotation R, translation t)
    that aligns point set P to point set Q using the Kabsch algorithm.

    Args:
        P: The source point set of shape [N, 3].
        Q: The target point set of shape [N, 3].
        mask: A boolean mask [N] indicating which points to use.
        allow_reflections: If False, ensures the rotation is proper (det(R)=1).

    Returns:
        A tuple (R, t) where R is the [3, 3] rotation matrix and t is the
        [3] translation vector, such that Q ≈ R @ P + t.
    """
    if mask is not None:
        P_masked = P[mask]
        Q_masked = Q[mask]
    else:
        P_masked, Q_masked = P, Q

    # Center the point sets
    p_centroid = torch.mean(P_masked, dim=0)
    q_centroid = torch.mean(Q_masked, dim=0)
    P_centered = P_masked - p_centroid
    Q_centered = Q_masked - q_centroid

    # Compute the covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, _, V = torch.linalg.svd(H)
    # Note: linalg.svd returns V, not V.T
    Vt = V.T

    # Compute rotation matrix
    R = Vt @ U.T

    # Handle reflections if necessary
    if not allow_reflections and torch.linalg.det(R) < 0:
        Vt_copy = Vt.clone()
        Vt_copy[-1, :] *= -1
        R = Vt_copy @ U.T

    # Compute translation
    t = q_centroid - (R @ p_centroid)

    return R, t


def build_edge_frames(
    edge_index: torch.Tensor,
    coords: torch.Tensor,
) -> torch.Tensor:
    """
    Builds right-handed orthonormal frames F_ij for each edge.
    Follows the procedure from Section 1.5 of the workplan.
    """
    i_nodes, j_nodes = edge_index
    ca_coords = coords[:, 1]

    r_ij = ca_coords[j_nodes] - ca_coords[i_nodes]

    # Tangent vector
    a_ij = F.normalize(r_ij, dim=-1)

    # Use the C-N vector as a robust, local bias to define the frame's orientation
    c_i = coords[i_nodes, 2]
    n_i = coords[i_nodes, 0]
    tangent_bias = F.normalize(c_i - n_i, dim=-1)

    # Gram-Schmidt to get an orthonormal basis
    b_ij_unnormalized = tangent_bias - torch.sum(tangent_bias * a_ij, dim=-1, keepdim=True) * a_ij
    b_ij = F.normalize(b_ij_unnormalized, dim=-1)

    c_ij = torch.cross(a_ij, b_ij, dim=-1)

    # Stack into frames [E, 3, 3] and ensure they are right-handed
    frames = torch.stack([a_ij, b_ij, c_ij], dim=1)
    return frames
