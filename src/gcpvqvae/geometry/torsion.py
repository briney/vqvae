"""Torsion angle computations for protein backbones."""

from __future__ import annotations

import torch


def _dihedral_angle(p0, p1, p2, p3):
    """
    Computes the dihedral angle between four points.

    The angle is computed in radians, in the range [-pi, pi].
    The points are assumed to be torch tensors.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(b1, v, dim=-1) * w, dim=-1)

    return torch.atan2(y, x)


def backbone_torsions(coords: torch.Tensor, mask: torch.Tensor):
    """
    Compute φ/ψ/ω torsions for a backbone.

    Args:
        coords: A tensor of backbone coordinates of shape [L, 3, 3],
                representing the N, CA, C atoms.
        mask: A boolean tensor of shape [L] indicating valid residues.

    Returns:
        A tensor of shape [L, 6] containing the sin and cos of the
        φ, ψ, and ω angles. Missing angles are set to 0.
    """
    L = coords.shape[0]

    # Extract atom coordinates
    n_coords = coords[:, 0, :]
    ca_coords = coords[:, 1, :]
    c_coords = coords[:, 2, :]

    # Pad with zeros for boundary conditions
    n_coords_padded = torch.cat([torch.zeros(1, 3), n_coords, torch.zeros(1, 3)], dim=0)
    ca_coords_padded = torch.cat([torch.zeros(1, 3), ca_coords, torch.zeros(1, 3)], dim=0)
    c_coords_padded = torch.cat([torch.zeros(1, 3), c_coords, torch.zeros(1, 3)], dim=0)

    # Phi: C(i-1) - N(i) - CA(i) - C(i)
    phi = _dihedral_angle(
        c_coords_padded[:-2],
        n_coords_padded[1:-1],
        ca_coords_padded[1:-1],
        c_coords_padded[1:-1]
    )

    # Psi: N(i) - CA(i) - C(i) - N(i+1)
    psi = _dihedral_angle(
        n_coords_padded[1:-1],
        ca_coords_padded[1:-1],
        c_coords_padded[1:-1],
        n_coords_padded[2:]
    )

    # Omega: CA(i) - C(i) - N(i+1) - CA(i+1)
    omega = _dihedral_angle(
        ca_coords_padded[1:-1],
        c_coords_padded[1:-1],
        n_coords_padded[2:],
        ca_coords_padded[2:]
    )

    # Create a mask for valid angle calculations
    # Phi requires C(i-1), so the first residue is invalid
    # Psi requires N(i+1), so the last residue is invalid
    # Omega requires N(i+1) and CA(i+1), so the last residue is invalid
    mask_phi = torch.ones(L, dtype=torch.bool)
    mask_phi[0] = False
    mask_psi = torch.ones(L, dtype=torch.bool)
    mask_psi[-1] = False
    mask_omega = torch.ones(L, dtype=torch.bool)
    mask_omega[-1] = False

    # Combine with the input mask
    phi_mask = mask & mask_phi
    psi_mask = mask & mask_psi
    omega_mask = mask & mask_omega

    # Set invalid angles to 0
    phi = torch.where(phi_mask, phi, torch.zeros_like(phi))
    psi = torch.where(psi_mask, psi, torch.zeros_like(psi))
    omega = torch.where(omega_mask, omega, torch.zeros_like(omega))

    # Get sin/cos pairs
    torsions = torch.stack([
        torch.sin(phi), torch.cos(phi),
        torch.sin(psi), torch.cos(psi),
        torch.sin(omega), torch.cos(omega),
    ], dim=-1)

    # Final check to zero out rows corresponding to masked residues
    torsions = torsions * mask.unsqueeze(-1)

    return torsions
