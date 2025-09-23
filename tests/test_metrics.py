from __future__ import annotations

import torch

from gcpvqvae.geometry.metrics import gdt_ts


def test_gdt_ts_perfect_alignment() -> None:
    coords = torch.zeros((5, 3, 3))
    score = gdt_ts(coords, coords)
    assert torch.isclose(score, torch.tensor(1.0))


def test_gdt_ts_thresholds() -> None:
    coords_a = torch.zeros((2, 3, 3))
    coords_b = coords_a.clone()

    coords_b[0, 1, 0] = 0.5  # within all thresholds
    coords_b[1, 1, 0] = 5.0  # only within the 8 Å threshold

    score = gdt_ts(coords_a, coords_b)
    expected = torch.tensor((0.5 + 0.5 + 0.5 + 1.0) / 4.0)
    assert torch.isclose(score, expected)
