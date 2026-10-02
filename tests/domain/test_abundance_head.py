"""Tests for the abundance regression head (HEAD workstream).

CPU-only, synthetic, fast. Cover the acceptance criteria that touch HEAD code:
AC1 (masked loss), AC2 (ft_last leaves fc1 frozen), standardiser round-trip,
finetune-returns-a-copy, determinism (AC9), and an end-to-end synthetic check
that fine-tuning closes the DR19->DR17 gap as n grows.
"""

from __future__ import annotations

import numpy as np
import torch

from lightsurf.domain.models.abundance_head import (
    AbundanceHead,
    Standardiser,
    finetune_head,
    fit_head,
    masked_mse,
    predict,
)

N_T = 9


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- AC1: masked loss
def test_masked_mse_ignores_masked_value():
    torch.manual_seed(0)
    pred = torch.randn(8, N_T, requires_grad=True)
    target = torch.randn(8, N_T)
    mask = torch.ones(8, N_T, dtype=torch.bool)
    mask[:, 3] = False  # element 3 fully masked

    loss_a = masked_mse(pred, target, mask)
    # change the masked column arbitrarily -> loss must be identical
    target_b = target.clone()
    target_b[:, 3] = 999.0
    loss_b = masked_mse(pred, target_b, mask)
    assert torch.allclose(loss_a, loss_b)

    # zero gradient into the masked output row
    pred.grad = None
    loss_a.backward()
    assert torch.allclose(pred.grad[:, 3], torch.zeros(8))
    assert pred.grad[:, 0].abs().sum() > 0  # unmasked columns do get gradient


def test_masked_mse_all_masked_is_zero_no_nan():
    pred = torch.randn(4, N_T, requires_grad=True)
    target = torch.full((4, N_T), float("nan"))
    mask = torch.zeros(4, N_T, dtype=torch.bool)
    loss = masked_mse(pred, target, mask)
    assert float(loss.item()) == 0.0
    loss.backward()
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------- standardiser round-trip
def test_standardiser_round_trip():
    rng = _rng(1)
    Y = rng.normal(0.3, 2.0, size=(100, N_T)).astype("float32")
    M = np.ones((100, N_T), dtype=bool)
    std = Standardiser.fit(Y, M)
    Yz = std.transform(Y)
    assert np.allclose(std.inverse(Yz), Y, atol=1e-4)
    # standardised columns are ~0 mean, ~1 std
    assert np.allclose(Yz.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(Yz.std(axis=0), 1, atol=1e-4)


def test_standardiser_ignores_masked():
    Y = np.zeros((10, N_T), dtype="float32")
    M = np.ones((10, N_T), dtype=bool)
    Y[0, 0] = 1e6  # outlier that is masked out
    M[0, 0] = False
    std = Standardiser.fit(Y, M)
    assert abs(std.mean[0]) < 1.0  # outlier excluded


# ------------------------------------------------- AC2: ft_last freezes first layer
def _toy_data(n=200, seed=0):
    rng = _rng(seed)
    Z = rng.normal(size=(n, 256)).astype("float32")
    W = rng.normal(size=(256, N_T)).astype("float32") * 0.1
    Y = (Z @ W + rng.normal(scale=0.05, size=(n, N_T))).astype("float32")
    M = np.ones((n, N_T), dtype=bool)
    return Z, Y, M


def test_finetune_last_leaves_fc1_unchanged():
    Z, Y, M = _toy_data()
    base, std = fit_head(Z, Y, M, epochs=30, lr=1e-3, weight_decay=0.0, seed=0, device="cpu")
    w0 = base.fc1.weight.detach().clone()
    b0 = base.fc1.bias.detach().clone()
    Z2, Y2, M2 = _toy_data(seed=5)
    ft = finetune_head(base, std, Z2, Y2, M2, arm="last", epochs=30, lr=1e-3,
                       weight_decay=0.0, seed=1, device="cpu")
    assert torch.allclose(ft.fc1.weight.detach(), w0)
    assert torch.allclose(ft.fc1.bias.detach(), b0)
    # fc2 did change
    assert not torch.allclose(ft.fc2.weight.detach(), base.fc2.weight.detach())


def test_finetune_returns_copy_base_unchanged():
    Z, Y, M = _toy_data()
    base, std = fit_head(Z, Y, M, epochs=20, lr=1e-3, weight_decay=0.0, seed=0, device="cpu")
    snap = {k: v.detach().clone() for k, v in base.state_dict().items()}
    Z2, Y2, M2 = _toy_data(seed=5)
    ft = finetune_head(base, std, Z2, Y2, M2, arm="full", epochs=20, lr=1e-3,
                       weight_decay=0.0, seed=1, device="cpu")
    assert ft is not base
    for k, v in base.state_dict().items():
        assert torch.allclose(v, snap[k]), f"base param {k} mutated"


# --------------------------------------------------------------- AC9: determinism
def test_determinism_same_seed_same_predictions():
    Z, Y, M = _toy_data()
    h1, s1 = fit_head(Z, Y, M, epochs=25, lr=1e-3, weight_decay=0.0, seed=7, device="cpu")
    h2, s2 = fit_head(Z, Y, M, epochs=25, lr=1e-3, weight_decay=0.0, seed=7, device="cpu")
    p1 = predict(h1, s1, Z, device="cpu")
    p2 = predict(h2, s2, Z, device="cpu")
    assert np.allclose(p1, p2)


# ------------------------------------------------ end-to-end: fine-tune closes gap
def test_end_to_end_finetune_closes_gap():
    """Base head learns DR19; full fine-tune on DR17 drops RMSE well below
    base_n0 as n grows (small-n overfits, large-n closes the gap)."""
    rng = _rng(42)
    n = 5000
    Z = rng.normal(size=(n, 256)).astype("float32")
    W = (rng.normal(size=(256, N_T)) * 0.1).astype("float32")
    y19 = (Z @ W + rng.normal(scale=0.03, size=(n, N_T))).astype("float32")
    # known quadratic DR19->DR17 shift in a synthetic teff (not an input)
    teff = rng.uniform(3200, 4100, size=n).astype("float32")
    t = (teff - 3650.0) / 500.0
    shift = (0.4 + 0.15 * t + 0.05 * t**2)[:, None].astype("float32")
    y17 = (y19 + shift).astype("float32")
    M = np.ones((n, N_T), dtype=bool)

    base, std = fit_head(Z[:2000], y19[:2000], M[:2000], epochs=250, lr=1e-3,
                         weight_decay=1e-4, seed=0, device="cpu", batch_size=128)

    test = slice(4000, 5000)     # scoring set, disjoint from base + reservoir
    res = slice(2000, 4000)      # fine-tune reservoir

    def rmse(pred, y):
        return float(np.sqrt(((pred - y) ** 2).mean()))

    rmse0 = rmse(predict(base, std, Z[test], device="cpu"), y17[test])

    rmses = {}
    for nft in (50, 2000):
        idx = rng.permutation(2000)[:nft]
        ft = finetune_head(base, std, Z[res][idx], y17[res][idx], M[res][idx],
                           arm="full", epochs=200, lr=1e-3, weight_decay=1e-4,
                           seed=0, device="cpu", batch_size=128)
        rmses[nft] = rmse(predict(ft, std, Z[test], device="cpu"), y17[test])

    # large-n fine-tune drops well below base_n0, and more data helps
    assert rmses[2000] < 0.5 * rmse0
    assert rmses[2000] < rmses[50]
