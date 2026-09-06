"""Smoke tests for the Phase-B disentangled conditional autoencoder (PyTorch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lightsurf.domain.models.disentangled_ae import DisentangledSpectralAE


@pytest.fixture
def tiny_ae(tmp_path: Path) -> DisentangledSpectralAE:
    return DisentangledSpectralAE(
        latent_dim=8,
        n_conditions=2,
        disentanglement_lambda=0.1,
        epochs=1,
        batch_size=4,
        verbose=0,
        checkpoint_path=str(tmp_path / "ae.pt"),
    )


def test_build_train_and_encode(
    tiny_ae: DisentangledSpectralAE, tmp_path: Any,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 64)).astype("float32")
    U = rng.standard_normal((8, 2)).astype("float32")

    history = tiny_ae.fit(X, U)
    assert history is not None

    z = tiny_ae.encode(X)
    assert z.shape == (8, 8)  # latent_dim


def test_encoder_shape(tiny_ae: DisentangledSpectralAE) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 64)).astype("float32")
    U = rng.standard_normal((4, 2)).astype("float32")
    tiny_ae.fit(X, U)
    z = tiny_ae.encode(X)
    assert z.shape == (4, 8)
