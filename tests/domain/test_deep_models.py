"""Smoke tests for the embedding-extraction hooks on CnnLstmAttentionModel.

Kept tiny (8 synthetic spectra, 64 flux bins, 1 epoch) so the tests run on
CPU in a few seconds — they only verify the latent-tap plumbing, not the
regression quality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lightsurf.domain.models.deep_models import CnnLstmAttentionModel


@pytest.fixture
def tiny_model(tmp_path: Path) -> CnnLstmAttentionModel:
    return CnnLstmAttentionModel(
        epochs=1,
        batch_size=4,
        cnn_filters=[8, 4],
        lstm_units=8,
        dense_units=[6, 4],
        output_dimension=1,
        verbose=0,
        checkpoint_path=str(tmp_path / "model.keras"),
    )


@pytest.fixture
def spectra() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((8, 64)).astype("float32"))


def test_embedding_extraction_shapes(
    tiny_model: CnnLstmAttentionModel, spectra: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(1)
    y = pd.Series(rng.standard_normal(8).astype("float32"))
    tiny_model.fit(spectra, y)

    expected = {"attention": 8, "dense_0": 6, "dense_1": 4}
    for layer, dim in expected.items():
        emb = tiny_model.predict_embeddings(spectra, layer)
        assert emb.shape == (8, dim), f"{layer}: {emb.shape}"


def test_embedding_model_requires_fit(tiny_model: CnnLstmAttentionModel) -> None:
    with pytest.raises(ValueError, match="not built"):
        tiny_model.embedding_model("attention")


def test_multitask_output_dimension(spectra: pd.DataFrame, tmp_path: Any) -> None:
    model = CnnLstmAttentionModel(
        epochs=1,
        batch_size=4,
        cnn_filters=[8, 4],
        lstm_units=8,
        dense_units=[6, 4],
        output_dimension=9,
        verbose=0,
        checkpoint_path=str(tmp_path / "multi.keras"),
    )
    rng = np.random.default_rng(2)
    y = pd.DataFrame(rng.standard_normal((8, 9)).astype("float32"))
    model.fit(spectra, y)
    assert model.model.output.shape[-1] == 9
    assert model.predict_embeddings(spectra, "attention").shape == (8, 8)
