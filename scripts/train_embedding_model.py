"""Train the multi-task CNN-LSTM-Attention model and export embeddings (PyTorch).

Single command for the GPU machine (replaces the TensorFlow path):

    uv run python scripts/train_embedding_model.py \
        --spectra data/raw_data/flux_abundances.csv \
        --layer attention \
        --epochs 100 \
        --model-out models/cnn_lstm_attention_model.pt \
        --embeddings-out data/embeddings/attention.parquet

The model regresses ALL APOGEE_ABUNDANCE_TARGETS simultaneously
(output_dimension = 9), so the latent layer is a shared chemical
representation. The exported parquet is keyed by APOGEE_ID and consumed by
the workshop repo's cluster.spectral module.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS, APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.models.deep_models import CnnLstmAttentionModel

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def flux_columns(columns: list[str]) -> list[str]:
    """Return the wavelength (flux) columns, dropping metadata + targets."""
    return [c for c in columns if c in WAVELENGTH_SET]


def load_training_data(
    path: str | Path, id_column: str = "FILE", standardize_flux: bool = False,
) -> tuple[pd.Series, np.ndarray, np.ndarray, list[str]]:
    """Read the flux CSV → (ids, spectra, multi-target labels, target names)."""
    df = pd.read_csv(path)
    cols = flux_columns(list(df.columns))
    targets = [t for t in APOGEE_ABUNDANCE_TARGETS if t in df.columns]
    if not cols:
        raise ValueError(f"no wavelength columns found in {path}")
    if not targets:
        raise ValueError(f"no abundance targets found in {path}")
    X = df[cols].to_numpy(dtype="float32")
    X = np.nan_to_num(X, nan=0.0)  # apStar bad pixels -> 0 flux
    if standardize_flux:
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - m) / s
    Y = df[targets].to_numpy(dtype="float32")
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    ids = df[id_column].astype(str)
    ids = (
        ids.str.removeprefix("apStar-1.3-apo25m-")
           .str.removeprefix("aspcapStar-dr17-")
           .str.replace(r"-\d{5}$", "", regex=True)
    )
    return ids[keep], X[keep], Y[keep], targets


@click.command()
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--layer", default="attention", show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--patience", default=5, show_default=True)
@click.option("--learning-rate", default=0.001, show_default=True)
@click.option("--loss", default="mse", show_default=True,
              type=click.Choice(["mse", "mae", "huber", "weighted_mse"]))
@click.option("--standardize-targets", is_flag=True, default=False, show_default=True)
@click.option("--standardize-flux", is_flag=True, default=False, show_default=True,
              help="Per-star zero-mean unit-var on the flux (DR19 apStar is raw ~1e4).")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--model-out", default="models/cnn_lstm_attention_model.pt")
@click.option("--embeddings-out", default="data/embeddings/attention.parquet")
def main(
    spectra: str, layer: str, epochs: int, patience: int, learning_rate: float,
    loss: str, standardize_targets: bool, standardize_flux: bool,
    batch_size: int, model_out: str, embeddings_out: str,
) -> None:
    ids, X, Y, targets = load_training_data(spectra, standardize_flux=standardize_flux)
    click.echo(
        f"training on {len(ids)} stars × {X.shape[1]} flux bins "
        f"→ {len(targets)} targets {targets}",
    )

    model = CnnLstmAttentionModel(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=patience,
        loss=loss,
        standardize_targets=standardize_targets,
        output_dimension=len(targets),
        checkpoint_path=model_out,
    )
    model.fit(pd.DataFrame(X), pd.DataFrame(Y))
    click.echo(f"💾 model → {model_out}")

    Z = model.predict_embeddings(pd.DataFrame(X), layer)
    out_path = Path(embeddings_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d]
    frame.to_parquet(out_path, index=False)
    click.echo(f"💾 {len(frame)} × {Z.shape[1]} embeddings → {out_path}")


if __name__ == "__main__":
    main()
