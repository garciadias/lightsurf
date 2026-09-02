"""Train the multi-task CNN-LSTM-Attention model and export embeddings.

Single command for the GPU machine (replaces the RandomizedSearchCV path for
the embedding use-case — a fixed, reproducible config):

    uv run python scripts/train_embedding_model.py \\
        --spectra data/raw_data/flux_abundances.csv \\
        --layer attention \\
        --epochs 100 \\
        --model-out models/cnn_lstm_attention_model.keras \\
        --embeddings-out data/embeddings/attention.parquet

The model regresses ALL APOGEE_ABUNDANCE_TARGETS simultaneously
(output_dimension = 9), so the latent layer is a shared chemical
representation. The exported parquet is keyed by APOGEE_ID (FILE prefix
stripped) and consumed by the workshop repo's cluster.spectral module.
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
    path: str | Path, id_column: str = "FILE",
) -> tuple[pd.Series, np.ndarray, np.ndarray, list[str]]:
    """Read the flux CSV → (ids, spectra, multi-target labels, target names).

    Rows with a non-finite flux value or a non-finite abundance target are
    dropped (APOGEE abundances have NaN for undetected lines).
    """
    df = pd.read_csv(path)
    cols = flux_columns(list(df.columns))
    targets = [t for t in APOGEE_ABUNDANCE_TARGETS if t in df.columns]
    if not cols:
        raise ValueError(f"no wavelength columns found in {path}")
    if not targets:
        raise ValueError(f"no abundance targets found in {path}")
    X = df[cols].to_numpy(dtype="float32")
    Y = df[targets].to_numpy(dtype="float32")
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return (
        df[id_column].astype(str).str.removeprefix("aspcapStar-dr17-")[keep],
        X[keep],
        Y[keep],
        targets,
    )


@click.command()
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--layer", default="attention", show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--model-out", default="models/cnn_lstm_attention_model.keras")
@click.option("--embeddings-out", default="data/embeddings/attention.parquet")
def main(
    spectra: str, layer: str, epochs: int, batch_size: int,
    model_out: str, embeddings_out: str,
) -> None:
    ids, X, Y, targets = load_training_data(spectra)
    click.echo(
        f"training on {len(ids)} stars × {X.shape[1]} flux bins "
        f"→ {len(targets)} targets {targets}",
    )

    model = CnnLstmAttentionModel(
        epochs=epochs,
        batch_size=batch_size,
        output_dimension=len(targets),
        checkpoint_path=model_out,
    )
    model.fit(pd.DataFrame(X), pd.DataFrame(Y))
    model.model.save(model_out)
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
