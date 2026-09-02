"""Export latent embeddings from a trained CNN-LSTM-Attention model.

Run on the machine where the model was trained (has the .keras artifact):

    uv run python scripts/export_embeddings.py \
        --model models/cnn_lstm_attention_model.keras \
        --spectra data/raw_data/flux_abundances.csv \
        --layer attention \
        --out data/embeddings/attention.parquet

The output is a parquet with an identifier column (FILE/APOGEE_ID) plus one
column per latent dimension — consumed downstream by the workshop repo's
`cluster.spectral` module.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import tensorflow as tf

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def flux_columns(columns: list[str]) -> list[str]:
    """Return the wavelength (flux) columns, dropping metadata + targets."""
    return [c for c in columns if c in WAVELENGTH_SET]


def load_spectra(
    path: str | Path, id_column: str = "FILE",
) -> tuple[pd.Series, np.ndarray]:
    """Read the flux CSV and return (identifiers, flux matrix)."""
    df = pd.read_csv(path)
    cols = flux_columns(list(df.columns))
    if not cols:
        raise ValueError(f"no wavelength columns found in {path}")
    return df[id_column], df[cols].to_numpy(dtype="float32")


@click.command()
@click.option(
    "--model", required=True, type=click.Path(exists=True),
    help="Trained .keras model.",
)
@click.option(
    "--spectra", required=True, type=click.Path(exists=True),
    help="flux CSV (flux_abundances.csv).",
)
@click.option(
    "--layer", default="attention", show_default=True,
    help="attention | dense_0 | dense_1.",
)
@click.option("--out", required=True, help="Output parquet path.")
@click.option("--id-column", default="FILE", show_default=True)
@click.option(
    "--strip-prefix", default="aspcapStar-dr17-", show_default=True,
    help="Prefix to strip from the identifier (set empty to keep it).",
)
def main(
    model: str, spectra: str, layer: str, out: str,
    id_column: str, strip_prefix: str,
) -> None:
    tf_model = tf.keras.models.load_model(model, compile=False)
    layer_out = tf_model.get_layer(layer).output
    embedder = tf.keras.Model(tf_model.input, layer_out, name=f"embedding_{layer}")

    ids, X = load_spectra(spectra, id_column)
    if strip_prefix:
        ids = ids.astype(str).str.removeprefix(strip_prefix)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    Z = embedder.predict(X, verbose=1)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({id_column: ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d]
    frame.to_parquet(out_path, index=False)
    click.echo(f"💾 {len(frame)} × {Z.shape[1]} embeddings → {out_path}")


if __name__ == "__main__":
    main()
