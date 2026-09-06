"""Export latent embeddings from a trained PyTorch CNN-LSTM-Attention model.

Run on the machine where the model was trained (has the .pt artifact):

    uv run python scripts/export_embeddings.py \
        --model models/cnn_lstm_attention_model.pt \
        --spectra data/raw_data/flux_abundances.csv \
        --layer attention \
        --out data/embeddings/attention.parquet
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.models.deep_models import CnnLstmAttention, _device

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def flux_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c in WAVELENGTH_SET]


def load_spectra(path: str | Path, id_column: str = "FILE"):
    df = pd.read_csv(path)
    cols = flux_columns(list(df.columns))
    if not cols:
        raise ValueError(f"no wavelength columns found in {path}")
    return df[id_column], df[cols].to_numpy(dtype="float32")


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--layer", default="attention", show_default=True)
@click.option("--out", required=True)
@click.option("--id-column", default="FILE", show_default=True)
@click.option("--strip-prefix", default="aspcapStar-dr17-", show_default=True)
def main(model: str, spectra: str, layer: str, out: str, id_column: str, strip_prefix: str) -> None:
    ids, X = load_spectra(spectra, id_column)
    n_features = X.shape[1]
    net = CnnLstmAttention(n_features=n_features)
    net.load_state_dict(torch.load(model, map_location=_device()))
    net.to(_device())
    net.eval()

    Xs = X.reshape(-1, 1, n_features)
    Z = []
    with torch.no_grad():
        for i in range(0, len(Xs), 64):
            xb = torch.from_numpy(Xs[i:i + 64]).to(_device())
            Z.append(net(xb)[layer].detach().cpu().numpy())
    Z = np.concatenate(Z, axis=0)

    if strip_prefix:
        ids = ids.astype(str).str.removeprefix(strip_prefix)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({id_column: ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d]
    frame.to_parquet(out_path, index=False)
    click.echo(f"💾 {len(frame)} × {Z.shape[1]} embeddings → {out_path}")


if __name__ == "__main__":
    main()
