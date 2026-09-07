"""Embed the DR17 spectra through the trained masked AE (no retrain).

The DR19 redux covers only APO-North; the 21-cluster DR17 sample lives in
/data/rgd/lightsurf/data/raw_data/flux_abundances.csv. Embed both through the
same masked AE so the latent covers all 25 clusters.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.models.masked_spectral_ae import MaskedSpectralAE

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_spectra(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in WAVELENGTH_SET]
    X = df[cols].to_numpy("float32")
    X = np.nan_to_num(X, nan=0.0)
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True) + 1e-8
    X = (X - m) / s
    ids = (df["FILE"].astype(str)
             .str.removeprefix("apStar-1.3-apo25m-")
             .str.removeprefix("aspcapStar-dr17-")
             .str.replace(r"-\d{5}$", "", regex=True))
    return ids, X


@click.command()
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--embeddings-out", required=True)
@click.option("--batch-size", default=256, show_default=True)
def main(spectra, model, embeddings_out, batch_size):
    ids, X = load_spectra(spectra)
    print(f"embedding {len(ids)} stars × {X.shape[1]} bins")
    device = _device()

    m = MaskedSpectralAE(n_features=X.shape[1], latent_dim=256)
    m.load_state_dict(torch.load(model, map_location=device))
    m.to(device)
    m.eval()

    Xt = torch.from_numpy(X.reshape(len(ids), 1, X.shape[1]))
    Z = []
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            xb = Xt[i:i + batch_size].to(device)
            Z.append(m.embed(xb).cpu().numpy())
    Z = np.concatenate(Z, axis=0)

    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d].astype("float32")
    Path(embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(embeddings_out, index=False)
    print(f"{len(frame)} × {Z.shape[1]} → {embeddings_out}")


if __name__ == "__main__":
    main()
