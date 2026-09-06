"""Pretrain the masked spectral autoencoder (foundation model) + export latents.

Self-supervised: mask contiguous wavelength blocks, reconstruct them, MSE on
the masked pixels only. No abundance labels. Exports the latent ``z`` keyed by
APOGEE_ID for downstream chemical-tagging clustering.

    uv run python scripts/pretrain_masked_ae.py \
        --spectra flux_out/flux_abundances.csv \
        --latent-dim 256 --mask-ratio 0.5 --block-size 200 \
        --epochs 100 --batch-size 64 \
        --model-out flux_out/masked_ae.pt \
        --embeddings-out flux_out/masked_latent.parquet
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.models.masked_spectral_ae import MaskedSpectralAE

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flux_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c in WAVELENGTH_SET]


def load_spectra(path: str | Path, standardize: bool = True):
    df = pd.read_csv(path)
    cols = flux_columns(list(df.columns))
    X = df[cols].to_numpy(dtype="float32")
    X = np.nan_to_num(X, nan=0.0)
    if standardize:
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - m) / s
    ids = df["FILE"].astype(str).str.removesuffix(".fits")
    return ids, X


@click.command()
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--latent-dim", default=256, show_default=True)
@click.option("--mask-ratio", default=0.5, show_default=True)
@click.option("--block-size", default=200, show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--learning-rate", default=1e-3, show_default=True)
@click.option("--patience", default=15, show_default=True)
@click.option("--model-out", default="models/masked_ae.pt")
@click.option("--embeddings-out", default="data/embeddings/masked_latent.parquet")
def main(spectra, latent_dim, mask_ratio, block_size, epochs, batch_size,
         learning_rate, patience, model_out, embeddings_out):
    ids, X = load_spectra(spectra)
    print(f"pretraining on {len(ids)} stars × {X.shape[1]} bins → {latent_dim}-d latent")
    n = len(ids)

    model = MaskedSpectralAE(n_features=X.shape[1], latent_dim=latent_dim,
                             mask_ratio=mask_ratio, block_size=block_size)
    device = _device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Xt = torch.from_numpy(X.reshape(n, 1, X.shape[1]))
    idx = torch.randperm(n)
    n_val = int(n * 0.2)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    Xv = Xt[val_idx].to(device)
    loader = DataLoader(TensorDataset(Xt[train_idx]), batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)
            mask, recon = out["mask"], out["recon"]
            loss = nn.functional.mse_loss(recon[mask], xb[mask])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item()) * len(xb)
        epoch_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), batch_size):
                xb = Xv[i:i + batch_size]
                out = model(xb)
                mask, recon = out["mask"], out["recon"]
                val_loss += float(nn.functional.mse_loss(recon[mask], xb[mask]).item()) * len(xb)
        val_loss /= len(Xv)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"model → {model_out}")

    # export latents (unmasked spectra)
    model.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = Xt[i:i + batch_size].to(device)
            Z.append(model.embed(xb).cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d]
    Path(embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(embeddings_out, index=False)
    print(f"{len(frame)} × {Z.shape[1]} latents → {embeddings_out}")


if __name__ == "__main__":
    main()
