"""Train the Phase-B disentangled autoencoder and export its latent.

Single command for the GPU machine (parallel to train_embedding_model.py):

    uv run python scripts/train_disentangled_ae.py \\
        --spectra data/raw_data/flux_abundances.csv \\
        --lambda 0.1 \\
        --epochs 100 \\
        --model-out models/disentangled_ae.keras \\
        --embeddings-out data/embeddings/disentangled.parquet

The latent z is the abundance-free chemical embedding (de Mijolla et al.
2021, arXiv:2103.06377), disentangled from Teff/logg by the gradient-reversal
head. The exported parquet is keyed by APOGEE_ID and consumed by the
workshop repo's cluster.spectral module.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_WAVELENGTH_AIR_STR
from lightsurf.domain.models.disentangled_ae import DisentangledSpectralAE

WAVELENGTH_SET = frozenset(APOGEE_WAVELENGTH_AIR_STR)


def load_conditions(path: str | Path) -> tuple[pd.Series, np.ndarray, np.ndarray]:
    """Read flux CSV -> (ids, spectra, [Teff, logg] conditions)."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in WAVELENGTH_SET]
    cond_cols = [c for c in ("TEFF", "LOGG") if c in df.columns]
    if not cols or len(cond_cols) != 2:
        raise ValueError(
            f"need flux columns and TEFF+LOGG; got {len(cols)} flux, "
            f"{len(cond_cols)} conditions in {path}",
        )
    X = df[cols].to_numpy(dtype="float32")
    U = df[cond_cols].to_numpy(dtype="float32")
    # standardise conditions so Teff/logg are on a comparable scale
    med = np.nanmedian(U, axis=0)
    scale = np.nanstd(U, axis=0)
    scale[scale == 0] = 1.0
    U = (U - med) / scale
    return df["FILE"].astype(str).str.removeprefix("aspcapStar-dr17-"), X, U


@click.command()
@click.option("--spectra", required=True, type=click.Path(exists=True))
@click.option("--lambda", "disentanglement_lambda", default=0.1, show_default=True)
@click.option("--latent-dim", default=16, show_default=True)
@click.option("--epochs", default=100, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--model-out", default="models/disentangled_ae.keras")
@click.option("--embeddings-out", default="data/embeddings/disentangled.parquet")
def main(
    spectra: str, disentanglement_lambda: float, latent_dim: int,
    epochs: int, batch_size: int, model_out: str, embeddings_out: str,
) -> None:
    ids, X, U = load_conditions(spectra)
    click.echo(
        f"training disentangled AE on {len(ids)} stars "
        f"× {X.shape[1]} flux bins, λ={disentanglement_lambda}",
    )

    model = DisentangledSpectralAE(
        latent_dim=latent_dim,
        n_conditions=2,
        disentanglement_lambda=disentanglement_lambda,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_path=model_out,
    )
    model.fit(X, U)
    model.model.save(model_out)
    click.echo(f"💾 model → {model_out}")

    Z = model.encode(X)
    out_path = Path(embeddings_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d]
    frame.to_parquet(out_path, index=False)
    click.echo(f"💾 {len(frame)} × {Z.shape[1]} latent → {out_path}")


if __name__ == "__main__":
    main()
