"""Download the missing DR17 aspcapStar + embed through the masked AE."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from lightsurf.domain.models.masked_spectral_ae import MaskedSpectralAE


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.option("--list", required=True, type=click.Path(exists=True))
@click.option("--outdir", default="aspcap_dr17")
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--embeddings-out", required=True)
@click.option("--workers", default=40)
def main(list, outdir, model, embeddings_out, workers):
    df = pd.read_csv(list)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    def dl(row):
        dest = out / f"aspcapStar-dr17-{row['APOGEE_ID']}.fits"
        if dest.exists() and dest.stat().st_size > 0:
            return "skip"
        r = subprocess.run(
            ["wget", "--no-check-certificate", "--tries=5", "--timeout=20",
             "-O", str(dest), row["url"]],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return "ok"
        dest.unlink(missing_ok=True)
        return "fail"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(dl, [r for _, r in df.iterrows()]))
    import collections
    print("download:", collections.Counter(res))

    # extract flux + embed
    ids, fluxes = [], []
    for f in sorted(out.glob("*.fits")):
        try:
            with fits.open(f) as hdul:
                flux = hdul[1].data
            if flux.ndim == 2:
                flux = flux[0]
            fluxes.append(flux.astype("float32"))
            ids.append(f.stem.removeprefix("aspcapStar-dr17-"))
        except Exception:
            continue
    if not fluxes:
        print("no spectra loaded")
        return
    X = np.array(fluxes)
    X = np.nan_to_num(X, nan=0.0)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    print(f"embedding {len(ids)} stars × {X.shape[1]} bins")

    m = MaskedSpectralAE(n_features=X.shape[1], latent_dim=256)
    m.load_state_dict(torch.load(model, map_location=_device()))
    m.to(_device())
    m.eval()
    Xt = torch.from_numpy(X.reshape(len(ids), 1, X.shape[1]))
    Z = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            Z.append(m.embed(Xt[i:i + 256].to(_device())).cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d].astype("float32")
    Path(embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(embeddings_out, index=False)
    print(f"{len(frame)} × {Z.shape[1]} → {embeddings_out}")


if __name__ == "__main__":
    main()
