"""Pretrain the masked AE on raw DR19 mwmStar spectra + embed everything.

Self-contained GPU pipeline for the uniform re-run. Reads mwmStar fits
directly (HDU APOGEE/APO preferred, LCO fallback), standardises per star,
pretrains the MaskedSpectralAE, then embeds every spectrum and writes a
parquet keyed by APOGEE_ID.

Run on the desktop (RTX 5090):
    uv run python pretrain_embed.py --spec-dir data/field \
        --model-out data/embeddings/masked_ae_rerun.pt \
        --embeddings-out data/embeddings/masked_latent_field.parquet
    # then embed the members with the saved checkpoint:
    uv run python pretrain_embed.py --spec-dir data/members \
        --checkpoint data/embeddings/masked_ae_rerun.pt \
        --embeddings-out data/embeddings/masked_latent_members.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from masked_spectral_ae import MaskedSpectralAE

N_BINS = 8575


def read_flux(path: Path) -> np.ndarray | None:
    """Combined APOGEE spectrum (raw, 8575 px); APO preferred, LCO fallback."""
    try:
        with fits.open(path, memmap=True) as h:
            fallback: np.ndarray | None = None
            for hdu in h:
                name = (hdu.name or "").upper()
                if not name.startswith("APOGEE") or hdu.data is None:
                    continue
                if "flux" not in hdu.columns.names or hdu.data.shape[0] == 0:
                    continue
                flux = np.asarray(hdu.data["flux"], dtype=np.float32)
                if flux.ndim == 2:
                    flux = flux[0]
                if flux.size == N_BINS and np.nanmedian(flux) > 0:
                    if name.endswith("APO"):
                        return flux
                    fallback = fallback or flux
            return fallback
    except Exception:
        return None


def load_specs(spec_dir: Path) -> tuple[list[str], np.ndarray]:
    """Read all fits; standardise per star. Returns (ids, X)."""
    ids, spectra = [], []
    for f in sorted(spec_dir.glob("*.fits")):
        flux = read_flux(f)
        if flux is None:
            continue
        ids.append(f.stem)
        spectra.append(flux)
    X = np.stack(spectra).astype("float32")
    X = np.nan_to_num(X, nan=0.0)
    m = X.mean(1, keepdims=True)
    s = X.std(1, keepdims=True) + 1e-8
    X = (X - m) / s
    return ids, X


def embed(model: MaskedSpectralAE, Xt: torch.Tensor, dev: torch.device,
          batch_size: int) -> np.ndarray:
    Z = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            Z.append(model.embed(Xt[i:i + batch_size].to(dev)).cpu().numpy())
    return np.concatenate(Z, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec-dir", required=True)
    p.add_argument("--model-out", required=True)
    p.add_argument("--embeddings-out", required=True)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--block-size", type=int, default=200)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None,
                   help="if set, skip training and just embed --spec-dir")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)

    ids, X = load_specs(Path(args.spec_dir))
    print(f"loaded {len(ids)} spectra x {X.shape[1]} bins", flush=True)
    n = len(ids)
    Xt = torch.from_numpy(X.reshape(n, 1, X.shape[1]))

    model = MaskedSpectralAE(n_features=X.shape[1], latent_dim=args.latent_dim,
                             mask_ratio=args.mask_ratio, block_size=args.block_size)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=dev))
        print(f"loaded checkpoint {args.checkpoint}; skipping training", flush=True)
    else:
        idx = torch.randperm(n)
        n_val = int(n * 0.2)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        Xv = Xt[val_idx].to(dev)
        loader = DataLoader(TensorDataset(Xt[train_idx]), batch_size=args.batch_size, shuffle=True)

        best_val, best_state, patience_ctr = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(dev)
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
                for i in range(0, len(Xv), args.batch_size):
                    xb = Xv[i:i + args.batch_size]
                    out = model(xb)
                    val_loss += float(nn.functional.mse_loss(
                        out["recon"][out["mask"]], xb[out["mask"]]).item()) * len(xb)
            val_loss /= len(Xv)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"epoch {epoch + 1}/{args.epochs} loss {epoch_loss:.4f} "
                      f"val {val_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)

            if val_loss < best_val:
                best_val, patience_ctr = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= args.patience:
                    print(f"early stop at epoch {epoch + 1}", flush=True)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_out)
        print(f"checkpoint -> {args.model_out}", flush=True)

    Z = embed(model, Xt, dev, args.batch_size)
    frame = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(Z.shape[1]):
        frame[f"z{d}"] = Z[:, d].astype("float32")
    Path(args.embeddings_out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.embeddings_out, index=False)
    print(f"{len(frame)} x {Z.shape[1]} -> {args.embeddings_out}", flush=True)


if __name__ == "__main__":
    main()
