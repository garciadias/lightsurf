"""Phase B: disentangled conditional autoencoder for abundance-free tagging.

PyTorch port of the de Mijolla, Ness, Viti & Wheeler (2021, arXiv:2103.06377)
idea in a minimal, CPU-testable form:

- encoder  ``z = enc(spectrum)`` — CNN + LSTM → chemical latent,
- decoder  ``x̂ = dec(z, Teff/logg)`` — reconstructs the spectrum,
- disentanglement head — gradient-reversal makes ``z`` uninformative about
  Teff/logg.

loss = MSE(spectrum, x̂)  +  λ · MSE(Teff/logg, û_from_reversed_latent)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradientReversal(torch.autograd.Function):
    """Reverses gradients (scale multiplies them) — the adversarial term."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


class DisentangledAE(nn.Module):
    """Encoder/decoder + disentanglement head."""

    def __init__(self, input_dim: int, latent_dim: int, n_conditions: int,
                 scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1), nn.PReLU(),
            nn.Conv1d(32, 16, 3, stride=2, padding=1), nn.PReLU(),
        )
        self.lstm = nn.LSTM(16, latent_dim, batch_first=True)
        self.latent = nn.Linear(latent_dim, latent_dim)

        self.dec1 = nn.Linear(latent_dim + n_conditions, 64)
        self.dec2 = nn.Linear(64, 128)
        self.dec_out = nn.Linear(128, input_dim)

        self.u1 = nn.Linear(latent_dim, 16)
        self.u_out = nn.Linear(16, n_conditions)

    def encode_only(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = self.encoder(spectrum)          # (B, 16, L')
        x = x.transpose(1, 2)               # (B, L', 16)
        x, _ = self.lstm(x)                 # (B, L', latent_dim)
        return self.latent(x[:, -1, :])     # (B, latent_dim)

    def forward(self, spectrum: torch.Tensor, conditions: torch.Tensor):
        z = self.encode_only(spectrum)
        dec_in = torch.cat([z, conditions], dim=1)
        y = F.relu(self.dec2(F.relu(self.dec1(dec_in))))
        recon = self.dec_out(y).unsqueeze(1)          # (B, 1, input_dim)

        rev = GradientReversal.apply(z, self.scale)
        u_hat = self.u_out(F.relu(self.u1(rev)))      # (B, n_conditions)
        return recon, u_hat, z


@dataclass
class DisentangledSpectralAE:
    latent_dim: int = 16
    n_conditions: int = 2
    disentanglement_lambda: float = 0.1
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    verbose: int = 1
    checkpoint_path: str = "models/disentangled_ae.pt"
    _model: DisentangledAE | None = None

    def _build(self, input_dim: int) -> DisentangledAE:
        model = DisentangledAE(
            input_dim, self.latent_dim, self.n_conditions,
            scale=self.disentanglement_lambda,
        ).to(_device())
        self._model = model
        return model

    def fit(self, X: np.ndarray, U: np.ndarray, validation_split: float = 0.2) -> dict[str, list[float]]:
        X = np.asarray(X, dtype="float32")
        U = np.asarray(U, dtype="float32")
        n, input_dim = X.shape
        model = self._build(input_dim)
        device = _device()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        mse = nn.MSELoss()

        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        n_val = int(n * validation_split)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        Xs = X.reshape(n, 1, input_dim)
        Xt = torch.from_numpy(Xs[train_idx]).to(device)
        Ut = torch.from_numpy(U[train_idx]).to(device)
        Xv = torch.from_numpy(Xs[val_idx]).to(device)
        Uv = torch.from_numpy(U[val_idx]).to(device)
        recon_t = torch.from_numpy(Xs[train_idx]).to(device)
        recon_v = torch.from_numpy(Xs[val_idx]).to(device)

        loader = DataLoader(TensorDataset(Xt, Ut, recon_t), batch_size=self.batch_size, shuffle=True)
        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, ub, rb in loader:
                optimizer.zero_grad()
                recon, u_hat, _ = model(xb, ub)
                loss = mse(recon, rb) + mse(u_hat, ub)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(xb)
            epoch_loss /= len(train_idx)
            model.eval()
            with torch.no_grad():
                rv, uv, _ = model(Xv, Uv)
                val_loss = float(mse(rv, recon_v).item() + mse(uv, Uv).item())
            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        return history

    def encode(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not built. Call fit() first.")
        X = np.asarray(X, dtype="float32")
        if X.ndim < 3:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        device = _device()
        self._model.eval()
        with torch.no_grad():
            z = self._model.encode_only(torch.from_numpy(X).to(device))
        return z.detach().cpu().numpy()
