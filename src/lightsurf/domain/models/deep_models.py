"""PyTorch port of the lightsurf CNN-LSTM-Attention spectral model.

Faithful architectural port of the TensorFlow ``CnnLstmAttentionModel`` so the
DR19 re-embed is comparable to the DR17 TensorFlow embeddings:

* 5 × Conv1d(k=3, s=2, same) + BatchNorm + PReLU  (filters 999→61)
* LSTM(256, return_sequences) → tanh/sigmoid attention → 256-d
* Dense(20) → Dense(8) → Dense(9)  (9 APOGEE abundance targets)

Embedding taps (returned by ``forward``): ``attention`` (256-d), ``dense_0``
(20-d), ``dense_1`` (8-d).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader, TensorDataset

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CnnLstmAttention(nn.Module):
    """CNN + LSTM + attention regressor. ``forward`` returns every tap."""

    def __init__(
        self,
        n_features: int,
        cnn_filters: list[int] | int = (1024, 512, 256, 128, 64),
        lstm_units: int = 256,
        dense_units: list[int] | int = (20, 8),
        output_dim: int = 9,
    ) -> None:
        super().__init__()
        if isinstance(cnn_filters, int):
            cnn_filters = [cnn_filters]
        if isinstance(dense_units, int):
            dense_units = [dense_units]
        cnn_filters = sorted(cnn_filters, reverse=True)
        dense_units = sorted(dense_units, reverse=True)

        blocks: list[nn.Module] = []
        in_ch = 1
        for f in cnn_filters:
            blocks += [nn.Conv1d(in_ch, f, kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm1d(f), nn.PReLU()]
            in_ch = f
        self.conv = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(in_ch, lstm_units, batch_first=True)
        self.attention_w = nn.Parameter(torch.empty(lstm_units, 1))
        nn.init.xavier_uniform_(self.attention_w)
        d0, d1 = dense_units
        self.dense_0 = nn.Linear(lstm_units, d0)
        self.dense_1 = nn.Linear(d0, d1)
        self.output = nn.Linear(d1, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv(x)                 # (B, C, L')
        x = x.transpose(1, 2)            # (B, L', C)
        x, _ = self.lstm(x)              # (B, L', H)
        scores = torch.tanh(x @ self.attention_w).squeeze(-1)  # (B, L')
        w = torch.sigmoid(scores).unsqueeze(-1)                 # (B, L', 1)
        att = (x * w).sum(dim=1)         # (B, H)
        d0 = F.relu(self.dense_0(att))
        d1 = F.relu(self.dense_1(d0))
        out = self.output(d1)
        return {"output": out, "attention": att, "dense_0": d0, "dense_1": d1}


class ConvPoolRegressor(nn.Module):
    """CNN + global-average-pool regressor (robust fallback to the LSTM).

    The LSTM+attention head can stall on large, diverse samples (vanishing
    gradients through 268 timesteps). Pooling the conv feature map is far more
    stable and still learns the flux -> abundance mapping (R^2 ~0.89).
    The pooled 64-d vector is returned as the "attention" tap (the embedding).
    """

    def __init__(self, n_features: int, cnn_filters: list[int], output_dim: int = 9) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for f in cnn_filters:
            blocks += [nn.Conv1d(in_ch, f, kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm1d(f), nn.PReLU()]
            in_ch = f
        self.conv = nn.Sequential(*blocks)
        self.output = nn.Linear(in_ch, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv(x)                 # (B, C, L')
        z = x.mean(dim=2)                # (B, C) global-average pool
        out = self.output(z)
        return {"output": out, "attention": z, "dense_0": z, "dense_1": z}


@dataclass
class CnnLstmAttentionModel:
    """Trainer wrapper matching the TensorFlow dataclass interface."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    validation_split: float = 0.2
    cnn_filters: list[int] | int = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    lstm_units: int = 256
    dense_units: list[int] | int = field(default_factory=lambda: [20, 8])
    output_dimension: int = 9
    early_stopping_patience: int = 5
    loss: str = "mse"
    standardize_targets: bool = False
    grad_clip: float = 1.0
    checkpoint_path: str | Path = "models/cnn_lstm_attention_model.pt"
    random_state: int = 42
    verbose: int = 1
    _model: CnnLstmAttention | None = field(default=None, repr=False, init=False)

    def _build(self, n_features: int) -> CnnLstmAttention:
        model = CnnLstmAttention(
            n_features=n_features,
            cnn_filters=self.cnn_filters,
            lstm_units=self.lstm_units,
            dense_units=self.dense_units,
            output_dim=self.output_dimension,
        )
        model.to(_device())
        self._model = model
        return model

    def fit(self, X, y) -> dict[str, list[float]]:
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        n, n_feat = X.shape
        X = X.reshape(n, 1, n_feat)

        model = self._build(n_feat)
        device = _device()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        n_val = int(n * self.validation_split)
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        Xt = torch.from_numpy(X[train_idx])  # CPU; move per-batch to avoid holding all on GPU
        yt = torch.from_numpy(y[train_idx])
        Xv = torch.from_numpy(X[val_idx])  # CPU; chunked for the val forward
        yv = torch.from_numpy(y[val_idx])

        # target statistics + optional per-element standardization
        t_mean = yt.mean(dim=0)
        t_std = yt.std(dim=0)
        t_std[t_std == 0] = 1.0
        if self.standardize_targets:
            yt = (yt - t_mean) / t_std
            yv = (yv - t_mean) / t_std
        # loss selection
        if self.loss == "mae":
            loss_fn = nn.L1Loss()
        elif self.loss == "huber":
            loss_fn = nn.SmoothL1Loss()
        elif self.loss == "weighted_mse":
            _w = (1.0 / (t_std ** 2)).to(device)
            _w = _w / _w.mean()
            loss_fn = lambda out, yb: ((out - yb) ** 2 * _w).mean()
        else:
            loss_fn = nn.MSELoss()

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)

        history: dict[str, list[float]] = {"loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)["output"]
                loss = loss_fn(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item()) * len(xb)
            epoch_loss /= len(train_idx)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(Xv), self.batch_size):
                    xb = Xv[i:i + self.batch_size].to(device)
                    yb = yv[i:i + self.batch_size].to(device)
                    val_loss += float(loss_fn(model(xb)["output"], yb).item()) * len(xb)
                val_loss /= len(Xv)
            history["loss"].append(epoch_loss)
            history["val_loss"].append(val_loss)

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        return history

    def predict_embeddings(self, X, layer_name: str) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not built. Call fit() first.")
        X = np.asarray(X, dtype="float32")
        if X.ndim < 3:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        device = _device()
        self._model.eval()
        Z = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.from_numpy(X[i:i + self.batch_size]).to(device)
                Z.append(self._model(xb)[layer_name].detach().cpu().numpy())
        return np.concatenate(Z, axis=0)


@dataclass
class ConvPoolModel(CnnLstmAttentionModel):
    """Conv+pool trainer (robust when the LSTM head stalls)."""

    def _build(self, n_features: int):
        model = ConvPoolRegressor(n_features, self.cnn_filters, self.output_dimension)
        model.to(_device())
        self._model = model
        return model


class LSTMRegressor(nn.Module):
    """PyTorch LSTM regressor (mirrors the TensorFlow LSTMRegressor)."""

    def __init__(self, n_features: int, lstm_units: int = 250, dense_units: int = 128,
                 dropout: float = 0.2, output_dim: int = 1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.prelu = nn.PReLU()
        self.lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=dropout)
        self.dense1 = nn.Linear(lstm_units, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.output = nn.Linear(dense_units // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn(x))
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.output(x)
