"""Abundance regression head on the frozen MaskedSpectralAE latent.

A small MLP (256 -> 128 -> 9, ReLU between) that predicts the nine
``APOGEE_ABUNDANCE_TARGETS`` from the cached latent ``z``. The encoder is
frozen throughout; this module only ever consumes latents.

Targets are standardised per element with the *base* training mean/std
(``Standardiser``, saved in the checkpoint). ``masked_mse`` averages only over
``mask == True`` entries; a batch with no valid entries yields loss 0 with no
NaN. Fine-tuning starts from a deep copy of the base weights, so the base head
is never mutated, and uses no validation split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS

N_TARGETS = len(APOGEE_ABUNDANCE_TARGETS)


def _device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Standardiser:
    """Per-element target mean/std of the base training set.

    Stored with the head weights so predictions come back in physical units.
    Standardisation ignores NaN/masked entries when computing the statistics.
    """

    mean: np.ndarray  # (9,)
    std: np.ndarray   # (9,)

    @classmethod
    def fit(cls, Y: np.ndarray, M: np.ndarray) -> "Standardiser":
        Y = np.asarray(Y, dtype=np.float64)
        M = np.asarray(M, dtype=bool)
        valid = M & np.isfinite(Y)
        mean = np.zeros(Y.shape[1], dtype=np.float64)
        std = np.ones(Y.shape[1], dtype=np.float64)
        for j in range(Y.shape[1]):
            col = Y[valid[:, j], j]
            if col.size > 0:
                mean[j] = col.mean()
                s = col.std()
                std[j] = s if s > 1e-8 else 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, Y: np.ndarray) -> np.ndarray:
        return (np.asarray(Y, dtype=np.float32) - self.mean) / self.std

    def inverse(self, Yz: np.ndarray) -> np.ndarray:
        return np.asarray(Yz, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardiser":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32))


class AbundanceHead(nn.Module):
    """256 -> 128 -> 9 MLP with a ReLU between the two linear layers."""

    def __init__(self, in_dim: int = 256, hidden: int = 128, out_dim: int = N_TARGETS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(z)))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over ``mask == True`` entries only.

    Masked target values never enter the loss, so they cannot affect it or leak
    gradient into the corresponding output. A batch with zero valid entries
    returns 0 with no NaN (the ``target`` at masked positions is replaced by the
    prediction, giving a zero squared error there; the sum is then divided by a
    count clamped to >= 1).
    """
    mask = mask.to(dtype=torch.bool)
    # neutralise masked (and any NaN) targets so they contribute exactly zero
    safe_target = torch.where(mask, target, pred.detach())
    sq = (pred - safe_target) ** 2
    sq = sq * mask.to(pred.dtype)
    denom = mask.to(pred.dtype).sum().clamp_min(1.0)
    return sq.sum() / denom


def _as_tensors(Z, Y, M, device):
    Zt = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=device)
    Yt = torch.as_tensor(np.asarray(Y, dtype=np.float32), device=device)
    Mt = torch.as_tensor(np.asarray(M), dtype=torch.bool, device=device)
    # mask out non-finite targets as well
    Mt = Mt & torch.isfinite(Yt)
    Yt = torch.nan_to_num(Yt, nan=0.0)
    return Zt, Yt, Mt


def _train_loop(head, Zt, Yt_std, Mt, *, epochs, lr, weight_decay, patience,
                val_idx=None, train_idx=None, seed=0, batch_size=256):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    if train_idx is None:
        train_idx = torch.arange(Zt.shape[0], device=Zt.device)
    best_state = None
    best_val = float("inf")
    patience_ctr = 0
    g = torch.Generator().manual_seed(seed)
    n_tr = len(train_idx)
    for _ in range(epochs):
        head.train()
        order = train_idx[torch.randperm(n_tr, generator=g).to(train_idx.device)]
        for i in range(0, n_tr, batch_size):
            bidx = order[i:i + batch_size]
            pred = head(Zt[bidx])
            loss = masked_mse(pred, Yt_std[bidx], Mt[bidx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

        if val_idx is not None and len(val_idx) > 0:
            head.eval()
            with torch.no_grad():
                vpred = head(Zt[val_idx])
                vloss = float(masked_mse(vpred, Yt_std[val_idx], Mt[val_idx]).item())
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def fit_head(Z, Y, M, *, epochs, lr, weight_decay, seed, val_frac=0.2,
             patience=15, device=None, batch_size=256) -> tuple[AbundanceHead, Standardiser]:
    """Train a fresh head on standardised targets with best-state early stopping.

    ``val_frac`` of the rows are held out (seeded) for the early-stopping metric.
    Minibatched (``batch_size``) so the loss gets many gradient steps per epoch.
    """
    device = _device(device)
    torch.manual_seed(seed)  # seed BEFORE head init so weights are reproducible
    std = Standardiser.fit(Y, M)
    Zt, Yt, Mt = _as_tensors(Z, Y, M, device)
    Yt_std = torch.as_tensor(std.transform(np.asarray(Y, dtype=np.float32)),
                             device=device, dtype=torch.float32)
    Yt_std = torch.nan_to_num(Yt_std, nan=0.0)

    n = Zt.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val].to(device)
    train_idx = perm[n_val:].to(device)
    if len(train_idx) == 0:  # degenerate tiny inputs: train on all
        train_idx, val_idx = perm.to(device), None

    head = AbundanceHead(in_dim=Zt.shape[1]).to(device)
    head = _train_loop(head, Zt, Yt_std, Mt, epochs=epochs, lr=lr,
                       weight_decay=weight_decay, patience=patience,
                       val_idx=val_idx, train_idx=train_idx, seed=seed,
                       batch_size=batch_size)
    return head, std


def finetune_head(head: AbundanceHead, std: Standardiser, Z, Y, M, *,
                  arm: str, epochs, lr, weight_decay, seed, device=None,
                  batch_size=256) -> AbundanceHead:
    """Fine-tune a deep copy of ``head`` (base head left untouched). No val split.

    ``arm == 'full'`` trains every head parameter; ``arm == 'last'`` trains only
    the final Linear (fc2) and freezes fc1. Targets are standardised with the
    base ``std`` (never re-fit). Minibatched; for tiny N this is effectively
    full-batch.
    """
    if arm not in ("full", "last"):
        raise ValueError(f"arm must be 'full' or 'last', got {arm!r}")
    device = _device(device)
    ft = copy.deepcopy(head).to(device)

    if arm == "last":
        for p in ft.fc1.parameters():
            p.requires_grad_(False)

    Zt, Yt, Mt = _as_tensors(Z, Y, M, device)
    Yt_std = torch.as_tensor(std.transform(np.asarray(Y, dtype=np.float32)),
                             device=device, dtype=torch.float32)
    Yt_std = torch.nan_to_num(Yt_std, nan=0.0)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    trainable = [p for p in ft.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    n = Zt.shape[0]
    for _ in range(epochs):
        ft.train()
        order = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            bidx = order[i:i + batch_size]
            pred = ft(Zt[bidx])
            loss = masked_mse(pred, Yt_std[bidx], Mt[bidx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    return ft


@torch.no_grad()
def predict(head: AbundanceHead, std: Standardiser, Z, device=None) -> np.ndarray:
    """Predict abundances in physical units (undoes the standardisation)."""
    device = _device(device)
    head = head.to(device)
    head.eval()
    Zt = torch.as_tensor(np.asarray(Z, dtype=np.float32), device=device)
    out_std = head(Zt).cpu().numpy()
    return std.inverse(out_std)


def save_head(path, head: AbundanceHead, std: Standardiser, extra: dict | None = None) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "standardiser": std.to_dict(),
        "in_dim": head.fc1.in_features,
        "hidden": head.fc1.out_features,
        "out_dim": head.fc2.out_features,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_head(path, device=None) -> tuple[AbundanceHead, Standardiser]:
    device = _device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    head = AbundanceHead(in_dim=payload["in_dim"], hidden=payload["hidden"],
                         out_dim=payload["out_dim"]).to(device)
    head.load_state_dict(payload["state_dict"])
    std = Standardiser.from_dict(payload["standardiser"])
    return head, std
