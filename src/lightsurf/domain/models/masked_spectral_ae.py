"""Masked spectral autoencoder (MAE-style) — self-supervised foundation model.

Masks contiguous wavelength blocks, encodes the *visible* pixels through a
conv stack into a latent ``z``, and reconstructs the masked pixels through
transposed convs. The latent is abundance-free (no ASPCAP labels), addressing
the circularity of the supervised embeddings (a supervised latent cannot carry
more chemical information than the elements it regresses).

The latent ``z`` is the embedding for chemical tagging. Pretrain on the full
DR19 sample, then cluster on ``z`` — no re-training, no labels.

Loss: MSE on the masked pixels only (the model must predict what it cannot see).
"""

from __future__ import annotations

import torch
from torch import nn


def _conv_out_len(length: int, n_convs: int, kernel: int = 3, stride: int = 2, pad: int = 1) -> int:
    for _ in range(n_convs):
        length = (length + 2 * pad - kernel) // stride + 1
    return length


def make_block_mask(
    batch: int, length: int, mask_ratio: float, block_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask: contiguous wavelength blocks, True = masked (hidden)."""
    mask = torch.zeros(batch, 1, length, dtype=torch.bool, device=device)
    n_blocks = max(1, int(length * mask_ratio / block_size))
    for i in range(batch):
        starts = torch.randint(0, max(1, length - block_size), (n_blocks,), generator=generator)
        for s in starts:
            mask[i, 0, s:s + block_size] = True
    return mask


class MaskedSpectralAE(nn.Module):
    def __init__(
        self,
        n_features: int = 8575,
        latent_dim: int = 256,
        cnn_filters: tuple[int, ...] = (1024, 512, 256, 128, 64),
        mask_ratio: float = 0.5,
        block_size: int = 200,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio
        self.block_size = block_size

        # encoder: conv stack 1 -> C channels, L -> L'
        blocks: list[nn.Module] = []
        in_ch = 1
        for f in cnn_filters:
            blocks += [nn.Conv1d(in_ch, f, 3, 2, 1), nn.BatchNorm1d(f), nn.PReLU()]
            in_ch = f
        self.encoder = nn.Sequential(*blocks)
        self._enc_len = _conv_out_len(n_features, len(cnn_filters))
        self.fc = nn.Linear(cnn_filters[-1], latent_dim)

        # decoder: latent -> transposed convs -> spectrum (then crop to n_features)
        self.dec_fc = nn.Linear(latent_dim, cnn_filters[-1] * self._enc_len)
        dec_blocks: list[nn.Module] = []
        in_ch = cnn_filters[-1]
        for f in reversed(cnn_filters):
            dec_blocks += [nn.ConvTranspose1d(in_ch, f, 3, 2, 1, output_padding=1),
                           nn.BatchNorm1d(f), nn.PReLU()]
            in_ch = f
        dec_blocks += [nn.ConvTranspose1d(in_ch, 1, 3, 2, 1, output_padding=1)]
        self.decoder = nn.Sequential(*dec_blocks)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)              # (B, C, L')
        h = h.mean(dim=2)                # (B, C) global-average pool
        return self.fc(h)                # (B, latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mask = make_block_mask(x.shape[0], x.shape[2], self.mask_ratio, self.block_size, device=x.device)
        x_masked = x * (~mask)           # zero the hidden blocks
        z = self.encode(x_masked)        # (B, latent_dim)
        recon = self.dec_fc(z)
        recon = recon.view(x.shape[0], -1, self._enc_len)
        recon = self.decoder(recon)      # (B, 1, L'')
        recon = recon[:, :, : self.n_features]
        return {"latent": z, "recon": recon, "mask": mask}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Latent embedding of the *unmasked* spectrum (downstream use)."""
        return self.encode(x)
