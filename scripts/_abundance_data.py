"""Shared data loading for the abundance-head scripts (S4-S7).

Reads the DATA-workstream artifacts (``labels.parquet``, ``splits.parquet``,
``finetune_targets.parquet``, ``gate.json``) and the read-only latents parquet,
and assembles the numpy matrices the head consumes. Kept separate from the
library module (which only knows about latents + targets) so the CLIs share one
schema-aware loader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS

TARGETS = list(APOGEE_ABUNDANCE_TARGETS)
Z_COLS = [f"z{i}" for i in range(256)]


def load_latents(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in Z_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"latents file missing columns: {missing[:3]}... ({len(missing)})")
    return df[["APOGEE_ID", *Z_COLS]]


def load_gate(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def label_kinds(gate: dict) -> dict[str, str]:
    """element -> 'real'|'synthetic' from gate.json."""
    return {e: gate["elements"][e]["verdict"] for e in TARGETS}


def _z_matrix(ids: pd.Series, latents: pd.DataFrame) -> np.ndarray:
    merged = ids.to_frame("APOGEE_ID").merge(latents, on="APOGEE_ID", how="left")
    Z = merged[Z_COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(Z).all():
        raise ValueError("some requested APOGEE_IDs have no latent row")
    return Z


class Dataset:
    """Assembled, index-aligned view of one split population.

    Attributes are all aligned by row:
      ids (n,), Z (n,256) f32, teff19 (n,) f32,
      Y19/M19 (n,9) DR19 labels+mask, Yt/Mt (n,9) target labels+mask.
    """

    def __init__(self, ids, Z, teff19, Y19, M19, Yt, Mt):
        self.ids = ids
        self.Z = Z
        self.teff19 = teff19
        self.Y19 = Y19
        self.M19 = M19
        self.Yt = Yt
        self.Mt = Mt

    def __len__(self):
        return len(self.ids)

    def subset(self, idx) -> "Dataset":
        return Dataset(self.ids[idx], self.Z[idx], self.teff19[idx],
                       self.Y19[idx], self.M19[idx], self.Yt[idx], self.Mt[idx])


def build_datasets(data_root: str | Path, latents_path: str | Path):
    """Return (splits_dict, gate) where splits_dict maps split name -> Dataset.

    Base split uses DR19 labels ({E}_19, mask {E}_19_ok & passes_quality).
    tune/reservoir/test carry the fine-tune targets ({E}_t, {E}_t_ok) too.
    """
    data_root = Path(data_root)
    labels = pd.read_parquet(data_root / "labels.parquet")
    splits = pd.read_parquet(data_root / "splits.parquet")
    ft = pd.read_parquet(data_root / "finetune_targets.parquet")
    gate = load_gate(data_root / "gate.json")
    latents = load_latents(latents_path)

    df = splits.merge(labels, on="APOGEE_ID", how="left")
    df = df.merge(ft, on="APOGEE_ID", how="left")

    out: dict[str, Dataset] = {}
    for split in ["base", "tune", "reservoir", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if sub.empty:
            continue
        ids = sub["APOGEE_ID"].reset_index(drop=True)
        Z = _z_matrix(ids, latents)
        teff19 = sub["teff19"].to_numpy(dtype=np.float32)
        passes = sub["passes_quality"].to_numpy(dtype=bool)

        Y19 = np.full((len(sub), len(TARGETS)), np.nan, dtype=np.float32)
        M19 = np.zeros((len(sub), len(TARGETS)), dtype=bool)
        Yt = np.full((len(sub), len(TARGETS)), np.nan, dtype=np.float32)
        Mt = np.zeros((len(sub), len(TARGETS)), dtype=bool)
        for j, e in enumerate(TARGETS):
            Y19[:, j] = sub[f"{e}_19"].to_numpy(dtype=np.float32)
            M19[:, j] = sub[f"{e}_19_ok"].to_numpy(dtype=bool) & passes
            if f"{e}_t" in sub.columns:
                Yt[:, j] = sub[f"{e}_t"].to_numpy(dtype=np.float32)
                Mt[:, j] = sub[f"{e}_t_ok"].fillna(False).to_numpy(dtype=bool)
        out[split] = Dataset(ids.to_numpy(), Z, teff19, Y19, M19, Yt, Mt)
    return out, gate
