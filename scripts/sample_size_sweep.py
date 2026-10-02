"""Sample-size sweep (S6): fine-tune / baseline arms vs N, scored on test.

For each N on the grid (capped at the reservoir size) and each seed 0..19, draw
N reservoir stars (rng seeded by (n, seed)), fit every arm, and score it on the
``test`` split per element against the fine-tune targets via EVAL's
``rmse_masked``. Rows are appended to ``sweep.parquet`` after each (n, seed) so a
crash loses at most one run; an existing run (n, seed, arm all present) is
skipped on resume.

Arms: ``base_n0`` (the base head, scored once at n=0), ``ft_full``, ``ft_last``
(fine-tuned copies), ``posthoc`` (EVAL post-hoc calibration) and ``ridge`` (EVAL
ridge on frozen latents). EVAL functions are imported, never reimplemented.

    python scripts/sample_size_sweep.py --data-root $R \
        --latents .../masked_latent_field.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _abundance_data import TARGETS, build_datasets, label_kinds  # noqa: E402

from lightsurf.domain.models.abundance_head import (  # noqa: E402
    finetune_head, load_head, predict,
)
from lightsurf.domain.services.evaluation.sample_size import (  # noqa: E402
    posthoc_calibration, ridge_baseline, rmse_masked,
)

N_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
SEEDS = list(range(20))
SWEEP_COLS = ["n", "seed", "arm", "element", "rmse", "bias", "n_used", "label_kind"]


def _bias_masked(pred, y, mask):
    """Per-element mean(pred - y) over valid rows (NaN if none)."""
    pred = np.asarray(pred, float); y = np.asarray(y, float); mask = np.asarray(mask, bool)
    out = np.full(len(TARGETS), np.nan)
    valid = mask & np.isfinite(y) & np.isfinite(pred)
    for j in range(len(TARGETS)):
        sel = valid[:, j]
        if sel.any():
            out[j] = float(np.mean(pred[sel, j] - y[sel, j]))
    return out


def _rows(n, seed, arm, pred, test, kinds, n_used):
    rmse = rmse_masked(pred, test.Yt, test.Mt)
    bias = _bias_masked(pred, test.Yt, test.Mt)
    return [{"n": n, "seed": seed, "arm": arm, "element": e,
             "rmse": float(rmse[j]), "bias": float(bias[j]),
             "n_used": int(n_used), "label_kind": kinds[e]}
            for j, e in enumerate(TARGETS)]


def _load_existing(path: Path) -> tuple[pd.DataFrame, set]:
    if path.exists():
        df = pd.read_parquet(path)
        done = set(zip(df["n"], df["seed"], df["arm"]))
        return df, done
    return pd.DataFrame(columns=SWEEP_COLS), set()


def _append(path: Path, rows: list[dict]):
    new = pd.DataFrame(rows, columns=SWEEP_COLS)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


def run_sweep(base_head, splits, kinds, config, out_path, *, n_grid=N_GRID,
              seeds=SEEDS, device="cpu", batch_size=128, head_std=None,
              progress=lambda *_: None):
    """Core sweep loop; importable for tests. ``base_head`` is a loaded head,
    ``head_std`` its Standardiser; ``config`` maps arm -> {epochs,lr,weight_decay}."""
    head, std = base_head, head_std
    reservoir, test = splits["reservoir"], splits["test"]
    n_res = len(reservoir)
    ns = [n for n in n_grid if n <= n_res]
    out_path = Path(out_path)
    _, done = _load_existing(out_path)

    # base_n0: score the base head on test once, at n=0.
    if (0, 0, "base_n0") not in done:
        pred0 = predict(head, std, test.Z, device=device)
        _append(out_path, _rows(0, 0, "base_n0", pred0, test, kinds, 0))
        done.add((0, 0, "base_n0"))
        progress("base_n0", 0, 0)

    # precompute base predictions on test (used by posthoc)
    pred_test_base = predict(head, std, test.Z, device=device)

    for n in ns:
        for seed in seeds:
            pending = {a for a in ("ft_full", "ft_last", "posthoc", "ridge")
                       if (n, seed, a) not in done}
            if not pending:
                continue
            rng = np.random.default_rng([n, seed])
            idx = rng.choice(n_res, size=n, replace=False)
            sub = reservoir.subset(idx)
            rows = []
            if "ft_full" in pending:
                cfg = config["full"]
                ft = finetune_head(head, std, sub.Z, sub.Yt, sub.Mt, arm="full",
                                   seed=seed, device=device, batch_size=batch_size, **cfg)
                rows += _rows(n, seed, "ft_full", predict(ft, std, test.Z, device=device),
                              test, kinds, n)
            if "ft_last" in pending:
                cfg = config["last"]
                ft = finetune_head(head, std, sub.Z, sub.Yt, sub.Mt, arm="last",
                                   seed=seed, device=device, batch_size=batch_size, **cfg)
                rows += _rows(n, seed, "ft_last", predict(ft, std, test.Z, device=device),
                              test, kinds, n)
            if "posthoc" in pending:
                pred_fit = predict(head, std, sub.Z, device=device)
                ph = posthoc_calibration(pred_fit, sub.teff19, sub.Yt, sub.Mt,
                                         pred_test_base, test.teff19)
                rows += _rows(n, seed, "posthoc", ph, test, kinds, n)
            if "ridge" in pending:
                rb = ridge_baseline(sub.Z, sub.Yt, sub.Mt, test.Z)
                rows += _rows(n, seed, "ridge", rb, test, kinds, n)
            _append(out_path, rows)
            for a in pending:
                done.add((n, seed, a))
            progress("arms", n, seed)
    return out_path


@click.command()
@click.option("--data-root", required=True, type=click.Path(exists=True))
@click.option("--latents", required=True, type=click.Path(exists=True))
@click.option("--base-head", default=None, help="defaults to $R/base_head.pt")
@click.option("--config", "config_path", default=None, help="defaults to $R/finetune_config.json")
@click.option("--out", "out_path", default=None, help="defaults to $R/sweep.parquet")
@click.option("--batch-size", default=128, show_default=True)
@click.option("--gpu/--no-gpu", default=True, show_default=True)
@click.option("--max-n", default=None, type=int, help="cap the N grid (debug)")
def main(data_root, latents, base_head, config_path, out_path, batch_size, gpu, max_n):
    device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    data_root = Path(data_root)
    base_head = base_head or str(data_root / "base_head.pt")
    config_path = config_path or str(data_root / "finetune_config.json")
    out_path = out_path or str(data_root / "sweep.parquet")

    head, std = load_head(base_head, device=device)
    with open(config_path) as fh:
        config = json.load(fh)
    splits, gate = build_datasets(data_root, latents)
    kinds = label_kinds(gate)

    grid = [n for n in N_GRID if (max_n is None or n <= max_n)]
    click.echo(f"reservoir={len(splits['reservoir'])} test={len(splits['test'])} "
               f"device={device} grid={[n for n in grid if n <= len(splits['reservoir'])]}")

    def prog(stage, n, seed):
        click.echo(f"  {stage} n={n} seed={seed}")

    run_sweep(head, splits, kinds, config, out_path, n_grid=grid, seeds=SEEDS,
              device=device, batch_size=batch_size, head_std=std, progress=prog)
    df = pd.read_parquet(out_path)
    click.echo(f"sweep.parquet: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
