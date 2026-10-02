"""Train the DR19 base abundance head (S4) and select fine-tune configs (S5).

S4 (``base``): fit the head on ``split=='base'`` latents with DR19 labels
(20% held out for early stopping), save ``base_head.pt`` and ``base_metrics.json``
(per-element RMSE & R^2 vs DR19 on the held-out base stars, and RMSE vs the
fine-tune targets on the test set, which is RMSE(0)).

S5 (``tune-config``): grid-search fine-tune epochs / lr / weight_decay for each
arm. For each config, fine-tune on reservoir subsets at a few N and score on the
``tune`` split against the fine-tune targets; pick per arm the config minimising
the median RMSE across N. Save ``finetune_config.json``.

    python scripts/train_abundance_head.py base \
        --data-root $R --latents .../masked_latent_field.parquet
    python scripts/train_abundance_head.py tune-config \
        --data-root $R --latents .../masked_latent_field.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _abundance_data import TARGETS, build_datasets  # noqa: E402

from lightsurf.domain.models.abundance_head import (  # noqa: E402
    fit_head, finetune_head, predict, save_head, load_head,
)
from lightsurf.domain.services.evaluation.sample_size import rmse_masked  # noqa: E402


def _device(use_gpu: bool) -> str:
    return "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"


def _val_split(n: int, val_frac: float, seed: int):
    """Reproduce fit_head's internal early-stopping split to report metrics on it."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * val_frac)
    return perm[n_val:], perm[:n_val]  # train_idx, val_idx


def _r2(pred, y):
    ss_res = np.sum((pred - y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


@click.group()
def cli():
    pass


@cli.command("base")
@click.option("--data-root", required=True, type=click.Path(exists=True))
@click.option("--latents", required=True, type=click.Path(exists=True))
@click.option("--epochs", default=300, show_default=True)
@click.option("--lr", default=1e-3, show_default=True)
@click.option("--weight-decay", default=1e-4, show_default=True)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--patience", default=15, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--gpu/--no-gpu", default=True, show_default=True)
def base(data_root, latents, epochs, lr, weight_decay, batch_size, patience, seed, gpu):
    device = _device(gpu)
    data_root = Path(data_root)
    splits, _gate = build_datasets(data_root, latents)
    ds = splits["base"]
    click.echo(f"base stars: {len(ds)}  device={device}")

    head, std = fit_head(ds.Z, ds.Y19, ds.M19, epochs=epochs, lr=lr,
                         weight_decay=weight_decay, seed=seed, val_frac=0.2,
                         patience=patience, device=device, batch_size=batch_size)

    # held-out base metrics on the same split fit_head used for early stopping
    _, val_idx = _val_split(len(ds), 0.2, seed)
    vds = ds.subset(val_idx)
    pred_val = predict(head, std, vds.Z, device=device)

    metrics = {"n_base": int(len(ds)), "n_heldout": int(len(val_idx)),
               "elements": {}, "config": {"epochs": epochs, "lr": lr,
               "weight_decay": weight_decay, "batch_size": batch_size,
               "patience": patience, "seed": seed}}
    rmse19 = rmse_masked(pred_val, vds.Y19, vds.M19)
    for j, e in enumerate(TARGETS):
        m = vds.M19[:, j] & np.isfinite(vds.Y19[:, j])
        r2 = _r2(pred_val[m, j], vds.Y19[m, j]) if m.sum() > 1 else float("nan")
        metrics["elements"][e] = {"rmse19_heldout": float(rmse19[j]),
                                  "r2_19_heldout": r2, "n_heldout_valid": int(m.sum())}

    # RMSE(0): base head scored on the test split vs the fine-tune targets
    if "test" in splits:
        tds = splits["test"]
        pred_test = predict(head, std, tds.Z, device=device)
        rmse0 = rmse_masked(pred_test, tds.Yt, tds.Mt)
        for j, e in enumerate(TARGETS):
            metrics["elements"][e]["rmse0_test_vs_target"] = float(rmse0[j])

    save_head(data_root / "base_head.pt", head, std,
              extra={"trained_on": "base", "seed": seed})
    with open(data_root / "base_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    click.echo(f"wrote {data_root/'base_head.pt'} and base_metrics.json")
    for e in TARGETS:
        el = metrics["elements"][e]
        click.echo(f"  {e:6s} R2(DR19)={el['r2_19_heldout']:.3f} "
                   f"rmse19={el['rmse19_heldout']:.3f} "
                   f"rmse0={el.get('rmse0_test_vs_target', float('nan')):.3f}")


@cli.command("tune-config")
@click.option("--data-root", required=True, type=click.Path(exists=True))
@click.option("--latents", required=True, type=click.Path(exists=True))
@click.option("--base-head", default=None, help="defaults to $R/base_head.pt")
@click.option("--epochs-grid", default="100,300", show_default=True)
@click.option("--lr-grid", default="1e-3,3e-3", show_default=True)
@click.option("--wd-grid", default="0.0,1e-4", show_default=True)
@click.option("--tune-seeds", default=3, show_default=True)
@click.option("--batch-size", default=128, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--gpu/--no-gpu", default=True, show_default=True)
def tune_config(data_root, latents, base_head, epochs_grid, lr_grid, wd_grid,
                tune_seeds, batch_size, seed, gpu):
    device = _device(gpu)
    data_root = Path(data_root)
    base_head = base_head or str(data_root / "base_head.pt")
    head, std = load_head(base_head, device=device)
    splits, _gate = build_datasets(data_root, latents)
    reservoir, tune = splits["reservoir"], splits["tune"]
    n_res = len(reservoir)

    epochs_list = [int(x) for x in epochs_grid.split(",")]
    lr_list = [float(x) for x in lr_grid.split(",")]
    wd_list = [float(x) for x in wd_grid.split(",")]
    n_values = [n for n in (20, 100, 500) if n <= n_res]
    if not n_values:
        n_values = [n_res]
    click.echo(f"reservoir={n_res} tune={len(tune)} device={device} "
               f"N={n_values} grid={len(epochs_list)*len(lr_list)*len(wd_list)}")

    grid_report = []
    best = {}
    for arm in ("full", "last"):
        best_score, best_cfg = float("inf"), None
        for ep in epochs_list:
            for lr in lr_list:
                for wd in wd_list:
                    per_n = []
                    for n in n_values:
                        seed_scores = []
                        for s in range(tune_seeds):
                            rng = np.random.default_rng([n, s, 7])
                            idx = rng.choice(n_res, size=n, replace=False)
                            sub = reservoir.subset(idx)
                            ft = finetune_head(head, std, sub.Z, sub.Yt, sub.Mt,
                                               arm=arm, epochs=ep, lr=lr,
                                               weight_decay=wd, seed=s,
                                               device=device, batch_size=batch_size)
                            pred = predict(ft, std, tune.Z, device=device)
                            rmse = rmse_masked(pred, tune.Yt, tune.Mt)
                            seed_scores.append(float(np.nanmedian(rmse)))
                        per_n.append(float(np.median(seed_scores)))
                    score = float(np.median(per_n))
                    cfg = {"epochs": ep, "lr": lr, "weight_decay": wd}
                    grid_report.append({"arm": arm, **cfg, "per_n": per_n,
                                        "median_rmse": score})
                    if score < best_score:
                        best_score, best_cfg = score, cfg
        best[arm] = {**best_cfg, "selected_median_rmse": best_score}
        click.echo(f"  arm={arm} best={best_cfg} median_rmse={best_score:.4f}")

    out = {arm: {"epochs": best[arm]["epochs"], "lr": best[arm]["lr"],
                 "weight_decay": best[arm]["weight_decay"]} for arm in ("full", "last")}
    with open(data_root / "finetune_config.json", "w") as fh:
        json.dump(out, fh, indent=2)
    with open(data_root / "finetune_grid.json", "w") as fh:
        json.dump({"n_values": n_values, "tune_seeds": tune_seeds,
                   "grid": grid_report, "selected": best}, fh, indent=2)
    click.echo(f"wrote {data_root/'finetune_config.json'} and finetune_grid.json")


if __name__ == "__main__":
    cli()
