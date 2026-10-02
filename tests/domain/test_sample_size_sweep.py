"""Tests for the sample-size sweep (S6): resume logic and schema (AC7, AC9).

Builds tiny synthetic parquet fixtures matching the DATA contract schemas,
trains a quick base head, and exercises ``run_sweep``: appended rows, resume
skips completed (n, seed, arm), N above the reservoir is absent (not padded),
and a rerun reproduces the same rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _abundance_data import TARGETS, build_datasets, label_kinds  # noqa: E402
from lightsurf.domain.models.abundance_head import (  # noqa: E402
    draw_reservoir_subset, fit_head,
)
import sample_size_sweep as sw  # noqa: E402


def _make_fixtures(root: Path, n_stars=140):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    ids = [f"S{i:04d}" for i in range(n_stars)]
    Z = rng.normal(size=(n_stars, 256)).astype("float32")
    lat = pd.DataFrame({"APOGEE_ID": ids})
    for d in range(256):
        lat[f"z{d}"] = Z[:, d]
    lat.to_parquet(root / "latents.parquet", index=False)

    W = (rng.normal(size=(256, len(TARGETS))) * 0.1).astype("float32")
    y19 = Z @ W + rng.normal(scale=0.03, size=(n_stars, len(TARGETS)))
    teff = rng.uniform(3200, 4100, size=n_stars)
    shift = (0.4 + 0.1 * (teff - 3650) / 500)[:, None]
    y17 = y19 + shift

    lab = {"APOGEE_ID": ids, "sdss_id": np.arange(n_stars),
           "teff19": teff.astype("float32"), "logg19": np.full(n_stars, 4.6, "float32"),
           "snr19": np.full(n_stars, 100.0, "float32"), "star_bad19": np.zeros(n_stars, bool),
           "snr17": np.full(n_stars, 100.0, "float32"), "star_bad17": np.zeros(n_stars, bool),
           "is_mdwarf": np.ones(n_stars, bool), "passes_quality": np.ones(n_stars, bool)}
    ftt = {"APOGEE_ID": ids}
    for j, e in enumerate(TARGETS):
        lab[f"{e}_19"] = y19[:, j].astype("float32")
        lab[f"{e}_19_err"] = np.full(n_stars, 0.02, "float32")
        lab[f"{e}_19_ok"] = np.ones(n_stars, bool)
        lab[f"{e}_17"] = y17[:, j].astype("float32")
        lab[f"{e}_17_err"] = np.full(n_stars, 0.02, "float32")
        lab[f"{e}_17_ok"] = np.ones(n_stars, bool)
        ftt[f"{e}_t"] = y17[:, j].astype("float32")
        ftt[f"{e}_t_ok"] = np.ones(n_stars, bool)
    pd.DataFrame(lab).to_parquet(root / "labels.parquet", index=False)
    pd.DataFrame(ftt).to_parquet(root / "finetune_targets.parquet", index=False)

    # splits: base 60, tune 20, test 30, reservoir 30
    split = (["base"] * 60 + ["tune"] * 20 + ["test"] * 30 + ["reservoir"] * 30)
    pd.DataFrame({"APOGEE_ID": ids, "split": split}).to_parquet(
        root / "splits.parquet", index=False)

    gate = {"thresholds": {"min_sys_rms": 0.03, "n_boot": 1000, "ci": 0.95},
            "elements": {e: {"verdict": ("synthetic" if j == 2 else "real"),
                             "n": 50, "sys_rms": 0.1, "ci": [0.05, 0.2],
                             "poly_coef": [0, 0, 0], "resid_rms": 0.05,
                             "synthetic": None} for j, e in enumerate(TARGETS)}}
    (root / "gate.json").write_text(__import__("json").dumps(gate))


@pytest.fixture()
def fixtures(tmp_path):
    _make_fixtures(tmp_path)
    splits, gate = build_datasets(tmp_path, tmp_path / "latents.parquet")
    base, std = fit_head(splits["base"].Z, splits["base"].Y19, splits["base"].M19,
                         epochs=40, lr=1e-3, weight_decay=1e-4, seed=0,
                         device="cpu", batch_size=32)
    cfg = {"full": {"epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
           "last": {"epochs": 20, "lr": 1e-3, "weight_decay": 0.0}}
    return tmp_path, splits, label_kinds(gate), base, std, cfg


def test_sweep_schema_and_arms(fixtures):
    root, splits, kinds, base, std, cfg = fixtures
    out = root / "sweep.parquet"
    sw.run_sweep(base, splits, kinds, cfg, out, n_grid=[10, 20], seeds=[0, 1],
                 device="cpu", batch_size=32, head_std=std)
    df = pd.read_parquet(out)
    assert list(df.columns) == sw.SWEEP_COLS
    assert set(df.arm.unique()) == {"base_n0", "ft_full", "ft_last", "posthoc", "ridge"}
    # base_n0 scored once (9 element rows)
    assert len(df[df.arm == "base_n0"]) == len(TARGETS)
    # synthetic flag propagated (element index 2 = CA_FE)
    assert (df[df.element == TARGETS[2]]["label_kind"] == "synthetic").all()


def test_sweep_absent_above_reservoir(fixtures):
    """AC7: N cells that exceed the reservoir are absent, not padded."""
    root, splits, kinds, base, std, cfg = fixtures
    out = root / "sweep.parquet"
    # reservoir is 30; 50 and 100 must not appear
    sw.run_sweep(base, splits, kinds, cfg, out, n_grid=[10, 50, 100], seeds=[0],
                 device="cpu", batch_size=32, head_std=std)
    df = pd.read_parquet(out)
    assert set(df[df.arm != "base_n0"]["n"].unique()) == {10}


def test_sweep_resume_skips_and_reproduces(fixtures):
    root, splits, kinds, base, std, cfg = fixtures
    out = root / "sweep.parquet"
    sw.run_sweep(base, splits, kinds, cfg, out, n_grid=[10, 20], seeds=[0, 1],
                 device="cpu", batch_size=32, head_std=std)
    df1 = pd.read_parquet(out).sort_values(sw.SWEEP_COLS).reset_index(drop=True)

    # drop one (n, seed) block of arms, simulate a crash mid-run
    mask = ~((df1.n == 20) & (df1.seed == 1) & (df1.arm.isin(["ridge", "posthoc"])))
    df1[mask].to_parquet(out, index=False)
    before = len(df1[mask])

    sw.run_sweep(base, splits, kinds, cfg, out, n_grid=[10, 20], seeds=[0, 1],
                 device="cpu", batch_size=32, head_std=std)
    df2 = pd.read_parquet(out).sort_values(sw.SWEEP_COLS).reset_index(drop=True)
    # refilled exactly the removed rows, no duplicates
    assert len(df2) == len(df1)
    assert len(df2) > before
    keys = ["n", "seed", "arm", "element"]
    assert not df2.duplicated(keys).any()
    # ft arms reproduce to float tolerance (determinism, AC9)
    merged = df1.merge(df2, on=keys, suffixes=("_a", "_b"))
    assert np.allclose(merged.rmse_a, merged.rmse_b, atol=1e-5, equal_nan=True)


def test_draw_reservoir_subset_matches_sweep_rng_contract():
    """F5: the draw is a pure function replicating the sweep's seeded rule."""
    ids = np.array([f"S{i:04d}" for i in range(120)])
    for n, seed in [(10, 0), (20, 7), (120, 3), (1, 19)]:
        got_a = draw_reservoir_subset(ids, n, seed)
        got_b = draw_reservoir_subset(ids, n, seed)
        np.testing.assert_array_equal(got_a, got_b)  # pure / deterministic
        assert len(set(got_a)) == n  # without replacement
        # matches the historical inline draw exactly
        rng = np.random.default_rng([n, seed])
        want = ids[rng.choice(len(ids), size=n, replace=False)]
        np.testing.assert_array_equal(got_a, want)
    with pytest.raises(ValueError):
        draw_reservoir_subset(ids, 121, 0)
