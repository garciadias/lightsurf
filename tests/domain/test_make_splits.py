"""Tests for the S3 splits (scripts/make_splits.py)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from click.testing import CliRunner

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS as TARGETS

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location("make_splits", _SCRIPTS / "make_splits.py")
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def _make_data(tmp, n_pool=1200, n_other=800):
    rng = np.random.default_rng(0)
    pool_ids = [f"P{i:05d}" for i in range(n_pool)]
    other_ids = [f"O{i:05d}" for i in range(n_other)]
    aid = pool_ids + other_ids
    teff = np.concatenate([rng.uniform(3000, 4100, n_pool), rng.uniform(4500, 6000, n_other)])
    labels = pd.DataFrame({
        "APOGEE_ID": aid,
        "teff19": teff,
        "logg19": np.concatenate([np.full(n_pool, 4.6), np.full(n_other, 4.5)]),
        "snr19": 100.0,
        "star_bad19": False,
        "is_mdwarf": [True] * n_pool + [False] * n_other,
        "passes_quality": True,
    })
    ft = pd.DataFrame({"APOGEE_ID": pool_ids})
    for E in TARGETS:
        ft[f"{E}_t"] = rng.normal(0, 0.2, n_pool)
        ft[f"{E}_t_ok"] = True
    lp = tmp / "labels.parquet"
    fp = tmp / "finetune_targets.parquet"
    labels.to_parquet(lp)
    ft.to_parquet(fp)
    return lp, fp


def test_splits_disjoint_covering_and_sizes(tmp_path):
    lp, fp = _make_data(tmp_path)
    r = CliRunner().invoke(ms.main, ["--labels", str(lp), "--finetune-targets", str(fp),
                                     "--out-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    sp = pd.read_parquet(tmp_path / "splits.parquet")
    # covering
    assert len(sp) == 2000
    assert set(sp["split"]) <= {"base", "tune", "reservoir", "test"}
    vc = sp["split"].value_counts().to_dict()
    assert vc["test"] == 300
    assert vc["tune"] == 100
    assert vc["reservoir"] == 800
    assert vc["base"] == 800
    # disjoint by construction: each id appears once
    assert sp["APOGEE_ID"].is_unique
    # all non-pool (O*) stars are base
    assert (sp[sp["APOGEE_ID"].str.startswith("O")]["split"] == "base").all()


def test_test_set_is_teff_stratified(tmp_path):
    lp, fp = _make_data(tmp_path)
    CliRunner().invoke(ms.main, ["--labels", str(lp), "--finetune-targets", str(fp),
                                 "--out-dir", str(tmp_path)])
    sp = pd.read_parquet(tmp_path / "splits.parquet")
    labels = pd.read_parquet(lp)
    m = sp.merge(labels, on="APOGEE_ID")
    pool = m[m["is_mdwarf"]]
    test = pool[pool["split"] == "test"]
    # each teff decile of the pool should contribute ~30 test stars (stratified)
    deciles = pd.qcut(pool["teff19"], 10, labels=False)
    pool = pool.assign(dec=deciles.to_numpy())
    test_dec = pool[pool["split"] == "test"]["dec"].value_counts()
    assert test_dec.min() >= 20 and test_dec.max() <= 40
    assert len(test) == 300


def test_reproducible(tmp_path):
    lp, fp = _make_data(tmp_path)
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    out1.mkdir(); out2.mkdir()
    for o in (out1, out2):
        CliRunner().invoke(ms.main, ["--labels", str(lp), "--finetune-targets", str(fp),
                                     "--out-dir", str(o)])
    a = pd.read_parquet(out1 / "splits.parquet").sort_values("APOGEE_ID").reset_index(drop=True)
    b = pd.read_parquet(out2 / "splits.parquet").sort_values("APOGEE_ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
