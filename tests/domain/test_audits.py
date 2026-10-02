"""Synthetic-fixture tests for the adversarial audit scripts.

These build small self-consistent artifacts in ``tmp_path`` and assert that:
  * each audit PASSES on a clean fixture, and
  * each audit FAILS (exit non-zero / flags the issue) when the fixture is
    tampered with exactly the defect the audit is meant to catch.

The tamper cases double as the audit-side of the mutation table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import audit_gate  # noqa: E402
import audit_leakage  # noqa: E402
import audit_sweep  # noqa: E402
from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS  # noqa: E402
from lightsurf.domain.services.evaluation.sample_size import (  # noqa: E402
    ridge_baseline,
    rmse_masked,
)

ELEMS = APOGEE_ABUNDANCE_TARGETS
N_ELEM = len(ELEMS)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def make_labels(n=240, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [f"S{i:04d}" for i in range(n)]
    teff = rng.uniform(3200, 4050, n)  # all <= 4100
    logg = rng.uniform(4.2, 5.0, n)    # all >= 4
    df = pd.DataFrame({
        "APOGEE_ID": ids,
        "sdss_id": np.arange(n),
        "teff19": teff, "logg19": logg,
        "snr19": rng.uniform(80, 200, n), "star_bad19": False,
        "snr17": rng.uniform(80, 200, n), "star_bad17": False,
        "teff17": teff + rng.normal(0, 30, n),  # DR17 Teff present (must be ignored)
        "logg17": logg + rng.normal(0, 0.05, n),
    })
    for k, el in enumerate(ELEMS):
        x19 = rng.normal(0, 0.2, n)
        # element 0 has a strong Teff systematic (-> real); others flat (-> synthetic)
        if k == 0:
            delta = 0.12 + 8e-4 * (teff - 3600)
        else:
            delta = rng.normal(0, 0.005, n)
        df[f"{el}_19"] = x19
        df[f"{el}_19_err"] = 0.02
        df[f"{el}_19_ok"] = True
        df[f"{el}_17"] = x19 + delta
        df[f"{el}_17_err"] = 0.02
        df[f"{el}_17_ok"] = True
    df["is_mdwarf"] = (df["teff19"] <= 4100) & (df["logg19"] >= 4)
    df["passes_quality"] = (df["snr19"] >= 70) & (~df["star_bad19"])
    return df


def make_splits(labels: pd.DataFrame, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = labels[labels["is_mdwarf"] & labels["passes_quality"]].copy()
    # stratified test by teff19 deciles
    edges = np.quantile(pool["teff19"], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    pool["dec"] = np.digitize(pool["teff19"], edges[1:-1])
    test_ids = []
    per = 6
    for d, g in pool.groupby("dec"):
        take = min(per, len(g))
        test_ids += list(g.sample(take, random_state=int(d)).APOGEE_ID)
    rest = [i for i in pool.APOGEE_ID if i not in set(test_ids)]
    rng.shuffle(rest)
    tune_ids = rest[:20]
    reservoir_ids = rest[20:]
    assign = {}
    for i in test_ids:
        assign[i] = "test"
    for i in tune_ids:
        assign[i] = "tune"
    for i in reservoir_ids:
        assign[i] = "reservoir"
    for i in labels.APOGEE_ID:
        assign.setdefault(i, "base")
    return pd.DataFrame({"APOGEE_ID": list(assign), "split": list(assign.values())})


def make_finetune(labels: pd.DataFrame) -> pd.DataFrame:
    out = {"APOGEE_ID": labels["APOGEE_ID"].tolist()}
    for el in ELEMS:
        out[f"{el}_t"] = labels[f"{el}_17"].to_numpy()
        out[f"{el}_t_ok"] = labels[f"{el}_17_ok"].to_numpy()
    return pd.DataFrame(out)


def make_gate(labels: pd.DataFrame) -> dict:
    """Build a gate.json using the SAME recipe audit_gate recomputes."""
    elements = {}
    for el in ELEMS:
        r = audit_gate.recompute_element(labels, el, 0.03, 300, 0.95, seed=111)
        elements[el] = {
            "verdict": r["verdict"], "n": r["n"],
            "sys_rms": r["sys_rms"] if np.isfinite(r["sys_rms"]) else None,
            "ci": r["ci"], "resid_rms": r.get("resid_rms"),
            "poly_coef": r.get("poly_coef"),
            "synthetic": None if r["verdict"] == "real" else {"seed": 0},
        }
    return {"thresholds": {"min_sys_rms": 0.03, "n_boot": 1000, "ci": 0.95},
            "elements": elements}


def make_latents(labels: pd.DataFrame, dim=8, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(labels)
    Z = rng.normal(size=(n, dim))
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(dim)])
    df.insert(0, "APOGEE_ID", labels["APOGEE_ID"].to_numpy())
    return df


def make_sweep(labels, splits, latents, finetune, ns=(10, 20), seeds=(0, 1, 2)) -> pd.DataFrame:
    """Build a ridge sweep using default_draw + ridge_baseline, so audit_sweep
    (which falls back to default_draw) recomputes it exactly."""
    zcols = [c for c in latents.columns if c.startswith("z")]
    lat = latents.set_index("APOGEE_ID")
    ft = finetune.set_index("APOGEE_ID")
    reservoir = splits.loc[splits["split"] == "reservoir", "APOGEE_ID"].to_numpy()
    test_ids = splits.loc[splits["split"] == "test", "APOGEE_ID"].to_numpy()

    def targ(ids):
        sub = ft.loc[list(ids)]
        y = np.column_stack([sub[f"{e}_t"].to_numpy(float) for e in ELEMS])
        m = np.column_stack([sub[f"{e}_t_ok"].to_numpy(bool) for e in ELEMS])
        return y, m

    Z_eval = lat.loc[list(test_ids), zcols].to_numpy(float)
    y_eval, m_eval = targ(test_ids)
    rows = []
    # base_n0: a flat baseline rmse per element
    for el in ELEMS:
        rows.append(dict(n=0, seed=0, arm="base_n0", element=el, rmse=0.3,
                         bias=0.0, n_used=0, label_kind="real"))
    for n in ns:
        for s in seeds:
            fit_ids = audit_sweep.default_draw(reservoir, n, s)
            Z_fit = lat.loc[list(fit_ids), zcols].to_numpy(float)
            y_fit, m_fit = targ(fit_ids)
            pred = ridge_baseline(Z_fit, y_fit, m_fit, Z_eval)
            rmse = rmse_masked(pred, y_eval, m_eval)
            for j, el in enumerate(ELEMS):
                rows.append(dict(n=n, seed=s, arm="ridge", element=el,
                                 rmse=float(rmse[j]), bias=0.0, n_used=n,
                                 label_kind="real"))
    return pd.DataFrame(rows)


@pytest.fixture
def artifacts(tmp_path):
    labels = make_labels()
    splits = make_splits(labels)
    finetune = make_finetune(labels)
    latents = make_latents(labels)
    gate = make_gate(labels)
    sweep = make_sweep(labels, splits, latents, finetune)
    labels.to_parquet(tmp_path / "labels.parquet")
    splits.to_parquet(tmp_path / "splits.parquet")
    finetune.to_parquet(tmp_path / "finetune_targets.parquet")
    latents.to_parquet(tmp_path / "latents.parquet")
    sweep.to_parquet(tmp_path / "sweep.parquet")
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    return tmp_path


# --------------------------------------------------------------------------- #
# Leakage audit (AC3)
# --------------------------------------------------------------------------- #
def test_leakage_clean_passes(artifacts):
    assert audit_leakage.main(["--R", str(artifacts)]) == 0


def test_leakage_catches_test_star_in_base(artifacts):
    splits = pd.read_parquet(artifacts / "splits.parquet")
    test_id = splits.loc[splits["split"] == "test", "APOGEE_ID"].iloc[0]
    # leak: also add it as a base row (duplicate -> non-disjoint + base contam)
    splits = pd.concat([splits, pd.DataFrame([{"APOGEE_ID": test_id, "split": "base"}])],
                       ignore_index=True)
    splits.to_parquet(artifacts / "splits.parquet")
    assert audit_leakage.main(["--R", str(artifacts)]) == 1


def test_leakage_catches_non_mdwarf_in_pool(artifacts):
    labels = pd.read_parquet(artifacts / "labels.parquet")
    splits = pd.read_parquet(artifacts / "splits.parquet")
    res_id = splits.loc[splits["split"] == "reservoir", "APOGEE_ID"].iloc[0]
    labels.loc[labels["APOGEE_ID"] == res_id, "is_mdwarf"] = False
    labels.to_parquet(artifacts / "labels.parquet")
    assert audit_leakage.main(["--R", str(artifacts)]) == 1


def test_leakage_catches_coverage_gap(artifacts):
    splits = pd.read_parquet(artifacts / "splits.parquet")
    splits = splits.iloc[5:]  # drop a few labels rows -> uncovered
    splits.to_parquet(artifacts / "splits.parquet")
    assert audit_leakage.main(["--R", str(artifacts)]) == 1


# --------------------------------------------------------------------------- #
# Teff-leak (AC4)
# --------------------------------------------------------------------------- #
def test_teff_leak_static_flags_dr17_token(tmp_path):
    src = tmp_path / "src" / "lightsurf" / "domain" / "models"
    src.mkdir(parents=True)
    (src / "abundance_head.py").write_text(
        "def calib(df):\n    t = df['teff17']  # leak!\n    return t\n"
    )
    res = audit_leakage.teff_leak_static(tmp_path)
    assert res["passed"] is False
    assert any("teff17" in i.lower() for i in res["issues"])


def test_teff_leak_static_clean_passes(tmp_path):
    src = tmp_path / "src" / "lightsurf" / "domain" / "models"
    src.mkdir(parents=True)
    (src / "abundance_head.py").write_text(
        "def calib(df):\n    t = df['teff19']  # DR19 only\n    return t\n"
    )
    res = audit_leakage.teff_leak_static(tmp_path)
    assert res["passed"] is True


def test_teff_leak_runtime_invariant(artifacts):
    labels = pd.read_parquet(artifacts / "labels.parquet")
    res = audit_leakage.teff_leak_runtime(labels)
    assert res["passed"] is True


# --------------------------------------------------------------------------- #
# Gate audit (AC5 / D13a)
# --------------------------------------------------------------------------- #
def test_gate_clean_agrees(artifacts):
    assert audit_gate.main(["--R", str(artifacts), "--n-boot", "200"]) == 0


def test_gate_catches_flipped_verdict(artifacts):
    gate = json.loads((artifacts / "gate.json").read_text())
    el0 = ELEMS[0]
    gate["elements"][el0]["verdict"] = (
        "synthetic" if gate["elements"][el0]["verdict"] == "real" else "real")
    (artifacts / "gate.json").write_text(json.dumps(gate))
    assert audit_gate.main(["--R", str(artifacts), "--n-boot", "200"]) == 1


def test_gate_catches_wrong_sys_rms(artifacts):
    gate = json.loads((artifacts / "gate.json").read_text())
    el0 = ELEMS[0]
    if gate["elements"][el0]["sys_rms"]:
        gate["elements"][el0]["sys_rms"] *= 1.5  # >10% off
        (artifacts / "gate.json").write_text(json.dumps(gate))
        assert audit_gate.main(["--R", str(artifacts), "--n-boot", "200"]) == 1


def test_gate_element0_is_real(artifacts):
    labels = pd.read_parquet(artifacts / "labels.parquet")
    r = audit_gate.recompute_element(labels, ELEMS[0], 0.03, 200, 0.95, seed=7)
    assert r["verdict"] == "real"
    r1 = audit_gate.recompute_element(labels, ELEMS[1], 0.03, 200, 0.95, seed=7)
    assert r1["verdict"] == "synthetic"


# --------------------------------------------------------------------------- #
# Sweep audit (AC7 / AC9)
# --------------------------------------------------------------------------- #
def test_sweep_clean_matches(artifacts):
    rc = audit_sweep.main(["--R", str(artifacts),
                           "--latents", str(artifacts / "latents.parquet"),
                           "--frac", "1.0"])
    assert rc == 0


def test_sweep_catches_corrupted_cell(artifacts):
    sweep = pd.read_parquet(artifacts / "sweep.parquet")
    mask = (sweep["arm"] == "ridge")
    idx = sweep[mask].index[0]
    sweep.loc[idx, "rmse"] = sweep.loc[idx, "rmse"] + 1.0  # corrupt
    sweep.to_parquet(artifacts / "sweep.parquet")
    rc = audit_sweep.main(["--R", str(artifacts),
                           "--latents", str(artifacts / "latents.parquet"),
                           "--frac", "1.0"])
    assert rc == 1
