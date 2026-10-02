"""Tests for the S2 preflight gate (scripts/abundance_gate.py)."""
import importlib.util
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_spec = importlib.util.spec_from_file_location("abundance_gate", _SCRIPTS / "abundance_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_gate_passes_on_injected_teff_trend():
    rng = np.random.default_rng(1)
    n = 500
    teff = rng.uniform(3000, 4100, n)
    t = teff - np.median(teff)
    # strong, mean-zero Teff trend -> systematic the permutation null destroys
    delta = 0.15 * (t / 600.0) + rng.normal(0, 0.01, n)
    info = gate.gate_element(t, delta, np.random.default_rng(0))
    assert info["sys_rms"] >= gate.MIN_SYS_RMS
    assert info["sys_rms"] > info["null_sys_rms_p95"]
    assert info["verdict"] == "real"


def test_gate_fails_on_pure_noise():
    rng = np.random.default_rng(2)
    n = 500
    teff = rng.uniform(3000, 4100, n)
    t = teff - np.median(teff)
    delta = rng.normal(0, 0.008, n)  # no structure, tiny scatter
    info = gate.gate_element(t, delta, np.random.default_rng(0))
    assert info["sys_rms"] < gate.MIN_SYS_RMS
    assert info["verdict"] == "synthetic"


def test_gate_passes_on_constant_offset_via_mean_ci():
    rng = np.random.default_rng(3)
    n = 500
    teff = rng.uniform(3000, 4100, n)
    t = teff - np.median(teff)
    delta = 0.1 + rng.normal(0, 0.01, n)  # pure offset, no trend
    info = gate.gate_element(t, delta, np.random.default_rng(0))
    assert info["sys_rms"] >= gate.MIN_SYS_RMS
    assert info["offset_excludes_0"]
    assert info["verdict"] == "real"


def test_make_synthetic_matches_target_sys_rms():
    rng = np.random.default_rng(4)
    n = 400
    t = rng.uniform(3000, 4100, n) - 3500.0
    y19 = rng.normal(0, 0.2, n)
    err = np.full(n, 0.03)
    target = 0.05
    _, params = gate.make_synthetic(t, y19, err, err, target, seed=7)
    corr = params["a"] + params["b"] * t + params["c"] * t**2
    assert np.isclose(np.sqrt(np.mean(corr**2)), target, rtol=1e-6)
    assert params["sigma"] > 0
