"""Hand-computed tests for the EVAL sample-size library.

Run only these (the repo has unrelated pre-existing failures)::

    PYTHONPATH=src .venv/bin/python -m pytest -q --no-cov \
        tests/domain/test_sample_size.py tests/domain/test_audits.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS
from lightsurf.domain.services.evaluation.sample_size import (
    gap_closed,
    minimum_n,
    posthoc_calibration,
    ridge_baseline,
    rmse_masked,
)

N_ELEM = len(APOGEE_ABUNDANCE_TARGETS)


def _zeros(n):
    return np.zeros((n, N_ELEM))


# --------------------------------------------------------------------------- #
# rmse_masked
# --------------------------------------------------------------------------- #
def test_rmse_masked_hand_value():
    pred = _zeros(3)
    y = _zeros(3)
    mask = np.zeros((3, N_ELEM), dtype=bool)
    # element 0: two valid rows with errors 3 and 4 -> RMSE = sqrt(12.5)
    pred[0, 0], y[0, 0], mask[0, 0] = 3.0, 0.0, True
    pred[1, 0], y[1, 0], mask[1, 0] = 0.0, 4.0, True
    pred[2, 0], y[2, 0], mask[2, 0] = 9.0, 0.0, False  # masked -> ignored
    out = rmse_masked(pred, y, mask)
    assert out[0] == pytest.approx(np.sqrt(12.5))
    # every other element has no valid row -> NaN (never 0)
    assert np.isnan(out[1:]).all()


def test_rmse_masked_all_masked_is_nan_not_zero():
    pred = _zeros(4)
    y = _zeros(4)
    mask = np.zeros((4, N_ELEM), dtype=bool)
    out = rmse_masked(pred, y, mask)
    assert np.isnan(out).all()


def test_rmse_masked_single_row():
    pred = _zeros(1)
    y = _zeros(1)
    mask = np.zeros((1, N_ELEM), dtype=bool)
    pred[0, 2], y[0, 2], mask[0, 2] = 5.0, 2.0, True
    out = rmse_masked(pred, y, mask)
    assert out[2] == pytest.approx(3.0)


def test_rmse_masked_nan_target_on_valid_row_treated_as_masked():
    # Documented choice: (mask True, target NaN) is dropped, not raised.
    pred = _zeros(2)
    y = _zeros(2)
    mask = np.zeros((2, N_ELEM), dtype=bool)
    pred[0, 0], y[0, 0], mask[0, 0] = 2.0, np.nan, True  # dropped
    pred[1, 0], y[1, 0], mask[1, 0] = 2.0, 0.0, True  # kept, err 2
    out = rmse_masked(pred, y, mask)
    assert out[0] == pytest.approx(2.0)


def test_rmse_masked_nan_pred_propagates():
    pred = _zeros(1)
    y = _zeros(1)
    mask = np.zeros((1, N_ELEM), dtype=bool)
    pred[0, 0], y[0, 0], mask[0, 0] = np.nan, 1.0, True
    out = rmse_masked(pred, y, mask)
    assert np.isnan(out[0])


def test_rmse_masked_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse_masked(np.zeros((3, N_ELEM)), np.zeros((2, N_ELEM)),
                    np.zeros((3, N_ELEM), dtype=bool))


# --------------------------------------------------------------------------- #
# gap_closed
# --------------------------------------------------------------------------- #
def test_gap_closed_hand_value():
    assert gap_closed(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert gap_closed(0.2, 1.0, 0.2) == pytest.approx(1.0)


def test_gap_closed_zero_denominator_is_nan():
    assert np.isnan(gap_closed(0.5, 1.0, 1.0))


def test_gap_closed_negative_denominator_is_nan():
    # n_max RMSE worse than base -> denom < 0 -> NaN.
    assert np.isnan(gap_closed(0.4, 0.5, 1.0))


def test_gap_closed_tiny_denominator_is_nan():
    assert np.isnan(gap_closed(0.5, 1.0, 1.0 - 1e-15))


def test_gap_closed_nan_rmse_n_is_nan():
    assert np.isnan(gap_closed(np.nan, 1.0, 0.0))


# --------------------------------------------------------------------------- #
# minimum_n  (constructed df with a known answer)
# --------------------------------------------------------------------------- #
def _sweep_df(element, arm_rows, base_rmse=1.0):
    """Build a sweep-schema df. arm_rows: list of (n, seed, rmse) for `arm`."""
    rows = []
    rows.append(dict(n=0, seed=0, arm="base_n0", element=element,
                     rmse=base_rmse, bias=0.0, n_used=0, label_kind="real"))
    for n, seed, rmse in arm_rows:
        rows.append(dict(n=n, seed=seed, arm="posthoc", element=element,
                         rmse=rmse, bias=0.0, n_used=n, label_kind="real"))
    return pd.DataFrame(rows)


def test_minimum_n_smallest_passing():
    # rmse_0=1.0; n_max=20 median rmse=0.2 -> denom=0.8.
    # n=10 seeds rmse [0.2,0.2,0.2,0.2,0.9] -> G>=0.9 for 4/5 = 0.8 >= 0.8 PASS
    el = APOGEE_ABUNDANCE_TARGETS[0]
    arm_rows = [
        (10, 0, 0.2), (10, 1, 0.2), (10, 2, 0.2), (10, 3, 0.2), (10, 4, 0.9),
        (20, 0, 0.2), (20, 1, 0.2), (20, 2, 0.2), (20, 3, 0.2), (20, 4, 0.2),
    ]
    df = _sweep_df(el, arm_rows)
    assert minimum_n(df, el, "posthoc") == 10


def test_minimum_n_skips_failing_small_n():
    # n=10 only 3/5 pass -> fail; n=20 all pass -> answer 20.
    el = APOGEE_ABUNDANCE_TARGETS[0]
    arm_rows = [
        (10, 0, 0.2), (10, 1, 0.2), (10, 2, 0.2), (10, 3, 0.9), (10, 4, 0.9),
        (20, 0, 0.2), (20, 1, 0.2), (20, 2, 0.2), (20, 3, 0.2), (20, 4, 0.2),
    ]
    df = _sweep_df(el, arm_rows)
    assert minimum_n(df, el, "posthoc") == 20


def test_minimum_n_none_when_never_reached():
    # Even n_max fraction below frac_seeds (spread too wide) -> None.
    el = APOGEE_ABUNDANCE_TARGETS[0]
    arm_rows = [
        (10, 0, 0.2), (10, 1, 0.2), (10, 2, 0.2), (10, 3, 0.9), (10, 4, 0.9),
        (20, 0, 0.2), (20, 1, 0.2), (20, 2, 0.2), (20, 3, 0.9), (20, 4, 0.9),
    ]
    df = _sweep_df(el, arm_rows)
    assert minimum_n(df, el, "posthoc") is None


def test_minimum_n_no_arm_rows_returns_none():
    el = APOGEE_ABUNDANCE_TARGETS[0]
    df = _sweep_df(el, [])  # only base_n0
    assert minimum_n(df, el, "posthoc") is None


def test_minimum_n_missing_base_raises():
    el = APOGEE_ABUNDANCE_TARGETS[0]
    df = pd.DataFrame([
        dict(n=10, seed=0, arm="posthoc", element=el, rmse=0.2,
             bias=0.0, n_used=10, label_kind="real"),
    ])
    with pytest.raises(ValueError):
        minimum_n(df, el, "posthoc")


# --------------------------------------------------------------------------- #
# posthoc_calibration
# --------------------------------------------------------------------------- #
def test_posthoc_exact_linear_recovery():
    # y = 2 + 3*p - 1*t on element 0; >=6 rows -> full fit recovers exactly.
    rng = np.random.default_rng(0)
    n_fit = 20
    p = rng.normal(size=n_fit)
    t = rng.normal(size=n_fit)
    pred_fit = np.zeros((n_fit, N_ELEM))
    y_fit = np.zeros((n_fit, N_ELEM))
    mask_fit = np.zeros((n_fit, N_ELEM), dtype=bool)
    pred_fit[:, 0] = p
    y_fit[:, 0] = 2.0 + 3.0 * p - 1.0 * t
    mask_fit[:, 0] = True

    n_eval = 5
    pe = rng.normal(size=n_eval)
    te = rng.normal(size=n_eval)
    pred_eval = np.zeros((n_eval, N_ELEM))
    pred_eval[:, 0] = pe
    out = posthoc_calibration(pred_fit, t, y_fit, mask_fit, pred_eval, te)
    expected = 2.0 + 3.0 * pe - 1.0 * te
    assert np.allclose(out[:, 0], expected, atol=1e-8)
    # untouched element has no valid fit rows -> NaN predictions
    assert np.isnan(out[:, 1]).all()


def test_posthoc_offset_only_fallback():
    # n_valid < 3 -> predict mean(y_fit).
    n_fit = 2
    pred_fit = np.zeros((n_fit, N_ELEM))
    y_fit = np.zeros((n_fit, N_ELEM))
    mask_fit = np.zeros((n_fit, N_ELEM), dtype=bool)
    y_fit[:, 0] = [4.0, 6.0]
    mask_fit[:, 0] = True
    pred_eval = np.zeros((3, N_ELEM))
    out = posthoc_calibration(pred_fit, np.zeros(n_fit), y_fit, mask_fit,
                              pred_eval, np.zeros(3))
    assert np.allclose(out[:, 0], 5.0)


def test_posthoc_linear_fallback_recovers():
    # 3 <= n_valid < 6 -> linear [1,p,t]; exact for a linear y with 5 rows.
    n_fit = 5
    p = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t = np.array([1.0, 0.0, 2.0, 1.0, 3.0])
    pred_fit = np.zeros((n_fit, N_ELEM))
    y_fit = np.zeros((n_fit, N_ELEM))
    mask_fit = np.zeros((n_fit, N_ELEM), dtype=bool)
    pred_fit[:, 0] = p
    y_fit[:, 0] = 1.0 + 2.0 * p + 0.5 * t
    mask_fit[:, 0] = True
    pe = np.array([5.0, 6.0])
    te = np.array([2.0, 4.0])
    pred_eval = np.zeros((2, N_ELEM))
    pred_eval[:, 0] = pe
    out = posthoc_calibration(pred_fit, t, y_fit, mask_fit, pred_eval, te)
    assert np.allclose(out[:, 0], 1.0 + 2.0 * pe + 0.5 * te, atol=1e-8)


def test_posthoc_all_masked_is_nan():
    pred_fit = np.zeros((4, N_ELEM))
    y_fit = np.zeros((4, N_ELEM))
    mask_fit = np.zeros((4, N_ELEM), dtype=bool)
    pred_eval = np.zeros((3, N_ELEM))
    out = posthoc_calibration(pred_fit, np.zeros(4), y_fit, mask_fit,
                              pred_eval, np.zeros(3))
    assert out.shape == (3, N_ELEM)
    assert np.isnan(out).all()


def test_posthoc_nan_target_on_valid_row_dropped():
    # One masked-valid row has NaN target -> dropped; remaining 2 define offset.
    pred_fit = np.zeros((3, N_ELEM))
    y_fit = np.zeros((3, N_ELEM))
    mask_fit = np.zeros((3, N_ELEM), dtype=bool)
    y_fit[:, 0] = [np.nan, 4.0, 6.0]
    mask_fit[:, 0] = True
    pred_eval = np.zeros((1, N_ELEM))
    out = posthoc_calibration(pred_fit, np.zeros(3), y_fit, mask_fit,
                              pred_eval, np.zeros(1))
    # 2 valid rows -> offset fallback -> mean(4,6)=5
    assert out[0, 0] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# ridge_baseline
# --------------------------------------------------------------------------- #
def test_ridge_shape_and_recovery_low_alpha():
    rng = np.random.default_rng(1)
    n_fit, dim = 200, 20  # n > dim so near-OLS is well posed
    Z = rng.normal(size=(n_fit, dim))
    w = rng.normal(size=dim)
    Z_fit = Z
    y_fit = np.zeros((n_fit, N_ELEM))
    mask_fit = np.zeros((n_fit, N_ELEM), dtype=bool)
    y_fit[:, 0] = Z @ w
    mask_fit[:, 0] = True
    Z_eval = rng.normal(size=(6, dim))
    out = ridge_baseline(Z_fit, y_fit, mask_fit, Z_eval, alpha=1e-6)
    assert out.shape == (6, N_ELEM)
    # near-OLS on noiseless linear data -> close to truth
    truth = Z_eval @ w
    assert np.corrcoef(out[:, 0], truth)[0, 1] > 0.99
    assert np.isnan(out[:, 1]).all()  # unfit element


def test_ridge_all_masked_is_nan():
    Z_fit = np.zeros((5, 256))
    y_fit = np.zeros((5, N_ELEM))
    mask_fit = np.zeros((5, N_ELEM), dtype=bool)
    Z_eval = np.zeros((3, 256))
    out = ridge_baseline(Z_fit, y_fit, mask_fit, Z_eval)
    assert out.shape == (3, N_ELEM)
    assert np.isnan(out).all()


def test_ridge_small_sample_uses_default_alpha():
    # n_valid < 5 -> alpha=1.0 path still runs and returns finite predictions.
    rng = np.random.default_rng(2)
    Z_fit = rng.normal(size=(4, 256))
    y_fit = np.zeros((4, N_ELEM))
    mask_fit = np.zeros((4, N_ELEM), dtype=bool)
    y_fit[:, 0] = rng.normal(size=4)
    mask_fit[:, 0] = True
    Z_eval = rng.normal(size=(2, 256))
    out = ridge_baseline(Z_fit, y_fit, mask_fit, Z_eval)
    assert np.isfinite(out[:, 0]).all()
