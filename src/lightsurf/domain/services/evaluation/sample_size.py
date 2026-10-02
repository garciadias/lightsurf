"""Evaluation library for the DR19->DR17 abundance sample-size study.

Implements the Evaluation API (EVAL) from
``docs/specs/abundance-head-contracts.md`` (section "Evaluation API (EVAL)").
HEAD's sweep imports these helpers; it must not reimplement them.

All element-wise arrays are ordered by
``lightsurf.constants.APOGEE_ABUNDANCE_TARGETS`` (9 elements, that order).

Documented edge-case choices (adversarial worker, chosen to be consistent
with the spec's masking philosophy; recorded here and in progress-eval.md):

* ``rmse_masked`` returns ``NaN`` (never 0) for an element whose mask selects
  no rows. A ``NaN`` *target* on a ``mask==True`` row is TREATED AS MASKED
  (the row is dropped), not raised: the contract says ``Y`` may contain NaN
  and ``M`` is the authority on validity, so an inconsistent (True, NaN) pair
  is resolved in favour of the mask. A ``NaN`` *prediction* on a valid row is
  a genuine model failure and is left to propagate to a ``NaN`` RMSE so it
  surfaces loudly. The fit helpers (posthoc/ridge) apply the same rule:
  a fit row counts only when ``mask`` is True AND the target is finite.
* ``gap_closed`` returns ``NaN`` when the denominator ``rmse_0 - rmse_nmax``
  is non-finite, <= 0, or below ``_DENOM_TOL`` (1e-12). ``minimum_n`` then
  treats such a seed as "not reached".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS

N_ELEM = len(APOGEE_ABUNDANCE_TARGETS)

# Denominator tolerance for gap_closed: below this the base/n_max gap is too
# small to define a meaningful fraction closed, so G is undefined (NaN).
_DENOM_TOL = 1e-12

# Post-hoc polynomial terms, in order: [1, p, t, p^2, t^2, p*t].
_POSTHOC_FULL_TERMS = 6
_POSTHOC_LINEAR_TERMS = 3  # [1, p, t]


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def rmse_masked(pred, y, mask) -> np.ndarray:
    """Per-element RMSE over ``mask==True`` rows.

    Parameters
    ----------
    pred, y : array-like, shape (n, 9)
        Predictions and targets in physical units.
    mask : array-like of bool, shape (n, 9)
        True where the element is valid for that row.

    Returns
    -------
    np.ndarray, shape (9,)
        RMSE per element. ``NaN`` (never 0) for an element with no valid row.
        A ``mask==True`` row whose target is ``NaN`` is dropped (treated as
        masked). A ``NaN`` prediction on a kept row propagates to ``NaN``.
    """
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != y.shape or pred.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape}, y {y.shape}, mask {mask.shape}"
        )
    if pred.ndim != 2 or pred.shape[1] != N_ELEM:
        raise ValueError(f"expected (n, {N_ELEM}) arrays, got {pred.shape}")

    # A (True, NaN-target) pair is resolved in favour of the mask: drop it.
    valid = mask & np.isfinite(y)
    out = np.full(N_ELEM, np.nan, dtype=float)
    for j in range(N_ELEM):
        sel = valid[:, j]
        if not sel.any():
            continue  # leave NaN
        diff = pred[sel, j] - y[sel, j]
        out[j] = float(np.sqrt(np.mean(diff**2)))
    return out


def gap_closed(rmse_n, rmse_0, rmse_nmax) -> float:
    """Fraction of the base->n_max RMSE gap closed at sample size n.

    ``G = (rmse_0 - rmse_n) / (rmse_0 - rmse_nmax)``.

    Returns ``NaN`` when the denominator is non-finite, <= 0, or below
    ``_DENOM_TOL`` (i.e. the base and n_max RMSE are effectively equal, so the
    gap is undefined). ``minimum_n`` treats a ``NaN`` G as "threshold not met".
    """
    rmse_n = float(rmse_n)
    rmse_0 = float(rmse_0)
    rmse_nmax = float(rmse_nmax)
    denom = rmse_0 - rmse_nmax
    if not np.isfinite(denom) or denom <= _DENOM_TOL:
        return float("nan")
    if not np.isfinite(rmse_n):
        return float("nan")
    return (rmse_0 - rmse_n) / denom


def minimum_n(
    df: pd.DataFrame,
    element: str,
    arm: str,
    thresh: float = 0.9,
    frac_seeds: float = 0.8,
) -> int | None:
    """Smallest N at which ``arm`` closes >= ``thresh`` of the gap in >=
    ``frac_seeds`` of seeds, for ``element``.

    ``df`` has the ``sweep.parquet`` schema
    (``n, seed, arm, element, rmse, bias, n_used, label_kind``).

    * ``rmse_0`` = median of the ``base_n0`` rows' rmse for ``element``.
    * ``n_max`` = the largest ``n`` present for this ``arm``/``element``.
    * ``rmse_nmax`` = median across seeds of ``arm``'s rmse at ``n_max``.

    For each ``n`` ascending, the fraction of seeds whose per-seed
    ``gap_closed(rmse_n_seed, rmse_0, rmse_nmax) >= thresh`` is computed;
    the smallest ``n`` with fraction ``>= frac_seeds`` is returned, else
    ``None``.

    Subtle point (documented deliberately): by construction the median G at
    ``n_max`` is ~1, so the criterion AT ``n_max`` is really a statement about
    seed-to-seed spread (how many seeds stay near the plateau), not about
    whether the plateau was reached. A seed whose G is ``NaN`` (undefined gap,
    see ``gap_closed``) counts as NOT meeting the threshold.

    Raises
    ------
    ValueError
        If there are no ``base_n0`` rows for ``element`` (rmse_0 undefined).

    Returns
    -------
    int | None
        The minimum N, or ``None`` if no N on the grid reaches the criterion
        (including when the arm/element has no rows at all).
    """
    base = df[(df["arm"] == "base_n0") & (df["element"] == element)]
    if base.empty:
        raise ValueError(
            f"no base_n0 rows for element {element!r}; rmse_0 is undefined"
        )
    rmse_0 = float(np.nanmedian(base["rmse"].to_numpy(dtype=float)))

    sub = df[(df["arm"] == arm) & (df["element"] == element)]
    if sub.empty:
        return None

    ns = np.sort(sub["n"].unique())
    n_max = int(ns[-1])
    at_nmax = sub[sub["n"] == n_max]["rmse"].to_numpy(dtype=float)
    rmse_nmax = float(np.nanmedian(at_nmax))

    for n in ns:
        rows = sub[sub["n"] == n]["rmse"].to_numpy(dtype=float)
        if rows.size == 0:
            continue
        g = np.array([gap_closed(r, rmse_0, rmse_nmax) for r in rows])
        # NaN G -> not meeting threshold.
        frac = np.mean(np.where(np.isnan(g), False, g >= thresh))
        if frac >= frac_seeds:
            return int(n)
    return None


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _standardise_fit(x: np.ndarray):
    """Return (mean, std) for standardising a 1-D fit vector; std==0 -> 1."""
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0 or not np.isfinite(sd):
        sd = 1.0
    return mu, sd


def posthoc_calibration(
    pred_fit, teff_fit, y_fit, mask_fit, pred_eval, teff_eval
) -> np.ndarray:
    """Per-element post-hoc calibration of base-head predictions.

    Least-squares fit of ``y`` on ``[1, p, t, p^2, t^2, p*t]`` using only
    ``mask_fit==True`` rows with finite target, where ``p`` is the base-head
    prediction and ``t`` is ``teff19``. Both ``p`` and ``t`` are standardised
    on the fit set (per element) purely for numerical conditioning; the same
    transform is applied to the eval set before prediction, so predictions are
    returned in physical ``y`` units.

    Degenerate-size fallback (documented):

    * ``n_valid < 3``  -> offset only: predict ``mean(y_fit)``.
    * ``3 <= n_valid < 6`` -> linear: fit ``[1, p, t]``.
    * ``n_valid >= 6`` -> full quadratic with cross term.
    * ``n_valid == 0``  -> that element's eval predictions are all ``NaN``.

    Parameters
    ----------
    pred_fit, pred_eval : array-like, shape (n_fit, 9) / (n_eval, 9)
        Base-head predictions.
    teff_fit, teff_eval : array-like, shape (n_fit,) / (n_eval,)
        DR19 Teff (shared across elements).
    y_fit : array-like, shape (n_fit, 9)
        DR17/target abundances; NaN allowed.
    mask_fit : array-like of bool, shape (n_fit, 9)

    Returns
    -------
    np.ndarray, shape (n_eval, 9)
    """
    pred_fit = np.asarray(pred_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    mask_fit = np.asarray(mask_fit, dtype=bool)
    teff_fit = np.asarray(teff_fit, dtype=float).reshape(-1)
    pred_eval = np.asarray(pred_eval, dtype=float)
    teff_eval = np.asarray(teff_eval, dtype=float).reshape(-1)

    n_eval = pred_eval.shape[0]
    out = np.full((n_eval, N_ELEM), np.nan, dtype=float)

    for j in range(N_ELEM):
        valid = mask_fit[:, j] & np.isfinite(y_fit[:, j]) & np.isfinite(
            pred_fit[:, j]
        ) & np.isfinite(teff_fit)
        n_valid = int(valid.sum())
        yv = y_fit[valid, j]
        if n_valid == 0:
            continue  # leave NaN column

        if n_valid < 3:
            out[:, j] = float(np.mean(yv))
            continue

        pv = pred_fit[valid, j]
        tv = teff_fit[valid]
        p_mu, p_sd = _standardise_fit(pv)
        t_mu, t_sd = _standardise_fit(tv)
        ps = (pv - p_mu) / p_sd
        ts = (tv - t_mu) / t_sd

        pe = (pred_eval[:, j] - p_mu) / p_sd
        te = (teff_eval - t_mu) / t_sd

        if n_valid < 6:  # linear [1, p, t]
            A = np.column_stack([np.ones_like(ps), ps, ts])
            Ae = np.column_stack([np.ones(n_eval), pe, te])
        else:  # full [1, p, t, p^2, t^2, p*t]
            A = np.column_stack([np.ones_like(ps), ps, ts, ps**2, ts**2, ps * ts])
            Ae = np.column_stack(
                [np.ones(n_eval), pe, te, pe**2, te**2, pe * te]
            )

        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        out[:, j] = Ae @ coef

    return out


def ridge_baseline(Z_fit, y_fit, mask_fit, Z_eval, alpha=None) -> np.ndarray:
    """Per-element ridge regression on the frozen latents.

    For each element, sklearn ``Ridge`` is fit on ``mask_fit==True`` rows with
    finite target, after standardising ``Z`` on the fit set.

    * ``alpha is None`` and ``n_valid >= 5`` -> ``RidgeCV`` chooses alpha by
      leave-one-out over ``logspace(-3, 3, 13)``.
    * ``alpha is None`` and ``n_valid < 5``  -> ``alpha = 1.0``.
    * ``alpha`` given -> that alpha is used directly.
    * ``n_valid == 0`` -> that element's eval predictions are all ``NaN``.

    Returns
    -------
    np.ndarray, shape (n_eval, 9)
    """
    from sklearn.linear_model import Ridge, RidgeCV

    Z_fit = np.asarray(Z_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    mask_fit = np.asarray(mask_fit, dtype=bool)
    Z_eval = np.asarray(Z_eval, dtype=float)

    n_eval = Z_eval.shape[0]
    out = np.full((n_eval, N_ELEM), np.nan, dtype=float)
    alphas = np.logspace(-3, 3, 13)

    for j in range(N_ELEM):
        valid = mask_fit[:, j] & np.isfinite(y_fit[:, j])
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue

        Zv = Z_fit[valid]
        yv = y_fit[valid, j]

        mu = Zv.mean(axis=0)
        sd = Zv.std(axis=0)
        sd[sd == 0.0] = 1.0
        Zs = (Zv - mu) / sd
        Ze = (Z_eval - mu) / sd

        if alpha is None:
            if n_valid >= 5:
                model = RidgeCV(alphas=alphas)  # leave-one-out by default
            else:
                model = Ridge(alpha=1.0)
        else:
            model = Ridge(alpha=float(alpha))

        model.fit(Zs, yv)
        out[:, j] = model.predict(Ze)

    return out
