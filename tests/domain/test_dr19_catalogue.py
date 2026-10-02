"""Tests for the DR19 catalogue loader / cross-match helpers (study S1)."""
import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS as TARGETS
from lightsurf.domain.services.data.dr19_catalogue import (
    mwmstar_url,
    resolve_duplicates,
    star_bad_from_aspcapflag,
    to_xfe,
    valid_value,
)


def test_mwmstar_url_rule():
    # XX = 4th & 3rd last digits, YY = last two digits of the sdss_id.
    assert mwmstar_url(54400160).endswith("star/01/60/mwmStar-0.6.0-54400160.fits")
    assert mwmstar_url(117524931).endswith("star/49/31/mwmStar-0.6.0-117524931.fits")
    assert mwmstar_url(54487187).endswith("star/71/87/mwmStar-0.6.0-54487187.fits")
    # short id is zero-padded to >=4 digits
    assert "star/00/07/" in mwmstar_url(7)


def test_star_bad_bit23():
    flags = np.array([0, 1 << 23, (1 << 23) | 5, 1 << 22], dtype=np.int64)
    assert star_bad_from_aspcapflag(flags).tolist() == [False, True, True, False]


def test_valid_value():
    x = np.array([0.1, np.nan, -9999.0, -10000.0, -1.0])
    assert valid_value(x).tolist() == [True, False, False, False, True]


def test_resolve_duplicates_prefers_apo_then_snr():
    df = pd.DataFrame({
        "APOGEE_ID": ["A", "A", "A", "B", "B"],
        "telescope": ["lco25m", "apo25m", "apo25m", "lco25m", "lco25m"],
        "snr19": [500.0, 100.0, 80.0, 90.0, 120.0],
    })
    out = resolve_duplicates(df).set_index("APOGEE_ID")
    # A: APO preferred even though LCO has higher SNR; within APO take snr=100
    assert out.loc["A", "telescope"] == "apo25m"
    assert out.loc["A", "snr19"] == 100.0
    # B: only LCO, take highest snr
    assert out.loc["B", "snr19"] == 120.0


def test_to_xfe_conversion_and_error_propagation():
    # FE_H stays [Fe/H]; others become X_H - FE_H, errors in quadrature.
    row = {"fe_h": 0.2, "e_fe_h": 0.01, "fe_h_flags": 0}
    for xh in ["c_h", "ca_h", "k_h", "mg_h", "ni_h", "o_h", "si_h", "ti_h"]:
        row[xh] = 0.5
        row["e_" + xh] = 0.03
        row[xh + "_flags"] = 0
    df = pd.DataFrame([row])
    out = to_xfe(df)
    assert out["FE_H_19"].iloc[0] == 0.2
    assert np.isclose(out["MG_FE_19"].iloc[0], 0.5 - 0.2)
    assert np.isclose(out["MG_FE_19_err"].iloc[0], np.hypot(0.03, 0.01))
    assert bool(out["MG_FE_19_ok"].iloc[0])


def test_to_xfe_masks_bad_flags_and_invalid():
    row = {"fe_h": 0.2, "e_fe_h": 0.01, "fe_h_flags": 0}
    for xh in ["c_h", "ca_h", "k_h", "mg_h", "ni_h", "o_h", "si_h", "ti_h"]:
        row[xh] = 0.5
        row["e_" + xh] = 0.03
        row[xh + "_flags"] = 0
    row["mg_h_flags"] = 1  # element flagged
    row["o_h"] = np.nan     # invalid value
    df = pd.DataFrame([row])
    out = to_xfe(df)
    assert not bool(out["MG_FE_19_ok"].iloc[0])
    assert not bool(out["O_FE_19_ok"].iloc[0])
    assert np.isnan(out["O_FE_19"].iloc[0])


def test_to_xfe_fe_bad_kills_all_xfe():
    row = {"fe_h": np.nan, "e_fe_h": 0.01, "fe_h_flags": 0}
    for xh in ["c_h", "ca_h", "k_h", "mg_h", "ni_h", "o_h", "si_h", "ti_h"]:
        row[xh] = 0.5
        row["e_" + xh] = 0.03
        row[xh + "_flags"] = 0
    out = to_xfe(pd.DataFrame([row]))
    for E in TARGETS:
        if E != "FE_H":
            assert not bool(out[f"{E}_19_ok"].iloc[0])
