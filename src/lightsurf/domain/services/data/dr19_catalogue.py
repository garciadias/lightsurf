"""DR19 astra ASPCAP catalogue loader and DR19<->DR17 cross-match (study stage S1).

Reads the DR19 ``astraAllStarASPCAP-0.6.0`` summary FITS, resolves one row per
star (prefer APO telescope, then highest SNR), converts the native ``[X/H]``
abundances to ``[X/Fe]``, and joins against the DR17 subset and the cached
latent APOGEE_IDs to produce ``labels.parquet`` (contract schema) plus a
``dr19_mdwarf_candidates.parquet`` list for the S8 expansion stage.

Astra specifics (verified against astraAllStarASPCAP-0.6.0, HDU 2, 1,095,480 rows):
  * identifier: ``sdss4_apogee_id`` holds the full 2MASS designation WITH the
    leading ``2M`` for the field stars (e.g. ``2M00001687+5...``); it matches
    the latents/DR17 APOGEE_ID directly (verified: 39,945/39,945 overlap). A
    minority of rows carry a non-2M designation, which simply will not match.
  * ``sdss_id`` (int64) gives the mwmStar path.
  * abundances are ``[X/H]`` (columns ``fe_h``, ``mg_h`` ...); we convert to
    ``[X/Fe] = X_H - FE_H`` and propagate errors in quadrature. ``FE_H`` stays
    ``[Fe/H]``.
  * STAR_BAD equivalent: the boolean ``flag_bad`` column (astra's overall bad
    flag; ``result_flags`` is the underlying bitmask and ``flag_warn`` the warn
    level -- ``flag_bad`` is the direct STAR_BAD analogue, documented in the
    astra datamodel). Per-element flags: ``<x>_h_flags`` bitmasks.
  * telescope values: ``apo25m``, ``lco25m``, ``apo1m``; APO = startswith 'apo'.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from astropy.io import fits

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS as TARGETS

log = logging.getLogger(__name__)

# DR17 STAR_BAD is bit 23 of ASPCAPFLAG.
# Verified against the DR17 APOGEE bitmask datamodel:
# https://www.sdss4.org/dr17/irspec/apogee-bitmasks/  (ASPCAPFLAG, bit 23 = STAR_BAD)
DR17_STAR_BAD_BIT = 23

# DR17 [X/Fe] target name  ->  DR19 astra [X/H] base column name.
ASTRA_XH = {
    "FE_H": "fe_h",
    "C_FE": "c_h",
    "CA_FE": "ca_h",
    "K_FE": "k_h",
    "MG_FE": "mg_h",
    "NI_FE": "ni_h",
    "O_FE": "o_h",
    "SI_FE": "si_h",
    "TI_FE": "ti_h",
}

MWMSTAR_URL = (
    "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/spectra/star/"
    "{xx}/{yy}/mwmStar-0.6.0-{sdss_id}.fits"
)


# --------------------------------------------------------------------------- #
# small reusable helpers
# --------------------------------------------------------------------------- #
def _native(a: np.ndarray) -> np.ndarray:
    """Return array in native byte order (FITS tables are big-endian)."""
    a = np.asarray(a)
    if a.dtype.byteorder not in ("=", "|"):
        a = a.byteswap().view(a.dtype.newbyteorder("="))
    return a


def valid_value(x: np.ndarray) -> np.ndarray:
    """Finite and not a -9999 style sentinel."""
    x = np.asarray(x, dtype=float)
    return np.isfinite(x) & (x > -9999.0)


def mwmstar_url(sdss_id: int) -> str:
    """mwmStar URL for a DR19 ``sdss_id``.

    Directory ``XX/YY`` uses the zero-padded id: ``XX`` = 4th and 3rd last
    digits, ``YY`` = last two digits.
    """
    s = f"{int(sdss_id):04d}"
    return MWMSTAR_URL.format(xx=s[-4:-2], yy=s[-2:], sdss_id=int(sdss_id))


def star_bad_from_aspcapflag(aspcapflag: np.ndarray) -> np.ndarray:
    """DR17 STAR_BAD = bit 23 of ASPCAPFLAG."""
    return ((np.asarray(aspcapflag).astype(np.int64) >> DR17_STAR_BAD_BIT) & 1).astype(bool)


# --------------------------------------------------------------------------- #
# DR19 astra catalogue
# --------------------------------------------------------------------------- #
def load_astra_raw(cat_path: str | Path) -> pd.DataFrame:
    """Load the astra HDU with the per-star rows into a native DataFrame."""
    with fits.open(cat_path) as h:
        hdu = max(h[1:], key=lambda x: getattr(x, "header", {}).get("NAXIS2", 0))
        d = hdu.data
        cols: dict[str, np.ndarray] = {}
        cols["APOGEE_ID"] = np.char.strip(d["sdss4_apogee_id"].astype(str))
        cols["sdss_id"] = _native(d["sdss_id"]).astype(np.int64)
        cols["telescope"] = np.char.strip(d["telescope"].astype(str))
        cols["teff19"] = _native(d["teff"]).astype(float)
        cols["logg19"] = _native(d["logg"]).astype(float)
        cols["snr19"] = _native(d["snr"]).astype(float)
        cols["star_bad19"] = _native(d["flag_bad"]).astype(bool)
        for xh in sorted(set(ASTRA_XH.values())):
            cols[xh] = _native(d[xh]).astype(float)
            cols["e_" + xh] = _native(d["e_" + xh]).astype(float)
            cols[xh + "_flags"] = _native(d[xh + "_flags"]).astype(np.int64)
    return pd.DataFrame(cols)


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per APOGEE_ID: prefer APO telescope, then highest SNR.

    The frozen encoder was trained on APO-preferred spectra, so when a star has
    both APO and LCO rows we keep the APO one even if LCO has higher SNR; SNR is
    the tie-breaker within a telescope preference.
    """
    d = df.copy()
    d["_is_apo"] = d["telescope"].str.startswith("apo").astype(int)
    d = d.sort_values(["_is_apo", "snr19"], ascending=[False, False])
    d = d.drop_duplicates("APOGEE_ID", keep="first").drop(columns="_is_apo")
    return d.reset_index(drop=True)


def to_xfe(df: pd.DataFrame) -> pd.DataFrame:
    """Add DR19 ``{E}_19``, ``{E}_19_err``, ``{E}_19_ok`` columns ([X/Fe])."""
    d = df.copy()
    fe = d["fe_h"].to_numpy()
    e_fe = d["e_fe_h"].to_numpy()
    fe_ok = valid_value(fe) & (d["fe_h_flags"].to_numpy() == 0)
    for E in TARGETS:
        xh = ASTRA_XH[E]
        x = d[xh].to_numpy()
        ex = d["e_" + xh].to_numpy()
        xflag0 = d[xh + "_flags"].to_numpy() == 0
        if E == "FE_H":
            val = np.where(valid_value(x), x, np.nan)
            err = np.where(valid_value(ex), ex, np.nan)
            ok = valid_value(x) & xflag0
        else:
            both = valid_value(x) & valid_value(fe)
            val = np.where(both, x - fe, np.nan)
            err = np.where(both, np.sqrt(ex**2 + e_fe**2), np.nan)
            # per-element ok: element flag clear AND Fe valid+clear (needed for
            # the [X/Fe] conversion) AND the resulting value finite.
            ok = both & xflag0 & fe_ok
        d[f"{E}_19"] = val
        d[f"{E}_19_err"] = err
        d[f"{E}_19_ok"] = ok
    return d


def build_dr19_catalogue(cat_path: str | Path) -> pd.DataFrame:
    """Full DR19 catalogue: one row per APOGEE_ID with [X/Fe], url, flags."""
    raw = load_astra_raw(cat_path)
    raw = resolve_duplicates(raw)
    cat = to_xfe(raw)
    cat["mwmstar_url"] = [mwmstar_url(s) for s in cat["sdss_id"].to_numpy()]
    keep = ["APOGEE_ID", "sdss_id", "teff19", "logg19", "snr19", "star_bad19", "mwmstar_url"]
    keep += [c for E in TARGETS for c in (f"{E}_19", f"{E}_19_err", f"{E}_19_ok")]
    return cat[keep]


# --------------------------------------------------------------------------- #
# DR17 side
# --------------------------------------------------------------------------- #
def build_dr17_labels(dr17_subset_path: str | Path) -> pd.DataFrame:
    """DR17 [X/Fe] labels, errors, masks keyed by APOGEE_ID (already deduped)."""
    s = pd.read_parquet(dr17_subset_path)
    out = pd.DataFrame({"APOGEE_ID": s["APOGEE_ID"].astype(str)})
    out["snr17"] = s["SNR"].astype(float)
    out["star_bad17"] = star_bad_from_aspcapflag(s["ASPCAPFLAG"].to_numpy())
    for E in TARGETS:
        val = s[E].to_numpy().astype(float)
        err = s[E + "_ERR"].to_numpy().astype(float)
        flag = s[E + "_FLAG"].to_numpy().astype(np.int64)
        v = valid_value(val)
        out[f"{E}_17"] = np.where(v, val, np.nan)
        out[f"{E}_17_err"] = np.where(valid_value(err), err, np.nan)
        out[f"{E}_17_ok"] = v & (flag == 0)
    return out


# --------------------------------------------------------------------------- #
# cross-match -> labels.parquet
# --------------------------------------------------------------------------- #
def build_labels(cat_path, latents_path, dr17_subset_path) -> pd.DataFrame:
    """labels.parquet: one row per APOGEE_ID present in the latents file."""
    ids = pd.read_parquet(latents_path, columns=["APOGEE_ID"])
    ids["APOGEE_ID"] = ids["APOGEE_ID"].astype(str).str.strip()
    cat = build_dr19_catalogue(cat_path)
    dr17 = build_dr17_labels(dr17_subset_path)

    df = ids.merge(cat, on="APOGEE_ID", how="left").merge(dr17, on="APOGEE_ID", how="left")
    df["is_mdwarf"] = (df["teff19"] <= 4100) & (df["logg19"] >= 4)
    df["passes_quality"] = (df["snr19"] >= 70) & (~df["star_bad19"].fillna(True))
    # tidy dtypes for the contract
    df["sdss_id"] = df["sdss_id"].astype("Int64")
    for c in ("star_bad19", "star_bad17", "is_mdwarf", "passes_quality"):
        df[c] = df[c].fillna(False) if c != "star_bad19" else df[c]
    for E in TARGETS:
        for c in (f"{E}_19_ok", f"{E}_17_ok"):
            df[c] = df[c].fillna(False).astype(bool)
    return df


def build_mdwarf_candidates(cat_path) -> pd.DataFrame:
    """All DR19 M dwarfs in the catalogue (teff<=4100, logg>=4, snr>=70, not bad)."""
    cat = build_dr19_catalogue(cat_path)
    m = (
        (cat["teff19"] <= 4100)
        & (cat["logg19"] >= 4)
        & (cat["snr19"] >= 70)
        & (~cat["star_bad19"])
    )
    return cat.loc[m, ["APOGEE_ID", "sdss_id", "mwmstar_url"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@click.command()
@click.option("--catalogue", required=True, help="astraAllStarASPCAP-0.6.0.fits.gz path")
@click.option("--latents", required=True, help="masked_latent_field.parquet path")
@click.option("--dr17-subset", required=True, help="dr17_subset.parquet path")
@click.option("--out-dir", required=True, help="output dir ($R)")
def main(catalogue, latents, dr17_subset, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log.info("building labels.parquet")
    labels = build_labels(catalogue, latents, dr17_subset)
    labels.to_parquet(out / "labels.parquet", index=False)
    print(f"labels.parquet: {labels.shape}")
    print(f"  dr19 matched: {labels['sdss_id'].notna().sum()}")
    print(f"  dr17 matched: {labels['snr17'].notna().sum()}")
    print(f"  is_mdwarf: {int(labels['is_mdwarf'].sum())}")
    print(f"  passes_quality: {int(labels['passes_quality'].sum())}")

    cand = build_mdwarf_candidates(catalogue)
    cand.to_parquet(out / "dr19_mdwarf_candidates.parquet", index=False)
    print(f"dr19_mdwarf_candidates.parquet: {cand.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
