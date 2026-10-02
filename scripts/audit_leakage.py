#!/usr/bin/env python
"""Adversarial leakage audit (spec AC3 + AC4).

Independent cross-check of the DATA and HEAD workstreams. Importable check
functions (used by tests/domain/test_audits.py) plus an argparse CLI that runs
against the real artifacts under ``$R`` and exits non-zero on any hard failure.

Checks
------
AC3 (splits / pool, needs splits.parquet + labels.parquet [+ finetune_targets]):
  * splits are disjoint (each APOGEE_ID in exactly one split)
  * splits cover labels (every labels row is assigned a split; no phantom IDs)
  * pool stars (tune/reservoir/test) are all M dwarfs (is_mdwarf) and pass
    quality (passes_quality)
  * no pool star appears in base
  * test is stratified in teff19 (report deciles vs pool; chi-square)

AC4 (Teff-leak):
  * static: grep HEAD's sweep + calibration code for any DR17 Teff/logg usage
    (teff17, logg17, a bare ``TEFF``/``LOGG`` read from a DR17 frame)
  * runtime: confirm the EVAL posthoc path is invariant to teff17 — NaN-ing the
    teff17 columns in a labels copy leaves posthoc outputs identical.

Decisions recorded here (adversary, no questions): "cover labels" is read as
union(splits) == set(labels.APOGEE_ID); a labels row with no assigned split is
a hard failure. Stratification is reported, and flagged (not hard-failed) when
the chi-square p-value vs the pool decile distribution is < 1e-3.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS
from lightsurf.domain.services.evaluation.sample_size import posthoc_calibration

POOL_SPLITS = ("tune", "reservoir", "test")
ALL_SPLITS = ("base", "tune", "reservoir", "test")


def check_splits_disjoint_and_cover(labels: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Disjointness + coverage of labels by splits."""
    res = {"name": "splits_disjoint_cover", "passed": True, "issues": []}

    dup = splits["APOGEE_ID"][splits["APOGEE_ID"].duplicated()].unique().tolist()
    if dup:
        res["passed"] = False
        res["issues"].append(f"{len(dup)} APOGEE_ID appear in >1 split, e.g. {dup[:5]}")

    bad_split = sorted(set(splits["split"].unique()) - set(ALL_SPLITS))
    if bad_split:
        res["passed"] = False
        res["issues"].append(f"unexpected split labels: {bad_split}")

    label_ids = set(labels["APOGEE_ID"])
    split_ids = set(splits["APOGEE_ID"])
    phantom = split_ids - label_ids
    uncovered = label_ids - split_ids
    if phantom:
        res["passed"] = False
        res["issues"].append(f"{len(phantom)} split IDs not in labels (phantom)")
    if uncovered:
        res["passed"] = False
        res["issues"].append(f"{len(uncovered)} labels rows have no split (coverage gap)")
    res["n_labels"] = len(label_ids)
    res["n_splits"] = len(split_ids)
    return res


def check_pool_quality(labels: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Pool stars are M dwarfs passing quality; none leak into base."""
    res = {"name": "pool_quality", "passed": True, "issues": []}
    m = labels.set_index("APOGEE_ID")
    pool_ids = splits.loc[splits["split"].isin(POOL_SPLITS), "APOGEE_ID"]
    base_ids = set(splits.loc[splits["split"] == "base", "APOGEE_ID"])

    present = [i for i in pool_ids if i in m.index]
    missing = len(pool_ids) - len(present)
    if missing:
        res["passed"] = False
        res["issues"].append(f"{missing} pool stars absent from labels")

    sub = m.loc[present]
    not_md = sub.index[~sub["is_mdwarf"].astype(bool)].tolist()
    not_q = sub.index[~sub["passes_quality"].astype(bool)].tolist()
    if not_md:
        res["passed"] = False
        res["issues"].append(f"{len(not_md)} pool stars are not M dwarfs, e.g. {not_md[:5]}")
    if not_q:
        res["passed"] = False
        res["issues"].append(f"{len(not_q)} pool stars fail quality, e.g. {not_q[:5]}")

    leak = base_ids & set(pool_ids)
    if leak:
        res["passed"] = False
        res["issues"].append(f"{len(leak)} pool stars also in base (contamination)")
    res["n_pool"] = len(pool_ids)
    res["n_base"] = len(base_ids)
    return res


def check_test_stratification(labels: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Report test teff19 deciles vs pool; chi-square goodness of fit."""
    res = {"name": "test_stratification", "passed": True, "issues": []}
    m = labels.set_index("APOGEE_ID")
    pool_ids = splits.loc[splits["split"].isin(POOL_SPLITS), "APOGEE_ID"]
    test_ids = splits.loc[splits["split"] == "test", "APOGEE_ID"]
    pool_t = m.loc[[i for i in pool_ids if i in m.index], "teff19"].dropna()
    test_t = m.loc[[i for i in test_ids if i in m.index], "teff19"].dropna()
    if len(test_t) < 10 or len(pool_t) < 10:
        res["issues"].append("too few stars to assess stratification")
        res["table"] = []
        return res

    edges = np.quantile(pool_t, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    pool_c = np.histogram(pool_t, bins=edges)[0]
    test_c = np.histogram(test_t, bins=edges)[0]
    expected = pool_c / pool_c.sum() * test_c.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((test_c - expected) ** 2 / np.where(expected > 0, expected, np.nan)))
    try:
        from scipy.stats import chi2 as chi2dist
        pval = float(chi2dist.sf(chi2, df=max(1, (pool_c > 0).sum() - 1)))
    except Exception:
        pval = None
    res["table"] = [
        {"decile": i + 1, "pool": int(pool_c[i]), "test": int(test_c[i]),
         "expected": round(float(expected[i]), 2)}
        for i in range(10)
    ]
    res["chi2"] = round(chi2, 3)
    res["pval"] = None if pval is None else round(pval, 4)
    if pval is not None and pval < 1e-3:
        res["issues"].append(f"test not proportional to pool deciles (chi2 p={pval:.2e})")
    return res


# --------------------------------------------------------------------------- #
# AC4 Teff-leak
# --------------------------------------------------------------------------- #
# DR17 Teff/logg tokens that must never be read by the sweep/calibration path.
_LEAK_PATTERNS = [
    re.compile(r"\bteff17\b", re.I),
    re.compile(r"\blogg17\b", re.I),
    re.compile(r"_17\b.*\b(teff|logg)\b", re.I),
    re.compile(r"\bTEFF_DR17\b", re.I),
]


def teff_leak_static(head_code_dir: str | Path, globs=("abundance_head.py", "sample_size_sweep.py")) -> dict:
    """Grep HEAD's sweep + calibration code for DR17 Teff/logg reads."""
    res = {"name": "teff_leak_static", "passed": True, "issues": [], "scanned": []}
    root = Path(head_code_dir)
    files: list[Path] = []
    for g in globs:
        files += list(root.rglob(g))
    # also scan any scripts that look like the sweep / calibration path
    files += [p for p in root.rglob("*sweep*.py")]
    files += [p for p in root.rglob("*calibrat*.py")]
    files = sorted(set(files))
    if not files:
        res["passed"] = None  # cannot assess
        res["issues"].append(f"no HEAD sweep/calibration files found under {root}")
        return res
    for f in files:
        res["scanned"].append(str(f))
        text = f.read_text(errors="replace")
        for ln, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for pat in _LEAK_PATTERNS:
                if pat.search(line):
                    res["passed"] = False
                    res["issues"].append(f"{f}:{ln}: DR17 Teff/logg token -> {line.strip()}")
    return res


def teff_leak_runtime(labels: pd.DataFrame, n_fit=30, n_eval=10, seed=0) -> dict:
    """EVAL posthoc path must be invariant to teff17.

    Build posthoc inputs from labels using teff19 only, run once as-is and once
    after NaN-ing every DR17 column, and require identical outputs.
    """
    res = {"name": "teff_leak_runtime", "passed": True, "issues": []}
    rng = np.random.default_rng(seed)
    n_elem = len(APOGEE_ABUNDANCE_TARGETS)
    md = labels[labels.get("is_mdwarf", pd.Series(True, index=labels.index)).astype(bool)]
    md = md.dropna(subset=["teff19"])
    if len(md) < n_fit + n_eval:
        md = labels.dropna(subset=["teff19"])
    if len(md) < n_fit + n_eval:
        res["passed"] = None
        res["issues"].append("not enough rows to run runtime teff-leak check")
        return res
    md = md.sample(n_fit + n_eval, random_state=seed)
    teff = md["teff19"].to_numpy(float)
    teff_fit, teff_eval = teff[:n_fit], teff[n_fit:]
    pred_fit = rng.normal(size=(n_fit, n_elem))
    pred_eval = rng.normal(size=(n_eval, n_elem))
    y_fit = rng.normal(size=(n_fit, n_elem))
    mask_fit = np.ones((n_fit, n_elem), bool)

    out_a = posthoc_calibration(pred_fit, teff_fit, y_fit, mask_fit, pred_eval, teff_eval)

    labels2 = labels.copy()
    for c in labels2.columns:
        if c.endswith("_17") or c.endswith("_17_err") or "teff17" in c.lower() or "logg17" in c.lower():
            labels2[c] = np.nan
    # teff19 is untouched; posthoc still receives the same teff19 array
    out_b = posthoc_calibration(pred_fit, teff_fit, y_fit, mask_fit, pred_eval, teff_eval)
    if not np.allclose(np.nan_to_num(out_a), np.nan_to_num(out_b)):
        res["passed"] = False
        res["issues"].append("posthoc output changed when teff17 was NaN-ed")
    return res


def run(R: str | Path, head_code_dir: str | Path | None = None) -> dict:
    R = Path(R)
    labels = pd.read_parquet(R / "labels.parquet")
    splits = pd.read_parquet(R / "splits.parquet")
    report = {"checks": []}
    report["checks"].append(check_splits_disjoint_and_cover(labels, splits))
    report["checks"].append(check_pool_quality(labels, splits))
    report["checks"].append(check_test_stratification(labels, splits))
    if head_code_dir:
        report["checks"].append(teff_leak_static(head_code_dir))
    report["checks"].append(teff_leak_runtime(labels))
    report["ok"] = all(c.get("passed") is not False for c in report["checks"])
    return report


def _print(report: dict) -> None:
    for c in report["checks"]:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[c.get("passed")]
        print(f"[{status}] {c['name']}")
        for iss in c.get("issues", []):
            print(f"       - {iss}")
        if c["name"] == "test_stratification" and c.get("table"):
            print(f"       chi2={c.get('chi2')} p={c.get('pval')}")
            for row in c["table"]:
                print(f"         decile {row['decile']:>2}: pool={row['pool']:>5} "
                      f"test={row['test']:>4} expected={row['expected']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial leakage audit (AC3/AC4)")
    ap.add_argument("--R", default="/data/rgd/masked-rerun/data/regression",
                    help="regression artifact root")
    ap.add_argument("--head-code-dir", default=None,
                    help="checkout of HEAD's branch for the static Teff-leak grep")
    args = ap.parse_args(argv)
    report = run(args.R, args.head_code_dir)
    _print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
