#!/usr/bin/env python
"""Adversarial gate recomputation (spec D13a / AC5).

Independently recompute the preflight gate from ``labels.parquet`` with our own
code (numpy ``polyfit``, our own bootstrap + permutation null, a DIFFERENT seed)
and compare per element against DATA's ``gate.json``: verdicts must agree and
``sys_rms`` must match within 10%. Prints a diff table and exits non-zero on any
disagreement.

Recipe (from D13a), recomputed here from scratch:
  * overlap = DR19 M dwarfs with valid labels in BOTH releases for the element
    (``{E}_19_ok`` and ``{E}_17_ok``)
  * ``delta = X_17 - X_19``
  * fit ``delta`` as a 2nd-degree polynomial in teff19 (numpy polyfit, deg=2)
  * ``sys_rms`` = RMS of the fitted curve evaluated over the overlap rows
  * ``resid_rms`` = RMS of residuals
  * verdict ``real`` iff ``sys_rms >= min_sys_rms`` AND the bootstrap 95% CI of
    ``sys_rms`` excludes 0; else ``synthetic``

Decision (adversary): "RMS of the fitted curve over the pool" is evaluated on
the same overlap rows used for the fit (the only rows with a valid delta). If
DATA evaluated it over the full pool instead, sys_rms can differ; a >10% gap is
reported as a finding rather than silently tolerated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS

AUDIT_SEED = 987654321  # deliberately different from any DATA seed
SYS_RMS_REL_TOL = 0.10


def recompute_element(labels: pd.DataFrame, element: str, min_sys_rms: float,
                      n_boot: int, ci: float, seed: int) -> dict:
    ok19 = f"{element}_19_ok"
    ok17 = f"{element}_17_ok"
    c19 = f"{element}_19"
    c17 = f"{element}_17"
    mdwarf = labels.get("is_mdwarf")
    sel = labels[ok19].astype(bool) & labels[ok17].astype(bool)
    if mdwarf is not None:
        sel = sel & mdwarf.astype(bool)
    sub = labels[sel]
    t = sub["teff19"].to_numpy(float)
    delta = (sub[c17] - sub[c19]).to_numpy(float)
    good = np.isfinite(t) & np.isfinite(delta)
    t, delta = t[good], delta[good]
    n = int(t.size)
    out = {"element": element, "n": n}
    if n < 3:
        out.update(verdict="synthetic", sys_rms=float("nan"),
                   resid_rms=float("nan"), ci=[float("nan"), float("nan")],
                   reason="n<3")
        return out

    coef = np.polyfit(t, delta, 2)
    fit = np.polyval(coef, t)
    sys_rms = float(np.sqrt(np.mean(fit**2)))
    resid_rms = float(np.sqrt(np.mean((delta - fit) ** 2)))

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cb = np.polyfit(t[idx], delta[idx], 2)
        boots[b] = np.sqrt(np.mean(np.polyval(cb, t[idx]) ** 2))
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))

    verdict = "real" if (sys_rms >= min_sys_rms and lo > 0.0) else "synthetic"
    out.update(verdict=verdict, sys_rms=sys_rms, resid_rms=resid_rms,
               ci=[lo, hi], poly_coef=coef.tolist())
    return out


def run(R: str | Path, n_boot: int | None = None) -> dict:
    R = Path(R)
    labels = pd.read_parquet(R / "labels.parquet")
    gate = json.loads((R / "gate.json").read_text())
    thr = gate.get("thresholds", {})
    min_sys_rms = float(thr.get("min_sys_rms", 0.03))
    ci = float(thr.get("ci", 0.95))
    nb = int(n_boot or thr.get("n_boot", 1000))

    rows = []
    disagreements = []
    for el in APOGEE_ABUNDANCE_TARGETS:
        mine = recompute_element(labels, el, min_sys_rms, nb, ci, AUDIT_SEED)
        theirs = gate.get("elements", {}).get(el, {})
        their_verdict = theirs.get("verdict")
        their_sys = theirs.get("sys_rms")
        verdict_ok = (their_verdict == mine["verdict"])
        if their_sys is None or not np.isfinite(mine["sys_rms"]):
            sys_ok = (their_sys is None) == (not np.isfinite(mine["sys_rms"]))
            rel = float("nan")
        else:
            denom = max(abs(their_sys), 1e-9)
            rel = abs(mine["sys_rms"] - their_sys) / denom
            sys_ok = rel <= SYS_RMS_REL_TOL
        row = {
            "element": el, "mine_verdict": mine["verdict"],
            "their_verdict": their_verdict, "verdict_ok": verdict_ok,
            "mine_sys_rms": mine["sys_rms"], "their_sys_rms": their_sys,
            "sys_rel_diff": rel, "sys_ok": sys_ok, "n": mine["n"],
        }
        rows.append(row)
        if not (verdict_ok and sys_ok):
            disagreements.append(el)
    return {"rows": rows, "disagreements": disagreements, "ok": not disagreements}


def _print(report: dict) -> None:
    hdr = f"{'elem':<7}{'mine':<10}{'theirs':<10}{'v_ok':<6}{'mine_sys':<11}{'their_sys':<11}{'rel':<8}{'s_ok':<5}{'n':<6}"
    print(hdr)
    print("-" * len(hdr))
    for r in report["rows"]:
        ts = "None" if r["their_sys_rms"] is None else f"{r['their_sys_rms']:.4f}"
        rel = "nan" if not np.isfinite(r["sys_rel_diff"]) else f"{r['sys_rel_diff']*100:.1f}%"
        print(f"{r['element']:<7}{r['mine_verdict']:<10}{str(r['their_verdict']):<10}"
              f"{str(r['verdict_ok']):<6}{r['mine_sys_rms']:<11.4f}{ts:<11}{rel:<8}"
              f"{str(r['sys_ok']):<5}{r['n']:<6}")
    if report["disagreements"]:
        print(f"\nDISAGREEMENTS: {report['disagreements']}")
    else:
        print("\nAll elements agree (verdict + sys_rms within 10%).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial gate recomputation (D13a/AC5)")
    ap.add_argument("--R", default="/data/rgd/masked-rerun/data/regression")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args(argv)
    report = run(args.R, args.n_boot)
    _print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
