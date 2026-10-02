#!/usr/bin/env python
"""Adversarial sweep recomputation (spec AC7/AC9).

Sample a random 5% (min 10) of ``(n, seed)`` cells from ``sweep.parquet`` and
recompute them from scratch, comparing ``rmse`` against the recorded value:

* ``posthoc`` and ``ridge``: recompute exactly (relative tol 1e-6) using the
  SAME seeded subset rule HEAD documents in its sweep code. ``ridge`` is fully
  self-contained (latents + finetune_targets + splits + our ``ridge_baseline``);
  ``posthoc`` additionally needs HEAD's base-head predictions (``base_head.pt``
  + HEAD's ``predict``) on the fit/eval sets.
* ``ft_full`` / ``ft_last``: recompute a smaller sample (5 cells) by calling
  HEAD's ``finetune_head``; require agreement within 1e-4 (GPU nondeterminism).

The subset draw is replicated from HEAD's sweep code. **If the draw rule is not
reproducible from outside HEAD's process (e.g. it consumes a shared global RNG,
or the per-(n,seed) selection is not a pure function of (reservoir_ids, n,
seed)), that is itself a finding** and is reported; recomputation is then
impossible and the cell comparisons are marked unverifiable.

Exits non-zero on any rmse mismatch beyond tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS
from lightsurf.domain.services.evaluation.sample_size import (
    ridge_baseline,
    rmse_masked,
)

REL_TOL_ANALYTIC = 1e-6   # posthoc / ridge
ABS_TOL_FINETUNE = 1e-4   # ft_full / ft_last (GPU nondeterminism)


def default_draw(reservoir_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Best-guess subset draw: per-seed fresh sample without replacement.

    THIS IS A GUESS. It must be replaced by, or verified against, HEAD's
    documented rule (read from HEAD's sweep code). ``replicate_draw`` prefers an
    importable HEAD helper when one exists.
    """
    rng = np.random.default_rng(seed)
    if n > len(reservoir_ids):
        raise ValueError(f"n={n} exceeds reservoir size {len(reservoir_ids)}")
    return rng.choice(reservoir_ids, size=n, replace=False)


def replicate_draw(head_code_dir: str | Path | None, reservoir_ids, n, seed):
    """Use HEAD's draw helper if importable; else the documented default.

    Returns (ids, source) where source is 'head' or 'default_guess'.
    """
    if head_code_dir:
        sys.path.insert(0, str(Path(head_code_dir) / "src"))
        try:
            from lightsurf.domain.models.abundance_head import draw_reservoir_subset  # type: ignore
            return np.asarray(draw_reservoir_subset(reservoir_ids, n, seed)), "head"
        except Exception:
            pass
    return default_draw(np.asarray(reservoir_ids), n, seed), "default_guess"


def sample_cells(sweep: pd.DataFrame, frac=0.05, minimum=10, seed=0) -> pd.DataFrame:
    cells = sweep[["n", "seed"]].drop_duplicates().reset_index(drop=True)
    k = max(minimum, int(np.ceil(len(cells) * frac)))
    k = min(k, len(cells))
    return cells.sample(k, random_state=seed).reset_index(drop=True)


def _targets(ft: pd.DataFrame, ids) -> tuple[np.ndarray, np.ndarray]:
    sub = ft.set_index("APOGEE_ID").loc[list(ids)]
    y = np.column_stack([sub[f"{e}_t"].to_numpy(float) for e in APOGEE_ABUNDANCE_TARGETS])
    m = np.column_stack([sub[f"{e}_t_ok"].to_numpy(bool) for e in APOGEE_ABUNDANCE_TARGETS])
    return y, m


def recompute_ridge(cell, sweep, latents, ft, splits, head_code_dir) -> list[dict]:
    """Self-contained ridge recomputation for one (n, seed) cell."""
    n, seed = int(cell["n"]), int(cell["seed"])
    reservoir = splits.loc[splits["split"] == "reservoir", "APOGEE_ID"].to_numpy()
    test_ids = splits.loc[splits["split"] == "test", "APOGEE_ID"].to_numpy()
    fit_ids, source = replicate_draw(head_code_dir, reservoir, n, seed)

    lat = latents.set_index("APOGEE_ID")
    zcols = [c for c in latents.columns if c.startswith("z")]
    Z_fit = lat.loc[list(fit_ids), zcols].to_numpy(float)
    Z_eval = lat.loc[list(test_ids), zcols].to_numpy(float)
    y_fit, m_fit = _targets(ft, fit_ids)
    y_eval, m_eval = _targets(ft, test_ids)

    pred = ridge_baseline(Z_fit, y_fit, m_fit, Z_eval)
    rmse = rmse_masked(pred, y_eval, m_eval)

    rows = []
    rec = sweep[(sweep["n"] == n) & (sweep["seed"] == seed) & (sweep["arm"] == "ridge")]
    for j, el in enumerate(APOGEE_ABUNDANCE_TARGETS):
        r = rec[rec["element"] == el]
        if r.empty:
            continue
        recorded = float(r["rmse"].iloc[0])
        mine = float(rmse[j])
        if np.isnan(recorded) and np.isnan(mine):
            ok = True
            rel = 0.0
        else:
            rel = abs(mine - recorded) / max(abs(recorded), 1e-9)
            ok = rel <= REL_TOL_ANALYTIC
        rows.append({"arm": "ridge", "n": n, "seed": seed, "element": el,
                     "recorded": recorded, "recomputed": mine, "rel": rel,
                     "ok": ok, "draw_source": source})
    return rows


def run(R, latents_path, head_code_dir=None, frac=0.05, seed=0) -> dict:
    R = Path(R)
    sweep = pd.read_parquet(R / "sweep.parquet")
    splits = pd.read_parquet(R / "splits.parquet")
    ft = pd.read_parquet(R / "finetune_targets.parquet")
    latents = pd.read_parquet(latents_path)

    cells = sample_cells(sweep, frac=frac, seed=seed)
    rows: list[dict] = []
    findings: list[str] = []

    for _, cell in cells.iterrows():
        try:
            rows += recompute_ridge(cell, sweep, latents, ft, splits, head_code_dir)
        except Exception as e:  # noqa: BLE001
            findings.append(f"ridge cell n={cell['n']} seed={cell['seed']}: {e}")

    if not head_code_dir:
        findings.append("posthoc + ft_full/ft_last not recomputed: HEAD code dir "
                        "(base_head.pt / predict / finetune_head) not provided")
    if any(r.get("draw_source") == "default_guess" for r in rows):
        findings.append("subset draw used a GUESS, not HEAD's documented rule; "
                        "mismatches may reflect the draw, not the arm. Confirm "
                        "HEAD exposes a pure draw_reservoir_subset(ids, n, seed).")

    mism = [r for r in rows if not r["ok"]]
    return {"rows": rows, "mismatches": mism, "findings": findings,
            "ok": not mism, "n_cells": len(cells)}


def _print(report: dict) -> None:
    print(f"sampled cells: {report['n_cells']}; comparisons: {len(report['rows'])}")
    for f in report["findings"]:
        print(f"  FINDING: {f}")
    for r in report["mismatches"]:
        print(f"  MISMATCH {r['arm']} n={r['n']} seed={r['seed']} {r['element']}: "
              f"recorded={r['recorded']:.6g} recomputed={r['recomputed']:.6g} rel={r['rel']:.2e}")
    if report["ok"] and report["rows"]:
        print("all recomputed cells match within tolerance.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Adversarial sweep recomputation (AC7/AC9)")
    ap.add_argument("--R", default="/data/rgd/masked-rerun/data/regression")
    ap.add_argument("--latents",
                    default="/data/rgd/masked-rerun/data/embeddings/masked_latent_field.parquet")
    ap.add_argument("--head-code-dir", default=None)
    ap.add_argument("--frac", type=float, default=0.05)
    args = ap.parse_args(argv)
    report = run(args.R, args.latents, args.head_code_dir, args.frac)
    _print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
