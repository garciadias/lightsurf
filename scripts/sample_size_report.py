"""Sample-size report (S7): G(N) curves, minimum N, fine-tune vs post-hoc.

Reads ``sweep.parquet`` (+ ``base_metrics.json``, ``gate.json``) and writes
``report.md`` plus PNGs under ``$R`` (or ``--report-dir``):

* G(N) per element and arm (median and 10-90% band across seeds), using EVAL's
  ``gap_closed``.
* Minimum N per element via EVAL's ``minimum_n`` (spec D11).
* Fine-tune vs post-hoc at that N: fraction of seeds where ``ft_full`` RMSE <
  ``posthoc`` RMSE (paired by seed).
* base_metrics summary. Synthetic elements are flagged from ``label_kind``.

    python scripts/sample_size_report.py --data-root $R --report-dir $R
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _abundance_data import TARGETS  # noqa: E402

from lightsurf.domain.services.evaluation.sample_size import (  # noqa: E402
    gap_closed, minimum_n,
)

ARMS = ["base_n0", "ft_full", "ft_last", "posthoc", "ridge"]
FT_ARMS = ["ft_full", "ft_last", "posthoc", "ridge"]


def _g_curve(df, element, arm):
    """Per-N (ns, median G, p10, p90) for one element/arm."""
    base = df[(df.arm == "base_n0") & (df.element == element)]["rmse"]
    rmse_0 = float(np.nanmedian(base)) if len(base) else float("nan")
    sub = df[(df.arm == arm) & (df.element == element)]
    if sub.empty:
        return np.array([]), np.array([]), np.array([]), np.array([])
    ns = np.sort(sub.n.unique())
    n_max = int(ns[-1])
    rmse_nmax = float(np.nanmedian(sub[sub.n == n_max]["rmse"]))
    med, lo, hi = [], [], []
    for n in ns:
        rows = sub[sub.n == n]["rmse"].to_numpy(float)
        gs = np.array([gap_closed(r, rmse_0, rmse_nmax) for r in rows])
        gs = gs[np.isfinite(gs)]
        if gs.size == 0:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            med.append(float(np.median(gs)))
            lo.append(float(np.percentile(gs, 10)))
            hi.append(float(np.percentile(gs, 90)))
    return ns, np.array(med), np.array(lo), np.array(hi)


def _ft_vs_posthoc(df, element, n):
    """Fraction of seeds where ft_full rmse < posthoc rmse at N (paired)."""
    f = df[(df.arm == "ft_full") & (df.element == element) & (df.n == n)][["seed", "rmse"]]
    p = df[(df.arm == "posthoc") & (df.element == element) & (df.n == n)][["seed", "rmse"]]
    m = f.merge(p, on="seed", suffixes=("_ft", "_ph")).dropna()
    if m.empty:
        return float("nan"), 0
    frac = float((m["rmse_ft"] < m["rmse_ph"]).mean())
    return frac, int(len(m))


@click.command()
@click.option("--data-root", required=True, type=click.Path(exists=True))
@click.option("--sweep", "sweep_path", default=None, help="defaults to $R/sweep.parquet")
@click.option("--report-dir", default=None, help="defaults to $R")
@click.option("--thresh", default=0.9, show_default=True)
@click.option("--frac-seeds", default=0.8, show_default=True)
def main(data_root, sweep_path, report_dir, thresh, frac_seeds):
    data_root = Path(data_root)
    sweep_path = Path(sweep_path or data_root / "sweep.parquet")
    report_dir = Path(report_dir or data_root)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sweep_path)
    kinds = {e: df[df.element == e]["label_kind"].iloc[0]
             if (df.element == e).any() else "?" for e in TARGETS}
    base_metrics = {}
    bm_path = data_root / "base_metrics.json"
    if bm_path.exists():
        base_metrics = json.loads(bm_path.read_text())

    # --- per-element G(N) figures ---
    png_paths = []
    for e in TARGETS:
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm in FT_ARMS:
            ns, med, lo, hi = _g_curve(df, e, arm)
            if ns.size == 0:
                continue
            ax.plot(ns, med, marker="o", label=arm)
            ax.fill_between(ns, lo, hi, alpha=0.15)
        ax.axhline(thresh, ls="--", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("N (fine-tune stars)"); ax.set_ylabel("G(N) gap closed")
        tag = "SYNTHETIC" if kinds.get(e) == "synthetic" else "real"
        ax.set_title(f"{e}  [{tag}]")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = report_dir / f"gofn_{e}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        png_paths.append(p)

    # --- summary figure: minimum N per element/arm heat-ish table skipped; curves suffice ---

    # --- minimum N table + ft vs posthoc ---
    lines = ["# Sample-size report (DR19 -> DR17)", ""]
    lines.append(f"Sweep rows: {len(df)}  |  elements: {len(TARGETS)}  |  "
                 f"arms: {sorted(df.arm.unique())}")
    lines.append("")
    lines.append("Synthetic elements are flagged; their correction is a fixed "
                 "Teff polynomial, not a real DR19->DR17 delta.")
    lines.append("")
    lines.append("## Minimum N per element (D11: G>=%.2f in >=%.0f%% of seeds)"
                 % (thresh, 100 * frac_seeds))
    lines.append("")
    header = "| element | kind | min N (ft_full) | min N (ft_last) | "
    header += "min N (posthoc) | min N (ridge) | ft_full<posthoc @minN |"
    lines.append(header)
    lines.append("|" + "---|" * 7)
    for e in TARGETS:
        cells = [e, kinds.get(e, "?")]
        minns = {}
        for arm in ["ft_full", "ft_last", "posthoc", "ridge"]:
            try:
                mn = minimum_n(df, e, arm, thresh=thresh, frac_seeds=frac_seeds)
            except ValueError:
                mn = None
            minns[arm] = mn
            cells.append("—" if mn is None else str(mn))
        mn_full = minns["ft_full"]
        if mn_full is not None:
            frac, npair = _ft_vs_posthoc(df, e, mn_full)
            cells.append(f"{frac:.2f} (seeds={npair})" if npair else "—")
        else:
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # --- base metrics summary ---
    lines.append("## Base head metrics (held-out DR19)")
    lines.append("")
    lines.append("| element | kind | R2(DR19) | RMSE19 | RMSE(0) test |")
    lines.append("|---|---|---|---|---|")
    for e in TARGETS:
        el = base_metrics.get("elements", {}).get(e, {})
        r2 = el.get("r2_19_heldout", float("nan"))
        rm = el.get("rmse19_heldout", float("nan"))
        r0 = el.get("rmse0_test_vs_target", float("nan"))
        lines.append(f"| {e} | {kinds.get(e,'?')} | {r2:.3f} | {rm:.3f} | {r0:.3f} |")
    lines.append("")
    lines.append("## Figures")
    for p in png_paths:
        lines.append(f"- ![{p.stem}]({p.name})")
    lines.append("")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines))
    click.echo(f"wrote {report_path} and {len(png_paths)} PNGs")


if __name__ == "__main__":
    main()
