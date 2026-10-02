"""S2 preflight gate for the DR19->DR17 abundance correction (spec D13a).

Per element, on the M dwarf pool (``is_mdwarf & passes_quality`` with both the
DR17 and DR19 labels valid for that element):

  delta = X_DR17 - X_DR19
  fit delta as a 2nd-degree polynomial in centered DR19 Teff (T - T0)
  sys_rms  = RMS of the fitted curve over the pool  (systematic part)
  resid_rms = RMS of the residuals                  (scatter)

Bootstrap (1000 seeded resamples) gives the 95% CI of sys_rms, of the mean
offset, and of the linear/quadratic coefficients. A permutation null (1000
seeded draws shuffling delta across Teff) gives the 95th percentile of sys_rms
under no real Teff structure.

Pass rule (stricter than the bare D13a default, as instructed):
  sys_rms >= 0.03  AND
  ( sys_rms > null p95   OR   the 95% CI of the mean offset excludes 0 )

sys_rms is always >= 0 so "CI excludes 0" is weak on its own; the permutation
null and the mean-offset CI give the real test of signal.

Failing elements fall back to a synthetic Teff-polynomial target (D13a):
  y' = y_DR19 + a + b*(T-T0) + c*(T-T0)^2 + eps,  eps ~ N(0, sigma)
with sigma = median sqrt(err17^2 + err19^2) and a,b,c drawn once from a fixed
seed then rescaled so the synthetic correction's sys_rms matches the median
sys_rms of the passing elements (or 0.05 dex if none pass).
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS as TARGETS

MIN_SYS_RMS = 0.03
N_BOOT = 1000
CI = 0.95
SEED = 0
DEFAULT_SYNTH_SYS_RMS = 0.05


def fit_curve(t_centered: np.ndarray, delta: np.ndarray):
    """Return (coef_ascending[c0,c1,c2], yhat) for a 2nd-degree poly fit."""
    coef_desc = np.polyfit(t_centered, delta, 2)  # [c2, c1, c0]
    yhat = np.polyval(coef_desc, t_centered)
    coef_asc = coef_desc[::-1].copy()  # [c0, c1, c2]
    return coef_asc, yhat


def sys_rms_of(t_centered: np.ndarray, delta: np.ndarray) -> float:
    _, yhat = fit_curve(t_centered, delta)
    return float(np.sqrt(np.mean(yhat**2)))


def gate_element(t_centered, delta, rng):
    """Full gate statistics for one element."""
    coef, yhat = fit_curve(t_centered, delta)
    sys_rms = float(np.sqrt(np.mean(yhat**2)))
    resid_rms = float(np.sqrt(np.mean((delta - yhat) ** 2)))
    mean_off = float(np.mean(delta))
    n = len(delta)

    lo_q, hi_q = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    boot_sys, boot_off, boot_c1, boot_c2 = [], [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        tc, dd = t_centered[idx], delta[idx]
        c, yh = fit_curve(tc, dd)
        boot_sys.append(np.sqrt(np.mean(yh**2)))
        boot_off.append(np.mean(dd))
        boot_c1.append(c[1])
        boot_c2.append(c[2])
    ci_sys = [float(np.percentile(boot_sys, lo_q)), float(np.percentile(boot_sys, hi_q))]
    ci_off = [float(np.percentile(boot_off, lo_q)), float(np.percentile(boot_off, hi_q))]
    ci_c1 = [float(np.percentile(boot_c1, lo_q)), float(np.percentile(boot_c1, hi_q))]
    ci_c2 = [float(np.percentile(boot_c2, lo_q)), float(np.percentile(boot_c2, hi_q))]

    # permutation null: shuffle delta across Teff
    null_sys = []
    for _ in range(N_BOOT):
        perm = rng.permutation(delta)
        null_sys.append(sys_rms_of(t_centered, perm))
    null_p95 = float(np.percentile(null_sys, 95))

    offset_excludes_0 = (ci_off[0] > 0) or (ci_off[1] < 0)
    verdict_real = (sys_rms >= MIN_SYS_RMS) and ((sys_rms > null_p95) or offset_excludes_0)

    return {
        "n": int(n),
        "sys_rms": sys_rms,
        "resid_rms": resid_rms,
        "mean_offset": mean_off,
        "ci": ci_sys,  # contract: bootstrap CI of sys_rms
        "ci_mean_offset": ci_off,
        "ci_coef_linear": ci_c1,
        "ci_coef_quad": ci_c2,
        "poly_coef": [float(c) for c in coef],  # [c0,c1,c2] in centered Teff
        "null_sys_rms_p95": null_p95,
        "offset_excludes_0": bool(offset_excludes_0),
        "verdict": "real" if verdict_real else "synthetic",
    }


def make_synthetic(t_centered, y19, err17, err19, target_sys_rms, seed):
    """Draw & rescale a,b,c so the correction sys_rms ~ target; return params+target."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(0, 0.05), rng.normal(0, 1e-4), rng.normal(0, 1e-7)
    corr = a + b * t_centered + c * t_centered**2
    cur = np.sqrt(np.mean(corr**2))
    scale = target_sys_rms / cur if cur > 0 else 1.0
    a, b, c = a * scale, b * scale, c * scale
    comb = np.sqrt(np.asarray(err17, float) ** 2 + np.asarray(err19, float) ** 2)
    sigma = float(np.nanmedian(comb))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.05
    eps = rng.normal(0, sigma, size=len(t_centered))
    y_syn = y19 + a + b * t_centered + c * t_centered**2 + eps
    params = {"a": float(a), "b": float(b), "c": float(c), "sigma": sigma, "seed": int(seed)}
    return y_syn, params


@click.command()
@click.option("--labels", required=True)
@click.option("--out-dir", required=True)
def main(labels, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(labels)
    pool = df[df["is_mdwarf"] & df["passes_quality"]].copy()
    T0 = float(np.median(pool["teff19"].to_numpy()))

    rng = np.random.default_rng(SEED)
    elements = {}
    for E in TARGETS:
        m = pool[f"{E}_17_ok"] & pool[f"{E}_19_ok"]
        sub = pool[m]
        if len(sub) < 10:
            elements[E] = {"n": int(len(sub)), "verdict": "synthetic", "sys_rms": None,
                           "ci": None, "poly_coef": None, "resid_rms": None,
                           "note": "too few overlap stars", "synthetic": None}
            continue
        t = sub["teff19"].to_numpy() - T0
        delta = sub[f"{E}_17"].to_numpy() - sub[f"{E}_19"].to_numpy()
        elements[E] = gate_element(t, delta, rng)

    # synthetic target sys_rms from passing elements
    passing = [v["sys_rms"] for v in elements.values() if v["verdict"] == "real" and v["sys_rms"]]
    target_sys = float(np.median(passing)) if passing else DEFAULT_SYNTH_SYS_RMS

    # build finetune targets over the whole M dwarf pool
    ft = pd.DataFrame({"APOGEE_ID": pool["APOGEE_ID"].to_numpy()})
    tc_pool = pool["teff19"].to_numpy() - T0
    for i, E in enumerate(TARGETS):
        info = elements[E]
        if info["verdict"] == "real":
            ft[f"{E}_t"] = pool[f"{E}_17"].to_numpy()
            ft[f"{E}_t_ok"] = pool[f"{E}_17_ok"].to_numpy()
            info["synthetic"] = None
        else:
            y19 = pool[f"{E}_19"].to_numpy()
            y_syn, params = make_synthetic(
                tc_pool, y19, pool[f"{E}_17_err"].to_numpy(),
                pool[f"{E}_19_err"].to_numpy(), target_sys, seed=SEED + 100 + i)
            params["T0"] = T0
            ft[f"{E}_t"] = y_syn
            # synthetic target defined where DR19 base label is valid
            ft[f"{E}_t_ok"] = pool[f"{E}_19_ok"].to_numpy()
            info["synthetic"] = params
    ft.to_parquet(out / "finetune_targets.parquet", index=False)

    gate = {
        "thresholds": {
            "min_sys_rms": MIN_SYS_RMS, "n_boot": N_BOOT, "ci": CI,
            "pass_rule": "sys_rms>=0.03 AND (sys_rms>null_p95 OR mean_offset CI excludes 0)",
            "null": "permutation: shuffle delta across teff, 1000 draws, p95",
            "T0_teff": T0, "poly_basis": "centered Teff (T - T0_teff)",
            "synthetic_target_sys_rms": target_sys,
        },
        "elements": elements,
    }
    (out / "gate.json").write_text(json.dumps(gate, indent=2))

    # plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, E in zip(axes.ravel(), TARGETS):
        m = pool[f"{E}_17_ok"] & pool[f"{E}_19_ok"]
        sub = pool[m]
        info = elements[E]
        if len(sub) >= 10:
            t = sub["teff19"].to_numpy()
            delta = sub[f"{E}_17"].to_numpy() - sub[f"{E}_19"].to_numpy()
            ax.scatter(t, delta, s=4, alpha=0.3, color="steelblue")
            xs = np.linspace(t.min(), t.max(), 100)
            coef = info["poly_coef"]  # ascending, centered
            ys = coef[0] + coef[1] * (xs - T0) + coef[2] * (xs - T0) ** 2
            ax.plot(xs, ys, "r-", lw=2)
            ax.set_title(f"{E} [{info['verdict']}] n={info['n']} "
                         f"sys={info['sys_rms']:.3f} p95={info['null_sys_rms_p95']:.3f}")
        else:
            ax.set_title(f"{E} [synthetic] n<10")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("DR19 Teff")
        ax.set_ylabel(f"{E}_17 - {E}_19")
    fig.tight_layout()
    fig.savefig(out / "gate.png", dpi=110)

    print(f"pool (M dwarf & passes_quality): {len(pool)}")
    print(f"{'elem':8s} {'n':>6s} {'mean_off':>9s} {'sys_rms':>8s} {'null_p95':>9s} verdict")
    for E in TARGETS:
        v = elements[E]
        if v.get("sys_rms"):
            print(f"{E:8s} {v['n']:6d} {v['mean_offset']:9.4f} {v['sys_rms']:8.4f} "
                  f"{v['null_sys_rms_p95']:9.4f} {v['verdict']}")
        else:
            print(f"{E:8s} {v['n']:6d} {'-':>9s} {'-':>8s} {'-':>9s} {v['verdict']}")
    print(f"synthetic_target_sys_rms={target_sys:.4f}")


if __name__ == "__main__":
    main()
