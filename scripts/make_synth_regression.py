"""Generate a synthetic $R-schema dataset for dry-running the abundance scripts.

Writes labels.parquet, finetune_targets.parquet, splits.parquet, gate.json and a
latents parquet whose schema matches the real DATA artifacts, so every S4-S7 CLI
can be exercised end to end (GPU path + resume) without the real data.

    python scripts/make_synth_regression.py --out /data/.../regression_synth \
        --n-stars 4000
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS

TARGETS = list(APOGEE_ABUNDANCE_TARGETS)


@click.command()
@click.option("--out", required=True, type=click.Path())
@click.option("--n-stars", default=4000, show_default=True)
@click.option("--test-frac", default=0.12, show_default=True)
@click.option("--tune-frac", default=0.05, show_default=True)
@click.option("--reservoir-frac", default=0.33, show_default=True)
@click.option("--seed", default=0, show_default=True)
def main(out, n_stars, test_frac, tune_frac, reservoir_frac, seed):
    rng = np.random.default_rng(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ids = [f"SYN{i:05d}" for i in range(n_stars)]

    Z = rng.normal(size=(n_stars, 256)).astype("float32")
    lat = pd.concat(
        [pd.DataFrame({"APOGEE_ID": ids}),
         pd.DataFrame(Z, columns=[f"z{d}" for d in range(256)])], axis=1)
    lat.to_parquet(out / "masked_latent_field.parquet", index=False)

    W = (rng.normal(size=(256, len(TARGETS))) * 0.1).astype("float32")
    y19 = Z @ W + rng.normal(scale=0.03, size=(n_stars, len(TARGETS)))
    teff = rng.uniform(3200, 4100, size=n_stars)
    t = (teff - 3650) / 500
    # per-element DR19->DR17 shift: mostly constant + small quadratic in teff
    shift = (0.4 + 0.12 * t + 0.05 * t**2)[:, None] + rng.normal(0, 0.1, size=(1, len(TARGETS)))
    y17 = y19 + shift

    lab = {"APOGEE_ID": ids, "sdss_id": np.arange(n_stars, dtype="int64"),
           "teff19": teff.astype("float32"), "logg19": np.full(n_stars, 4.6, "float32"),
           "snr19": np.full(n_stars, 110.0, "float32"), "star_bad19": np.zeros(n_stars, bool),
           "snr17": np.full(n_stars, 110.0, "float32"), "star_bad17": np.zeros(n_stars, bool),
           "is_mdwarf": np.ones(n_stars, bool), "passes_quality": np.ones(n_stars, bool)}
    ftt = {"APOGEE_ID": ids}
    for j, e in enumerate(TARGETS):
        ok = rng.random(n_stars) > 0.1  # ~10% masked per element
        lab[f"{e}_19"] = y19[:, j].astype("float32")
        lab[f"{e}_19_err"] = np.full(n_stars, 0.02, "float32")
        lab[f"{e}_19_ok"] = ok
        lab[f"{e}_17"] = y17[:, j].astype("float32")
        lab[f"{e}_17_err"] = np.full(n_stars, 0.02, "float32")
        lab[f"{e}_17_ok"] = ok
        ftt[f"{e}_t"] = y17[:, j].astype("float32")
        ftt[f"{e}_t_ok"] = ok
    pd.DataFrame(lab).to_parquet(out / "labels.parquet", index=False)
    pd.DataFrame(ftt).to_parquet(out / "finetune_targets.parquet", index=False)

    perm = rng.permutation(n_stars)
    n_test = int(n_stars * test_frac)
    n_tune = int(n_stars * tune_frac)
    n_res = int(n_stars * reservoir_frac)
    split = np.array(["base"] * n_stars, dtype=object)
    split[perm[:n_test]] = "test"
    split[perm[n_test:n_test + n_tune]] = "tune"
    split[perm[n_test + n_tune:n_test + n_tune + n_res]] = "reservoir"
    pd.DataFrame({"APOGEE_ID": ids, "split": split}).to_parquet(
        out / "splits.parquet", index=False)

    gate = {"thresholds": {"min_sys_rms": 0.03, "n_boot": 1000, "ci": 0.95},
            "elements": {e: {"verdict": ("synthetic" if j in (2, 8) else "real"),
                             "n": 200, "sys_rms": 0.12, "ci": [0.08, 0.18],
                             "poly_coef": [0.4, 0.12, 0.05], "resid_rms": 0.05,
                             "synthetic": (None if j not in (2, 8) else
                                           {"a": 0.4, "b": 0.12, "c": 0.05,
                                            "T0": 3650, "sigma": 0.05, "seed": 1})}
                         for j, e in enumerate(TARGETS)}}
    (out / "gate.json").write_text(json.dumps(gate, indent=2))

    counts = pd.Series(split).value_counts().to_dict()
    click.echo(f"wrote synthetic $R to {out}  splits={counts}")


if __name__ == "__main__":
    main()
