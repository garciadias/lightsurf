"""S3 splits for the abundance sample-size study (contract splits.parquet).

Pool = M dwarfs passing quality with at least one ``{E}_t_ok`` (from
finetune_targets.parquet). Within the pool: 300 test stars stratified in
teff19 deciles, 100 tune stars, the rest reservoir. Every other star in
labels.parquet (M dwarf or not) is ``base``. Seed 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lightsurf.constants import APOGEE_ABUNDANCE_TARGETS as TARGETS

SEED = 0
N_TEST = 300
N_TUNE = 100


@click.command()
@click.option("--labels", required=True)
@click.option("--finetune-targets", required=True)
@click.option("--out-dir", required=True)
def main(labels, finetune_targets, out_dir):
    out = Path(out_dir)
    df = pd.read_parquet(labels)
    ft = pd.read_parquet(finetune_targets)

    ok_cols = [f"{E}_t_ok" for E in TARGETS]
    ft["_any_ok"] = ft[ok_cols].any(axis=1)
    pool_ids = set(ft.loc[ft["_any_ok"], "APOGEE_ID"])

    # pool must also be M dwarf & passes_quality (finetune is already the pool,
    # but re-assert against labels for safety).
    md_pass = set(df.loc[df["is_mdwarf"] & df["passes_quality"], "APOGEE_ID"])
    pool_ids = pool_ids & md_pass
    pool = df[df["APOGEE_ID"].isin(pool_ids)].copy()

    # stratified test in teff19 deciles
    deciles = pd.qcut(pool["teff19"], 10, labels=False, duplicates="drop")
    n_test = min(N_TEST, len(pool) - N_TUNE - 1)
    rest_idx, test_idx = train_test_split(
        np.arange(len(pool)), test_size=n_test, random_state=SEED, stratify=deciles)
    test_ids = set(pool.iloc[test_idx]["APOGEE_ID"])

    rng = np.random.default_rng(SEED)
    rest_ids = pool.iloc[rest_idx]["APOGEE_ID"].to_numpy()
    rng.shuffle(rest_ids)
    n_tune = min(N_TUNE, len(rest_ids))
    tune_ids = set(rest_ids[:n_tune])
    reservoir_ids = set(rest_ids[n_tune:])

    def label(aid):
        if aid in test_ids:
            return "test"
        if aid in tune_ids:
            return "tune"
        if aid in reservoir_ids:
            return "reservoir"
        return "base"

    splits = pd.DataFrame({"APOGEE_ID": df["APOGEE_ID"].to_numpy()})
    splits["split"] = [label(a) for a in splits["APOGEE_ID"]]
    splits.to_parquet(out / "splits.parquet", index=False)

    counts = splits["split"].value_counts().to_dict()
    # reservoir per-element availability
    res = ft[ft["APOGEE_ID"].isin(reservoir_ids)]
    per_elem = {E: int(res[f"{E}_t_ok"].sum()) for E in TARGETS}

    report = {
        "seed": SEED,
        "pool_size": int(len(pool)),
        "counts": {k: int(v) for k, v in counts.items()},
        "reservoir_per_element_ok": per_elem,
    }
    (out / "splits_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # sanity: disjoint & covering
    assert len(splits) == len(df)
    assert splits["split"].isin(["base", "tune", "reservoir", "test"]).all()
    assert test_ids.isdisjoint(tune_ids) and test_ids.isdisjoint(reservoir_ids)
    assert tune_ids.isdisjoint(reservoir_ids)
    print("OK disjoint & covering")


if __name__ == "__main__":
    main()
