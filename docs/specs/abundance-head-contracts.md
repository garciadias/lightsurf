# Contracts between workstreams (abundance head)

Companion to `abundance-head-dr19-to-dr17.md`. These file schemas and function
signatures are fixed so the three workstreams can be built in parallel. Change
them only through the judge (the orchestrating session).

## Locations

- Data root on desk: `/data/rgd/masked-rerun/data/regression/` (call it `$R`).
- Latents (read-only input): `/data/rgd/masked-rerun/data/embeddings/masked_latent_field.parquet`,
  columns `APOGEE_ID, z0..z255`.
- Catalogue download: `/data/rgd/masked-rerun/data/catalogues/astraAllStarASPCAP-0.6.0.fits.gz`.
- Run code on desk from a git clone at `/data/rgd/lightsurf-abund/<workstream>/`,
  never from copies. Python: `/data/rgd/masked-rerun/.venv/bin/python` with
  `PYTHONPATH=src` (CUDA torch 2.11 for the RTX 5090). Install missing pure
  Python deps into that venv with `uv pip install --python /data/rgd/masked-rerun/.venv/bin/python <pkg>`.
- `TARGETS` = `lightsurf.constants.APOGEE_ABUNDANCE_TARGETS` (9 elements, that order).

## Files

### `$R/labels.parquet` (workstream DATA)
One row per `APOGEE_ID` present in the latents file.

| column | type | meaning |
|---|---|---|
| APOGEE_ID | str | key, same string as in the latents parquet |
| sdss_id | int64 | DR19 id |
| teff19, logg19, snr19 | float | DR19 astra ASPCAP |
| star_bad19 | bool | DR19 STAR_BAD bit set |
| `{E}_19`, `{E}_19_err` | float | DR19 [X/Fe] (FE_H is [Fe/H]); NaN if invalid |
| `{E}_19_ok` | bool | valid and per-element flag clear |
| `{E}_17`, `{E}_17_err` | float | DR17 same quantities |
| `{E}_17_ok` | bool | as above, DR17 |
| snr17, star_bad17 | float, bool | DR17 |
| is_mdwarf | bool | `teff19 <= 4100 and logg19 >= 4` |
| passes_quality | bool | `snr19 >= 70 and not star_bad19` (and DR17 equivalents when the DR17 label is used) |

### `$R/gate.json` (DATA)
`{"thresholds": {"min_sys_rms": 0.03, "n_boot": 1000, "ci": 0.95},
  "elements": {E: {"verdict": "real"|"synthetic", "n": int, "sys_rms": float,
  "ci": [lo, hi], "poly_coef": [c0, c1, c2], "resid_rms": float,
  "synthetic": null | {"a","b","c","T0","sigma","seed"}}}}`

### `$R/finetune_targets.parquet` (DATA)
`APOGEE_ID, {E}_t (float), {E}_t_ok (bool)` for every M dwarf in the pool.
`{E}_t` is the DR17 value for `real` elements and the synthetic Teff polynomial
target for `synthetic` ones (D13a). Downstream code never re-derives targets.

### `$R/splits.parquet` (DATA)
`APOGEE_ID, split` with split in `base | tune | reservoir | test`. Pool = M dwarfs
passing quality with at least one `{E}_t_ok`. Test is stratified in teff19.
Sizes in the pilot: test 300, tune 100, reservoir the rest of the pool. Seed 0.

### `$R/base_head.pt`, `$R/base_metrics.json` (HEAD)
### `$R/finetune_config.json` (HEAD), `{arm: {"epochs": int, "lr": float, "weight_decay": float}}`
### `$R/sweep.parquet` (HEAD)
`n, seed, arm, element, rmse, bias, n_used, label_kind`; arm in
`base_n0 | ft_full | ft_last | posthoc | ridge`. Appended per run.

## Library API (HEAD) in `src/lightsurf/domain/models/abundance_head.py`

```python
class AbundanceHead(nn.Module):          # 256 -> 128 -> 9, ReLU between
    def __init__(self, in_dim=256, hidden=128, out_dim=9): ...
def masked_mse(pred, target, mask) -> Tensor   # mean over mask==True entries only
def fit_head(Z, Y, M, *, epochs, lr, weight_decay, seed, val_frac=0.2,
             patience=15, device=None) -> tuple[AbundanceHead, Standardiser]
def finetune_head(head, std, Z, Y, M, *, arm: "full"|"last", epochs, lr,
                  weight_decay, seed, device=None) -> AbundanceHead  # deep copy; no val split
def predict(head, std, Z, device=None) -> np.ndarray   # physical units
def draw_reservoir_subset(ids, n, seed) -> np.ndarray  # pure; the sweep's per-cell draw (F5)
```
`Z` float32 (n, 256), `Y` float32 (n, 9) with NaN allowed, `M` bool (n, 9).
`Standardiser` stores per-element mean/std of the base training targets and is
saved with the weights.

## Evaluation API (EVAL) in `src/lightsurf/domain/services/evaluation/sample_size.py`

```python
def rmse_masked(pred, y, mask) -> np.ndarray            # per element (9,)
def gap_closed(rmse_n, rmse_0, rmse_nmax) -> float
def minimum_n(df, element, arm, thresh=0.9, frac_seeds=0.8) -> int | None
def posthoc_calibration(pred_fit, teff_fit, y_fit, mask_fit, pred_eval, teff_eval) -> np.ndarray
    # per element: y ~ 2nd-degree polynomial in (pred, teff), incl. cross term
def ridge_baseline(Z_fit, y_fit, mask_fit, Z_eval, alpha=None) -> np.ndarray
```
HEAD's sweep imports these; it does not reimplement them.
