# Spec: abundance regression head and DR19 to DR17 sample-size analysis

Status: agreed in interview (2026-10-02). Not implemented.

## Goal

Add a regression head on top of the frozen `MaskedSpectralAE` latent that
predicts the nine `APOGEE_ABUNDANCE_TARGETS`. Train it on DR19 ASPCAP labels,
then fine-tune it on N M dwarfs carrying DR17 ASPCAP labels. The question is
the smallest N at which the fine-tuned head reproduces the DR19 to DR17
pipeline correction on held-out M dwarfs.

Two claims to support:

1. The frozen latent carries enough chemistry to predict DR19 abundances.
2. A small labelled sample is enough to adapt the head to a different
   pipeline's scale, and we can name that minimum N per element.

## Evidence the design rests on

- Encoder: `src/lightsurf/domain/models/masked_spectral_ae.py:42`. 5 conv
  blocks (1024 to 64), global-average pool (line 80), `Linear(64, 256)`
  (line 65). `embed()` at line 94 encodes the unmasked spectrum.
- Checkpoint: `desk:/data/rgd/masked-rerun/data/embeddings/masked_ae_rerun.pt`,
  pretrained on 39,945 DR19 mwmStar field spectra. Cached latents:
  `masked_latent_field.parquet` (39,945 x 257, `APOGEE_ID` + `z0..z255`) and
  `masked_latent_members.parquet` (994 x 257).
- Preprocessing the checkpoint expects: raw mwmStar `flux`, APO HDU preferred,
  NaN to 0, standardised per star (`scripts/pretrain_embed.py:35-72`). Any new
  spectra must go through exactly this path.
- All 40,000 field IDs and 985 member IDs cross-match into
  `allStar-dr17-synspec_rev1.fits` by `APOGEE_ID` (measured, 100%).
- Using the DR17 cut, 2,478 embedded field rows are M dwarfs, and 906 have all
  nine abundances. `CA_FE` and `TI_FE` are valid for 37%, the rest for 61%.
  `ASPCAPFLAG == 0` holds for 1.2%.
- No regression head exists in the repo today. `deep_models.py:31` is the
  separate supervised CNN-LSTM and is not reused.

## Decisions

| # | Decision |
|---|----------|
| D1 | Pilot on the M dwarfs already embedded, then download and embed more DR19 mwmStar M dwarfs with the frozen checkpoint. DR17 aspcapStar spectra are not used (domain shift). |
| D2 | Map APOGEE_ID to mwmStar URL by rebuilding from `astraAllStarASPCAP-0.6.0.fits.gz` (`sdss_id` gives the `spectra/star/XX/YY/` path). |
| D3 | Base head labels: DR19 astra ASPCAP. Fine-tune and test labels: DR17 ASPCAP. Target mapping is DR19 to DR17. |
| D4 | Predict all nine `APOGEE_ABUNDANCE_TARGETS` with a per-element masked loss. |
| D5 | Base training includes M dwarfs, except a fixed M dwarf pool carved out first. All fine-tune, tuning and test stars come from that pool. |
| D6 | M dwarf = DR19 `TEFF <= 4100` and `LOGG >= 4`. DR19 Teff is the only Teff used as an input anywhere (calibration, stratification). DR17 supplies target abundances only. |
| D7 | Quality: `SNR >= 70`, drop `STAR_BAD`, per-element flag set masks that element only. Same rules on both releases. No `ASPCAPFLAG == 0` requirement. |
| D8 | Encoder frozen throughout; all training runs on cached latents. Head: MLP 256 to 128 to 9. |
| D9 | Fine-tune arms: (a) whole head, (b) last linear layer only. Both start from base weights. |
| D10 | Baselines at every N: post-hoc calibration (per element, DR17 as a 2nd-degree polynomial in base prediction and DR19 Teff, fitted on the same N stars) and ridge on frozen latents (same N stars). The base head unchanged is the N = 0 point. |
| D11 | Metric: `G(N) = (RMSE(0) - RMSE(N)) / (RMSE(0) - RMSE(N_max))`, against DR17 labels on the fixed test set, per element. Minimum N = smallest N with `G >= 0.9` in at least 80% of seeds. Report whether the fine-tune beats post-hoc calibration at that N. |
| D12 | Fixed Teff-stratified test set (300 pilot, 1000 after expansion), used only for scoring. N grid 10, 20, 50, 100, 200, 500, then 1000, 2000, 5000 after expansion. 20 seeds per N, fresh subset per seed. No validation split from N; epochs and weight decay tuned once on a separate tuning pool, then frozen. |
| D13 | Preflight gate (D13a below) runs before any training. Elements without a real correction fall back to a synthetic one. |
| D14 | Code in this worktree, run on `desk` from a git checkout. Outputs under `desk:/data/rgd/masked-rerun/data/regression/`. |

### D13a. Gate and fallback

On the overlap (DR19 M dwarfs with valid labels in both releases), per element:

- `delta = X_DR17 - X_DR19`.
- Fit `delta` as a 2nd-degree polynomial in DR19 Teff. The systematic part is
  the RMS of the fitted curve over the pool; the residual is the scatter.
- Pass if the systematic RMS is at least 0.03 dex and its bootstrap 95%
  interval (1000 resamples) excludes 0. These thresholds are defaults chosen
  here, not by the user, and are recorded in the gate report so they can be
  changed.

Failing elements use a synthetic target instead of DR17:
`y' = y_DR19 + a + b*(T - T0) + c*(T - T0)^2 + eps`, with `T` the DR19 Teff,
`T0` the pool median Teff, coefficients drawn once per element from a fixed
seed and recorded, and `eps ~ N(0, sigma)` where sigma is the median combined
reported DR17 and DR19 error for that element. The report marks every element
as `real` or `synthetic`.

## Stages

Each stage ends in a file a later stage reads, so stages can be rerun alone.

### S1. DR19 catalogue and cross-match
- Download `https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarASPCAP-0.6.0.fits.gz` (1.17 GB) to `desk:/data/rgd/masked-rerun/data/catalogues/`.
- Inspect columns before writing the loader. Verify: identifier columns (`sdss_id`, the APOGEE_ID equivalent), whether abundances are `[X/H]` or `[X/Fe]` (convert to `[X/Fe]` as `X_H - FE_H` if needed), flag and error column names, telescope field, and duplicate rows per star (keep highest SNR, as DR17 handling at `combine_fits.py:42` keeps the first).
- Output: `labels.parquet`, one row per `APOGEE_ID`: DR19 Teff, logg, SNR, nine abundances, errors, masks, `sdss_id`, mwmStar URL; and the same nine from DR17 with masks.

### S2. Preflight gate
- Output: `gate.json` and `gate.png` (delta vs DR19 Teff per element), one verdict per element per D13a.

### S3. Splits
- From the embedded field, select DR19 M dwarfs passing D7. Carve the pool: test (Teff-stratified), tuning, and fine-tune reservoir. Everything else is base training.
- Output: `splits.parquet` (`APOGEE_ID`, `split`), seeded.

### S4. Base head
- New module `src/lightsurf/domain/models/abundance_head.py`: `AbundanceHead(nn.Module)` (256 to 128 to 9), `masked_mse(pred, target, mask)`, target standardisation stored with the weights.
- Train on base split latents with DR19 labels. 20% of the base split for early stopping (large here, so no small-N issue).
- Output: `base_head.pt`, `base_metrics.json` (per element RMSE and R^2 vs DR19 on held-out base stars, and vs DR17 on the M dwarf test set, which is RMSE(0)).

### S5. Hyperparameters on the tuning pool
- Grid over fine-tune epochs and weight decay for both arms; pick by tuning-pool RMSE vs DR17. Output: `finetune_config.json`, then frozen.

### S6. Sweep
- For each N, seed, arm (whole head, last layer, post-hoc calibration, ridge): draw N from the reservoir, fit, score on test vs DR17.
- Output: `sweep.parquet` with columns `n, seed, arm, element, rmse, bias, label_kind (real|synthetic)`. Append per run so a crash loses one run.

### S7. Report
- `G(N)` curves per element and arm (median and 10 to 90% band across seeds), minimum N table, fine-tune vs post-hoc comparison at that N. Output: `report.md` plus figures.

### S8. Expansion (after the pilot passes)
- Select DR19 M dwarfs from `labels.parquet` not yet embedded, download mwmStar via the S1 URLs (reuse the logic in `desk:/data/rgd/masked-rerun/download_field.py`, moved into the repo), embed with `pretrain_embed.py --checkpoint`, append to the latent store, rerun S3 to S7 with test 1000 and the extended N grid. Base training stays fixed unless the user decides otherwise.

## Code layout

- `src/lightsurf/domain/models/abundance_head.py`: head, masked loss, fit and predict helpers.
- `src/lightsurf/domain/services/data/dr19_catalogue.py`: S1 loader and cross-match.
- `scripts/abundance_gate.py`, `scripts/train_abundance_head.py`, `scripts/sample_size_sweep.py`, `scripts/sample_size_report.py`: S2 to S7 entry points, click CLIs like `scripts/pretrain_masked_ae.py`.
- Move `masked_spectral_ae.py`, `pretrain_embed.py` and `download_field.py` provenance into the repo so the desk run uses repo code (the desk copy at `/data/rgd/masked-rerun/` diverged from `scripts/`).
- Tests in `tests/domain/test_abundance_head.py`.

## Acceptance criteria

1. `masked_mse` ignores masked entries: a test with one element fully masked gives the same loss and zero gradient for that output whatever its value.
2. Encoder is frozen: no encoder parameter changes during head training (test compares state dicts).
3. No leakage: test asserts the base split, tuning pool, reservoir and test set are disjoint by `APOGEE_ID`, and no test or tuning ID appears in base training.
4. DR17 Teff is never an input: the calibration and stratification code only read DR19 Teff (test with DR17 Teff set to NaN still runs).
5. Gate report exists before any sweep run and lists every element as `real` or `synthetic` with its thresholds.
6. Base head reports R^2 vs DR19 per element on held-out base stars. Claim 1 is supported only for elements with R^2 >= 0.5 (default threshold, to revisit once numbers exist).
7. `sweep.parquet` holds 20 seeds for every (N, arm, element) cell that fits in the reservoir; cells that do not fit are absent, not padded.
8. Report gives, per element: minimum N per D11 or "not reached within grid", and whether the whole-head fine-tune beats post-hoc calibration at that N.
9. Reproducible: rerunning S3 to S6 with the same seeds reproduces `sweep.parquet` to float tolerance.
10. `pytest` passes.

## Open points, not blocking

- DR19 column names and `[X/H]` vs `[X/Fe]` are unverified until S1 opens the file.
- Pilot reservoir size depends on DR19 M dwarf counts and masks; if it is below 500 after carving test (300) and tuning, the pilot grid stops at the largest N that fits.
- DR19 also ships `astraAllStarSlam-0.6.0` and `astraAllStarMDwarfType-0.6.0`. SLAM gives M dwarf oriented labels and could later serve as the "careful small sample" target instead of DR17. Out of scope here.
