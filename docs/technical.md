# Technical reference

This document explains how the pipeline is built and why. For commands see
`usage.md`; for experiment history see `results.md`.

## 1. Problem and data

- Competition: Kaggle Playground S6E9, "Predicting Electric Vehicle
  Purchases".
- Target: `Will_Buy_EV` (`Yes`/`No`), encoded as 1/0.
- Metric: ROC AUC.
- Train: 668,665 rows, 13 feature columns plus `id` and target; 116,779
  positives (17.5%). Test: 286,571 rows. No missing values in the
  competition files.
- Data is synthetic, derived from a 10,000-row CC0 survey dataset. Digit
  and value-frequency artifacts left by the generator turned out to be
  usable signal.

## 2. Module map

| Module | Responsibility |
|---|---|
| `config.py` | Paths, target/id names, fold count, seeds, thread cap. |
| `utils.py` | Seeding, stratified folds, submission validation. |
| `features.py` | Two feature spaces and fold-local target encoding. |
| `train_gbdt.py` | Cross-validated training for LGBM/XGB/CatBoost. |
| `tune_xgb.py`, `tune_lgbm.py`, `tune_cat.py` | Single-fold Optuna parameter search. |
| `ensemble.py` | Simple OOF mean / rank / AUC-weighted blends. |
| `stack.py` | Leak-free linear stacking (logreg, ridge, NNLS). |
| `stack_nl.py` | Nonlinear stacking experiment (histogram GBDT meta-model). |
| `recipe.py`, `recipe_v2.py`, `cell_prior.py`, `mlp_prior.py` | Logistic/MLP prior experiments (distinct from the `--features v2` space), all held. |
| `mlp.py` | Entity-embedding MLP diversity model, held. |
| `eda.py` | Initial exploratory pass. |

## 3. Data flow

`train_gbdt.run_cv` is the core loop:

1. Read train (and optionally test), split off the target, keep `id`.
2. Concatenate train and test **without the target** and featurize jointly.
   All feature steps are target-free, so this leaks nothing and guarantees
   identical category sets and bin edges on both sides.
3. Convert non-numeric, non-category columns to pandas `category` (native
   categorical handling in all three libraries).
4. Optionally add fold-local target encoding (v1 `--te` or `--te-triple`).
   Encoders are fitted on the training fold only; validation and test are
   pure transforms.
5. For each fold: fit, predict validation (OOF) and test (averaged over
   folds), print fold AUC.
6. Save `outputs/oof_<model><tag>_seed<seed>.npy`,
   `outputs/test_...npy` and `submissions/sub_...csv`.
7. Print pooled OOF AUC, which is the only number used for selection.

Tags are compositional: `_dv` (v2 features), `_te`/`_te3` (target
encoding), `_tuned` (params JSON), `_rec` (prior), plus `--out-suffix`.
This prevents a new run from overwriting an existing artifact.

## 4. Feature spaces

### v1 (legacy, 85 model features, 18 categorical)

`build_features` composes, in order:

- `base_clean`: cast string columns to `category`.
- `add_ev_features`: missing-value flags and median imputation (no-op on the
  competition data), charging totals/differences, income/commute/car and
  charging/car ratios, products, and 10-quantile bins of age, income and
  commute.
- `add_crosses`: string concatenation of 12 categorical pairs (nine pure
  categorical combinations such as subsidy x concern and city x car, plus
  three binned-numeric crosses such as age bin x subsidy).
- `add_group_stats`: per-category mean of four numerics, computed on the
  joint frame (target-free).
- `add_freq`: joint-frame value frequency for the cross and singleton
  columns used by target encoding.

The current revision has 85 model features, or 86 columns including `id`.
The S3-era base members were built from an earlier 62-feature revision that
was never committed, so they cannot be regenerated from this code (see
section 10). Do not modify this space in place.

### v2 (recipe feature space, 70 model features, 6 categorical)

`build_features_v2` composes:

- `base_clean`.
- `add_dvorkin_interactions`: multiplier and ratio crosses around
  environmental concern, subsidy, income and commute. Subsidy, home charging
  and high range anxiety are binarized before multiplying.
- `add_digit_features`: for each of seven numeric columns, eight features
  `floor(x / 10^k) mod 10` for `k = -4..3`. These expose rounding and
  quantization artifacts left by the generator.
- `add_freq_all`: normalized value frequency for every column, including the
  digit features (this is a large block of low-cardinality features).
- `drop_const_corr`: removes constant columns and numeric columns that are
  perfectly correlated with an earlier one.

`target_encode_triple_fold` adds 39 fold-local columns (13 source columns
times three regression smoothings: `auto`, 10, 100). It uses
`sklearn.preprocessing.TargetEncoder` with `shuffle=True, cv=5`, fitted on
the outer training fold only. The internal CV is what makes the training
transform honest; validation and test are transformed directly.

Design constraints:

- Everything except target encoding is target-free, so joint train+test
  featurization is safe. The built frame has 71 columns including `id`.
- v2 is opt-in via `--features v2`; v1 remains the default.
- v2 is materially more expensive (about 18 s for joint featurization versus
  6 s for v1) and larger in memory.

## 5. Cross-validation and leak prevention

- `make_folds` uses `StratifiedKFold(shuffle=True, random_state=seed)`.
  Class balance makes stratification mandatory.
- The legacy-space base models use seeds 42, 7 and 2026; the recipe-space
  models use 42 and 7. Each seed is a different partition, so their OOF
  vectors are genuine held-out predictions on the full training set.
- The stack does **not** reuse a base split. `stack.main` imposes a fresh
  `StratifiedKFold(5, seed=meta_seed)` over the fixed OOF matrix. This is
  required because base models used heterogeneous splits.
- Meta-model selection is by pooled meta-OOF AUC, never by test-time
  fit quality. `ensemble.py` prints an in-sample logistic "stack AUC" line
  that is diagnostic only and must not be used for selection.
- Ridge predictions are clipped to [0, 1]; test predictions are averaged
  across the five fold-models rather than fitted once on all rows.

Known residual risk: OOF vectors were produced by models trained on
different partitions, so a meta-fold's training rows may have been seen by
some base models. This is the standard cost of stacking heterogeneous OOF
and is why the meta-partition is fresh rather than shared.

## 6. Models

### LightGBM

Defaults in `train_gbdt.LGBM_PARAMS`: binary objective, AUC metric, learning
rate 0.05, 127 leaves, minimum 50 per leaf, feature and bagging fractions
0.8, L2 1.0, up to 3000 rounds with 200-round AUC early stopping. Native
categorical features.

### XGBoost

`XGB_PARAMS`: `binary:logistic`, AUC metric, learning rate 0.03, depth 7,
subsample/colsample 0.8, min child weight 20, lambda 10, `hist` tree method,
up to 3000 rounds with 200-round early stopping. `num_boost_round` and
`early_stopping_rounds` may be overridden through the params JSON.

### CatBoost

`CAT_PARAMS`: Logloss loss, AUC eval, learning rate 0.04, depth 7, L2 10,
random strength 1.0, 3000 iterations with 200-round early stopping.
CatBoost 1.2.10 supports `eval_metric='AUC'` with `task_type='GPU'`: the
metric is computed on CPU with a sampling period of 5, but early stopping
still keys off it, so no Logloss proxy is needed. GPU CatBoost nevertheless
scores roughly 0.0007 lower than CPU on this data because of tree-building
numerics; prefer CPU when score matters and GPU only when time does.

### Tuned XGBoost

`tune_xgb.py` runs Optuna on a single fold as a fast proxy. The saved params
are untracked. Reproduce the two used in the final pool:

- `outputs/best_xgb.json` (v1 space): depth 5, min child weight 61,
  subsample 0.9828, colsample 0.6717, lambda 26.18, alpha 0.8364,
  learning rate 0.0473.
- `outputs/best_xgb_dv.json` (v2 space): depth 7, min child weight 10,
  subsample 0.9, colsample 0.9, alpha 0.071, lambda 2.0, learning rate
  0.005, `max_bin` 1024, 10000 rounds, 500-round early stopping, seed 42.
  The `deterministic_histogram` key in the saved file is inert under
  XGBoost 3.4 and can be omitted.

The v2 configuration is a low-learning-rate, long-run setup; it needs the
long early-stopping patience to converge.

### MLP

`mlp.py` builds entity embeddings for categorical and binned columns with
standardized numerics fitted on training folds only. Its best OOF was 0.9386,
well below the GBDTs, and NNLS gave it zero weight; it is kept as a
diversity experiment, not as part of the final ensemble. It requires the
optional `requirements-mlp.txt`.

## 7. Ensembling

### Blend (`ensemble.py`)

Plain mean, rank average, and AUC-derived weights. The weighted mean uses
`clip(auc - 0.5)` weights. Useful for quick checks; superseded by stacking.

### Linear stack (`stack.py`)

Imposes a fresh 5-fold partition over the OOF matrix and evaluates:

- Logistic regression over a `C` grid (0.1 to 10).
- Ridge over an `alpha` grid (0.1 to 100), predictions clipped to [0, 1].
- NNLS with normalized weights, reported both per-fold averaged and
  fitted on the full matrix.

The best learner by pooled meta-OOF is saved. On the final pool ridge with
`alpha=10` won or tied across all four stack rounds; NNLS corroborated the
useful members by giving them non-zero weights.

### Nonlinear stack (`stack_nl.py`)

Histogram gradient boosting meta-model over the same OOF matrix, with the
same fresh-partition rule. It underperformed ridge (0.94556 versus 0.94567),
which is evidence that the pool's signal is close to linearly combinable.
Kept for the record; not used in any submission.

## 8. Gate and submission policy

- From S13 onward, a candidate is submitted only if its pooled validation
  AUC strictly beats the standing best, with at least one of: a margin of
  0.0001 or more, or agreement across two meta-seeds. Earlier rounds used a
  strict-beats rule without the margin.
- At most one or two submissions per idea; the best single file per round.
- The strict gate rejected the majority of experiments. The S12 result
  (+0.00001 validation, −0.00001 leaderboard) is why the margin rule was
  added: sub-0.00005 differences are noise on this board.

## 9. Operational constraints and failure modes

- **Memory.** The host has 7 GB of RAM and `N_JOBS=4`. Folds run
  sequentially; do not parallelize them.
- **10-fold v2 OOM.** A 10-fold recipe-space run silently dies after the
  first fold because `TargetEncoder` with internal CV plus per-fold frame
  copies exceeds available memory. Use 5 folds. There is no swap-related
  exception; the process simply disappears.
- **GPU numerics.** GPU XGBoost and CatBoost both diverge measurably from
  their CPU equivalents on this data (same-feature-space, same-seed OOF
  correlation about 0.999 for XGB; CatBoost about 0.0007 lower). GPU is a
  speed lever for search, not a score lever.
- **CPU/GPU comparisons.** Only compare runs built on the same feature
  space. An earlier "parity failure" (correlation 0.998) compared runs from
  before and after the v1 feature set grew from 62 to 85 model features.
- **Target encoding memory.** `TargetEncoder(cv=5)` fits five internal
  models per smoothing level, three levels per fold. This is the dominant
  memory cost in v2.
- **Kaggle CLI.** Foreground `kaggle competitions submit` sometimes hangs
  with no output. Background the command and poll the submission list;
  uploads then complete in about 15 seconds.

## 10. Reproducing untracked artifacts

`outputs/` and `submissions/` are excluded from version control. Parameter
JSONs live there and must be recreated from the values in section 6.
Prediction arrays can be regenerated from the commands in `usage.md`, with
one exception: the S3-era base members inside S13/S14 were trained on an
earlier 62-feature revision of the v1 space that predates the first commit.
The committed `build_features` (85 model features) reproduces the later
tuning and seed experiments, not those original members. Rebuilding the
exact final ensemble therefore requires the archived `outputs/*.npy` files;
a from-scratch rerun yields an equivalent but not identical stack. CPU runs
are reproducible in practice for a fixed seed and library version; GPU runs
differ slightly from CPU and from each other because of kernel ordering and
GPU tree-building numerics.

## 11. Extension points

- **New feature space.** Add a `build_features_vN` function and extend the
  `--features` choice; keep existing spaces untouched.
- **New base model.** Add a branch to `_fit_predict` and `predict_with` in
  `train_gbdt.py`; OOF/test artifacts then enter the stack unchanged.
- **New stacker.** Follow `stack_nl.py`: fresh partition, pooled meta-OOF
  selection, fold-averaged test predictions.
- **New target encoder.** Add a fold-local function next to
  `target_encode_triple_fold`; never fit an encoder outside the fold.
