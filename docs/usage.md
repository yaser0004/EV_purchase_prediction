# Usage runbook

Conventions: run from `source/`; `--seed` selects the StratifiedKFold
split; outputs land in `outputs/` (OOF/test `.npy`, params `.json`, run
`.log`) and `submissions/` (Kaggle-ready CSVs). Neither directory is
versioned. `--out-suffix` tags a run so prior files are never overwritten.

## P0 — seed-averaged base models

```bash
python -m src.train_gbdt --model lgbm --train data/raw/train.csv --test data/raw/test.csv --seed 2026
python -m src.train_gbdt --model xgb  --train data/raw/train.csv --test data/raw/test.csv --seed 2026
python -m src.train_gbdt --model cat  --train data/raw/train.csv --test data/raw/test.csv --seed 2026 --device gpu
python -m src.train_gbdt --model xgb  --train data/raw/train.csv --test data/raw/test.csv --seed 7 --params-json outputs/best_xgb.json
python -m src.train_gbdt --model xgb  --train data/raw/train.csv --test data/raw/test.csv --seed 2026 --params-json outputs/best_xgb.json
```

Notes: CatBoost 1.2.10 supports AUC early stopping on GPU (the metric is
computed on CPU with a sampling period, but stopping still keys off it), so
`--device gpu` no longer falls back to Logloss. GPU CatBoost still scores
slightly lower than CPU on this data; prefer CPU when score matters. Keep
folds sequential on small machines.

## Blending (exact-OOF verified)

```bash
python -m src.ensemble outputs/oof_lgbm_seed42.npy outputs/oof_xgb_seed42.npy outputs/oof_cat_seed42.npy \
  --y data/raw/train.csv --out-name blend
```

Add `--tests <files...> --test-ids data/raw/test.csv` to also write
`submissions/sub_blend.csv` (OOF-weighted mean of test preds). The printed
in-sample `stack AUC` line is diagnostic only — never use it for selection.

## Stacking (Step E)

```bash
python -m src.stack --oofs <oof...> --tests <test...> \
  --y data/raw/train.csv --test-ids data/raw/test.csv \
  --out-name stack_sN --seed 42 --models logreg ridge nnls
```

Imposes one fresh 5-fold partition over the fixed OOF matrix (base models
used heterogeneous seeds, so their splits must not be reused), tunes
`C`/`alpha` by pooled meta-OOF AUC, and averages the 5 fold-models for
test. Submit only if meta-OOF strictly beats the running best.

Nonlinear variant (experimental, held; same discipline):

```bash
python -m src.stack_nl --oofs <oof...> --tests <test...> \
  --y data/raw/train.csv --test-ids data/raw/test.csv \
  --out-name stack_nl --seed 42
```

## Recipe prior + residual tracks

```bash
python -m src.recipe --seed 42 --c 1.0
# head start (XGB base_margin):
python -m src.train_gbdt --model xgb --train data/raw/train.csv --test data/raw/test.csv \
  --seed 42 --device gpu --params-json outputs/best_xgb.json \
  --prior-train outputs/oof_recipe_seed42.npy --prior-test outputs/test_recipe_seed42.npy --out-suffix _h
# clue (prior as feature only): add --prior-feature --no-prior-margin
# recipe prior v2 (Env-as-cat + multiplier crosses, OOF 0.93865): `python -m src.recipe_v2 --seed 42`
# augments any stack pool as a 15th member, but S7 tied S3 (NNLS weight 0.0) — held.
# cell-smooth prior (J1, OOF 0.93803 — dominated by prior v2, held): `python -m src.cell_prior --seed 42`
# ingredient-only MLP prior (I, GPU, OOF 0.93854; S8 tied S3, held):
#   `python -m src.mlp_prior --seed 42 --max-epochs 60 --patience 8`
```

## Recipe feature space (`--features v2`) — digit/freq/triple-TE XGB

Replicates the public single-XGB recipe: decimal-digit features (synthetic
rounding tells) + joint frequency of every column + fold-local triple target
encoding (sklearn, smooth auto/10/100) + long low-lr XGB run. The default
`--features v1` preserves the legacy 85-feature space; v2 is opt-in and
never overwrites v1 files (tag `_dv...`).

```bash
python -m src.train_gbdt --model xgb --train data/raw/train.csv --test data/raw/test.csv \
  --seed 42 --folds 5 --features v2 --te-triple \
  --params-json outputs/best_xgb_dv.json --device gpu --out-suffix _dv5
```

`outputs/best_xgb_dv.json` is untracked (like `best_xgb.json`) — recreate
as: max_depth 7, min_child_weight 10, subsample/colsample 0.9,
reg_alpha 0.071, reg_lambda 2.0, learning_rate 0.005, max_bin 1024,
num_boost_round 10000, early_stopping_rounds 500, random_state/seed 42.
(The `deterministic_histogram` key in the saved file is inert under
XGBoost 3.4.)

Notes: `--te-triple` also works on v1 (uses the 13 raw TE columns);
10-fold v2 exceeds 7GB RAM (silent OOM after fold 1) — use 5-fold.
Reference runs: single OOF 0.94564 (seed 42) and 0.94568 (seed 7);
final stack S14 0.94587 outpaced both, scoring public 0.94599 and private
0.94495. The pre-commit S3 pool arrays needed for exact S14 reproduction
are committed in `backup/s3_pool_62feat/`.

## MLP diversity model

```bash
python -m src.mlp --train data/raw/train.csv --test data/raw/test.csv \
  --seed 42 --out-suffix _big --max-epochs 80 --patience 10 --batch 4096 --lr 0.001
```

Entity embeddings for the 21 categorical/binned columns, standardized
numerics from train-fold stats only. Checked into the stack like any OOF.
Requires the optional `requirements-mlp.txt`.

## Submission discipline

Respect the competition's daily submission limit (10 per day for this
one). House rule: at most 1-2 per idea, only past the strict OOF gate, best
single file per round.
