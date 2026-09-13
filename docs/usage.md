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

Notes: CatBoost on GPU uses a Logloss proxy (`train_gbdt.py`, GPU has no
AUC metric); CPU CatBoost early-stops on AUC directly. Keep folds
sequential on small machines.

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

## Recipe prior + residual tracks

```bash
python -m src.recipe --seed 42 --c 1.0
# head start (XGB base_margin):
python -m src.train_gbdt --model xgb --train data/raw/train.csv --test data/raw/test.csv \
  --seed 42 --device gpu --params-json outputs/best_xgb.json \
  --prior-train outputs/oof_recipe_seed42.npy --prior-test outputs/test_recipe_seed42.npy --out-suffix _h
# clue (prior as feature only): add --prior-feature --no-prior-margin
```

## MLP diversity model

```bash
python -m src.mlp --train data/raw/train.csv --test data/raw/test.csv \
  --seed 42 --out-suffix _big --max-epochs 80 --patience 10 --batch 4096 --lr 0.001
```

Entity embeddings for the 21 categorical/binned columns, standardized
numerics from train-fold stats only. Checked into the stack like any OOF.

## Submission discipline

Kaggle allows 10 submissions/day. House rule: at most 1–2 per idea, only
past the strict OOF gate, best single file per round.
