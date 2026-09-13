# EV Purchase Prediction (Playground Series S6E9)

Gradient-boosted-tree pipeline predicting `Will_Buy_EV` (AUC). All
experiments are leak-free: joint target-free featurization, stratified
K-fold OOF predictions, and a strict gate — nothing is submitted unless its
pooled OOF beats the current best.

## Setup

```bash
pip install -r requirements.txt
# competition files go here (never committed):
#   data/raw/train.csv  data/raw/test.csv
```

Run everything from `source/`. Folds are sequential by design (low-RAM
laptop); GPU flags exist for the heavy trainers.

## Pipeline stages

1. EDA: `python -m src.eda`
2. Base models (5-fold OOF + fold-averaged test preds):
   `python -m src.train_gbdt --model {lgbm,xgb,cat} --train data/raw/train.csv --test data/raw/test.csv --seed 42 [--device gpu] [--te] [--params-json outputs/best_xgb.json] [--out-suffix _tag]`
3. Tune XGB (single-fold Optuna proxy):
   `python -m src.tune_xgb --trials 40 --seed 42 --device gpu --out-name best_xgb.json`
4. Blend saved OOFs: `python -m src.ensemble <oof...> --y data/raw/train.csv --tests <test...> --test-ids data/raw/test.csv --out-name blend`
5. Leak-free CV stacking (fresh meta-partition, LogReg/Ridge/NNLS):
   `python -m src.stack --oofs <oof...> --tests <test...> --y data/raw/train.csv --test-ids data/raw/test.csv --out-name stack --seed 42`
6. Recipe prior (smooth cross-fitted LogReg, for XGB `base_margin` / feature):
   `python -m src.recipe --seed 42`
7. Embedding MLP (GPU, diversity model):
   `python -m src.mlp --train data/raw/train.csv --test data/raw/test.csv --seed 42`

Details: `docs/usage.md`. Experiment history and scores: `docs/results.md`.
