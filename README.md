# EV Purchase Prediction (Kaggle Playground S6E9)

This repository is a complete solution to the
[Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)
competition: predict whether a survey respondent will buy an electric
vehicle (`Will_Buy_EV`, Yes/No) from 13 demographic, financial and
attitudinal columns. The pipeline and the current feature spaces are
reproducible in practice, and the pre-commit S3 pool predictions needed to
rebuild the exact final ensemble are preserved in `backup/`.

The competition is scored by **ROC AUC**, which measures how well a model
ranks buyers above non-buyers rather than how accurate any single yes/no
guess is. The training set has 668,665 rows and 116,779 buyers (17.5%);
the test set has 286,571 rows.

## Final result

| Metric | Score |
|---|---|
| Best submission (private leaderboard) | **0.94495** |
| Same submission (public leaderboard) | 0.94599 |
| Rank | 1291 of 3575 (top 36%) |
| Winner (private) | 0.94602 |
| Top-10 cutoff (private) | 0.94574 |

The submission is `submissions/sub_stack_s14.csv` (written locally; the
`submissions/` directory is not versioned). On the public leaderboard the
score improved from 0.94146 on the first working blend to 0.94599.

## How the solution works

The pipeline follows the same shape throughout: build features, train
several models on held-out folds, combine their predictions, and only
accept a change if it genuinely improves validation.

1. **Feature engineering.** Raw survey columns are expanded into numeric
   signals: ratios and products between income, commute, charging access and
   environmental concern; frequency counts that reveal how unusual a value
   is; and, most importantly, the individual decimal digits of every numeric
   column and triple target encoding. The digit and frequency features
   capture artifacts of how the synthetic dataset was generated, which turns
   out to be the single biggest source of signal in this competition.
2. **Base models.** Three gradient-boosted tree libraries (LightGBM, XGBoost,
   CatBoost) are trained with 5-fold stratified cross-validation. The
   legacy-space models used three seeds; the recipe-space model used two.
   Every model produces out-of-fold (OOF) predictions on the training rows
   plus fold-averaged predictions on the test rows.
3. **Stacking.** The OOF predictions become the inputs of a second-level
   model. A regularized linear model (ridge regression) learns the best
   weights for combining them; it reliably beat logistic regression and
   non-negative least squares. A fresh cross-validation split is imposed on
   the OOF matrix so the second level never sees the base models' split
   structure.
4. **Strict gate.** Nothing is submitted unless its pooled validation AUC
   strictly beats the current best. Most ideas in the experiment log were
   held back by this rule, which is why the final scores are honest rather
   than the best-looking noisy number.

### Why the validation is trustworthy

All target-free feature engineering is computed jointly over train and
test, so no label information leaks into the features. Target encoding
is fitted inside each training fold only. OOF predictions are generated on
rows the model never saw during training. The final stack is then validated
on a second, independent partition. Public and private leaderboard scores
tracked the validation estimates closely, including the roughly 0.001
public-to-private drop we observed.

## What actually moved the score

| Change | Gain |
|---|---|
| Recipe-space features (digits, all-column frequency, triple target encoding) | +0.0036 OOF (+0.004 public LB) |
| Stacking over blending (ridge meta-model) | +0.0002 |
| Extra seeds and fold variants | noise level |
| LGBM/CatBoost retuning, GPU variants, extra model families | noise or worse |

The honest conclusion is that this dataset has a low ceiling for
conventional modelling. The top dozen teams finished within 0.001 of one
another, and the top score required finding the synthetic generator's
footprints rather than tuning models harder.

## Repository layout

```text
source/                 project root (the Git repository)
├── src/                pipeline code
│   ├── features.py     feature engineering (v1 legacy + v2 recipe space)
│   ├── train_gbdt.py   LightGBM / XGBoost / CatBoost cross-validation
│   ├── stack.py        leak-free linear stacking (logreg, ridge, NNLS)
│   ├── stack_nl.py     nonlinear stacking experiment
│   ├── ensemble.py     simple OOF-weighted blends
│   ├── recipe*.py      logistic recipe-prior experiments
│   ├── cell_prior.py   cell-smoothed prior experiment
│   ├── mlp*.py         neural-network diversity models
│   └── tune_*.py       Optuna parameter search
├── backup/             pre-commit 62-feature S3 pool predictions
├── requirements*.txt   core and optional (MLP) dependencies
├── docs/               technical reference, runbook, results ledger
├── data/raw/           competition CSVs (not versioned)
├── outputs/            OOF/test prediction arrays, logs (not versioned)
└── submissions/        Kaggle-ready CSVs (not versioned)
```

## Quick start

```bash
pip install -r requirements.txt
# place the competition files (never committed) at:
#   data/raw/train.csv  data/raw/test.csv

# train a legacy-space model
python -m src.train_gbdt --model lgbm --train data/raw/train.csv \
  --test data/raw/test.csv --seed 42

# train the recipe-space model (the one that mattered)
# recreate outputs/best_xgb_dv.json first (values in docs/technical.md, section 6)
python -m src.train_gbdt --model xgb --train data/raw/train.csv \
  --test data/raw/test.csv --seed 42 --folds 5 --features v2 \
  --te-triple --params-json outputs/best_xgb_dv.json --device gpu

# stack saved OOF predictions
python -m src.stack --oofs outputs/oof_lgbm_seed42.npy \
  outputs/oof_xgb_dv_te3_tuned_dv5_seed42.npy \
  --y data/raw/train.csv --out-name demo
```

Drop `--device gpu` to run on CPU, and install only `requirements.txt` unless
you also want the neural-network track (`requirements-mlp.txt`).

Recipe-space runs at 10 folds exceed the 7 GB of RAM used during
development; use 5 folds on a laptop. See `docs/usage.md` for every command.

## Documentation

- [`docs/technical.md`](docs/technical.md): architecture, feature spaces,
  cross-validation design, model configuration, failure modes and limits.
- [`docs/usage.md`](docs/usage.md): command-by-command runbook.
- [`docs/results.md`](docs/results.md): chronological experiment ledger with
  validation and leaderboard scores.

## Conclusion

The most valuable lesson was not a model trick but a workflow one: a strict,
leak-free validation gate made it possible to try many cheap ideas, reject
the noisy ones, and recognise the one real improvement when it appeared.
The second lesson is about where signal lives in synthetic tabular data.
Conventional feature engineering reached a hard plateau around 0.942; the
extra 0.004 came only after treating the data as a generated artifact and
encoding its rounding and frequency fingerprints. Model capacity, GPU speed
and additional seeds all failed to move the score, while understanding the
generator succeeded. A competitive score was reached around 0.945 (0.94599
public, 0.94495 private), and closing the remaining 0.001 would require
recovering the exact data-generating recipe rather than further ensembling.
