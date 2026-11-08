# Pre-commit S3 pool predictions

These 28 arrays are the OOF and fold-averaged test predictions for the 14
base models that fed the S3 stack, and therefore S13/S14. They were trained
on an earlier 62-feature revision of the v1 space that was never committed.
The current `build_features` produces 85 model features and cannot regenerate
them, which makes this directory the only exact record of the S3 pool.

- `oof_*.npy`: 668,665-length OOF probabilities in train row order.
- `test_*.npy`: 286,571-length test probabilities in test row order.
- `cat` seeds 7 and 2026 are the GPU-active copies; `cat` seed 42 is CPU.
- `SHA256SUMS` verifies every file.

Members: `lgbm`, `xgb`, `cat` at seeds 42/7/2026; `xgb_tuned` at seeds
42/7/2026; `xgb_te` at seeds 42/7.

To rebuild S14, combine these 14 members with the two recipe-space OOFs from
`outputs/` (regenerable from the committed v2 code):

```bash
python -m src.stack \
  --oofs backup/s3_pool_62feat/oof_*.npy \
  outputs/oof_xgb_dv_te3_tuned_dv5_seed42.npy \
  outputs/oof_xgb_dv_te3_tuned_dv5_seed7.npy \
  --tests backup/s3_pool_62feat/test_*.npy \
  outputs/test_xgb_dv_te3_tuned_dv5_seed42.npy \
  outputs/test_xgb_dv_te3_tuned_dv5_seed7.npy \
  --y data/raw/train.csv --test-ids data/raw/test.csv \
  --out-name stack_s14 --seed 42
```

These are derived predictions from competition data; handle them under the
competition's data terms.
