# Results ledger (OOF AUC / public LB)

Gate throughout: submit only if pooled OOF strictly beats the best.

## Blends

| Submission | Inputs | OOF | LB |
|---|---|---|---|
| blend3 | lgbm/xgb/cat seed42 | 0.94179 | 0.94146 |
| blend6 | + seed7 of each | 0.94177 | 0.94136 |
| blend7 | blend6 + xgb_tuned42 | 0.94182 | 0.94143 |
| blend8 | tuned-xgb seeds 42/7/2026 (mean) | **0.94190** | 0.94161 |
| blend9 | 12-model mean | 0.94183 | — (held) |

Best single: `xgb_tuned42` 0.94176.

## Stacking (Step E)

| Tier | Pool | Meta-OOF | LB |
|---|---|---|---|
| S1 | tuned×3 | 0.94190 (tie) | — (held) |
| S2 | tuned×3 + te×2 (te zero-weighted) | 0.94190 (tie) | — (held) |
| S3 | full 14, ridge a=10 | **0.94207** | **0.94188** ✅ best |
| S4 | sparse 6 (NNLS-selected) | 0.94196 | — (held) |

Robustness: S3 rerun under meta-seed 7 → 0.94208.

## Later tracks (all held — nothing beat S3)

- Deep features (freq encodings, group means, binned crosses; 86 cols):
  XGB-deep 0.94151; retuned on new space (40-trial Optuna) 0.94154.
- Embedding MLP (GPU): 0.93862 → 0.93863 after 4× capacity (ceiling, not
  capacity-bound); corr ~0.94 vs GBDTs but NNLS weight 0.0.
- Recipe prior (cross-fitted LogReg OOF 0.93809): head-start 0.94156,
  clue-only 0.94151, residual-boost on S3 margin 0.94157 (corr 0.998).
- Original 10k source dataset (CC0, aligned): +0.00003 as extra rows.
- Train/test shift check: adversarial AUC 0.5003 (none).

Standing best: `sub_stack_s3.csv`, LB **0.94188**. Open gap to the top of
the board (~0.9467) needs the true data-generating recipe or equivalent.
