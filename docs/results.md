# Results ledger (OOF AUC / public and private LB)

Gate: submit only if pooled OOF strictly beats the best. From S13 the rule
was tightened to require at least +0.0001 or agreement across two
meta-seeds.

## Blends

| Submission | Inputs | OOF | LB |
|---|---|---|---|
| blend3 | lgbm/xgb/cat seed42 | 0.94179 | 0.94146 |
| blend6 | + seed7 of each | 0.94177 | 0.94136 |
| blend7 | blend6 + xgb_tuned42 | 0.94182 | 0.94143 |
| blend8 | tuned-xgb seeds 42/7/2026 (mean) | **0.94190** | 0.94161 |
| blend9 | 12-model mean | 0.94183 | — (held) |

Best single at this stage: `xgb_tuned42` 0.94176.

## Stacking (Step E)

| Tier | Pool | Meta-OOF | LB |
|---|---|---|---|
| S1 | tuned×3 | 0.94190 (tie) | — (held) |
| S2 | tuned×3 + te×2 (te zero-weighted) | 0.94190 (tie) | — (held) |
| S3 | full 14, ridge a=10 | **0.94207** | **0.94188** (best at the time) |
| S4 | sparse 6 (NNLS-selected) | 0.94196 | — (held) |

Robustness: S3 rerun under meta-seed 7 → 0.94208.

## Later tracks (held at the time; superseded by L/M)

- Deep features (freq encodings, group means, binned crosses; 86 columns
  including `id`):
  XGB-deep 0.94151; retuned on new space (40-trial Optuna) 0.94154.
- Embedding MLP (GPU): 0.93862 → 0.93863 after 4× capacity (ceiling, not
  capacity-bound); corr ~0.94 vs GBDTs but NNLS weight 0.0.
- Recipe prior (cross-fitted LogReg OOF 0.93809): head-start 0.94156,
  clue-only 0.94151, residual-boost on S3 margin 0.94157 (corr 0.998).
- Original 10k source dataset (CC0, aligned): +0.00003 as extra rows.
- Train/test shift check: adversarial AUC 0.5003 (none).

Standing best at this point: `sub_stack_s3.csv`, LB **0.94188**. Open gap
to the top of the board (~0.9467) needs the true data-generating recipe or
equivalent.

## H. Recipe prior v2 (generator recovery) — 2026-09-14, HELD

- Diagnosis: target is near-rule-based — `Subsidy=No` rate 0.006 vs
  `Yes` 0.275; `High` anxiety vetoed (0/2069 except 3/125 at Env5+Yes);
  Env logits -4.6/-3.3/-1.5/-0.4/+0.8 (unequal steps); income 0.41→0.89
  across deciles inside the top cell. Prior v1 failed (0.93809) by coding
  Env numeric-linear with no multiplier crosses.
- H2 search (cross-fitted LogReg, seed 42): Env-as-cat 0.93842,
  +log/income²/commute² 0.93863, +IncQ×Env + Sub×Anx 0.93866, +Sub×Env
  0.93865 (flat in C). Smooth log-income slopes worse (0.93858);
  big-cell TE single worse (0.93640 — binning loses the smooth income
  gradient). Family caps ~0.9387.
- `src/recipe_v2.py` (NEW, defaults preserve prior v1): R4 ingredient set,
  joint target-free transforms, `oof/test_recipe_v2_seed42.npy`, OOF
  **0.93865** (+0.00056 over the prior v1) but corr 0.9975 vs prior v1,
  0.985 vs S3.
- S7 (S3 pool + prior v2): ridge_a10 0.94208 vs S3 0.94207 (+0.00001);
  meta-seed 7 rerun 0.94208 = S3-under-7 0.94208 (exact tie); NNLS
  weight 0.0 both seeds. Strict gate: HOLD `sub_stack_s7.csv`.
  Best stays S3.

## I/J. Cell-smooth + ingredient-MLP priors — 2026-09-14, HELD

- J1 `src/cell_prior.py` (NEW): per Subsidy×Anxiety×Env cell logistic
  on log_income + commute (30 cells, 15 fall back to smoothed rate —
  veto cells are near-all-negative). OOF 0.93803 < prior v2; corr 0.998 vs
  prior v2 (same family, weaker). J2/J3 skipped — no partial signal.
- I `src/mlp_prior.py` (NEW): 13 raw ingredients only, 64-32 net,
  GPU. OOF 0.93854 < prior v2; corr 0.94 vs S3 (diverse) but 0.998 vs the
  full 86-column MLP (same model, cleaner inputs).
- S8 (S3 pool + ingredient-MLP): ridge_a10 0.94207 = S3 exact tie,
  NNLS weight 0.0. HOLD `sub_stack_s8.csv`.
- Stop rule met: convex-LR, smooth-LR, cell-smooth, MLP families all
  cap ~0.9386 with zero stack residual. S3 was locked as the incumbent at
  this point.

## K. LGBM/CatBoost tuning — 2026-09-15, HELD

- Optuna single-fold proxy (`src/tune_lgbm.py`, `src/tune_cat.py`;
  mirrors `tune_xgb.py`): `outputs/best_lgbm.json`, `outputs/best_cat.json`.
- `lgbm_tuned42` (CPU) 0.94148 (+0.00009 over baseline — noise);
  `cat_tuned42` (GPU, initial Logloss-proxy run; see K2) 0.94101
  (-0.00068).
- S9 (S3 pool + both): ridge_a10 0.94207 = S3 tie both meta-seeds
  (seed-7: 0.94207 vs S3 0.94208); NNLS zeroes the new members.
  HOLD `sub_stack_s9.csv`. Only XGB had tuning headroom.

## K2. AUC-on-GPU correction — 2026-09-15, HELD

- CatBoost 1.2.10 supports AUC eval on GPU (CPU fallback, period 5);
  Logloss proxy removed from `train_gbdt.py`. But the AUC-early-stopped
  GPU rerun scores 0.94101 = proxy run exactly (corr 0.9999) — the
  CPU-vs-GPU gap is in tree-building numerics, not the metric.
- S10: ridge_a10 0.94206 < S3. HOLD `sub_stack_s10.csv`.

## K3. GPU XGB seeds + 10-fold — 2026-09-15, HELD

- Parity appears to fail: GPU `xgb_tuned42` 0.94151 vs CPU 0.94176
  (corr 0.9981; later shown to be a feature-space confound in K4).
  GPU hist numerics diverge — speed lever only, not score.
- GPU seeds 0/123/999: 0.94149/0.94147/0.94145; GPU 10-fold: 0.94158.
- S11 (S3 + 5 GPU members): ridge ties S3 both meta-seeds
  (0.94207 / 0.94208). HOLD `sub_stack_s11.csv`.

## K4. CPU XGB seeds + 10-fold — 2026-09-15, GATE PASS (submitted)

- Correction: K3 "parity failure" was a feature-space confound
  (`features.py` grew from 62 to 85 model features after the S3-pool runs).
  True same-seed CPU/GPU corr is 0.9993.
- CPU seeds 0/123/999: 0.94149/0.94146/0.94146; CPU 10-fold: 0.94158.
- S12 (S3 + 4 CPU members): ridge_a10 **0.94208** both meta-seeds
  (+0.00001 strict on 42, tie on 7); NNLS ~0.055 on seed0 twice.
  Submitted 2026-09-15 → LB **0.94187** (−0.00001 vs S3). Noise.
  S3 remained the standing best at this point.

## L. Recipe space — 2026-09-15, GATE PASS (submitted)

- Replicated `evgendvorkin/s6e9-single-xgb-cv-0-94583`: digit features
  + all-column freq + triple TE + long-run XGB (`--features v2`,
  `--te-triple`, `outputs/best_xgb_dv.json`). Legacy space untouched.
- `xgb_dv` single (GPU, 5-fold): OOF **0.94564**.
- S13 (S3 + dv): ridge_a10 **0.94567**/**0.94568** (seeds 42/7);
  NNLS 0.92 on dv. `sub_stack_s13.csv` submitted 2026-09-15 → public
  **0.94596** (private 0.94493). New best at the time (+0.00408 over S3).

## M. Second dv seed + nonlinear stacker — 2026-09-15, GATE PASS

- `xgb_dv` seed7 (GPU, 5-fold): OOF **0.94568**; seed-mean (42+7)
  **0.94586** (seeds corr 0.9969 — genuine split diversity).
- Nonlinear stacker (`src/stack_nl.py`, NEW; HGB meta-learner, same
  fresh-partition discipline): 0.94556 < ridge 0.94567 — linear wins,
  HOLD `sub_stack_nl13.csv`. Pool signal is linear-combinable.
- S14 (S3 + dv42 + dv7): ridge_a10 **0.94587** both meta-seeds
  (identical — first time); NNLS 0.43/0.53 across dv seeds.
  Submitted 2026-09-16 → public **0.94599**, private **0.94495**.

## Final standing (competition closed 2026-09-30)

- Best submission: S14 (`sub_stack_s14.csv`), private **0.94495**,
  rank **1291 / 3575** (top 36%).
- Winner: Chris Deotte, private **0.94602**; top-10 cutoff **0.94574**
  (public cutoff was 0.94655 — final scoring compressed the band).
- Public-to-private drop about 0.001 (S14: 0.94599 → 0.94495; S13:
  0.94596 → 0.94493). The gated stack held its relative position; absolute
  scores shifted because the private test sample is a fresh split.
- Open gap to top-10: 0.00079. Every other family we tested (LGBM/Cat
  retuning, GPU variants, extra seeds, nonlinear stacking, priors) landed
  within noise of the ridge stack; the recipe-space features were the only
  structural win (+0.0036 OOF, +0.004 public LB).
