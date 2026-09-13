"""Stratified K-fold training for LGBM/XGB/CatBoost. Sequential folds (low RAM)."""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import ID_COL, N_JOBS, N_SPLITS, OUTPUTS, TARGET
from .features import build_features
from .utils import make_folds, seed_all

LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",  # early-stop on competition metric, not logloss
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    n_jobs=N_JOBS,
)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=10.0,
    tree_method="hist",
    n_jobs=N_JOBS,
)
CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=10.0,
    random_strength=1.0,
    verbose=False,
    allow_writing_files=False,
)


def _fit_predict(model_name: str, X_tr, y_tr, X_va, y_va, cat_cols,
                 device: str = "cpu", overrides: dict | None = None,
                 tr_margin=None, va_margin=None):
    overrides = overrides or {}
    if model_name == "lgbm":
        import lightgbm as lgb

        lgb_params = {**LGBM_PARAMS, **overrides}
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
        booster = lgb.train(
            {**lgb_params, "num_boost_round": 3000},
            dtr,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        importances = booster.feature_importance(importance_type="gain")
        return booster.predict(X_va), booster, importances
    if model_name == "xgb":
        import xgboost as xgb

        xgb_params = {**XGB_PARAMS, **overrides}
        if device == "gpu":
            xgb_params["device"] = "cuda"
        dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True,
                          base_margin=tr_margin)
        dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True,
                          base_margin=va_margin)
        booster = xgb.train(
            xgb_params,
            dtr,
            num_boost_round=3000,
            evals=[(dva, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        return (
            booster.predict(dva),
            booster,
            None,
        )
    if model_name == "cat":
        from catboost import CatBoostClassifier, Pool

        cat_params = {**CAT_PARAMS, **overrides}
        if device == "gpu":
            # AUC eval is not implemented for GPU; Logloss is the proxy.
            cat_params.update(task_type="GPU", devices="0", eval_metric="Logloss")
        clf = CatBoostClassifier(**cat_params, iterations=3000,
                                 early_stopping_rounds=200)
        clf.fit(Pool(X_tr, y_tr, cat_features=cat_cols),
                eval_set=Pool(X_va, y_va, cat_features=cat_cols),
                verbose=False)
        return clf.predict_proba(X_va)[:, 1], clf, None
    raise ValueError(f"unknown model {model_name}")


def predict_with(model_name: str, model, X, te_margin=None):
    if model_name == "lgbm":
        return model.predict(X)
    if model_name == "xgb":
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X, enable_categorical=True,
                                         base_margin=te_margin))
    if model_name == "cat":
        return model.predict_proba(X)[:, 1]
    raise ValueError(f"unknown model {model_name}")


def run_cv(model_name: str, train_path: Path, seed: int = 42,
           n_splits: int = N_SPLITS, test_path: Path | None = None,
           device: str = "cpu", use_te: bool = False,
           overrides: dict | None = None, out_suffix: str = "",
           prior_train: Path | None = None, prior_test: Path | None = None,
           prior_as_feature: bool = False, use_prior_margin: bool = True):
    from .utils import check_submission
    from .config import SUBMISSIONS

    seed_all(seed)
    df = pd.read_csv(train_path)
    y_raw = df[TARGET]
    import pandas.api.types as _t
    y = y_raw.astype(int) if _t.is_numeric_dtype(y_raw) else (y_raw == "Yes").astype(int)
    train_ids = df[ID_COL]
    n_train = len(df)

    frames = [df.drop(columns=[TARGET])]
    test_ids = None
    if test_path is not None:
        tdf = pd.read_csv(test_path)
        test_ids = tdf[ID_COL]
        frames.append(tdf)
    # Joint featurization: no target statistics used, so no leakage;
    # guarantees identical categories/bins across train and test.
    X_all = build_features(pd.concat(frames, ignore_index=True))
    feat_cols = [c for c in X_all.columns if c != ID_COL]
    for c in feat_cols:
        if not _t.is_numeric_dtype(X_all[c]) and str(X_all[c].dtype) != "category":
            X_all[c] = X_all[c].astype("category")
    # Recipe prior: cross-fitted OOF probs for train, averaged for test.
    # As base_margin (head start, XGB) and/or as an extra feature (clue).
    use_prior = prior_train is not None
    if use_prior:
        assert prior_test is not None, "prior_test required with prior_train"
        pt = np.load(prior_train).astype(np.float64)
        pe = np.load(prior_test).astype(np.float64)
        assert len(pt) == n_train and len(pe) == len(X_all) - n_train, \
            (pt.shape, pe.shape, n_train, len(X_all))
        assert np.isfinite(pt).all() and np.isfinite(pe).all()
        eps = 1e-6
        if use_prior_margin:
            mtrain = np.log(np.clip(pt, eps, 1 - eps) /
                            (1 - np.clip(pt, eps, 1 - eps))).astype(np.float32)
            mtest = np.log(np.clip(pe, eps, 1 - eps) /
                           (1 - np.clip(pe, eps, 1 - eps))).astype(np.float32)
        else:
            mtrain = mtest = None
        if prior_as_feature:
            X_all["__recipe_prior__"] = np.concatenate([pt, pe]).astype(
                np.float32)
            feat_cols = [c for c in X_all.columns if c != ID_COL]
    else:
        mtrain = mtest = None
    cat_cols = [c for c in feat_cols if str(X_all[c].dtype) == "category"]
    X = X_all.iloc[:n_train][feat_cols].reset_index(drop=True)
    X_test = X_all.iloc[n_train:][feat_cols].reset_index(drop=True) \
        if test_ids is not None else None

    oof = np.zeros(n_train)
    test_pred = np.zeros(len(X_test)) if X_test is not None else None
    tag = ""
    if use_te:
        from .features import target_encode_fold, te_columns
        tag = "_te"
    if overrides:
        tag += "_tuned"
    if use_prior:
        tag += "_rec"
    tag += out_suffix
    for fold, (tr, va) in enumerate(make_folds(y, n_splits, seed)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        X_te_cur = X_test
        fit_cat = cat_cols
        if use_te:
            te_list = [X_test] if X_test is not None else []
            X_tr, X_va, te_outs = target_encode_fold(
                X_tr, y.iloc[tr], X_va, te_list, te_columns(X)
            )
            X_te_cur = te_outs[0] if te_outs else None
        p_va, model, _ = _fit_predict(
            model_name, X_tr, y.iloc[tr], X_va, y.iloc[va], fit_cat,
            device, overrides,
            mtrain[tr] if mtrain is not None else None,
            mtrain[va] if mtrain is not None else None,
        )
        oof[va] = p_va
        auc = roc_auc_score(y.iloc[va], p_va)
        print(f"[{model_name}{tag} seed={seed}] fold={fold} AUC={auc:.5f}", flush=True)
        if test_pred is not None:
            test_pred += predict_with(
                model_name, model, X_te_cur,
                mtest if model_name == "xgb" and mtrain is not None else None,
            ) / n_splits
        del model
        gc.collect()
    print(f"[{model_name}{tag} seed={seed}] OOF AUC={roc_auc_score(y, oof):.5f}")
    out = OUTPUTS / f"oof_{model_name}{tag}_seed{seed}.npy"
    np.save(out, oof)
    print(f"saved {out}")
    if test_pred is not None:
        tp = OUTPUTS / f"test_{model_name}{tag}_seed{seed}.npy"
        np.save(tp, test_pred)
        sub = pd.DataFrame({ID_COL: test_ids.values, TARGET: test_pred})
        check_submission(sub, test_ids)
        sp = SUBMISSIONS / f"sub_{model_name}{tag}_seed{seed}.csv"
        sub.to_csv(sp, index=False)
        print(f"saved {tp} and {sp}")
    return oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "xgb", "cat"], required=True)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=N_SPLITS)
    ap.add_argument("--test", type=Path, default=None)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--te", action="store_true",
                    help="add leak-free OOF target encodings + cross features")
    ap.add_argument("--params-json", type=Path, default=None,
                    help="JSON file with param overrides for the model")
    ap.add_argument("--out-suffix", default="",
                    help="extra tag suffix for outputs (keeps prior files intact)")
    ap.add_argument("--prior-train", type=Path, default=None,
                    help="npy of cross-fitted train prior probs (XGB base_margin)")
    ap.add_argument("--prior-test", type=Path, default=None,
                    help="npy of test prior probs (averaged fold models)")
    ap.add_argument("--prior-feature", action="store_true",
                    help="also append the prior as an input column (clue)")
    ap.add_argument("--no-prior-margin", action="store_true",
                    help="use prior as feature only, no base_margin head start")
    args = ap.parse_args()
    import json
    ov = json.loads(args.params_json.read_text()) if args.params_json else None
    run_cv(args.model, args.train, args.seed, args.folds, args.test,
           args.device, args.te, ov, args.out_suffix,
           args.prior_train, args.prior_test, args.prior_feature,
           not args.no_prior_margin)
