"""Nonlinear CV stacking over base OOF files (follows stack.py discipline).

Meta-features are fixed full-train OOF vectors, so a FRESH meta-partition
is imposed instead of reusing any base split. Test transform averages
per-fold meta-models. CPU-only; leaves stack.py behavior untouched.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import ID_COL, N_SPLITS, OUTPUTS, SUBMISSIONS, TARGET
from .ensemble import load_y
from .stack import _check_matrix
from .utils import check_submission, make_folds, seed_all


def _fold_predict_hgb(M, y, T, folds, lr=0.05, max_depth=3, max_iter=500):
    oof = np.zeros(len(y))
    tsum = np.zeros(T.shape[0]) if T is not None else None
    for tr, va in folds:
        clf = HistGradientBoostingClassifier(
            learning_rate=lr, max_depth=max_depth, max_iter=max_iter,
            early_stopping="auto", validation_fraction=0.1,
            n_iter_no_change=50, random_state=0,
        )
        clf.fit(M[tr], y[tr])
        oof[va] = clf.predict_proba(M[va])[:, 1]
        if tsum is not None:
            tsum += clf.predict_proba(T)[:, 1] / len(folds)
    return oof, tsum


def main(oofs, tests, y_path, test_ids_path, out_name, seed=42):
    seed_all(seed)
    names = [Path(f).name for f in oofs]
    M = np.column_stack([np.load(f) for f in oofs])
    T = np.column_stack([np.load(f) for f in tests]) if tests else None
    _check_matrix(M, T, names)

    y = load_y(str(y_path))
    folds = make_folds(pd.Series(y), N_SPLITS, seed)

    results = {}
    for lr, depth in ((0.05, 3), (0.1, 3), (0.05, None)):
        oof, t = _fold_predict_hgb(M, y, T, folds, lr, depth)
        auc = roc_auc_score(y, oof)
        results[f"hgb_lr{lr}_d{depth}"] = (auc, oof, t)
        print(f"hgb lr={lr} depth={depth} meta-OOF AUC={auc:.5f}", flush=True)

    best_key = max(results, key=lambda k: results[k][0])
    best_auc, best_oof, best_test = results[best_key]
    print(f"best={best_key} meta-OOF AUC={best_auc:.5f}")
    np.save(OUTPUTS / f"{out_name}_meta_oof.npy", best_oof)
    if best_test is not None and test_ids_path:
        ids = pd.read_csv(test_ids_path, usecols=[ID_COL])[ID_COL]
        sub = pd.DataFrame({ID_COL: ids.values, TARGET: np.clip(best_test, 0, 1)})
        check_submission(sub, ids)
        sp = SUBMISSIONS / f"sub_{out_name}.csv"
        sub.to_csv(sp, index=False)
        print(f"saved {sp}")
    return best_key, best_auc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oofs", nargs="+", type=Path, required=True)
    ap.add_argument("--tests", nargs="*", type=Path, default=None)
    ap.add_argument("--y", required=True)
    ap.add_argument("--test-ids", default=None)
    ap.add_argument("--out-name", default="stack_nl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.oofs, a.tests, a.y, a.test_ids, a.out_name, a.seed)
