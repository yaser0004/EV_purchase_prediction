"""Leak-free CV stacking over base OOF files (Step E).

Meta-features are fixed full-train OOF vectors from heterogeneous base
seeds/splits, so a FRESH meta-partition is imposed instead of reusing any
base split. Test transform averages per-fold meta-models.

Sklearn-only, CPU (meta-matrix is KBs; GPU reserved for base GBDT retrains).
Reuses config/utils/ensemble helpers; leaves ensemble.py behavior untouched.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from .config import ID_COL, N_SPLITS, OUTPUTS, SUBMISSIONS, TARGET
from .ensemble import load_y, rank_avg
from .utils import check_submission, make_folds, seed_all


def _check_matrix(M: np.ndarray, T: np.ndarray | None, names: list[str]) -> None:
    assert M.ndim == 2 and M.shape[0] > 0 and M.shape[1] == len(names), M.shape
    assert np.isfinite(M).all(), "NaN/inf in OOF matrix"
    assert ((M >= 0) & (M <= 1)).all(), "OOF probs out of [0,1]"
    if T is not None:
        assert T.shape[1] == M.shape[1], (T.shape, M.shape)
        assert np.isfinite(T).all(), "NaN/inf in test matrix"
        assert ((T >= 0) & (T <= 1)).all(), "test probs out of [0,1]"


def _fold_predict_logreg(M, y, T, C, folds):
    oof = np.zeros(len(y))
    tsum = np.zeros(T.shape[0]) if T is not None else None
    for tr, va in folds:
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(M[tr], y[tr])
        oof[va] = clf.predict_proba(M[va])[:, 1]
        if tsum is not None:
            tsum += clf.predict_proba(T)[:, 1] / len(folds)
    return oof, tsum


def _fold_predict_ridge(M, y, T, alpha, folds, positive=False):
    oof = np.zeros(len(y))
    tsum = np.zeros(T.shape[0]) if T is not None else None
    for tr, va in folds:
        reg = Ridge(alpha=alpha, positive=positive)
        reg.fit(M[tr], y[tr])
        oof[va] = np.clip(reg.predict(M[va]), 0, 1)
        if tsum is not None:
            tsum += np.clip(reg.predict(T), 0, 1) / len(folds)
    return oof, tsum


def _fold_predict_nnls(M, y, T, folds):
    oof = np.zeros(len(y))
    tsum = np.zeros(T.shape[0]) if T is not None else None
    coefs = []
    for tr, va in folds:
        w, _ = nnls(M[tr], y[tr])
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
        coefs.append(w)
        oof[va] = np.clip(M[va] @ w, 0, 1)
        if tsum is not None:
            tsum += np.clip(T @ w, 0, 1) / len(folds)
    w_full, _ = nnls(M, y)
    w_full = w_full / w_full.sum() if w_full.sum() > 0 else np.full_like(
        w_full, 1 / len(w_full))
    return oof, tsum, np.mean(coefs, axis=0), w_full


def main(oofs, tests, y_path, test_ids_path, out_name, seed=42,
         models=("logreg", "ridge", "nnls")):
    seed_all(seed)
    names = [Path(f).name for f in oofs]
    preds = [np.load(f) for f in oofs]
    assert len({len(p) for p in preds}) == 1, "OOF length mismatch"
    M = np.column_stack(preds)
    T = np.column_stack([np.load(f) for f in tests]) if tests else None
    if tests:
        assert len({len(np.load(f)) for f in tests}) == 1, "test length mismatch"
    _check_matrix(M, T, names)

    y = load_y(str(y_path))
    assert len(y) == len(M), (len(y), len(M))
    folds = make_folds(pd.Series(y), N_SPLITS, seed)

    mean_b = M.mean(axis=1)
    rank_b = rank_avg(list(M.T))
    w = np.array([roc_auc_score(y, M[:, j]) for j in range(M.shape[1])])
    w = np.clip(w - 0.5, 1e-6, None)
    w = w / w.sum()
    wtd_b = M @ w
    print(f"stacking {M.shape[1]} OOFs over {len(y)} rows, meta-seed={seed}")
    print(f"mean  AUC={roc_auc_score(y, mean_b):.5f}")
    print(f"rank  AUC={roc_auc_score(y, rank_b):.5f}")
    print(f"wtd   AUC={roc_auc_score(y, wtd_b):.5f}")
    print(f"weights={dict(zip(names, w.round(4)))}")

    results = {}
    if "logreg" in models:
        for C in (0.1, 0.5, 1.0, 5.0, 10.0):
            oof, t = _fold_predict_logreg(M, y, T, C, folds)
            auc = roc_auc_score(y, oof)
            results[f"logreg_C{C}"] = (auc, oof, t)
            print(f"logreg C={C:<5} meta-OOF AUC={auc:.5f}")
    if "ridge" in models:
        for a in (0.1, 1.0, 10.0, 100.0):
            oof, t = _fold_predict_ridge(M, y, T, a, folds)
            auc = roc_auc_score(y, oof)
            results[f"ridge_a{a}"] = (auc, oof, t)
            print(f"ridge  a={a:<6} meta-OOF AUC={auc:.5f}")
    if "nnls" in models:
        oof, t, w_cv, w_full = _fold_predict_nnls(M, y, T, folds)
        auc = roc_auc_score(y, oof)
        results["nnls"] = (auc, oof, t)
        print(f"nnls             meta-OOF AUC={auc:.5f}")
        print(f"nnls w_cv={dict(zip(names, w_cv.round(4)))}")
        print(f"nnls w_full={dict(zip(names, w_full.round(4)))}")

    best_key = max(results, key=lambda k: results[k][0])
    best_auc, best_oof, best_test = results[best_key]
    print(f"best={best_key} meta-OOF AUC={best_auc:.5f}")
    np.save(OUTPUTS / f"{out_name}_meta_oof.npy", best_oof)
    print(f"saved {OUTPUTS / f'{out_name}_meta_oof.npy'}")
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
    ap.add_argument("--out-name", default="stack")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+",
                    choices=["logreg", "ridge", "nnls"],
                    default=["logreg", "ridge", "nnls"])
    a = ap.parse_args()
    main(a.oofs, a.tests, a.y, a.test_ids, a.out_name, a.seed, tuple(a.models))
