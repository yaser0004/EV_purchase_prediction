"""Leak-free blending of OOF files: mean / rank-average / OOF-weighted."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import ID_COL, OUTPUTS, SUBMISSIONS, TARGET


def rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    ranks = np.mean([rankdata(p) / len(p) for p in preds], axis=0)
    return ranks


def load_y(y_path: str) -> np.ndarray:
    y_raw = pd.read_csv(y_path).iloc[:, -1]
    import pandas.api.types as _t
    return y_raw.astype(int).to_numpy() if _t.is_numeric_dtype(y_raw) \
        else (y_raw == "Yes").astype(int).to_numpy()


def main(files: list[Path], y_path: str | None = None,
         test_files: list[Path] | None = None,
         test_ids_path: str | None = None, out_name: str = "blend"):
    from .utils import check_submission

    preds = [np.load(f) for f in files]
    mean_blend = np.mean(preds, axis=0)
    rank_blend = rank_avg(preds)
    print(f"blending {len(preds)} OOF files")
    weights = None
    if y_path:
        y = load_y(y_path)
        print(f"mean  AUC={roc_auc_score(y, mean_blend):.5f}")
        print(f"rank  AUC={roc_auc_score(y, rank_blend):.5f}")
        w = np.array([roc_auc_score(y, p) for p in preds])
        w = np.clip(w - 0.5, 1e-6, None)
        weights = w / w.sum()
        print(f"weights={dict(zip([f.name for f in files], weights.round(4)))}")
        print(f"wtd   AUC={roc_auc_score(y, np.average(preds, axis=0, weights=weights)):.5f}")
        M = np.column_stack(preds)
        lr = LogisticRegression(max_iter=1000).fit(M, y)
        print(f"stack AUC={roc_auc_score(y, lr.predict_proba(M)[:, 1]):.5f}")
    np.save(OUTPUTS / f"{out_name}_mean.npy", mean_blend)
    np.save(OUTPUTS / f"{out_name}_rank.npy", rank_blend)
    if test_files and test_ids_path:
        tpreds = [np.load(f) for f in test_files]
        assert all(len(t) == len(tpreds[0]) for t in tpreds), "test length mismatch"
        blend = np.average(tpreds, axis=0, weights=weights) if weights is not None \
            else np.mean(tpreds, axis=0)
        ids = pd.read_csv(test_ids_path, usecols=[ID_COL])[ID_COL]
        sub = pd.DataFrame({ID_COL: ids.values, TARGET: blend})
        check_submission(sub, ids)
        sp = SUBMISSIONS / f"sub_{out_name}.csv"
        sub.to_csv(sp, index=False)
        print(f"saved {sp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--y", default=None)
    ap.add_argument("--tests", nargs="*", type=Path, default=None)
    ap.add_argument("--test-ids", default=None)
    ap.add_argument("--out-name", default="blend")
    a = ap.parse_args()
    main(a.files, a.y, a.tests, a.test_ids, a.out_name)
