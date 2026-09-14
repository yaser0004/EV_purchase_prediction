"""Per-cell smooth prior (track J1: symbolic surface search).

Hypothesis from H: the income gradient varies by cell (Env3 top/bottom
quintile ratio ~5x vs Env5 ~1.75x). H tested it stepwise (IncQxEnv
dummies) or globally (log-income slopes) — never as per-cell smooth
curves. Here each Subsidy x Anxiety x Env cell (30) gets its own tiny
logistic on log_income + commute, cross-fitted so every train row's
prior comes from fold-models that never saw it.

Leak-free: scalers fit on fold-train rows only; test averages fold
models. Cells with <10 fold-train positives fall back to the smoothed
fold-train cell rate (veto cells are near-all-negative).

Usage: python -m src.cell_prior --seed 42
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import DATA_RAW, N_SPLITS, OUTPUTS, TARGET
from .utils import make_folds, seed_all

CELL_COLS = ["Subsidy_Available", "Range_Anxiety_Level",
             "Environmental_Concern_Level"]
MIN_POS = 10
SMOOTH = 100.0


def _num(df: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(df[col], errors="coerce")
    return np.asarray(v.fillna(v.median()), dtype=np.float64)


def main(seed: int = 42):
    seed_all(seed)
    tr = pd.read_csv(DATA_RAW / "train.csv")
    te = pd.read_csv(DATA_RAW / "test.csv")
    y = (tr[TARGET] == "Yes").astype(int).to_numpy()

    cell_tr = (tr[CELL_COLS].astype("string").agg("__".join, axis=1))
    cell_te = (te[CELL_COLS].astype("string").agg("__".join, axis=1))
    li_tr = np.log(_num(tr, "Annual_Income_USD"))
    cm_tr = _num(tr, "Daily_Commute_km")
    li_te = np.log(_num(te, "Annual_Income_USD"))
    cm_te = _num(te, "Daily_Commute_km")

    oof = np.zeros(len(tr))
    tsum = np.zeros(len(te))
    folds = make_folds(pd.Series(y), N_SPLITS, seed)
    for i, (itr, iva) in enumerate(folds):
        mu = np.array([li_tr[itr].mean(), cm_tr[itr].mean()])
        sd = np.array([li_tr[itr].std(), cm_tr[itr].std()]) + 1e-9
        Z_tr = np.column_stack([(li_tr[itr] - mu[0]) / sd[0],
                                (cm_tr[itr] - mu[1]) / sd[1]])
        Z_va = np.column_stack([(li_tr[iva] - mu[0]) / sd[0],
                                (cm_tr[iva] - mu[1]) / sd[1]])
        Z_te = np.column_stack([(li_te - mu[0]) / sd[0],
                                (cm_te - mu[1]) / sd[1]])
        gm = float(y[itr].mean())
        va_cells = cell_tr.iloc[iva].to_numpy()
        oof_va = np.empty(len(iva))
        for cell in np.unique(cell_tr.iloc[itr].to_numpy()):
            m_tr = (cell_tr.iloc[itr].to_numpy() == cell)
            m_va = (va_cells == cell)
            m_te = (cell_te.to_numpy() == cell)
            yc = y[itr][m_tr]
            n_pos = int(yc.sum())
            if n_pos < MIN_POS or yc.mean() in (0.0, 1.0):
                rate = (yc.sum() + SMOOTH * gm) / (len(yc) + SMOOTH)
                if m_va.any():
                    oof_va[m_va] = rate
                if m_te.any():
                    tsum[m_te] += rate / len(folds)
                continue
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(Z_tr[m_tr], yc)
            if m_va.any():
                oof_va[m_va] = clf.predict_proba(Z_va[m_va])[:, 1]
            if m_te.any():
                tsum[m_te] += clf.predict_proba(Z_te[m_te])[:, 1] / len(folds)
        # test cells never seen in fold-train (should not happen): global mean
        oof[iva] = oof_va
        fold_auc = roc_auc_score(y[iva], oof[iva])
        n_fallback = sum(
            1 for cell in np.unique(va_cells)
            if int(y[itr][(cell_tr.iloc[itr].to_numpy() == cell)].sum()) < MIN_POS)
        print(f"[cell_prior] fold={i} AUC={fold_auc:.5f} "
              f"fallback_cells={n_fallback}", flush=True)
    print(f"[cell_prior] OOF AUC={roc_auc_score(y, oof):.5f}")
    np.save(OUTPUTS / f"oof_cell_seed{seed}.npy", oof)
    np.save(OUTPUTS / f"test_cell_seed{seed}.npy", tsum)
    print(f"saved outputs/oof_cell_seed{seed}.npy + test_cell_seed{seed}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.seed)
