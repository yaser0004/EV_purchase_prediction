"""Cross-fitted 'recipe' prior (Deotte Fable lever, no external data).

A smooth low-variance prior P(Yes) from key ingredients, cross-fitted so
every train row's prior comes from fold-models that never saw it.
Consumed as XGB base_margin (head start) and/or as a feature (clue).
Target touches y only inside train folds; test prior averages fold models.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import ID_COL, N_SPLITS, OUTPUTS, TARGET
from .utils import make_folds, seed_all

RECIPE_CATS = [
    "Subsidy_Available",
    "Range_Anxiety_Level",
    "Home_Charging_Possible",
    "City_Type",
    "Current_Car_Type",
    "Gender",
]
RECIPE_NUMS = [
    "Annual_Income_USD",
    "Environmental_Concern_Level",
    "Daily_Commute_km",
    "Age",
]


def recipe_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Target-free ingredient matrix: one-hots + median-imputed numerics."""
    parts = []
    for c in RECIPE_CATS:
        if c in df.columns:
            parts.append(pd.get_dummies(df[c].astype("string"), prefix=c,
                                        dtype=np.float32))
    for c in RECIPE_NUMS:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            parts.append(pd.DataFrame(
                {c: v.fillna(v.median()).astype(np.float32)}))
    return pd.concat(parts, axis=1).fillna(0.0)


def main(seed: int = 42, c: float = 1.0):
    from .config import DATA_RAW
    seed_all(seed)
    tr = pd.read_csv(DATA_RAW / "train.csv")
    te = pd.read_csv(DATA_RAW / "test.csv")
    y = (tr[TARGET] == "Yes").astype(int).to_numpy()
    X_all, X_allt = recipe_matrix(tr).align(recipe_matrix(te), join="outer",
                                           axis=1, fill_value=0.0)
    X = X_all.to_numpy(dtype=np.float32)
    Xt = X_allt.to_numpy(dtype=np.float32)
    n_train = len(tr)
    oof = np.zeros(n_train)
    tsum = np.zeros(len(te))
    folds = make_folds(pd.Series(y), N_SPLITS, seed)
    for i, (itr, iva) in enumerate(folds):
        sc = StandardScaler().fit(X[itr])
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(sc.transform(X[itr]), y[itr])
        oof[iva] = clf.predict_proba(sc.transform(X[iva]))[:, 1]
        tsum += clf.predict_proba(sc.transform(Xt))[:, 1] / len(folds)
        print(f"[recipe] fold={i} AUC={roc_auc_score(y[iva], oof[iva]):.5f}",
              flush=True)
    print(f"[recipe] OOF AUC={roc_auc_score(y, oof):.5f}")
    np.save(OUTPUTS / f"oof_recipe_seed{seed}.npy", oof)
    np.save(OUTPUTS / f"test_recipe_seed{seed}.npy", tsum)
    print(f"saved outputs/oof_recipe_seed{seed}.npy + test_recipe_seed{seed}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.0)
    a = ap.parse_args()
    main(a.seed, a.c)
