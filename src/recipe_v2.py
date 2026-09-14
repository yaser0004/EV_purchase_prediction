"""Cross-fitted 'recipe v2' prior (track H: generator recovery).

Extends recipe.py: Environmental_Concern_Level as categorical (logits are
unequally spaced), log/income + commute curvature, and the multiplier
crosses Subsidy x Anxiety / Subsidy x Env plus Income-quintile x Env.

All transforms are joint train+test and target-free (quantile edges,
medians, one-hot alignment), so no leakage; y is touched only inside
train folds. Test prior averages the 5 fold models.

H2 search (seed 42, 5-fold pooled OOF): v1 0.93810 -> v2 0.93866.
Usage: python -m src.recipe_v2 --seed 42 [--c 1.0]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import DATA_RAW, N_SPLITS, OUTPUTS, TARGET
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
    "Daily_Commute_km",
    "Age",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
]


def recipe_v2_matrix(tr: pd.DataFrame, te: pd.DataFrame):
    """Joint target-free ingredient matrix for train/test frames."""
    joint = pd.concat([tr, te], ignore_index=True)
    n_tr = len(tr)

    parts = []
    for c in RECIPE_CATS:
        if c in joint.columns:
            parts.append(pd.get_dummies(joint[c].astype("string"), prefix=c,
                                        dtype=np.float32))
    if "Environmental_Concern_Level" in joint.columns:
        parts.append(pd.get_dummies(joint["Environmental_Concern_Level"].astype("string"),
                                    prefix="Env", dtype=np.float32))
    for c in RECIPE_NUMS:
        if c in joint.columns:
            v = pd.to_numeric(joint[c], errors="coerce")
            parts.append(pd.DataFrame(
                {c: np.asarray(v.fillna(v.median()), dtype=np.float32)}))

    inc = np.asarray(pd.to_numeric(joint["Annual_Income_USD"], errors="coerce")
                     .fillna(pd.to_numeric(joint["Annual_Income_USD"],
                                           errors="coerce").median()),
                     dtype=np.float64)
    cm = np.asarray(pd.to_numeric(joint["Daily_Commute_km"], errors="coerce")
                    .fillna(pd.to_numeric(joint["Daily_Commute_km"],
                                          errors="coerce").median()),
                    dtype=np.float64)
    parts.append(pd.DataFrame({
        "log_income": np.log(inc).astype(np.float32),
        "income_std_sq": (((inc - inc.mean()) / inc.std()) ** 2).astype(np.float32),
        "commute_std_sq": (((cm - cm.mean()) / cm.std()) ** 2).astype(np.float32),
    }))

    sub = joint["Subsidy_Available"].astype("string")
    anx = joint["Range_Anxiety_Level"].astype("string")
    env = joint["Environmental_Concern_Level"].astype("string")
    parts.append(pd.get_dummies(sub + "__" + anx, prefix="Sub_Anx", dtype=np.float32))
    parts.append(pd.get_dummies(sub + "__" + env, prefix="Sub_Env", dtype=np.float32))
    inc_q = pd.qcut(joint["Annual_Income_USD"], 5,
                    labels=["q1", "q2", "q3", "q4", "q5"])
    parts.append(pd.get_dummies(inc_q.astype("string") + "__" + env,
                                prefix="IncQ_Env", dtype=np.float32))

    full = pd.concat(parts, axis=1).fillna(0.0)
    return (full.iloc[:n_tr].reset_index(drop=True),
            full.iloc[n_tr:].reset_index(drop=True))


def main(seed: int = 42, c: float = 1.0):
    seed_all(seed)
    tr = pd.read_csv(DATA_RAW / "train.csv")
    te = pd.read_csv(DATA_RAW / "test.csv")
    y = (tr[TARGET] == "Yes").astype(int).to_numpy()
    X_all, X_allt = recipe_v2_matrix(tr, te)
    X = X_all.to_numpy(dtype=np.float32)
    Xt = X_allt.to_numpy(dtype=np.float32)
    n_train = len(tr)
    oof = np.zeros(n_train)
    tsum = np.zeros(len(te))
    folds = make_folds(pd.Series(y), N_SPLITS, seed)
    for i, (itr, iva) in enumerate(folds):
        sc = StandardScaler().fit(X[itr])
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(sc.transform(X[itr]), y[itr])
        oof[iva] = clf.predict_proba(sc.transform(X[iva]))[:, 1]
        tsum += clf.predict_proba(sc.transform(Xt))[:, 1] / len(folds)
        print(f"[recipe_v2] fold={i} AUC={roc_auc_score(y[iva], oof[iva]):.5f}",
              flush=True)
    print(f"[recipe_v2] OOF AUC={roc_auc_score(y, oof):.5f}")
    np.save(OUTPUTS / f"oof_recipe_v2_seed{seed}.npy", oof)
    np.save(OUTPUTS / f"test_recipe_v2_seed{seed}.npy", tsum)
    print(f"saved outputs/oof_recipe_v2_seed{seed}.npy + test_recipe_v2_seed{seed}.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.0)
    a = ap.parse_args()
    main(a.seed, a.c)
