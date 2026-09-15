"""Optuna tuning for CatBoost (CPU, AUC early-stop) on a single fold.
Usage: python -m src.tune_cat --trials 30
Writes outputs/best_cat.json
"""
import argparse
import json
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from src.config import ID_COL, OUTPUTS, TARGET
from src.features import build_features
from src.train_gbdt import _fit_predict
from src.utils import seed_all


def load():
    from src.config import DATA_RAW
    df = pd.read_csv(DATA_RAW / "train.csv")
    import pandas.api.types as _t
    y = df[TARGET].astype(int) if _t.is_numeric_dtype(df[TARGET]) \
        else (df[TARGET] == "Yes").astype(int)
    X = build_features(df.drop(columns=[TARGET]))
    fc = [c for c in X.columns if c != ID_COL]
    cat = [c for c in fc if str(X[c].dtype) == "category"]
    return X[fc], y, cat


def main(trials: int, seed: int = 42, out_name: str = "best_cat.json"):
    import optuna
    seed_all(seed)
    X, y, cat = load()
    tr, va = next(StratifiedKFold(5, shuffle=True,
                                  random_state=seed).split(X, y))
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    def objective(trial: "optuna.Trial") -> float:
        ov = {
            "depth": trial.suggest_int("depth", 5, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08),
        }
        p_va, _, _ = _fit_predict("cat", X_tr, y_tr, X_va, y_va, cat,
                                   "cpu", ov)
        return roc_auc_score(y_va, p_va)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"best fold0 AUC={study.best_value:.5f}")
    print(json.dumps(study.best_params, indent=2))
    (OUTPUTS / out_name).write_text(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-name", default="best_cat.json")
    a = ap.parse_args()
    main(a.trials, a.seed, a.out_name)
