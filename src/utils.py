"""Small helpers: seeding, folds, submission validation."""
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import ID_COL, TARGET


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_folds(
    y: pd.Series, n_splits: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def check_submission(
    sub: pd.DataFrame, test_ids: pd.Series, target: str = TARGET
) -> None:
    assert list(sub.columns) == [ID_COL, target], sub.columns.tolist()
    assert len(sub) == len(test_ids), (len(sub), len(test_ids))
    assert sub[ID_COL].equals(test_ids.reset_index(drop=True)), "id order mismatch"
    assert sub[target].between(0, 1).all(), "probabilities out of [0,1]"
    assert sub[target].notna().all(), "NaN probabilities"
