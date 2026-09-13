"""01 EDA: shapes, target, missing, train/test shift. Usage:
python -m src.eda  (run from source/ with data/raw/train.csv present)
"""
import pandas as pd

from .config import DATA_RAW, ID_COL, TARGET


def main():
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    print(f"train {train.shape} test {test.shape}")
    print(f"columns: {list(train.columns)}")
    print(f"target balance:\n{train[TARGET].value_counts(normalize=True)}")
    miss = train.isna().mean().sort_values(ascending=False)
    print("missing train>0:\n", miss[miss > 0].head(15).to_string())
    miss_t = test.isna().mean().sort_values(ascending=False)
    print("missing test>0:\n", miss_t[miss_t > 0].head(15).to_string())
    print(f"train id unique={train[ID_COL].is_unique} "
          f"test id unique={test[ID_COL].is_unique}")
    overlap = set(train[ID_COL]) & set(test[ID_COL])
    print(f"id overlap train/test: {len(overlap)}")
    for c in train.columns:
        if c in (ID_COL, TARGET):
            continue
        if c in test.columns and train[c].dtype == object:
            tr, te = set(train[c].dropna().unique()), set(test[c].dropna().unique())
            if tr != te:
                print(f"cat shift {c}: train-only={sorted(tr-te)[:5]} "
                      f"test-only={sorted(te-tr)[:5]}")
    num = train.select_dtypes(include="number").columns.drop(
        [c for c in [TARGET] if c in train.columns], errors="ignore"
    )
    shift = pd.DataFrame({
        "train_mean": train[num].mean(numeric_only=True),
        "test_mean": test[[c for c in num if c in test.columns]].mean(numeric_only=True),
    })
    shift["rel_diff"] = (shift["test_mean"] - shift["train_mean"]).abs() / (
        shift["train_mean"].abs() + 1e-9)
    print("top numeric shift:\n", shift.sort_values("rel_diff", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
