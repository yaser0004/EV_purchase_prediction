"""Feature engineering: generic base + EV-specific interactions (all guarded)."""
import numpy as np
import pandas as pd


def base_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    import pandas.api.types as _t
    for c in df.columns:
        if not _t.is_numeric_dtype(df[c]) and str(df[c].dtype) != "category":
            df[c] = df[c].astype("category")
    return df


def add_ev_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additive only; every block checks column existence so pipeline
    works before/after seeing the exact 31 competition columns."""
    df = df.copy()

    def has(*cols):
        return all(c in df.columns for c in cols)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Missing flags (synthetic ~2% missing in Income/Commute in original)
    for c in list(df.columns):
        if df[c].isna().any():
            df[f"{c}_isna"] = df[c].isna().astype("int8")

    # Median impute numerics (categoricals left for native GBDT handling)
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    if has("Charging_Stations_Near_Home", "Charging_Stations_Near_Work"):
        df["total_charging"] = (
            df["Charging_Stations_Near_Home"] + df["Charging_Stations_Near_Work"]
        )
        df["charging_diff"] = (
            df["Charging_Stations_Near_Home"] - df["Charging_Stations_Near_Work"]
        )
        df["charging_zero_flag"] = (df["total_charging"] == 0).astype("int8")

    if has("Annual_Income_USD", "Number_of_Cars_Owned"):
        df["income_per_car"] = df["Annual_Income_USD"] / (
            df["Number_of_Cars_Owned"] + 1
        )
    if has("Daily_Commute_km", "Number_of_Cars_Owned"):
        df["commute_per_car"] = df["Daily_Commute_km"] / (
            df["Number_of_Cars_Owned"] + 1
        )
    if has("Annual_Income_USD", "Daily_Commute_km"):
        df["income_x_commute"] = df["Annual_Income_USD"] * df["Daily_Commute_km"]
    if has("Age", "Annual_Income_USD"):
        df["age_x_income"] = df["Age"] * df["Annual_Income_USD"]
    if has("Age", "Annual_Income_USD"):
        df["income_per_age"] = df["Annual_Income_USD"] / (df["Age"] + 1)
    if has("total_charging", "Number_of_Cars_Owned"):
        df["charging_per_car"] = df["total_charging"] / (
            df["Number_of_Cars_Owned"] + 1
        )
    if has("Daily_Commute_km", "total_charging"):
        df["commute_x_charging"] = df["Daily_Commute_km"] * df["total_charging"]

    # Quantile bins for key numerics (helps trees, cheap)
    for c in ("Age", "Annual_Income_USD", "Daily_Commute_km"):
        if c in df.columns:
            try:
                df[f"{c}_bin"] = pd.qcut(
                    df[c], 10, labels=False, duplicates="drop"
                ).astype("Int8")
            except ValueError:
                pass

    return df


# Candidate pairs for crossing + target encoding. Guarded by existence.
CROSS_PAIRS = [
    ("Subsidy_Available", "Environmental_Concern_Level"),
    ("Subsidy_Available", "Range_Anxiety_Level"),
    ("Subsidy_Available", "Home_Charging_Possible"),
    ("Subsidy_Available", "City_Type"),
    ("Environmental_Concern_Level", "Range_Anxiety_Level"),
    ("City_Type", "Current_Car_Type"),
    ("Gender", "City_Type"),
    ("Home_Charging_Possible", "Range_Anxiety_Level"),
]

# Binned-numeric crosses (bins are built in add_ev_features, so these
# only materialize when run through build_features, not on raw frames).
BIN_CROSS_PAIRS = [
    ("Age_bin", "Subsidy_Available"),
    ("Annual_Income_USD_bin", "City_Type"),
    ("Daily_Commute_km_bin", "Home_Charging_Possible"),
    ("Current_Car_Type", "Subsidy_Available"),
]

ALL_CROSS_PAIRS = CROSS_PAIRS + BIN_CROSS_PAIRS

# Singles worth target-encoding (high-cardinality or strong signal).
TE_SINGLETONS = [
    "Subsidy_Available",
    "Environmental_Concern_Level",
    "Range_Anxiety_Level",
    "Home_Charging_Possible",
    "City_Type",
    "Current_Car_Type",
    "Gender",
]


def add_crosses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a, b in ALL_CROSS_PAIRS:
        if a in df.columns and b in df.columns:
            df[f"{a}__{b}"] = (
                df[a].astype("string") + "__" + df[b].astype("string")
            ).astype("category")
    return df


def te_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in TE_SINGLETONS if c in df.columns]
    cols += [f"{a}__{b}" for a, b in ALL_CROSS_PAIRS
             if f"{a}__{b}" in df.columns]
    return cols


# Unsupervised group means: numeric signal per category level, computed on
# the joint train+test frame. Target-free, so no leakage; helps linear/NN
# models that cannot discover group structure on their own.
GROUP_STAT_NUMS = [
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Age",
    "total_charging",
]


def add_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    import pandas.api.types as _t
    for g in TE_SINGLETONS:
        if g not in df.columns:
            continue
        try:
            gb = df.groupby(g, observed=True)
        except (TypeError, ValueError):
            continue
        for c in GROUP_STAT_NUMS:
            if c not in df.columns or not _t.is_numeric_dtype(df[c]):
                continue
            try:
                df[f"{c}_mean_by_{g}"] = gb[c].transform("mean").astype("float32")
            except (TypeError, ValueError):
                continue
    return df


def add_freq(df: pd.DataFrame) -> pd.DataFrame:
    """Joint-frame category frequency (target-free rarity signal)."""
    df = df.copy()
    n = len(df)
    if n == 0:
        return df
    for c in te_columns(df):
        try:
            vc = df[c].value_counts(dropna=False)
            df[f"{c}_freq"] = (df[c].map(vc / n)).astype("float32")
        except (TypeError, ValueError):
            continue
    return df


def target_encode_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_te_list: list[pd.DataFrame],
    cols: list[str],
    alpha: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Leak-free smoothed target encoding: stats from tr only.
    Unseen categories fall back to the fold global mean."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    outs = [t.copy() for t in X_te_list]
    global_mean = float(y_tr.mean())
    tmp = pd.DataFrame({"__y": y_tr.values})
    for c in cols:
        tmp["__c"] = X_tr[c].to_numpy()
        agg = tmp.groupby("__c", observed=True)["__y"].agg(["sum", "count"])
        enc = (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)
        for frame in [X_tr, X_va, *outs]:
            frame[f"{c}_te"] = (
                frame[c].map(enc).fillna(global_mean).astype("float32")
            )
    return X_tr, X_va, outs


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_freq(add_group_stats(add_crosses(add_ev_features(base_clean(df)))))


# --- v2 recipe space: synthetic-artifact features (dvorkin-style) ---
# Joint-frame, target-free throughout (digits + frequencies), so no leakage.
# Kept separate from build_features so the legacy 86-col space (S3 pool)
# stays reproducible. TE is fold-local (see target_encode_triple_fold).

DVORKIN_NUMS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
DVORKIN_CATS = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
TE_DVORKIN_COLS = DVORKIN_NUMS + DVORKIN_CATS


def add_dvorkin_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplier/division crosses around Env/Subsidy/Income/Commute."""
    df = df.copy()

    def has(*cols):
        return all(c in df.columns for c in cols)

    def put(name, series):
        if name not in df.columns:
            df[name] = series

    if has("Environmental_Concern_Level", "Subsidy_Available"):
        sub = df["Subsidy_Available"]
        sub_n = (sub == "Yes").astype("float32") \
            if str(sub.dtype) in ("category", "object") else sub.astype("float32")
        env = df["Environmental_Concern_Level"].astype("float32")
        put("Env_Concern_x_Subsidy", env * sub_n)
        if has("Range_Anxiety_Level"):
            anx = df["Range_Anxiety_Level"]
            anx_n = (anx == "High").astype("float32") \
                if str(anx.dtype) in ("category", "object") else anx.astype("float32")
            put("Range_Anxiety_x_Subsidy", anx_n * sub_n)
        if has("Home_Charging_Possible"):
            hc = df["Home_Charging_Possible"]
            hc_n = (hc == "Yes").astype("float32") \
                if str(hc.dtype) in ("category", "object") else hc.astype("float32")
            put("Home_Charging_x_Subsidy", hc_n * sub_n)
        if has("Annual_Income_USD"):
            put("Subsidy_x_Income", sub_n * df["Annual_Income_USD"].astype("float32"))
    if has("Environmental_Concern_Level", "Annual_Income_USD"):
        put("Env_Concern_x_Income",
            df["Environmental_Concern_Level"].astype("float32")
            * df["Annual_Income_USD"].astype("float32"))
        put("Income_per_Concern",
            df["Annual_Income_USD"].astype("float32")
            / (df["Environmental_Concern_Level"].astype("float32") + 1))
    if has("Environmental_Concern_Level", "Daily_Commute_km"):
        put("Env_Concern_x_Commute",
            df["Environmental_Concern_Level"].astype("float32")
            * df["Daily_Commute_km"].astype("float32"))
        put("Commute_per_Concern",
            df["Daily_Commute_km"].astype("float32")
            / (df["Environmental_Concern_Level"].astype("float32") + 1))
    return df


def add_digit_features(
    df: pd.DataFrame, cols: list[str] = DVORKIN_NUMS
) -> pd.DataFrame:
    """Decimal digits at 10^k for k in [-4, 4): synthetic rounding tells."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        v = df[c]
        if str(v.dtype) in ("category", "object"):
            continue
        base = v.fillna(0).astype("float64").to_numpy()
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = ((base // (10.0 ** k)) % 10).astype("int8")
    return df


def add_freq_all(
    df: pd.DataFrame, exclude: tuple[str, ...] = ("id",)
) -> pd.DataFrame:
    """Normalized value frequency for EVERY column (joint frame)."""
    df = df.copy()
    n = len(df)
    if n == 0:
        return df
    for c in df.columns:
        if c in exclude or c.endswith("_freq"):
            continue
        try:
            vc = df[c].value_counts(dropna=False)
            df[f"{c}_freq"] = (df[c].map(vc / n)).astype("float32")
        except (TypeError, ValueError):
            continue
    return df


def drop_const_corr(
    df: pd.DataFrame, exclude: tuple[str, ...] = ("id",)
) -> pd.DataFrame:
    """Drop constant + perfectly-correlated numeric columns."""
    df = df.copy()
    keep = [c for c in df.columns if c not in exclude]
    nunique = df[keep].nunique(dropna=False)
    const = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=const)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    num = [c for c in num if c not in exclude]
    to_drop: list[str] = []
    if len(num) > 1:
        corr = df[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if bool((upper[c] == 1.0).any())]
        df = df.drop(columns=to_drop)
    return df


def target_encode_triple_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    X_te_list: list[pd.DataFrame],
    cols: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    """Leak-free triple target encoding (sklearn, smooth auto/10/100).

    Encoder is fit on the outer-train fold only (with its own internal
    5-fold for the train transform); val/test are pure transforms.
    """
    from sklearn.preprocessing import TargetEncoder

    X_tr = X_tr.copy()
    X_va = X_va.copy()
    outs = [t.copy() for t in X_te_list]
    cols = [c for c in cols if c in X_tr.columns]
    for smooth_val, smooth_name in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        te = TargetEncoder(shuffle=True, cv=5, smooth=smooth_val,
                           random_state=seed)
        tr_enc = np.asarray(te.fit_transform(X_tr[cols], y_tr), dtype="float32")
        va_enc = np.asarray(te.transform(X_va[cols]), dtype="float32")
        te_encs = [np.asarray(te.transform(t[cols]), dtype="float32") for t in outs]
        for i, c in enumerate(cols):
            name = f"TE_{c}_{smooth_name}"
            X_tr[name] = tr_enc[:, i]
            X_va[name] = va_enc[:, i]
            for frame, enc in zip(outs, te_encs):
                frame[name] = enc[:, i]
    return X_tr, X_va, outs


def build_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    return drop_const_corr(
        add_freq_all(add_digit_features(add_dvorkin_interactions(base_clean(df))))
    )
