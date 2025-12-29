import pandas as pd
import numpy as np

def make_train_val_frames(sales_df: pd.DataFrame, horizon: int = 28, id_col: str = "id"):
    """
    Split a wide sales table into train/validation matrices.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Wide table with an id column and day columns like d_1, d_2, ...
    horizon : int
        Number of trailing days reserved for validation.
    id_col : str
        Name of the identifier column.

    Returns
    -------
    (train_wide, val_wide) : Tuple[pd.DataFrame, pd.DataFrame]
        Train/val matrices aligned on the same index.
    """
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No day columns found (expected names like 'd_1', 'd_2', ...).")

    ids = sales_df[id_col].astype(str).values
    Y = sales_df[d_cols].astype(float).to_numpy()

    train_cols = d_cols[:-horizon]
    val_cols = d_cols[-horizon:]

    train_wide = pd.DataFrame(Y[:, : len(train_cols)], index=ids, columns=train_cols)
    val_wide = pd.DataFrame(Y[:, len(train_cols) :], index=ids, columns=val_cols)
    return train_wide, val_wide


def split_long_by_cutoff(df_long: pd.DataFrame, cutoff: int, horizon: int = 28):
    """
    Split a long-format table into train/validation blocks using a day cutoff.

    Expects columns: id, d_index, sales, plus any features.
    """
    if "d_index" not in df_long.columns:
        raise ValueError("df_long must contain 'd_index'.")

    train = df_long[df_long["d_index"] <= cutoff].copy()
    val = df_long[(df_long["d_index"] > cutoff) & (df_long["d_index"] <= cutoff + horizon)].copy()
    return train, val


def align_pred_wide(wide: pd.DataFrame, ids, d_cols):
    """Reindex a wide prediction matrix to match expected ids and columns."""
    wide = wide.copy()
    wide.index = wide.index.astype(str)
    return wide.reindex(index=pd.Index(ids, dtype=str), columns=d_cols)


def build_sales_long_features(
    calendar_df: pd.DataFrame,
    sales_wide_df: pd.DataFrame,
    sell_prices_df: pd.DataFrame,
    out_path=None,
    force_rebuild: bool = False,
):
    """
    Build long-format feature table with lags/rolls and cached write/read.

    Parameters
    ----------
    calendar_df : pd.DataFrame
        Calendar table with same length as day columns in sales_wide_df.
    sales_wide_df : pd.DataFrame
        M5-style wide sales table (id + d_# cols).
    sell_prices_df : pd.DataFrame
        Sell price table (store_id, item_id, wm_yr_wk, sell_price).
    out_path : Path-like or None
        If provided, cache is read/written here.
    force_rebuild : bool
        If True, rebuild even if cache exists.
    """
    from pathlib import Path

    out_path = Path(out_path) if out_path is not None else None

    day_cols = [c for c in sales_wide_df.columns if c.startswith("d_")]
    if not day_cols:
        raise ValueError("sales_wide_df must contain day columns like d_1, d_2, ...")
    required_last_day = max(int(c.replace("d_", "")) for c in day_cols)

    if out_path and out_path.exists() and not force_rebuild:
        sales_long = pd.read_csv(out_path)
        sales_long["d_index"] = sales_long["d_index"].astype(int)
        max_d = int(sales_long["d_index"].max())
        if max_d >= required_last_day:
            return sales_long

    calendar_df = calendar_df.copy()
    calendar_df["d_index"] = calendar_df.index + 1
    # Seasonality helpers
    calendar_df["weekofyear"] = pd.to_datetime(calendar_df["date"]).dt.isocalendar().week.astype(int)
    calendar_df["quarter"] = pd.to_datetime(calendar_df["date"]).dt.quarter.astype(int)
    # Sin/cos encodings for weekly and yearly cycles
    calendar_df["sin_week"] = np.sin(2 * np.pi * calendar_df["wday"] / 7)
    calendar_df["cos_week"] = np.cos(2 * np.pi * calendar_df["wday"] / 7)
    calendar_df["sin_year"] = np.sin(2 * np.pi * (calendar_df["d_index"] % 365) / 365)
    calendar_df["cos_year"] = np.cos(2 * np.pi * (calendar_df["d_index"] % 365) / 365)

    # Event proximity (days to next/prev event, coarse windows)
    for ev_col in ["event_name_1", "event_name_2"]:
        has_ev = calendar_df[ev_col].notna().astype(int)
        calendar_df[f"{ev_col}_prev_gap"] = has_ev.replace(0, np.nan)
        calendar_df[f"{ev_col}_next_gap"] = has_ev.replace(0, np.nan)
        # prev gap
        last = None
        prev_gaps = []
        for i, flag in enumerate(has_ev.tolist()):
            if flag == 1:
                last = i
                prev_gaps.append(0)
            else:
                prev_gaps.append(np.nan if last is None else i - last)
        # next gap
        next_gaps = [np.nan] * len(has_ev)
        next_idx = None
        for i in range(len(has_ev) - 1, -1, -1):
            if has_ev.iat[i] == 1:
                next_idx = i
                next_gaps[i] = 0
            else:
                next_gaps[i] = np.nan if next_idx is None else next_idx - i
        calendar_df[f"{ev_col}_prev_gap"] = prev_gaps
        calendar_df[f"{ev_col}_next_gap"] = next_gaps
        # Binary windows (within 3/7 days of event)
        calendar_df[f"{ev_col}_window_3"] = ((calendar_df[f"{ev_col}_prev_gap"] <= 3) | (calendar_df[f"{ev_col}_next_gap"] <= 3)).astype(int)
        calendar_df[f"{ev_col}_window_7"] = ((calendar_df[f"{ev_col}_prev_gap"] <= 7) | (calendar_df[f"{ev_col}_next_gap"] <= 7)).astype(int)

    id_cols = [c for c in sales_wide_df.columns if not c.startswith("d_")]
    sales_long = sales_wide_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d_col",
        value_name="sales",
    )
    sales_long["d_index"] = sales_long["d_col"].str.replace("d_", "", regex=False).astype(int)
    sales_long = sales_long.merge(calendar_df, on="d_index", how="left")

    sales_long["base_id"] = (
        sales_long["id"]
        .str.replace("_validation$", "", regex=True)
        .str.replace("_evaluation$", "", regex=True)
    )
    parts = sales_long["base_id"].str.split("_", expand=True)
    sales_long["item_id"] = parts[0] + "_" + parts[1] + "_" + parts[2]
    sales_long["store_id"] = parts[3] + "_" + parts[4]
    sales_long = sales_long.drop(columns=["base_id"])

    sales_long = sales_long.merge(
        sell_prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    sales_long = sales_long.sort_values(["store_id", "item_id", "d_index"])

    sales_long["sell_price_was_missing"] = sales_long["sell_price"].isna().astype(int)
    sales_long["sell_price"] = (
        sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.ffill().bfill())
    )
    sales_long["price_change"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].diff()
    prev_price = sales_long.groupby(["store_id", "item_id"])["sell_price"].shift(1)
    sales_long["price_pct_change"] = sales_long["price_change"] / prev_price.replace(0, np.nan)
    # Rolling price stats (weeks ~ 7 days)
    sales_long["price_roll_mean_4w"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.rolling(28, min_periods=1).mean())
    sales_long["price_roll_std_4w"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.rolling(28, min_periods=1).std())
    sales_long["price_roll_mean_8w"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.rolling(56, min_periods=1).mean())
    sales_long["price_roll_std_8w"] = sales_long.groupby(["store_id", "item_id"])["sell_price"].transform(lambda s: s.rolling(56, min_periods=1).std())
    # Simple promo flag: price below 97% of 8w mean
    sales_long["promo_flag"] = (sales_long["sell_price"] < 0.97 * sales_long["price_roll_mean_8w"]).astype(int)
    # Price level vs 8w mean and sharp change flags
    sales_long["price_level_8w"] = sales_long["sell_price"] / sales_long["price_roll_mean_8w"].replace(0, np.nan)
    sales_long["price_drop_flag"] = (sales_long["price_pct_change"] < -0.05).astype(int)
    sales_long["price_spike_flag"] = (sales_long["price_pct_change"] > 0.05).astype(int)

    sales_long = sales_long.sort_values(["id", "d_index"])
    lag_list = [1, 7, 13, 14, 15, 21, 28]
    for lag in lag_list:
        sales_long[f"lag_{lag}"] = sales_long.groupby("id")["sales"].shift(lag)

    # Leakage-safe rolling stats on lagged sales
    sales_long["sales_lag1"] = sales_long.groupby("id")["sales"].shift(1)
    roll_windows = [7, 14, 21, 28]
    for window in roll_windows:
        sales_long[f"roll_mean_{window}"] = (
            sales_long.groupby("id")["sales_lag1"].transform(lambda s: s.rolling(window).mean())
        )
        sales_long[f"roll_std_{window}"] = (
            sales_long.groupby("id")["sales_lag1"].transform(lambda s: s.rolling(window).std())
        )
    # Rolling zero share (past only)
    for window in [7, 14, 28]:
        sales_long[f"roll_zero_share_{window}"] = (
            sales_long.groupby("id")["sales_lag1"].transform(lambda s: s.rolling(window).apply(lambda x: np.mean(x == 0), raw=True))
        )
    # Time since last sale
    def time_since_last_sale(arr):
        out = []
        last = None
        for i, v in enumerate(arr):
            if v > 0:
                last = i
                out.append(0)
            else:
                out.append(np.nan if last is None else i - last)
        return out
    sales_long["time_since_last_sale"] = sales_long.groupby("id")["sales"].transform(time_since_last_sale)

    sales_long = sales_long.drop(columns=["sales_lag1"])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sales_long.to_csv(out_path, index=False)
    return sales_long


def load_or_build_features(
    calendar_df: pd.DataFrame,
    sales_wide_df: pd.DataFrame,
    sell_prices_df: pd.DataFrame,
    out_path: str,
    force_rebuild: bool = False,
    sample_n: int = 5,
):
    """
    Load the cached feature CSV if present (and not force_rebuild), otherwise build and cache it.
    Prints quick sanity info (shape, max d_index, missingness on key columns, sample rows).
    """
    from pathlib import Path

    out_path = Path(out_path)
    if out_path.exists() and not force_rebuild:
        df = pd.read_csv(out_path)
    else:
        df = build_sales_long_features(
            calendar_df=calendar_df,
            sales_wide_df=sales_wide_df,
            sell_prices_df=sell_prices_df,
            out_path=out_path,
            force_rebuild=force_rebuild,
        )

    df["d_index"] = df["d_index"].astype(int)
    print(f"Features ready: {out_path} | shape={df.shape} | max d_index={df['d_index'].max()}")

    # Simple corruption/consistency checks
    expected_last_day = max(int(c.replace("d_", "")) for c in sales_wide_df.columns if c.startswith("d_"))
    if df["d_index"].max() < expected_last_day:
        print(f"WARNING: feature cache ends at d_index {df['d_index'].max()} < expected {expected_last_day}")

    key_cols = [
        "sell_price",
        "sell_price_was_missing",
        "lag_1",
        "lag_7",
        "lag_13",
        "lag_14",
        "lag_15",
        "lag_21",
        "lag_28",
        "roll_mean_7",
        "roll_mean_14",
        "roll_mean_21",
        "roll_mean_28",
        "roll_zero_share_28",
    ]
    present = [c for c in key_cols if c in df.columns]
    if present:
        miss = df[present].isna().mean().sort_values(ascending=False)
        print("\nMissingness on key cols (fraction of rows):")
        print(miss.to_string())

    non_feature = {"id", "d_col", "d_index", "sales"}
    feature_cols = [c for c in df.columns if c not in non_feature]
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols)

    print("\nSample rows:")
    print(df.sample(sample_n, random_state=42))

    return df


__all__ = [
    "make_train_val_frames",
    "split_long_by_cutoff",
    "align_pred_wide",
    "build_sales_long_features",
    "load_or_build_features",
]
