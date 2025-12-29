from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX


def pred_naive(train_df: pd.DataFrame, horizon: int):
    last = train_df.iloc[:, -1].to_numpy(dtype=float)
    return np.repeat(last[:, None], horizon, axis=1)


def pred_seasonal_naive(train_df: pd.DataFrame, horizon: int, season: int = 7):
    last_season = train_df.iloc[:, -season:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / season))
    tiled = np.tile(last_season, reps)
    return tiled[:, :horizon]


def pred_moving_average(train_df: pd.DataFrame, horizon: int, window: int = 7):
    avg = train_df.iloc[:, -window:].to_numpy(dtype=float).mean(axis=1)
    return np.repeat(avg[:, None], horizon, axis=1)


def build_preprocess(num_cols: List[str], cat_cols: List[str]):
    ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ord_enc, cat_cols),
        ],
        remainder="drop",
    )


def build_histgbr_pipeline(num_cols: List[str], cat_cols: List[str], **kwargs):
    preprocess = build_preprocess(num_cols, cat_cols)
    model = HistGradientBoostingRegressor(**kwargs)
    return Pipeline([("prep", preprocess), ("model", model)])


def eval_model_against_val_wide(
    name: str,
    estimator,
    X_train,
    y_train,
    val_long_exact: pd.DataFrame,
    val_wide: pd.DataFrame,
    feature_cols: List[str],
):
    """
    Fit estimator, predict val grid, and compute wide predictions aligned to val_wide.
    Returns (summary dict, wide preds).
    """
    pipe = estimator
    pipe.fit(X_train[feature_cols], y_train)

    X_val_exact = val_long_exact[feature_cols]
    pred_long = pipe.predict(X_val_exact)
    pred_long = np.maximum(pred_long, 0.0)

    pred_df = val_long_exact[["id", "d_index"]].copy()
    pred_df["pred"] = pred_long
    Y_pred_wide = pred_df.pivot(index="id", columns="d_index", values="pred")
    Y_pred_wide.columns = [f"d_{int(c)}" for c in Y_pred_wide.columns]
    Y_pred_wide = Y_pred_wide.reindex(index=val_wide.index, columns=val_wide.columns)

    mask = ~Y_pred_wide.isna()
    Y_true = val_wide.to_numpy(dtype=float)[mask.to_numpy()]
    Y_pred = Y_pred_wide.to_numpy(dtype=float)[mask.to_numpy()]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    return {
        "model": name,
        "RMSE": rmse(Y_true, Y_pred),
        "MAE": mae(Y_true, Y_pred),
        "pipe": pipe,
        "Y_pred_wide": Y_pred_wide,
    }


def baseline_sarimax_wide(
    df: pd.DataFrame,
    cutoff: int,
    horizon: int,
    future_d_cols: List[str],
    ids,
    order=(1, 0, 0),
    seasonal_order=(0, 1, 1, 7),
    maxiter: int = 50,
    progress: bool = False,
):
    """
    Fit a SARIMAX per series on history up to cutoff and forecast horizon steps.
    Returns wide DataFrame aligned to ids x future_d_cols.
    """
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    preds = []
    ids = pd.Index(ids, dtype=str)
    iter_ids = ids
    if progress and tqdm is not None:
        iter_ids = tqdm(ids, desc="SARIMAX per series", mininterval=1.0)
    for rid in iter_ids:
        y = df[df["id"] == rid].sort_values("d_index")
        y_hist = pd.to_numeric(y[y["d_index"] <= cutoff]["sales"], errors="coerce").fillna(0.0).astype(float).values
        if len(y_hist) == 0:
            preds.append(np.zeros(horizon, dtype=float))
            continue
        try:
            model = SARIMAX(y_hist, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=maxiter)
            fc = res.forecast(steps=horizon)
            fc = np.maximum(fc, 0.0)
        except Exception:
            fc = np.zeros(horizon, dtype=float)
        preds.append(fc)

    preds = np.asarray(preds, dtype=float)
    return pd.DataFrame(preds, index=ids, columns=future_d_cols)


__all__ = [
    "pred_naive",
    "pred_seasonal_naive",
    "pred_moving_average",
    "build_preprocess",
    "build_histgbr_pipeline",
    "eval_model_against_val_wide",
    "baseline_sarimax_wide",
]
