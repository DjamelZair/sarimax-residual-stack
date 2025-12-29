from typing import Dict, List

import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def score_overall(models: Dict[str, pd.DataFrame], test_wide: pd.DataFrame, d_cols: List[str]):
    rows = []
    Y_true = test_wide.reindex(columns=d_cols).to_numpy(dtype=float)
    for name, pw in models.items():
        Y_hat = pw.reindex(index=test_wide.index, columns=d_cols).to_numpy(dtype=float)
        rows.append({"Model": name, "RMSE": rmse(Y_true, Y_hat), "MAE": mae(Y_true, Y_hat)})
    return pd.DataFrame(rows).sort_values("RMSE")


def score_by_group(models: Dict[str, pd.DataFrame], test_wide: pd.DataFrame, groups: pd.Series, d_cols: List[str], zero_share: pd.Series):
    group_order = ["G1 low intermittency", "G2 medium intermittency", "G3 high intermittency", "G4 very high intermittency"]
    rows = []
    for g in group_order:
        ids_g = groups[groups == g].index.astype(str)
        ids_g = ids_g.intersection(test_wide.index.astype(str))
        if len(ids_g) == 0:
            continue
        Yg_true = test_wide.loc[ids_g, d_cols].to_numpy(dtype=float)
        for name, pw in models.items():
            Yg_hat = pw.loc[ids_g, d_cols].to_numpy(dtype=float)
            rows.append({
                "group": g,
                "n_items": int(len(ids_g)),
                "zero_share_mean": float(pd.Series(zero_share, index=zero_share.index.astype(str)).loc[ids_g].mean()),
                "model": name,
                "RMSE": rmse(Yg_true, Yg_hat),
                "MAE": mae(Yg_true, Yg_hat),
            })
    return pd.DataFrame(rows).sort_values(["group", "RMSE"])


def residual_matrix(y_true_wide: pd.DataFrame, y_pred_wide: pd.DataFrame, d_cols: List[str]) -> pd.DataFrame:
    """
    Compute residuals (true - pred) in wide format aligned on ids and d_cols.
    """
    true = y_true_wide.reindex(columns=d_cols)
    pred = y_pred_wide.reindex(index=true.index.astype(str), columns=d_cols)
    resid = true.to_numpy(dtype=float) - pred.to_numpy(dtype=float)
    return pd.DataFrame(resid, index=true.index.astype(str), columns=d_cols)


def residual_stats_by_horizon(residuals_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize residual bias/scale per horizon column (e.g., d_1914 ... d_1941).
    """
    rows = []
    for col in residuals_wide.columns:
        r = residuals_wide[col].to_numpy(dtype=float)
        rows.append({
            "horizon": int(col.replace("d_", "")) if isinstance(col, str) and col.startswith("d_") else col,
            "day": col,
            "bias": float(np.nanmean(r)),
            "RMSE": float(np.sqrt(np.nanmean(r ** 2))),
            "MAE": float(np.nanmean(np.abs(r))),
        })
    return pd.DataFrame(rows).sort_values("horizon")


__all__ = ["rmse", "mae", "score_overall", "score_by_group", "residual_matrix", "residual_stats_by_horizon"]
