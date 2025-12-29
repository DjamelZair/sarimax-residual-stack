"""
Residual boosting experiment:
- Load cached long-format features and wide sales.
- Fit SARIMAX per series on the training window (no leakage).
- Build residual targets from in-sample one-step-ahead fits.
- Train a gradient boosting regressor on residuals with calendar/price/lag features.
- Predict residuals on the validation horizon and combine with SARIMAX forecasts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Make repository root importable when running as a script
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_prep import make_train_val_frames, load_or_build_features, split_long_by_cutoff
from src.eval import score_overall, score_by_group, residual_matrix, residual_stats_by_horizon
from src.models import baseline_sarimax_wide
from src.plots import plot_forecast_item, plot_horizon_delta

DEFAULT_SALES = ROOT / "data" / "sales_train_validation_afcs2025.csv"
DEFAULT_CAL = ROOT / "data" / "calendar_afcs2025.csv"
DEFAULT_PRICES = ROOT / "data" / "sell_prices_afcs2025.csv"
DEFAULT_CACHE = ROOT / "data" / "sales_features_with_calendar_prices.csv"
DEFAULT_SARIMAX_CACHE = ROOT / "data" / "sarimax_val_preds.npy"


def fit_sarimax_residuals(y_hist: np.ndarray, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7), maxiter: int = 30):
    """
    Fit SARIMAX on a single series and return in-sample predictions and residuals.
    One-step-ahead predictions prevent future leakage.
    """
    model = SARIMAX(y_hist, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False, maxiter=maxiter)
    fitted = res.predict(start=0, end=len(y_hist) - 1, dynamic=False)
    resid = y_hist - fitted
    return fitted, resid


def build_residual_training_frame(train_long: pd.DataFrame, train_wide: pd.DataFrame, order, seasonal_order, maxiter: int):
    """
    Compute SARIMAX residuals per series and merge with training features.
    """
    res_rows = []
    ids = train_wide.index.astype(str)
    d_idx_cols = [int(c.replace("d_", "")) for c in train_wide.columns]
    y_mat = train_wide.to_numpy(dtype=float)
    iter_ids = enumerate(ids)
    if tqdm is not None:
        iter_ids = tqdm(iter_ids, total=len(ids), desc="Residual targets", mininterval=1.0)
    for i, rid in iter_ids:
        y_hist = y_mat[i]
        try:
            _, resid = fit_sarimax_residuals(y_hist, order=order, seasonal_order=seasonal_order, maxiter=maxiter)
        except Exception:
            resid = np.zeros_like(y_hist, dtype=float)
        for d_int, r in zip(d_idx_cols, resid):
            res_rows.append({"id": rid, "d_index": d_int, "residual": float(r)})
    residuals_long = pd.DataFrame(res_rows)
    train_resid = train_long.merge(residuals_long, on=["id", "d_index"], how="inner")
    return train_resid


def add_top_event_buckets(train_df: pd.DataFrame, val_df: pd.DataFrame, col: str, k: int = 10):
    """
    Bucket rare event names into 'other', keep top-k frequent.
    """
    top = set(train_df[col].dropna().value_counts().head(k).index)
    for df in [train_df, val_df]:
        df[f"{col}_top"] = df[col].where(df[col].isin(top), "other").fillna("none")


def add_target_encoding(train_df: pd.DataFrame, val_df: pd.DataFrame, col: str, target: str):
    """
    Mean target encoding for a categorical column, fit on train, apply to train/val.
    """
    overall = train_df[target].mean()
    mapping = train_df.groupby(col)[target].mean()
    train_df[f"te_{col}_resid"] = train_df[col].map(mapping).fillna(overall)
    val_df[f"te_{col}_resid"] = val_df[col].map(mapping).fillna(overall)


def build_residual_model(train_resid: pd.DataFrame, model_name: str = "hgb", params: dict | None = None):
    """
    Train a residual model using simple ordinal encoding for categoricals.
    model_name: "hgb" (HistGradientBoosting) or "xgb" (XGBoost, if available).
    """
    feature_cols = [c for c in train_resid.columns if c not in {"id", "d_col", "d_index", "sales", "residual"}]
    cat_cols = [c for c in feature_cols if train_resid[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", enc, cat_cols),
        ],
        remainder="drop",
    )
    params = params or {}
    if model_name == "xgb" and XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=params.get("n_estimators", 600),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 8),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            objective="reg:squarederror",
            n_jobs=4,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=params.get("max_depth", 10),
            max_iter=params.get("max_iter", 800),
            learning_rate=params.get("learning_rate", 0.05),
            min_samples_leaf=params.get("min_samples_leaf", 10),
            l2_regularization=params.get("l2_regularization", 0.1),
        )
    pipe = Pipeline([("prep", pre), ("model", model)])
    X = train_resid[feature_cols]
    y = train_resid["residual"]
    pipe.fit(X, y)
    return pipe, feature_cols


def grid_search_residual_model(train_resid: pd.DataFrame, model_name: str, param_grid: list, sample_n: int = 250000, val_frac: float = 0.2, random_state: int = 42):
    """
    Lightweight grid search over residual models. Uses a random subsample for speed.
    param_grid: list of dicts with model hyperparams.
    """
    if sample_n and len(train_resid) > sample_n:
        train_resid = train_resid.sample(sample_n, random_state=random_state)

    feature_cols = [c for c in train_resid.columns if c not in {"id", "d_col", "d_index", "sales", "residual"}]
    cat_cols = [c for c in feature_cols if train_resid[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", enc, cat_cols),
        ],
        remainder="drop",
    )

    X_full = train_resid[feature_cols]
    y_full = train_resid["residual"]
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=val_frac, random_state=random_state)

    results = []
    best_pipe = None
    best_rmse = np.inf
    best_params = None

    iterator = param_grid
    if tqdm is not None:
        iterator = tqdm(param_grid, desc=f"Grid {model_name}", mininterval=1.0)
    for params in iterator:
        if model_name == "xgb" and XGB_AVAILABLE:
            model = XGBRegressor(
                n_estimators=params.get("n_estimators", 700),
                learning_rate=params.get("learning_rate", 0.05),
                max_depth=params.get("max_depth", 8),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                min_child_weight=params.get("min_child_weight", 1.0),
                reg_lambda=params.get("reg_lambda", 1.0),
                objective="reg:squarederror",
                n_jobs=4,
            )
        else:
            model = HistGradientBoostingRegressor(
                max_depth=params.get("max_depth", 10),
                max_iter=params.get("max_iter", 800),
                learning_rate=params.get("learning_rate", 0.05),
                min_samples_leaf=params.get("min_samples_leaf", 10),
                l2_regularization=params.get("l2_regularization", 0.1),
            )
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred_val = pipe.predict(X_val)
        rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
        results.append({"model": model_name, "params": params, "RMSE_val": rmse_val})
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_pipe = pipe
            best_params = params

    results_df = pd.DataFrame(results).sort_values("RMSE_val")
    return best_pipe, feature_cols, results_df, best_params


def grid_search_mlp(train_resid: pd.DataFrame, param_grid: list, sample_n: int = 250000, val_frac: float = 0.2, random_state: int = 42):
    """
    Grid search for an MLP residual model with scaling/imputation.
    """
    if sample_n and len(train_resid) > sample_n:
        train_resid = train_resid.sample(sample_n, random_state=random_state)

    feature_cols = [c for c in train_resid.columns if c not in {"id", "d_col", "d_index", "sales", "residual"}]
    cat_cols = [c for c in feature_cols if train_resid[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )

    X_full = train_resid[feature_cols]
    y_full = train_resid["residual"]
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=val_frac, random_state=random_state)

    results = []
    best_pipe = None
    best_rmse = np.inf
    best_params = None

    iterator = param_grid
    if tqdm is not None:
        iterator = tqdm(param_grid, desc="Grid mlp", mininterval=1.0)
    for params in iterator:
        mlp = MLPRegressor(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (128, 64)),
            activation="relu",
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("learning_rate_init", 1e-3),
            max_iter=params.get("max_iter", 100),
            batch_size=params.get("batch_size", 256),
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=5,
        )
        pipe = Pipeline([("prep", pre), ("model", mlp)])
        pipe.fit(X_tr, y_tr)
        pred_val = pipe.predict(X_val)
        rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
        results.append({"model": "mlp", "params": params, "RMSE_val": rmse_val})
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_pipe = pipe
            best_params = params

    results_df = pd.DataFrame(results).sort_values("RMSE_val")
    return best_pipe, feature_cols, results_df, best_params


def predict_residuals(pipe, feature_cols, val_long: pd.DataFrame, val_wide: pd.DataFrame):
    """
    Predict residuals on the validation block and return a wide matrix aligned to val_wide.
    """
    X_val = val_long[feature_cols]
    r_pred = pipe.predict(X_val)
    pred_df = val_long[["id", "d_index"]].copy()
    pred_df["residual_pred"] = r_pred
    resid_wide = pred_df.pivot(index="id", columns="d_index", values="residual_pred")
    resid_wide.columns = [f"d_{int(c)}" for c in resid_wide.columns]
    resid_wide = resid_wide.reindex(index=val_wide.index.astype(str), columns=val_wide.columns)
    return resid_wide


def run(sales_path: Path, calendar_path: Path, prices_path: Path, cache_path: Path, horizon: int, force_rebuild: bool):
    sales_wide = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    sell_prices = pd.read_csv(prices_path)

    # Load cached long-format features
    sales_long = load_or_build_features(
        calendar_df=calendar,
        sales_wide_df=sales_wide,
        sell_prices_df=sell_prices,
        out_path=cache_path,
        force_rebuild=force_rebuild,
        sample_n=3,
    )

    # Split train/val
    train_wide, val_wide = make_train_val_frames(sales_wide, horizon=horizon)
    zero_share = pd.Series((train_wide.to_numpy(dtype=float) == 0).mean(axis=1), index=train_wide.index, name="zero_share")
    q25, q50, q75 = zero_share.quantile([0.25, 0.50, 0.75]).values

    def bucket(z):
        if z <= q25:
            return "G1 low intermittency"
        elif z <= q50:
            return "G2 medium intermittency"
        elif z <= q75:
            return "G3 high intermittency"
        return "G4 very high intermittency"

    groups = zero_share.apply(bucket)
    cutoff = max(int(c.replace("d_", "")) for c in train_wide.columns)
    train_long, val_long = split_long_by_cutoff(sales_long, cutoff=cutoff, horizon=horizon)

    order = (1, 0, 0)
    seasonal_order = (0, 1, 1, 7)

    # Baseline SARIMAX forecasts on val
    sarimax_cache = DEFAULT_SARIMAX_CACHE
    sarimax_val = None
    if sarimax_cache.exists():
        arr = np.load(sarimax_cache)
        if arr.shape == (len(train_wide), len(val_wide.columns)):
            sarimax_val = pd.DataFrame(arr, index=train_wide.index, columns=val_wide.columns)
            print(f"Loaded cached SARIMAX forecasts from {sarimax_cache}")

    if sarimax_val is None:
        sarimax_val = baseline_sarimax_wide(
            df=sales_long,
            cutoff=cutoff,
            horizon=horizon,
            future_d_cols=val_wide.columns,
            ids=train_wide.index,
            order=order,
            seasonal_order=seasonal_order,
            maxiter=30,
            progress=True,
        )
        sarimax_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(sarimax_cache, sarimax_val.to_numpy(dtype=float))
        print(f"Saved SARIMAX forecasts to {sarimax_cache}")

    # Build residual targets on training window
    train_resid = build_residual_training_frame(train_long, train_wide, order=order, seasonal_order=seasonal_order, maxiter=30)
    # Enrich with top event buckets and target encodings
    add_top_event_buckets(train_resid, val_long, "event_name_1", k=10)
    add_top_event_buckets(train_resid, val_long, "event_name_2", k=5)
    add_target_encoding(train_resid, val_long, "item_id", "residual")
    add_target_encoding(train_resid, val_long, "store_id", "residual")
    add_target_encoding(train_resid, val_long, "event_name_1_top", "residual")
    add_target_encoding(train_resid, val_long, "event_name_2_top", "residual")
    models = {"SARIMAX[7]": sarimax_val}
    residual_predictions = {}

    # Model A: HistGradientBoosting residuals (with simple grid search)
    hgb_grid = [
        {"max_depth": 8, "max_iter": 700, "learning_rate": 0.05, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"max_depth": 10, "max_iter": 800, "learning_rate": 0.04, "min_samples_leaf": 15, "l2_regularization": 0.05},
        {"max_depth": 12, "max_iter": 900, "learning_rate": 0.03, "min_samples_leaf": 20, "l2_regularization": 0.1},
    ]
    pipe_hgb, feature_cols, hgb_results, best_hgb_params = grid_search_residual_model(train_resid, model_name="hgb", param_grid=hgb_grid, sample_n=250000, val_frac=0.2)
    resid_pred_hgb = predict_residuals(pipe_hgb, feature_cols, val_long, val_wide)
    residual_predictions["hgb"] = resid_pred_hgb
    resid_pred_hgb_centered = resid_pred_hgb.subtract(resid_pred_hgb.mean(axis=0), axis=1)
    residual_predictions["hgb_centered"] = resid_pred_hgb_centered

    # Model B: XGBoost residuals (if available) with small grid
    xgb_results = pd.DataFrame()
    if XGB_AVAILABLE:
        xgb_grid = [
            {"n_estimators": 700, "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1.0},
            {"n_estimators": 900, "learning_rate": 0.04, "max_depth": 10, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1.0},
        ]
        pipe_xgb, _, xgb_results, best_xgb_params = grid_search_residual_model(train_resid, model_name="xgb", param_grid=xgb_grid, sample_n=250000, val_frac=0.2)
        resid_pred_xgb = predict_residuals(pipe_xgb, feature_cols, val_long, val_wide)
        residual_predictions["xgb"] = resid_pred_xgb
        resid_pred_xgb_centered = resid_pred_xgb.subtract(resid_pred_xgb.mean(axis=0), axis=1)
        residual_predictions["xgb_centered"] = resid_pred_xgb_centered

    # Model C: MLP residuals
    mlp_grid = [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "learning_rate_init": 8e-4, "max_iter": 180, "batch_size": 256},
        {"hidden_layer_sizes": (96, 48), "alpha": 5e-3, "learning_rate_init": 6e-4, "max_iter": 220, "batch_size": 256},
    ]
    pipe_mlp, _, mlp_results, best_mlp_params = grid_search_mlp(train_resid, param_grid=mlp_grid, sample_n=250000, val_frac=0.2)
    resid_pred_mlp = predict_residuals(pipe_mlp, feature_cols, val_long, val_wide)
    residual_predictions["mlp"] = resid_pred_mlp
    resid_pred_mlp_centered = resid_pred_mlp.subtract(resid_pred_mlp.mean(axis=0), axis=1)
    residual_predictions["mlp_centered"] = resid_pred_mlp_centered

    # Combine predictions
    for name, resid_pred in residual_predictions.items():
        boosted = (sarimax_val + resid_pred).clip(lower=0.0)
        label = "SARIMAX + Tree residual boost" if name == "hgb" else f"SARIMAX + {name} residual boost"
        models[label] = boosted

    # Optional per-group head using best HGB params
    if best_hgb_params:
        grouped_pred = sarimax_val.copy()
        for g in ["G1 low intermittency", "G2 medium intermittency", "G3 high intermittency", "G4 very high intermittency"]:
            ids_g = groups[groups == g].index.astype(str)
            if len(ids_g) == 0:
                continue
            train_g = train_resid[train_resid["id"].isin(ids_g)]
            val_long_g = val_long[val_long["id"].isin(ids_g)]
            val_wide_g = val_wide.loc[ids_g]
            if train_g.empty or val_long_g.empty:
                continue
            pipe_g, _ = build_residual_model(train_g, model_name="hgb", params=best_hgb_params)
            resid_pred_g = predict_residuals(pipe_g, feature_cols, val_long_g, val_wide_g)
            grouped_pred.loc[ids_g] = (sarimax_val.loc[ids_g] + resid_pred_g).clip(lower=0.0)
        models["SARIMAX + hgb residual boost (per-group)"] = grouped_pred

    # Evaluate
    overall = score_overall(models, val_wide, d_cols=val_wide.columns.tolist())
    group_tbl = score_by_group(models, val_wide, groups, d_cols=val_wide.columns.tolist(), zero_share=zero_share)

    resid_base = residual_matrix(val_wide, sarimax_val, val_wide.columns.tolist())
    resid_stats_base = residual_stats_by_horizon(resid_base)

    resid_stats = {}
    resid_deltas = {}
    for name, pred_wide in models.items():
        if name == "SARIMAX[7]":
            continue
        resid_model = residual_matrix(val_wide, pred_wide, val_wide.columns.tolist())
        resid_stats[name] = residual_stats_by_horizon(resid_model)
        delta = resid_stats[name].copy()
        delta["bias_delta"] = resid_stats[name]["bias"] - resid_stats_base["bias"]
        delta["RMSE_delta"] = resid_stats[name]["RMSE"] - resid_stats_base["RMSE"]
        delta["MAE_delta"] = resid_stats[name]["MAE"] - resid_stats_base["MAE"]
        resid_deltas[name] = delta

    results_dir = ROOT / "results"
    boost_dir = results_dir / "boosting"
    plots_dir = results_dir / "plots" / "boosting"
    boost_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(boost_dir / "boost_overall.csv", index=False)
    group_tbl.to_csv(boost_dir / "boost_by_group.csv", index=False)
    resid_base.to_csv(boost_dir / "sarimax_residuals_val.csv")
    resid_stats_base.to_csv(boost_dir / "sarimax_residuals_by_horizon.csv", index=False)
    for name, stats_df in resid_stats.items():
        safe = name.lower().replace(" ", "_").replace("+", "").replace("/", "-")
        stats_df.to_csv(boost_dir / f"{safe}_residuals_by_horizon.csv", index=False)
        resid_deltas[name][["day", "bias_delta", "RMSE_delta", "MAE_delta"]].to_csv(boost_dir / f"{safe}_residual_deltas_by_horizon.csv", index=False)
    hgb_results.to_csv(boost_dir / "grid_hgb_results.csv", index=False)
    if not xgb_results.empty:
        xgb_results.to_csv(boost_dir / "grid_xgb_results.csv", index=False)
    mlp_results.to_csv(boost_dir / "grid_mlp_results.csv", index=False)

    # Permutation importance on primary residual model (HGB)
    resid_base_long = resid_base.reset_index().melt(id_vars="index", var_name="d_col", value_name="residual_base")
    resid_base_long = resid_base_long.rename(columns={"index": "id"})
    resid_base_long["d_index"] = resid_base_long["d_col"].str.replace("d_", "", regex=False).astype(int)
    val_with_resid = val_long.merge(resid_base_long[["id", "d_index", "residual_base"]], on=["id", "d_index"], how="inner")
    X_val_resid = val_with_resid[feature_cols]
    y_val_resid = val_with_resid["residual_base"]
    pi = permutation_importance(pipe_hgb, X_val_resid, y_val_resid, n_repeats=5, random_state=42, n_jobs=-1)
    pi_df = pd.DataFrame({"feature": feature_cols, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
    pi_df = pi_df.sort_values("importance_mean", ascending=False)
    pi_df.to_csv(boost_dir / "residual_model_permutation_importance.csv", index=False)

    print("\nOverall validation scores:")
    print(overall)
    print("\nPer-group validation scores (RMSE):")
    print(group_tbl.pivot(index="group", columns="model", values="RMSE"))
    print("\nPer-group validation scores (MAE):")
    print(group_tbl.pivot(index="group", columns="model", values="MAE"))
    print("\nPer-horizon residual delta vs SARIMAX (negative = improvement):")
    for name, delta in resid_deltas.items():
        print(f"\n{name}")
        print(delta[["day", "bias_delta", "RMSE_delta", "MAE_delta"]].to_string(index=False))
    print("\nTop permutation importance (residual model):")
    print(pi_df.head(12).to_string(index=False))
    print("\nSaved residual matrices to results/boosting/")
    print("\nGrid search (HGB) top results:")
    print(hgb_results.head(5).to_string(index=False))
    if not xgb_results.empty:
        print("\nGrid search (XGB) top results:")
        print(xgb_results.head(5).to_string(index=False))
    print("\nGrid search (MLP) top results:")
    print(mlp_results.head(5).to_string(index=False))

    # Quick visuals: per-horizon deltas and a few item-level forecasts
    for name, delta in resid_deltas.items():
        safe = name.lower().replace(" ", "_").replace("+", "").replace("/", "-")
        plot_horizon_delta(
            delta,
            title=f"Per-horizon delta vs SARIMAX ({name})",
            save_path=plots_dir / f"{safe}_horizon_delta.png",
            show=False,
        )

    # Sample a few items for forecast traces (focused comparisons)
    sample_ids = list(train_wide.index.astype(str)[:4])
    highlight_sets = [
        ["SARIMAX[7]", "SARIMAX + Tree residual boost"],
        ["SARIMAX[7]", "SARIMAX + hgb residual boost (per-group)"],
        ["SARIMAX[7]", "SARIMAX + mlp residual boost"],
    ]
    for rid in sample_ids:
        hist = train_wide.loc[rid].to_numpy(dtype=float)
        truth = val_wide.loc[rid].to_numpy(dtype=float)
        preds = {name: df.loc[rid].to_numpy(dtype=float) for name, df in models.items() if rid in df.index}
        for highlights in highlight_sets:
            safe = "-".join([h.replace(" ", "_").replace("+", "") for h in highlights])
            plot_forecast_item(
                rid,
                history=hist,
                y_true=truth,
                y_preds=preds,
                tail=90,
                title=f"{rid} forecast (val window)",
                save_path=plots_dir / f"{rid}_forecast_{safe}.png",
                show=False,
                legend_outside=True,
                highlight_models=highlights,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute long features even if cache exists.")
    args = parser.parse_args()
    run(args.sales, args.calendar, args.prices, args.cache, args.horizon, args.force_rebuild)


if __name__ == "__main__":
    main()
