"""
Run residual-boosted forecasts on the provided test split:
- Combine training history (d_1..d_1913) with test validation window (d_1914..d_1941)
- Fit SARIMAX per series and a residual booster (grid-searched HGB/XGB)
- Forecast the evaluation window (d_1942..d_1969), optionally round to integers
- Report overall/group metrics against the provided evaluation truth
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_prep import build_sales_long_features, align_pred_wide
from src.eval import score_overall, score_by_group, residual_matrix, residual_stats_by_horizon
from src.models import baseline_sarimax_wide
from src.plots import plot_horizon_delta
from scripts.train import (
    build_residual_training_frame,
    add_top_event_buckets,
    add_target_encoding,
    predict_residuals,
    grid_search_residual_model,
)

DEFAULT_SALES = ROOT / "data" / "sales_train_validation_afcs2025.csv"
DEFAULT_TEST_VAL = ROOT / "data" / "sales_test_validation_afcs2025.csv"
DEFAULT_TEST_EVAL = ROOT / "data" / "sales_test_evaluation_afcs_2025.csv"
DEFAULT_CAL = ROOT / "data" / "calendar_afcs2025.csv"
DEFAULT_PRICES = ROOT / "data" / "sell_prices_afcs2025.csv"
DEFAULT_CACHE = ROOT / "data" / "sales_test_features_with_calendar_prices.csv"


def load_combined_wide(train_path: Path, test_val_path: Path):
    train = pd.read_csv(train_path)
    test_val = pd.read_csv(test_val_path)
    # Align ids
    ids = pd.Index(test_val["id"].astype(str))
    train = train[train["id"].astype(str).isin(ids)].copy()
    train.index = train["id"].astype(str)
    test_val.index = test_val["id"].astype(str)

    train_d_cols = [c for c in train.columns if c.startswith("d_") and int(c.replace("d_", "")) < int(test_val.columns[1].replace("d_", ""))]
    val_d_cols = [c for c in test_val.columns if c.startswith("d_")]
    combined = pd.concat([train[train_d_cols], test_val[val_d_cols]], axis=1)
    combined.insert(0, "id", ids)
    return combined


def maybe_round_clip(df: pd.DataFrame, do_round: bool):
    arr = df.to_numpy(dtype=float)
    arr = np.clip(arr, 0.0, None)
    if do_round:
        arr = np.rint(arr)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def run(
    train_path: Path,
    test_val_path: Path,
    test_eval_path: Path,
    calendar_path: Path,
    prices_path: Path,
    cache_path: Path,
    horizon: int,
    round_preds: bool,
):
    print("Loading data and building combined wide matrix...")
    # Build combined wide and long features
    combined_wide = load_combined_wide(train_path, test_val_path)
    calendar = pd.read_csv(calendar_path)
    sell_prices = pd.read_csv(prices_path)
    eval_df = pd.read_csv(test_eval_path)
    future_cols = [c for c in eval_df.columns if c.startswith("d_")]

    # Extend for feature generation so eval horizon rows exist (sales remain NaN)
    combined_wide_features = combined_wide.copy()
    for col in future_cols:
        if col not in combined_wide_features.columns:
            combined_wide_features[col] = np.nan

    print("Building long-format features for test ids (force rebuild)...")
    # Build features for these ids only
    sales_long = build_sales_long_features(
        calendar_df=calendar,
        sales_wide_df=combined_wide_features,
        sell_prices_df=sell_prices,
        out_path=cache_path,
        force_rebuild=True,
    )
    sales_long["d_index"] = sales_long["d_index"].astype(int)

    print("Preparing train/eval splits...")
    # Train/eval splits
    cutoff = max(int(c.replace("d_", "")) for c in combined_wide.columns if c.startswith("d_"))
    if cutoff != 1941:
        cutoff = cutoff  # keep flexible if columns differ
    # Wide matrices
    train_d_cols = [c for c in combined_wide.columns if c.startswith("d_") and int(c.replace("d_", "")) <= cutoff]
    train_wide = combined_wide.set_index("id")[train_d_cols]
    eval_wide = eval_df.set_index("id")[future_cols].astype(float)

    zero_share = pd.Series((train_wide.to_numpy(dtype=float) == 0).mean(axis=1), index=train_wide.index, name="zero_share")
    q25, q50, q75 = zero_share.quantile([0.25, 0.50, 0.75]).values

    def bucket(z):
        if z <= q25:
            return "G1 low intermittency"
        elif z <= q50:
            return "G2 medium intermittency"
        elif z <= q75:
            return "G3 high intermittency"
        else:
            return "G4 very high intermittency"

    groups = zero_share.apply(bucket)

    print("Running SARIMAX baseline forecasting...")
    # SARIMAX baseline
    sarimax_df = baseline_sarimax_wide(
        df=sales_long,
        cutoff=cutoff,
        horizon=horizon,
        future_d_cols=future_cols,
        ids=train_wide.index,
        order=(1, 0, 0),
        seasonal_order=(0, 1, 1, 7),
        maxiter=30,
        progress=True,
    )
    sarimax_df = maybe_round_clip(sarimax_df, do_round=round_preds)

    print("Building residual targets and encodings...")
    # Residual targets
    train_long = sales_long[sales_long["d_index"] <= cutoff].copy()
    eval_long = sales_long[(sales_long["d_index"] > cutoff) & (sales_long["d_index"] <= cutoff + horizon)].copy()
    train_resid = build_residual_training_frame(train_long, train_wide, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7), maxiter=30)
    add_top_event_buckets(train_resid, eval_long, "event_name_1", k=10)
    add_top_event_buckets(train_resid, eval_long, "event_name_2", k=5)
    add_target_encoding(train_resid, eval_long, "item_id", "residual")
    add_target_encoding(train_resid, eval_long, "store_id", "residual")
    add_target_encoding(train_resid, eval_long, "event_name_1_top", "residual")
    add_target_encoding(train_resid, eval_long, "event_name_2_top", "residual")

    print("Grid search HGB residual model...")
    # Grid search HGB (and XGB if available)
    hgb_grid = [
        {"max_depth": 8, "max_iter": 700, "learning_rate": 0.05, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"max_depth": 10, "max_iter": 800, "learning_rate": 0.04, "min_samples_leaf": 15, "l2_regularization": 0.05},
        {"max_depth": 12, "max_iter": 900, "learning_rate": 0.03, "min_samples_leaf": 20, "l2_regularization": 0.1},
    ]
    pipe_hgb, feature_cols, hgb_results, _ = grid_search_residual_model(train_resid, model_name="hgb", param_grid=hgb_grid, sample_n=250000, val_frac=0.2)
    resid_pred_hgb = predict_residuals(pipe_hgb, feature_cols, eval_long, eval_wide)
    resid_pred_hgb_centered = resid_pred_hgb.subtract(resid_pred_hgb.mean(axis=0), axis=1)

    models = {
        "SARIMAX[7]": sarimax_df,
        "SARIMAX + Tree residual boost": maybe_round_clip(sarimax_df + resid_pred_hgb, do_round=round_preds),
        "SARIMAX + hgb_centered residual boost": maybe_round_clip(sarimax_df + resid_pred_hgb_centered, do_round=round_preds),
    }

    # Evaluate against test evaluation truth
    overall = score_overall(models, eval_wide, d_cols=future_cols)
    group_tbl = score_by_group(models, eval_wide, groups, d_cols=future_cols, zero_share=zero_share)

    resid_base = residual_matrix(eval_wide, sarimax_df, future_cols)
    resid_stats_base = residual_stats_by_horizon(resid_base)
    resid_deltas = {}
    for name, pred_wide in models.items():
        if name == "SARIMAX[7]":
            continue
        resid_model = residual_matrix(eval_wide, pred_wide, future_cols)
        stats = residual_stats_by_horizon(resid_model)
        delta = stats.copy()
        delta["bias_delta"] = stats["bias"] - resid_stats_base["bias"]
        delta["RMSE_delta"] = stats["RMSE"] - resid_stats_base["RMSE"]
        delta["MAE_delta"] = stats["MAE"] - resid_stats_base["MAE"]
        resid_deltas[name] = delta

    results_dir = ROOT / "results" / "boosting_test"
    plots_dir = ROOT / "results" / "plots" / "boosting_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall.to_csv(results_dir / "overall.csv", index=False)
    group_tbl.to_csv(results_dir / "by_group.csv", index=False)
    sarimax_df.to_csv(results_dir / "sarimax_preds.csv")
    for name, pred in models.items():
        safe = name.lower().replace(" ", "_").replace("+", "").replace("/", "-")
        pred.to_csv(results_dir / f"{safe}_preds.csv")
        if name != "SARIMAX[7]":
            resid_deltas[name][["day", "bias_delta", "RMSE_delta", "MAE_delta"]].to_csv(results_dir / f"{safe}_residual_deltas_by_horizon.csv", index=False)
            plot_horizon_delta(
                resid_deltas[name],
                title=f"Per-horizon delta vs SARIMAX ({name})",
                save_path=plots_dir / f"{safe}_horizon_delta.png",
                show=False,
            )

    print("\nOverall test scores:")
    print(overall)
    print("\nPer-group test scores (RMSE):")
    print(group_tbl.pivot(index="group", columns="model", values="RMSE"))
    print("\nPer-group test scores (MAE):")
    print(group_tbl.pivot(index="group", columns="model", values="MAE"))
    print("\nHGB grid search (residual model) top rows:")
    print(hgb_results.head(5).to_string(index=False))
    print(f"\nSaved predictions and metrics to {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_SALES, help="Path to base training wide CSV (with d_1..d_1913).")
    parser.add_argument("--test-val", type=Path, default=DEFAULT_TEST_VAL, help="Path to test validation wide CSV (d_1914..d_1941).")
    parser.add_argument("--test-eval", type=Path, default=DEFAULT_TEST_EVAL, help="Path to test evaluation wide CSV (d_1942..d_1969).")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Path to cache features for the test ids.")
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--round-preds", action="store_true", help="Round forecasts to nearest integer after clipping at 0.")
    args = parser.parse_args()
    run(args.train, args.test_val, args.test_eval, args.calendar, args.prices, args.cache, args.horizon, args.round_preds)


if __name__ == "__main__":
    main()
