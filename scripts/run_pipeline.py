"""
Lightweight CLI to run the pipeline steps:
    1) Load data (sales wide, calendar, prices)
    2) Build long-format features with lags/rolls (cached)
    3) Split train/val
    4) Run quick baselines (including SARIMAX[7] if cached) and print overall/group scores
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Make repository root importable when running as a script
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_prep import make_train_val_frames, load_or_build_features
from src.models import pred_naive, pred_seasonal_naive, pred_moving_average, baseline_sarimax_wide
from src.eval import score_overall, score_by_group, residual_matrix, residual_stats_by_horizon
from src.plots import plot_group_bar

# Default paths (so you can run without typing them)
DEFAULT_SALES = ROOT / "data" / "sales_train_validation_afcs2025.csv"
DEFAULT_CAL = ROOT / "data" / "calendar_afcs2025.csv"
DEFAULT_PRICES = ROOT / "data" / "sell_prices_afcs2025.csv"
DEFAULT_CACHE = ROOT / "data" / "sales_features_with_calendar_prices.csv"
DEFAULT_SARIMAX = ROOT / "data" / "sarimax_val_preds.npy"  # optional cached baseline


def maybe_load_sarimax(cache_path: Path, ids, d_cols):
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape == (len(ids), len(d_cols)):
            return pd.DataFrame(arr, index=ids, columns=d_cols)
    return None


def run(sales_path: Path, calendar_path: Path, prices_path: Path, horizon: int, cache_path: Path, force_rebuild: bool):
    sales_wide = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    sell_prices = pd.read_csv(prices_path)

    # Build/load long features (lags/rolls, price flags) with sanity checks
    sales_long = load_or_build_features(
        calendar_df=calendar,
        sales_wide_df=sales_wide,
        sell_prices_df=sell_prices,
        out_path=cache_path,
        force_rebuild=force_rebuild,
        sample_n=5,
    )

    # Split wide into train/val
    train_wide, val_wide = make_train_val_frames(sales_wide, horizon=horizon)

    # Zero-share groups for H2
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

    # Baseline predictions (aligned to val_wide)
    H = val_wide.shape[1]
    preds = {
        "Naive last value": pd.DataFrame(pred_naive(train_wide, H), index=train_wide.index, columns=val_wide.columns),
        "Seasonal naive period 7": pd.DataFrame(pred_seasonal_naive(train_wide, H, season=7), index=train_wide.index, columns=val_wide.columns),
        "Moving average window 28": pd.DataFrame(pred_moving_average(train_wide, H, window=28), index=train_wide.index, columns=val_wide.columns),
    }
    sarimax_df = maybe_load_sarimax(DEFAULT_SARIMAX, train_wide.index, val_wide.columns)
    if sarimax_df is not None:
        preds["SARIMAX[7]"] = sarimax_df
    else:
        # Fit SARIMAX on the fly if no cache is present; use last train day as cutoff
        last_train_day = max(int(c.replace("d_", "")) for c in train_wide.columns)
        sarimax_df = baseline_sarimax_wide(
            df=sales_long,
            cutoff=last_train_day,
            horizon=H,
            future_d_cols=val_wide.columns,
            ids=train_wide.index,
            order=(1, 0, 0),
            seasonal_order=(0, 1, 1, 7),
            maxiter=30,
        )
        preds["SARIMAX[7]"] = sarimax_df

    # Score overall and by group
    overall_tbl = score_overall(preds, val_wide, d_cols=val_wide.columns.tolist())
    group_tbl = score_by_group(preds, val_wide, groups, d_cols=val_wide.columns.tolist(), zero_share=zero_share)

    # Save metrics and plots
    results_dir = ROOT / "results"
    metrics_dir = results_dir / "metrics"
    plots_dir = results_dir / "plots"
    baseline_plots_dir = plots_dir / "baseline"
    baseline_metrics_dir = metrics_dir / "baseline"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    baseline_plots_dir.mkdir(parents=True, exist_ok=True)
    residual_metrics_dir = baseline_metrics_dir / "residuals"
    residual_metrics_dir.mkdir(parents=True, exist_ok=True)

    overall_tbl.to_csv(metrics_dir / "baseline_overall.csv", index=False)
    group_tbl.to_csv(metrics_dir / "baseline_by_group.csv", index=False)
    group_tbl.pivot(index="group", columns="model", values="RMSE").to_csv(baseline_metrics_dir / "baseline_group_rmse.csv")
    group_tbl.pivot(index="group", columns="model", values="MAE").to_csv(baseline_metrics_dir / "baseline_group_mae.csv")

    plot_group_bar(group_tbl, metric="RMSE", title="Baseline RMSE by intermittency group", save_path=baseline_plots_dir / "baseline_group_rmse.png", show=False)
    plot_group_bar(group_tbl, metric="MAE", title="Baseline MAE by intermittency group", save_path=baseline_plots_dir / "baseline_group_mae.png", show=False)

    # Per-model group bars
    for metric in ["RMSE", "MAE"]:
        for model_name in group_tbl["model"].unique():
            df_m = group_tbl[group_tbl["model"] == model_name]
            if df_m.empty:
                continue
            safe_model = model_name.lower().replace(" ", "_").replace("/", "-")
            plot_group_bar(
                df_m,
                metric=metric,
                title=f"{metric} by intermittency group ({model_name})",
                save_path=baseline_plots_dir / f"{metric.lower()}_{safe_model}.png",
                show=False,
            )

    # Residual diagnostics per horizon (helps with residual boosting / stacking)
    for model_name, pred_wide in preds.items():
        resid_wide = residual_matrix(val_wide, pred_wide, val_wide.columns.tolist())
        resid_stats = residual_stats_by_horizon(resid_wide)
        safe_model = model_name.lower().replace(" ", "_").replace("/", "-")
        resid_stats.to_csv(residual_metrics_dir / f"{safe_model}_residuals_by_horizon.csv", index=False)
        if model_name == "SARIMAX[7]":
            print("\nSARIMAX[7] residuals by horizon (mean bias / error across ids):")
            print(resid_stats[["day", "bias", "RMSE", "MAE"]].to_string(index=False))

    print("\nOverall baseline scores (val horizon):")
    print(overall_tbl)
    print("\nPer-group RMSE:")
    print(group_tbl.pivot(index="group", columns="model", values="RMSE"))
    print("\nPer-group MAE:")
    print(group_tbl.pivot(index="group", columns="model", values="MAE"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES, help="Path to wide sales CSV with id + d_# columns.")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CAL, help="Path to calendar CSV.")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="Path to sell_prices CSV.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Path to feature cache CSV.")
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute long features even if cache exists.")
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    run(args.sales, args.calendar, args.prices, args.horizon, args.cache, args.force_rebuild)


if __name__ == "__main__":
    main()
