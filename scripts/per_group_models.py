"""
Fit residual boosters per intermittency group, compute permutation importance per group,
and compare metrics against baseline SARIMAX and a global HGB booster.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

TEAL = "#1663A2"

from src.data_prep import load_or_build_features, make_train_val_frames, split_long_by_cutoff
from src.models import baseline_sarimax_wide
from src.eval import score_overall, score_by_group, residual_matrix, residual_stats_by_horizon
from scripts.train import (
    build_residual_training_frame,
    add_top_event_buckets,
    add_target_encoding,
    predict_residuals,
    grid_search_residual_model,
    build_residual_model,
)
from sklearn.inspection import permutation_importance

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_SALES = ROOT / "data" / "sales_train_validation_afcs2025.csv"
DEFAULT_CAL = ROOT / "data" / "calendar_afcs2025.csv"
DEFAULT_PRICES = ROOT / "data" / "sell_prices_afcs2025.csv"
DEFAULT_CACHE = ROOT / "data" / "sales_features_with_calendar_prices.csv"
DEFAULT_SARIMAX_CACHE = ROOT / "data" / "sarimax_val_preds.npy"


def bucket_groups(train_wide: pd.DataFrame):
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
    return groups, zero_share


def run(sales_path: Path, calendar_path: Path, prices_path: Path, cache_path: Path, horizon: int, force_rebuild: bool, round_preds: bool):
    print("Loading data...")
    sales_wide = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    sell_prices = pd.read_csv(prices_path)

    sales_long = load_or_build_features(
        calendar_df=calendar,
        sales_wide_df=sales_wide,
        sell_prices_df=sell_prices,
        out_path=cache_path,
        force_rebuild=force_rebuild,
        sample_n=3,
    )
    sales_long["d_index"] = sales_long["d_index"].astype(int)

    train_wide, val_wide = make_train_val_frames(sales_wide, horizon=horizon)
    cutoff = max(int(c.replace("d_", "")) for c in train_wide.columns)
    train_long, val_long = split_long_by_cutoff(sales_long, cutoff=cutoff, horizon=horizon)

    groups, zero_share = bucket_groups(train_wide)

    print("Running SARIMAX baseline...")
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
            order=(1, 0, 0),
            seasonal_order=(0, 1, 1, 7),
            maxiter=30,
            progress=True,
        )
        sarimax_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(sarimax_cache, sarimax_val.to_numpy(dtype=float))
        print(f"Saved SARIMAX forecasts to {sarimax_cache}")

    # Residual targets and encodings
    print("Building residual targets...")
    train_resid = build_residual_training_frame(train_long, train_wide, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7), maxiter=30)
    add_top_event_buckets(train_resid, val_long, "event_name_1", k=10)
    add_top_event_buckets(train_resid, val_long, "event_name_2", k=5)
    add_target_encoding(train_resid, val_long, "item_id", "residual")
    add_target_encoding(train_resid, val_long, "store_id", "residual")
    add_target_encoding(train_resid, val_long, "event_name_1_top", "residual")
    add_target_encoding(train_resid, val_long, "event_name_2_top", "residual")

    # Grid search best HGB
    print("Grid searching HGB (global)...")
    hgb_grid = [
        {"max_depth": 8, "max_iter": 700, "learning_rate": 0.05, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"max_depth": 10, "max_iter": 800, "learning_rate": 0.04, "min_samples_leaf": 15, "l2_regularization": 0.05},
        {"max_depth": 12, "max_iter": 900, "learning_rate": 0.03, "min_samples_leaf": 20, "l2_regularization": 0.1},
    ]
    pipe_hgb, feature_cols, hgb_results, best_hgb_params = grid_search_residual_model(train_resid, model_name="hgb", param_grid=hgb_grid, sample_n=250000, val_frac=0.2)
    resid_pred_global = predict_residuals(pipe_hgb, feature_cols, val_long, val_wide)

    models = {
        "SARIMAX[7]": sarimax_val,
        "SARIMAX + hgb residual boost": (sarimax_val + resid_pred_global).clip(lower=0.0),
    }

    # Per-group models + permutation importance
    perm_importances = []
    grouped_pred = sarimax_val.copy()
    grouped_pred_pruned = sarimax_val.copy()
    pruned_feature_logs = []
    plots_dir = ROOT / "results" / "per_group_models" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
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

        # Permutation importance on group
        base_resid_long = residual_matrix(val_wide_g, sarimax_val.loc[ids_g], val_wide_g.columns).reset_index().melt(id_vars="index", var_name="d_col", value_name="residual_base")
        base_resid_long = base_resid_long.rename(columns={"index": "id"})
        base_resid_long["d_index"] = base_resid_long["d_col"].str.replace("d_", "", regex=False).astype(int)
        val_with_resid = val_long_g.merge(base_resid_long[["id", "d_index", "residual_base"]], on=["id", "d_index"], how="inner")
        X_val_resid = val_with_resid[feature_cols]
        y_val_resid = val_with_resid["residual_base"]
        pi = permutation_importance(pipe_g, X_val_resid, y_val_resid, n_repeats=5, random_state=42, n_jobs=-1)
        pi_df = pd.DataFrame({"feature": feature_cols, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
        pi_df["group"] = g
        perm_importances.append(pi_df)

        # Prune to top features and refit per-group model
        top_features = (
            pi_df.sort_values("importance_mean", ascending=False)
            .head(15)["feature"]
            .tolist()
        )
        base_cols = {"id", "d_col", "d_index", "sales", "residual"}
        keep_cols = list(base_cols | set(top_features))
        train_g_pruned = train_g[[c for c in train_g.columns if c in keep_cols]]
        val_long_g_pruned = val_long_g[[c for c in val_long_g.columns if c in keep_cols]]
        pipe_g_pruned, feature_cols_pruned = build_residual_model(train_g_pruned, model_name="hgb", params=best_hgb_params)
        resid_pred_g_pruned = predict_residuals(pipe_g_pruned, feature_cols_pruned, val_long_g_pruned, val_wide_g)
        grouped_pred_pruned.loc[ids_g] = (sarimax_val.loc[ids_g] + resid_pred_g_pruned).clip(lower=0.0)

        pruned_feature_logs.append(pd.DataFrame({
            "group": g,
            "feature": top_features,
            "rank": np.arange(1, len(top_features) + 1),
        }))

        # Plot top feature importances per group
        top_pi = pi_df.sort_values("importance_mean", ascending=False).head(15)
        plt.figure(figsize=(6, 4))
        plt.barh(top_pi["feature"][::-1], top_pi["importance_mean"][::-1], xerr=top_pi["importance_std"][::-1], color=TEAL, alpha=0.9)
        plt.title(f"Permutation importance ({g})", fontsize=12, fontweight="bold")
        plt.xlabel("Importance (mean)")
        plt.tight_layout()
        safe_g = g.lower().replace(" ", "_")
        plt.savefig(plots_dir / f"pi_{safe_g}.png", bbox_inches="tight")
        plt.close()

    models["SARIMAX + hgb residual boost (per-group)"] = grouped_pred
    models["SARIMAX + hgb residual boost (per-group pruned)"] = grouped_pred_pruned

    # Evaluate
    overall = score_overall(models, val_wide, d_cols=val_wide.columns.tolist())
    group_tbl = score_by_group(models, val_wide, groups, d_cols=val_wide.columns.tolist(), zero_share=zero_share)

    results_dir = ROOT / "results" / "per_group_models"
    results_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(results_dir / "overall.csv", index=False)
    group_tbl.to_csv(results_dir / "by_group.csv", index=False)
    hgb_results.to_csv(results_dir / "grid_hgb_results.csv", index=False)
    if perm_importances:
        pd.concat(perm_importances).to_csv(results_dir / "permutation_importance_by_group.csv", index=False)
    if pruned_feature_logs:
        pd.concat(pruned_feature_logs).to_csv(results_dir / "pruned_features_by_group.csv", index=False)

    print("\nOverall validation scores:")
    print(overall)
    print("\nPer-group validation scores (RMSE):")
    print(group_tbl.pivot(index="group", columns="model", values="RMSE"))
    print("\nPer-group validation scores (MAE):")
    print(group_tbl.pivot(index="group", columns="model", values="MAE"))
    print("\nSaved metrics to", results_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--round-preds", action="store_true", help="(unused; outputs are clipped only)")
    args = parser.parse_args()
    run(args.sales, args.calendar, args.prices, args.cache, args.horizon, args.force_rebuild, args.round_preds)


if __name__ == "__main__":
    main()
