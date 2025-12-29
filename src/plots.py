import numpy as np
import matplotlib.pyplot as plt

# Palette
TEAL = "#1663A2"
MAGENTA = "#BC1052"
GOLD = "#34676F"
BLUEISH = "#CCAA00"
GRAY = "#4D4D4D"

MODEL_COLORS = {
    "SARIMAX": MAGENTA,
    "SARIMAX[7]": MAGENTA,
    "SARIMAX + Tree residual boost": TEAL,
    "SARIMAX + hgb residual boost": TEAL,
    "SARIMAX + hgb residual boost (per-group)": "#0F766E",
    "SARIMAX + hgb_centered residual boost": BLUEISH,
    "SARIMAX + xgb residual boost": GOLD,
    "SARIMAX + xgb_centered residual boost": "#7A3B9C",
    "SARIMAX + mlp residual boost": "#E67E22",
    "SARIMAX + mlp_centered residual boost": "#AF601A",
    "LightGBM (old/full)": TEAL,
    "LightGBM (pruned K=6)": BLUEISH,
    "Actual": GRAY,
}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def _maybe_save(save_path):
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")


def plot_forecast_item(item_id, history, y_true, y_preds: dict, tail=90, title=None, save_path=None, show=True, legend_outside=True, highlight_models=None):
    """
    Plot a single item forecast: history tail, truth, and multiple prediction series.

    y_preds: dict name -> np.ndarray aligned to y_true
    """
    hist_tail = history[-tail:] if len(history) > tail else history
    x_hist = np.arange(len(hist_tail))
    x_val = np.arange(len(hist_tail), len(hist_tail) + len(y_true))

    plt.figure(figsize=(10, 3.8))
    plt.plot(x_hist, hist_tail, linewidth=1.0, alpha=0.35, color=GRAY, label="history (tail)")
    plt.plot(x_val, y_true, linewidth=2.4, color=MODEL_COLORS.get("Actual", GRAY), label="actual")
    for name, arr in y_preds.items():
        if highlight_models and name not in highlight_models:
            continue
        plt.plot(
            x_val,
            np.asarray(arr, dtype=float),
            linewidth=2.0 if highlight_models is None or name in highlight_models else 1.1,
            label=name,
            color=MODEL_COLORS.get(name, None),
            alpha=0.9,
        )
    plt.axvline(x=len(hist_tail) - 1, linewidth=1.0, alpha=0.6, color=GRAY)
    plt.title(title or str(item_id))
    plt.xlabel("Time index")
    plt.ylabel("Units")
    plt.grid(True, alpha=0.2)
    if legend_outside:
        plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=9)
    else:
        plt.legend(frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 0.85, 1] if legend_outside else None)
    _maybe_save(save_path)
    if show:
        plt.show()
    plt.close()


def plot_horizon_lines(h, series_dict, title, ylabel="RMSE", save_path=None, show=True):
    fig = plt.figure(figsize=(7.2, 2.8))
    for label, (vals, color, ls) in series_dict.items():
        plt.plot(h, vals, linewidth=2.0, color=color, linestyle=ls, label=label)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Horizon day", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=4, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    _maybe_save(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_group_bar(df, metric="RMSE", title=None, save_path=None, show=True):
    x = np.arange(len(df))
    colors = [MAGENTA, TEAL, BLUEISH, GOLD][: len(df)]
    fig = plt.figure(figsize=(7.2, 2.8))
    plt.bar(x, df[metric].astype(float).values, color=colors, alpha=0.95)
    plt.xticks(x, df["group"].astype(str).values, rotation=18, ha="right")
    plt.ylabel(metric, fontsize=11)
    plt.title(title or f"{metric} by intermittency group", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    _maybe_save(save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_horizon_delta(delta_df: "pd.DataFrame", title: str, save_path=None, show=True):
    """
    Plot per-horizon delta metrics (vs SARIMAX). Expects columns: day, RMSE_delta, MAE_delta.
    """
    h = np.arange(len(delta_df))
    fig = plt.figure(figsize=(8.0, 3.0))
    plt.plot(h, delta_df["RMSE_delta"].astype(float).values, label="RMSE delta", color=TEAL, linewidth=2.0)
    plt.plot(h, delta_df["MAE_delta"].astype(float).values, label="MAE delta", color=MAGENTA, linewidth=2.0, linestyle="--")
    plt.axhline(0.0, color=GRAY, linewidth=1.0, alpha=0.6)
    plt.xticks(h, delta_df["day"].astype(str).values, rotation=60, ha="right")
    plt.ylabel("Delta (model - SARIMAX)")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    _maybe_save(save_path)
    if show:
        plt.show()
    plt.close(fig)


__all__ = [
    "plot_forecast_item",
    "plot_horizon_lines",
    "plot_group_bar",
    "plot_horizon_delta",
    "MODEL_COLORS",
    "TEAL",
    "MAGENTA",
    "GOLD",
    "BLUEISH",
    "GRAY",
]
