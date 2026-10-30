"""Matplotlib figures used by both the Streamlit app and the CLI."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "LSTM": "#d62728",
    "Last value": "#7f7f7f",
    "Seasonal naive (24h)": "#1f77b4",
    "Seasonal naive (168h)": "#2ca02c",
    "Weekly mean (4 wk)": "#ff7f0e",
}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})


def plot_overview(hourly: pd.DataFrame, b1: int, b2: int):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1.3]})
    obs = hourly["traffic"].where(hourly["observed"])
    daily = obs.resample("D").mean()
    a1.plot(daily.index, daily.values, lw=0.8, color="#333")
    t = hourly.index
    for lo, hi, c, lab in [(0, b1, "#2ca02c", "train"), (b1, b2, "#ff7f0e", "validation"),
                           (b2, len(t) - 1, "#d62728", "test")]:
        a1.axvspan(t[lo], t[hi], color=c, alpha=0.08, label=lab)
    a1.set_title("Daily mean traffic (observed hours only) and chronological split")
    a1.set_ylabel("vehicles / hour")
    a1.legend(loc="upper left", ncol=3)
    miss = (~hourly["observed"]).groupby(hourly.index.to_period("M")).mean() * 100
    a2.bar(miss.index.to_timestamp(), miss.values, width=25, color="#9467bd")
    a2.set_title("Missing sensor readings per month (%)")
    a2.set_ylabel("% hours")
    fig.tight_layout()
    return fig


def plot_history(history: dict):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(history["loss"], label="train loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Huber loss (scaled)")
    ax.set_title("LSTM training curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_bars(metrics: pd.DataFrame, cols=("MAE", "RMSE")):
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.4))
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.bar(metrics.index, metrics[c], color=[COLORS.get(m, "#999") for m in metrics.index])
        ax.set_title(f"{c} (vehicles/hour, lower is better)")
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.tight_layout()
    return fig


def plot_forecast(res, k: int, context: int = 72, models: list[str] | None = None):
    """One forecast window: recent history, the truth, and every model's prediction."""
    o = int(res.test_origins[k])
    H = res.test_y.shape[1]
    t = res.hourly.index
    lo = max(0, o - context + 1)
    hist = res.hourly["traffic"].iloc[lo: o + 1]
    hist_obs = res.hourly["observed"].iloc[lo: o + 1]
    fut_t = t[o + 1: o + H + 1]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hist.index, hist.values, color="#333", lw=1.4, label="history")
    ax.scatter(hist.index[~hist_obs.values], hist.values[~hist_obs.values], marker="x", color="#9467bd",
               zorder=5, label="imputed (sensor gap)")
    ax.plot(np.r_[hist.index[-1], fut_t], np.r_[hist.values[-1], res.test_y[k]], color="black", lw=2.2,
            marker="o", ms=4, label="actual")
    for m in models or res.models:
        ax.plot(np.r_[hist.index[-1], fut_t], np.r_[hist.values[-1], res.test_preds[m][k]],
                color=COLORS.get(m), lw=2 if m == "LSTM" else 1.2, ls="-" if m == "LSTM" else "--",
                marker="o" if m == "LSTM" else None, ms=4, label=m)
    ax.axvline(t[o], color="gray", ls=":")
    ax.set_title(f"Forecast issued at {t[o]:%a %d %b %Y %H:%M} for the next {H} h")
    ax.set_ylabel("vehicles / hour")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_breakdown(table: pd.DataFrame, metric: str, title: str):
    """Groups on the x-axis, one series per model."""
    numeric = pd.api.types.is_numeric_dtype(table.index)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if numeric:
        for m in table.columns:
            ax.plot(table.index, table[m], marker="o", ms=3, lw=2.2 if m == "LSTM" else 1.2,
                    color=COLORS.get(m), label=m)
    else:
        x = np.arange(len(table))
        w = 0.8 / len(table.columns)
        for i, m in enumerate(table.columns):
            ax.bar(x + i * w - 0.4 + w / 2, table[m], w, color=COLORS.get(m), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in table.index], rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (vehicles/hour)")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    return fig
