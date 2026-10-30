"""Regression metrics and error breakdowns by period.

Every forecast is stored in a tidy "long" frame: one row per
(model, forecast origin, horizon step).  Breakdowns are then simple group-bys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureSet, WindowSet

PERIOD_ORDER = ["Night (0-5)", "AM rush (6-9)", "Midday (10-15)", "PM rush (16-19)", "Evening (20-23)"]
DOW_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYTYPE_ORDER = ["Weekday", "Weekend", "Holiday"]
BREAKDOWNS = {            # label -> column in long frame
    "Hour of day": "hour",
    "Period of day": "period",
    "Day of week": "dow_name",
    "Day type (weekday / weekend / holiday)": "day_type",
    "Month": "month",
    "Forecast step (hours ahead)": "horizon",
}


def build_long_frame(preds: dict[str, np.ndarray], ws: WindowSet, fs: FeatureSet) -> pd.DataFrame:
    N, H = ws.y_raw.shape
    out_idx = (ws.origins[:, None] + np.arange(1, H + 1)[None, :]).ravel()
    tt = fs.index[out_idx]
    hour, dow = tt.hour.to_numpy(), tt.dayofweek.to_numpy()
    hol = fs.holiday[out_idx]
    period = np.select([hour <= 5, hour <= 9, hour <= 15, hour <= 19], PERIOD_ORDER[:4], PERIOD_ORDER[4])
    day_type = np.where(hol, "Holiday", np.where(dow >= 5, "Weekend", "Weekday"))
    base = {
        "origin_time": np.repeat(fs.index[ws.origins].to_numpy(), H),
        "horizon": np.tile(np.arange(1, H + 1), N),
        "target_time": tt.to_numpy(),
        "y": ws.y_raw.ravel(),
        "hour": hour,
        "dow_name": np.array(DOW_ORDER)[dow],
        "period": pd.Categorical(period, categories=PERIOD_ORDER, ordered=True),
        "day_type": pd.Categorical(day_type, categories=DAYTYPE_ORDER, ordered=True),
        "month": tt.strftime("%Y-%m").to_numpy(),
    }
    frames = []
    for name, p in preds.items():
        d = pd.DataFrame(base)
        d["model"] = name
        d["yhat"] = np.asarray(p).ravel()
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    long["abs_err"] = (long["y"] - long["yhat"]).abs()
    long["sq_err"] = (long["y"] - long["yhat"]) ** 2
    long["dow_name"] = pd.Categorical(long["dow_name"], categories=DOW_ORDER, ordered=True)
    return long


def overall_metrics(long: pd.DataFrame, model_order: list[str] | None = None) -> pd.DataFrame:
    g = long.groupby("model", sort=False)
    t = pd.DataFrame({
        "MAE": g["abs_err"].mean(),
        "RMSE": np.sqrt(g["sq_err"].mean()),
        "WAPE %": 100 * g["abs_err"].sum() / g["y"].sum(),
    })
    if model_order:
        t = t.reindex(model_order)
    baselines = t.drop(index="LSTM", errors="ignore")
    if "LSTM" in t.index and len(baselines):
        best = baselines["MAE"].idxmin()
        t["MAE vs best baseline %"] = np.nan
        t.loc["LSTM", "MAE vs best baseline %"] = 100 * (1 - t.loc["LSTM", "MAE"] / t.loc[best, "MAE"])
        t.attrs["best_baseline"] = best
    return t.round(2)


def grouped_metrics(long: pd.DataFrame, by: str, model_order: list[str]) -> dict[str, pd.DataFrame]:
    """{'MAE': pivot, 'RMSE': pivot, 'n': Series} with groups as rows and models as columns."""
    g = long.groupby([by, "model"], observed=True).agg(
        mae=("abs_err", "mean"), mse=("sq_err", "mean"), n=("abs_err", "size"))
    mae = g["mae"].unstack("model")[model_order]
    rmse = np.sqrt(g["mse"]).unstack("model")[model_order]
    n = g["n"].unstack("model")[model_order[0]]
    return {"MAE": mae, "RMSE": rmse, "n": n}
