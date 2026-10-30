"""Naive forecasters that every neural model must beat.

All of them see only observations at or before the forecast origin; the
seasonal ones raise an error if asked to look at an hour that is not yet known.
"""
from __future__ import annotations

import warnings

import numpy as np


def last_value(series: np.ndarray, origins: np.ndarray, H: int) -> np.ndarray:
    """Persistence: the next H hours equal the last observed hour."""
    return np.repeat(series[origins][:, None], H, axis=1)


def seasonal_naive(series: np.ndarray, origins: np.ndarray, H: int, season: int) -> np.ndarray:
    """y(t+h) = y(t+h-season). Needs H <= season so the lagged hour is already known."""
    if H > season:
        raise ValueError(f"horizon {H} > season {season}: lagged value would not be known yet (leakage)")
    idx = origins[:, None] + np.arange(1, H + 1)[None, :] - season
    out = np.full(idx.shape, np.nan)
    ok = idx >= 0
    out[ok] = series[idx[ok]]
    return out


def seasonal_mean(series: np.ndarray, origins: np.ndarray, H: int, season: int = 168, k: int = 4) -> np.ndarray:
    """Average of the same hour over the previous k seasons (a stronger, smoother baseline)."""
    if H > season:
        raise ValueError("horizon larger than season -> would use unknown values")
    acc = np.stack([seasonal_naive(series, origins, H, season * j) for j in range(1, k + 1)], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice at the very start of a series
        return np.nanmean(acc, axis=0)


def all_baselines(series: np.ndarray, origins: np.ndarray, H: int) -> dict[str, np.ndarray]:
    out = {"Last value": last_value(series, origins, H)}
    if H <= 24:
        out["Seasonal naive (24h)"] = seasonal_naive(series, origins, H, 24)
    if H <= 168:
        out["Seasonal naive (168h)"] = seasonal_naive(series, origins, H, 168)
        out["Weekly mean (4 wk)"] = seasonal_mean(series, origins, H, 168, 4)
    return out
