"""Feature engineering, chronological split and sliding-window construction.

Leakage rules enforced here
---------------------------
* The timeline is cut into train | val | test **in time order** (no shuffling).
* A window belongs to a split only if ALL of its target hours lie inside that
  split.  Its inputs may reach back into earlier splits (that is simply the past).
* Scaling statistics are fitted on the training segment only.
* Only calendar features (known in advance) and *past* observations are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import Config


# --------------------------------------------------------------------------- #
class Scaler:
    """Per-column standardiser whose statistics come from training data only."""

    def __init__(self):
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, name: str, values) -> "Scaler":
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        mean = float(v.mean()) if len(v) else 0.0
        std = float(v.std()) if len(v) else 1.0
        self.stats[name] = (mean, std if std > 1e-9 else 1.0)
        return self

    def transform(self, name: str, values) -> np.ndarray:
        m, s = self.stats[name]
        return (np.asarray(values, dtype=float) - m) / s

    def inverse(self, name: str, values) -> np.ndarray:
        m, s = self.stats[name]
        return np.asarray(values, dtype=float) * s + m


def holiday_flags(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    """Boolean array: is this hour on a public holiday (whole day flagged)."""
    days = pd.DatetimeIndex([])
    if cfg.use_us_holidays:
        days = USFederalHolidayCalendar().holidays(index.min().normalize(), index.max().normalize())
    if cfg.extra_holidays:
        days = days.union(pd.DatetimeIndex(pd.to_datetime(list(cfg.extra_holidays))))
    return np.asarray(index.normalize().isin(days))


@dataclass
class FeatureSet:
    feat: np.ndarray            # (T, F) float32 - model inputs, column 0 = scaled traffic
    names: list[str]
    scaler: Scaler
    traffic: np.ndarray         # (T,) raw (causally imputed) counts
    observed: np.ndarray        # (T,) bool - real reading?
    imputed: np.ndarray         # (T,) float 0/1
    holiday: np.ndarray         # (T,) bool
    index: pd.DatetimeIndex
    b1: int                     # first index of validation period
    b2: int                     # first index of test period


def build_features(hourly: pd.DataFrame, cfg: Config) -> FeatureSet:
    n = len(hourly)
    b1 = int(n * cfg.train_frac)
    b2 = int(n * (cfg.train_frac + cfg.val_frac))
    idx = hourly.index
    observed = hourly["observed"].to_numpy(bool)
    traffic = hourly["traffic"].to_numpy(float)

    # ---- scaler fitted on TRAIN rows with real readings only ---------------
    scaler = Scaler().fit("traffic", traffic[:b1][observed[:b1]])
    cols = {
        "traffic": scaler.transform("traffic", traffic),
        "imputed": (~observed).astype(float),
    }
    hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
    cols["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    cols["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    cols["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    cols["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    hol = holiday_flags(idx, cfg)
    cols["is_holiday"] = hol.astype(float)

    if cfg.use_weather:
        if "temp" in hourly:
            t = hourly["temp"].to_numpy(float)
            scaler.fit("temp", t[:b1])
            cols["temp"] = np.nan_to_num(scaler.transform("temp", t), nan=0.0)   # 0 == train mean
        if "rain" in hourly:
            r = np.log1p(np.clip(hourly["rain"].to_numpy(float), 0, None))
            scaler.fit("rain", r[:b1])
            cols["rain"] = np.nan_to_num(scaler.transform("rain", r), nan=0.0)

    names = list(cols)
    feat = np.column_stack([cols[k] for k in names]).astype(np.float32)
    return FeatureSet(feat, names, scaler, traffic, observed, (~observed).astype(float), hol, idx, b1, b2)


# --------------------------------------------------------------------------- #
@dataclass
class WindowSet:
    X: np.ndarray        # (N, L, F)
    y: np.ndarray        # (N, H) scaled targets
    y_raw: np.ndarray    # (N, H) targets in vehicles/hour
    origins: np.ndarray  # (N,) index of the LAST input hour ("forecast issued at")

    def __len__(self):
        return len(self.origins)

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(self.X[mask], self.y[mask], self.y_raw[mask], self.origins[mask])


def make_windows(fs: FeatureSet, cfg: Config, lo: int, hi: int) -> WindowSet:
    """All windows whose *targets* lie entirely inside [lo, hi).

    origin t = last input hour; inputs = t-L+1 .. t; targets = t+1 .. t+H.
    Windows are dropped when any target hour was not really observed (so no model
    is ever trained or scored against an imputed value) or when the input
    window is mostly imputed.
    """
    L, H = cfg.lookback, cfg.horizon
    first, last = max(lo - 1, L - 1), hi - H - 1
    if last < first:
        return WindowSet(np.empty((0, L, fs.feat.shape[1]), np.float32), np.empty((0, H), np.float32),
                         np.empty((0, H)), np.empty(0, int))
    origins = np.arange(first, last + 1)
    in_idx = origins[:, None] + np.arange(-L + 1, 1)[None, :]
    out_idx = origins[:, None] + np.arange(1, H + 1)[None, :]
    keep = fs.observed[out_idx].all(axis=1) & (fs.imputed[in_idx].mean(axis=1) <= cfg.max_imputed_frac)
    origins, in_idx, out_idx = origins[keep], in_idx[keep], out_idx[keep]
    return WindowSet(fs.feat[in_idx], fs.feat[out_idx, 0], fs.traffic[out_idx], origins)


def split_windows(fs: FeatureSet, cfg: Config):
    n = len(fs.index)
    return (make_windows(fs, cfg, 0, fs.b1),
            make_windows(fs, cfg, fs.b1, fs.b2),
            make_windows(fs, cfg, fs.b2, n))
