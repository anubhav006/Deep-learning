"""Leakage / correctness checks - none of these need TensorFlow."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines as B
from src import data as D
from src.config import Config
from src.features import build_features, make_windows, split_windows


@pytest.fixture(scope="module")
def hourly():
    return D.clean_hourly(D.make_synthetic(end="2017-06-30 23:00"))


def test_hourly_index_is_regular_and_flags_gaps(hourly):
    assert (hourly.index.to_series().diff().dropna() == np.timedelta64(1, "h")).all()
    assert hourly["observed"].sum() < len(hourly)          # gaps were injected
    assert hourly["traffic"].notna().all()                 # ...and causally filled


def test_causal_imputation_ignores_future():
    import pandas as pd
    s = pd.Series([5.0, np.nan, np.nan, 100.0])
    assert D.impute_causal(s, season=168).tolist() == [5.0, 5.0, 5.0, 100.0]  # not interpolated toward 100


def test_chronological_split_has_no_overlap(hourly):
    cfg = Config(lookback=48, horizon=6)
    fs = build_features(hourly, cfg)
    tr, va, te = split_windows(fs, cfg)
    H = cfg.horizon
    assert tr.origins.max() + H < fs.b1                       # all train targets before validation
    assert va.origins.min() + 1 >= fs.b1 and va.origins.max() + H < fs.b2
    assert te.origins.min() + 1 >= fs.b2
    assert tr.origins.max() + H < va.origins.min() + 1 <= te.origins.min()


def test_scaler_uses_training_rows_only(hourly):
    cfg = Config(lookback=48, horizon=6)
    a = build_features(hourly, cfg).scaler.stats["traffic"]
    changed = hourly.copy()
    changed.iloc[int(len(hourly) * 0.8):, changed.columns.get_loc("traffic")] *= 10   # corrupt test period
    b = build_features(changed, cfg).scaler.stats["traffic"]
    assert a == b


def test_targets_are_never_imputed(hourly):
    cfg = Config(lookback=48, horizon=6)
    fs = build_features(hourly, cfg)
    for ws in split_windows(fs, cfg):
        out_idx = ws.origins[:, None] + np.arange(1, cfg.horizon + 1)
        assert fs.observed[out_idx].all()


def test_window_inputs_end_at_origin(hourly):
    cfg = Config(lookback=48, horizon=6)
    fs = build_features(hourly, cfg)
    ws = make_windows(fs, cfg, fs.b2, len(fs.index))
    k = 10
    o = ws.origins[k]
    np.testing.assert_allclose(ws.X[k, -1], fs.feat[o])       # last input row == origin hour
    np.testing.assert_allclose(ws.y[k], fs.feat[o + 1: o + 7, 0])


def test_baselines_do_not_peek_at_future(hourly):
    series = hourly["traffic"].to_numpy()
    origins = np.array([2000, 3000, 4000])
    before = B.all_baselines(series, origins, 6)
    tampered = series.copy()
    tampered[4001:] = -1e6                                     # destroy everything after the last origin
    after = B.all_baselines(tampered, origins, 6)
    for name in before:
        np.testing.assert_allclose(before[name], after[name])


def test_seasonal_naive_rejects_horizon_longer_than_season():
    with pytest.raises(ValueError):
        B.seasonal_naive(np.arange(1000.0), np.array([500]), H=30, season=24)
