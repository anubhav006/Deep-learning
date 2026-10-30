"""End-to-end experiment: features -> windows -> LSTM -> baselines -> evaluation."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as B
from . import evaluate as E
from .config import Config
from .features import build_features, split_windows


@dataclass
class Results:
    cfg: Config
    quality: dict
    hourly: pd.DataFrame
    b1: int
    b2: int
    feature_names: list
    n_windows: dict
    history: dict
    models: list                    # display order, LSTM first
    test_origins: np.ndarray
    test_y: np.ndarray              # (N, H)
    test_preds: dict                # name -> (N, H)
    long_test: pd.DataFrame
    metrics_val: pd.DataFrame
    metrics_test: pd.DataFrame
    holiday: np.ndarray = field(default=None)


def _score_baselines(fs, ws, H):
    return B.all_baselines(fs.traffic, ws.origins, H)


def run_pipeline(hourly: pd.DataFrame, cfg: Config, callbacks=None) -> tuple[Results, object]:
    """Returns (results, trained_keras_model). Results is pickle-safe; the model is saved separately."""
    from .model import build_lstm, predict_raw, train_model

    fs = build_features(hourly, cfg)
    train_ws, val_ws, test_ws = split_windows(fs, cfg)
    if len(train_ws) < 200 or len(val_ws) < 50 or len(test_ws) < 50:
        raise ValueError(
            f"Not enough usable windows (train={len(train_ws)}, val={len(val_ws)}, test={len(test_ws)}). "
            "Use a longer series, a shorter lookback/horizon, or allow more imputed input."
        )

    model = build_lstm(cfg.lookback, fs.feat.shape[1], cfg.horizon, cfg.units, cfg.dropout, cfg.lr)
    history = train_model(model, train_ws, val_ws, cfg.epochs, cfg.batch_size, cfg.patience, cfg.seed, callbacks)

    def predict_all(ws):
        preds = {"LSTM": predict_raw(model, ws, fs.scaler)}
        preds.update(_score_baselines(fs, ws, cfg.horizon))
        ok = np.all([np.isfinite(p).all(axis=1) for p in preds.values()], axis=0)
        return ws.subset(ok), {k: v[ok] for k, v in preds.items()}

    val_ws, val_preds = predict_all(val_ws)
    test_ws, test_preds = predict_all(test_ws)
    models = list(test_preds)

    long_val = E.build_long_frame(val_preds, val_ws, fs)
    long_test = E.build_long_frame(test_preds, test_ws, fs)

    return Results(
        cfg=cfg, quality=hourly.attrs.get("quality", {}), hourly=hourly, b1=fs.b1, b2=fs.b2,
        feature_names=fs.names,
        n_windows={"train": len(train_ws), "val": len(val_ws), "test": len(test_ws)},
        history=history, models=models,
        test_origins=test_ws.origins, test_y=test_ws.y_raw, test_preds=test_preds,
        long_test=long_test,
        metrics_val=E.overall_metrics(long_val, models),
        metrics_test=E.overall_metrics(long_test, models),
        holiday=fs.holiday,
    ), model


def save_results(res: Results, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)


def load_results(path) -> Results:
    with open(path, "rb") as f:
        return pickle.load(f)
