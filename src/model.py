"""Keras LSTM for direct multi-step forecasting (one output unit per horizon hour)."""
from __future__ import annotations

import numpy as np


def build_lstm(lookback: int, n_features: int, horizon: int, units=(64, 32), dropout=0.2, lr=1e-3):
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(lookback, n_features))
    x = inp
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=i < len(units) - 1)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(horizon)(x)
    model = keras.Model(inp, out, name="traffic_lstm")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.Huber(), metrics=["mae"])
    return model


def train_model(model, train_ws, val_ws, epochs=30, batch_size=128, patience=5, seed=42, callbacks=None):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2)),
    ] + list(callbacks or [])
    hist = model.fit(
        train_ws.X, train_ws.y,
        validation_data=(val_ws.X, val_ws.y),
        epochs=epochs, batch_size=batch_size,
        shuffle=True,          # shuffling *windows inside train* is fine; the split itself is chronological
        callbacks=cbs, verbose=0,
    )
    return {k: [float(v) for v in vals] for k, vals in hist.history.items()}


def predict_raw(model, ws, scaler, batch_size=512) -> np.ndarray:
    """Predict and convert back to vehicles/hour (counts cannot be negative)."""
    p = model.predict(ws.X, batch_size=batch_size, verbose=0)
    return np.clip(scaler.inverse("traffic", p), 0, None)
