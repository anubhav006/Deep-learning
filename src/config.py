"""Single place for every tunable setting used by the pipeline, CLI and app."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- forecasting task -------------------------------------------------
    lookback: int = 72          # hours of history fed to the LSTM
    horizon: int = 6            # hours ahead to forecast (direct multi-step)

    # ---- chronological split (fractions of the timeline, never shuffled) ---
    train_frac: float = 0.70
    val_frac: float = 0.15      # remaining 15 % is the untouched test period

    # ---- features / data quality ------------------------------------------
    use_weather: bool = True            # temp + rain as extra *history* inputs
    max_imputed_frac: float = 0.25      # skip windows whose input is >25 % imputed
    use_us_holidays: bool = True        # pandas USFederalHolidayCalendar
    extra_holidays: tuple = field(default_factory=tuple)  # ("2017-10-19", ...)

    # ---- model -------------------------------------------------------------
    units: tuple = (64, 32)
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 128
    patience: int = 5           # early-stopping patience (on validation loss)
    seed: int = 42
