# 🚦 Road Traffic Flow Forecasting with an LSTM

Forecast hourly traffic counts for the next *H* hours from the previous *L* hours, and compare an
LSTM against naive/seasonal baselines. Includes a **Streamlit** app.
This is historical-count analytics for traffic planning, **not** live navigation advice.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                                   # interactive front-end
# or, headless:
python train.py --source uci --lookback 72 --horizon 6 --epochs 30
python train.py --source synthetic --epochs 10         # offline demo, no download
pytest -q                                              # leakage / correctness tests
```
Python 3.10-3.12 recommended (TensorFlow wheels).

## Dataset
**UCI "Metro Interstate Traffic Volume"** - hourly I-94 westbound counts (Minneapolis-St Paul), 2012-2018,
with temperature, rain and holiday info. It is downloaded automatically on first use into `data/`.
If your machine is offline, download it from <https://archive.ics.uci.edu/dataset/492>, unzip, and save the
CSV as `data/Metro_Interstate_Traffic_Volume.csv`.

You can also **upload any CSV** (timestamp + count column, optional temperature/rain) in the app or via
`--source csv --csv file.csv --ts-col ... --y-col ...`. Sub-hourly rows are summed per hour.

## Pipeline
| Step | Where | What |
|---|---|---|
| Clean | `src/data.py` | merge duplicate timestamps, regular hourly grid, flag missing hours, causal gap fill, remove impossible values |
| Features | `src/features.py` | scaled counts, `imputed` flag, sin/cos hour & weekday, holiday flag, optional temp/rain |
| Split | `src/features.py` | chronological 70/15/15 (train/val/test); windows assigned by where their *targets* fall |
| Windows | `src/features.py` | input `[t-L+1 … t]` → target `[t+1 … t+H]` (direct multi-step) |
| Model | `src/model.py` | 2×LSTM (64, 32) + dropout + dense head, Huber loss, early stopping on validation |
| Baselines | `src/baselines.py` | last value · seasonal naive 24 h · seasonal naive 168 h · mean of same hour over previous 4 weeks |
| Evaluate | `src/evaluate.py` | MAE, RMSE, WAPE overall and by hour / period of day / weekday / weekday-weekend-holiday / month / forecast step |
| UI | `app.py` | data quality view, training, results, forecast explorer, error analysis, notes |

## How future leakage is prevented
* Strict chronological split; test data is never touched during training or early stopping.
* Scaler statistics are fitted on training rows only.
* Gaps are filled **causally** (same hour last week, else last value). Interpolation is avoided because it uses the value after the gap.
* Only past observations and calendar features (known in advance) are inputs. Weather is used as history only.
* Baselines raise an error if they would need a value that is not yet known (e.g. 24 h seasonal naive with H > 24).
* `tests/test_no_leakage.py` checks split boundaries, scaler independence from test data, that targets are never imputed,
  window alignment, and that baseline forecasts don't change if the future is tampered with.

## Holidays
Holidays behave like weekends, so "Tuesday" alone over-predicts. A holiday flag (pandas' US federal calendar plus
optional custom dates) lets the model learn this. Holidays are rare, so their error is noisy; the app reports
**Weekday / Weekend / Holiday** errors separately. Seasonal baselines fail twice around a holiday (the day itself,
and the same weekday a week later). Non-US data needs its own holiday dates.

## Missing sensor readings
Hours without a reading are flagged (`observed=False`), filled causally, and the flag is fed to the LSTM.
Windows whose **target** contains an imputed hour are removed for *all* models (no scoring against invented
numbers), and windows with >25 % imputed input are skipped. The app shows the gap count, longest gap and
monthly missing-rate. Reported errors describe performance when the sensor works; multi-day outages cannot be forecast meaningfully.

## Output
`train.py` writes to `artifacts/`: `metrics_test.csv`, `lstm.keras`, `last_run.pkl` (reloadable in the app),
`overview.png`, `training_curve.png`, `forecast_example.png`, and `mae_by_*.png` breakdown plots.

## Notes on results
A smoke test on the built-in *synthetic* series (6 epochs, 48 h look-back, 6 h horizon) gave a test MAE of about
201 vs 222 for the best baseline (4-week same-hour mean) and 1,438 for last-value persistence.
Synthetic data is cleaner than real traffic, so **run the UCI dataset yourself for real numbers**; they will differ.
The LSTM is not guaranteed to beat a strong seasonal baseline: when it doesn't, that is a legitimate finding to report.

## Limitations / ideas
Single road segment · hourly resolution · no incidents, events or weather forecasts · one train/test split
(a rolling-origin backtest would be more robust) · add known-future calendar inputs for the horizon, try GRU/Transformer
or quantile outputs for prediction intervals.
