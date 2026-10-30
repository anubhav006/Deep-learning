"""Streamlit front-end:  streamlit run app.py"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import data as D
from src import evaluate as E
from src import plots as PL
from src.config import Config
from src.pipeline import load_results, run_pipeline, save_results

ARTIFACTS = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Traffic Flow Forecasting (LSTM)", page_icon="🚦", layout="wide")

st.title("🚦 Road Traffic Flow Forecasting with an LSTM")
st.caption("Historical-count analytics for traffic planning. This is a modelling demo, "
           "**not** live navigation or routing advice.")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and cleaning data...")
def cached_load(kind: str, file_bytes: bytes | None, ts_col, y_col, temp_col, rain_col):
    if kind == "csv":
        return D.load_source("csv", source=io.BytesIO(file_bytes), ts_col=ts_col, y_col=y_col,
                             temp_col=temp_col, rain_col=rain_col)
    return D.load_source(kind)


with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", ["UCI Metro Interstate (public)", "Synthetic demo", "Upload CSV"],
                      help="UCI = hourly I-94 westbound counts, Minnesota, 2012-2018. Downloaded on first use.")
    kind = {"UCI Metro Interstate (public)": "uci", "Synthetic demo": "synthetic", "Upload CSV": "csv"}[source]
    file_bytes = ts_col = y_col = temp_col = rain_col = None
    if kind == "csv":
        up = st.file_uploader("CSV with a timestamp and a count column", type="csv")
        if up is not None:
            file_bytes = up.getvalue()
            cols = list(pd.read_csv(io.BytesIO(file_bytes), nrows=5).columns)
            ts_col = st.selectbox("Timestamp column", cols)
            y_col = st.selectbox("Traffic count column", cols, index=min(1, len(cols) - 1))
            temp_col = st.selectbox("Temperature column (optional)", [None] + cols)
            rain_col = st.selectbox("Rain column (optional)", [None] + cols)

    st.header("2 · Task")
    horizon = st.slider("Forecast horizon (hours ahead)", 1, 24, 6)
    lookback = st.slider("Look-back window (hours)", 24, 336, 72, step=12)
    use_weather = st.checkbox("Use weather history (if available)", True)
    us_hol = st.checkbox("Flag US federal holidays", True)
    extra_hol = st.text_input("Extra holiday dates (YYYY-MM-DD, comma-separated)", "")

    st.header("3 · LSTM")
    u1 = st.select_slider("LSTM layer 1 units", [16, 32, 64, 96, 128], 64)
    u2 = st.select_slider("LSTM layer 2 units (0 = single layer)", [0, 16, 32, 64], 32)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
    epochs = st.slider("Max epochs", 3, 60, 20)
    batch = st.select_slider("Batch size", [32, 64, 128, 256], 128)
    lr = st.select_slider("Learning rate", [3e-4, 1e-3, 3e-3], 1e-3)
    seed = st.number_input("Random seed", 0, 9999, 42)

    train_clicked = st.button("🚀 Train & evaluate", type="primary", width="stretch")
    last = ARTIFACTS / "last_run.pkl"
    load_clicked = st.button("📂 Load last saved run", disabled=not last.exists(), width="stretch")

# ---- load data -------------------------------------------------------------
hourly = None
try:
    if kind != "csv" or file_bytes is not None:
        hourly = cached_load(kind, file_bytes, ts_col, y_col, temp_col, rain_col)
except Exception as e:   # noqa: BLE001
    st.error(f"Could not load data: {e}")
    if kind == "uci":
        st.info("The UCI download needs internet access. Download "
                "`Metro_Interstate_Traffic_Volume.csv(.gz)` from "
                "https://archive.ics.uci.edu/dataset/492 and place the CSV in the `data/` folder "
                "(named `Metro_Interstate_Traffic_Volume.csv`), or pick *Synthetic demo*.")
if hourly is None and kind == "csv":
    st.info("Upload a CSV in the sidebar to begin.")

if train_clicked and hourly is not None:
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras
    except ImportError:
        st.error("TensorFlow is not installed: `pip install tensorflow`")
        st.stop()

    cfg = Config(
        lookback=lookback, horizon=horizon, use_weather=use_weather, use_us_holidays=us_hol,
        extra_holidays=tuple(d.strip() for d in extra_hol.split(",") if d.strip()),
        units=tuple(u for u in (u1, u2) if u), dropout=dropout, epochs=epochs,
        batch_size=int(batch), lr=lr, seed=int(seed),
    )
    bar, status, chart = st.progress(0.0), st.empty(), st.empty()

    class Live(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.h = {"train loss": [], "val loss": []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.h["train loss"].append(logs.get("loss"))
            self.h["val loss"].append(logs.get("val_loss"))
            bar.progress(min(1.0, (epoch + 1) / cfg.epochs))
            status.write(f"Epoch {epoch + 1}/{cfg.epochs} - loss {logs.get('loss', 0):.4f} - "
                         f"val_loss {logs.get('val_loss', 0):.4f}")
            chart.line_chart(pd.DataFrame(self.h))

    try:
        res, model = run_pipeline(hourly, cfg, callbacks=[Live()])
    except ValueError as e:
        st.error(str(e))
        st.stop()
    ARTIFACTS.mkdir(exist_ok=True)
    save_results(res, ARTIFACTS / "last_run.pkl")
    model.save(ARTIFACTS / "lstm.keras")
    st.session_state["res"] = res
    bar.empty(); status.empty(); chart.empty()
    st.success("Training finished.")

if load_clicked:
    st.session_state["res"] = load_results(last)
    st.success("Loaded the last saved run.")

res = st.session_state.get("res")

tab_data, tab_res, tab_fc, tab_err, tab_notes = st.tabs(
    ["📊 Data", "🏁 Results", "🔮 Forecast explorer", "🔍 Error analysis", "📝 Notes"])

# --------------------------------------------------------------------------- #
with tab_data:
    if hourly is None:
        st.info("Choose a data source in the sidebar.")
    else:
        q = hourly.attrs["quality"]
        c = st.columns(5)
        c[0].metric("Hourly rows", f"{q['rows']:,}")
        c[1].metric("Missing hours", f"{q['missing_hours']:,}", f"{q['missing_pct']:.1f}%", delta_color="off")
        c[2].metric("Gaps", q["gaps"])
        c[3].metric("Longest gap (h)", q["longest_gap_hours"])
        c[4].metric("Duplicate rows merged", q["duplicates_removed"])
        st.caption(f"Span: {q['start']:%Y-%m-%d} → {q['end']:%Y-%m-%d}")
        n = len(hourly)
        b1, b2 = int(n * 0.70), int(n * 0.85)
        st.pyplot(PL.plot_overview(hourly, b1, b2))
        st.caption("Split shown is the default 70 / 15 / 15 chronological split: "
                   "train → validation → test, oldest to newest.")
        with st.expander("What the cleaning step does"):
            st.markdown("""
1. Collapse duplicate timestamps (the UCI file repeats an hour once per weather label).
2. Re-index to a **regular hourly grid** so a "lag of 24 steps" really means 24 hours.
3. Mark every hour with no sensor reading (`observed = False`).
4. Fill gaps **causally**: same hour last week, else carry the last value forward. No interpolation, because
   interpolation would use the value *after* the gap.
5. Impossible values (negative counts, 0 K temperatures, a 9 831 mm/h rain outlier) become missing.
6. Windows whose *target* contains an imputed hour are dropped, so no model is trained or scored on made-up data.
            """)
        with st.expander("Preview of the cleaned data"):
            preview = hourly.head(200).copy()
            preview.attrs = {}          # attrs hold Timestamps, which Streamlit cannot serialise
            st.dataframe(preview)

# --------------------------------------------------------------------------- #
with tab_res:
    if res is None:
        st.info("Press **Train & evaluate** in the sidebar (or load a saved run).")
    else:
        c = res.cfg
        st.subheader(f"Test-period accuracy: next {c.horizon} h, using the last {c.lookback} h")
        best = res.metrics_test.attrs.get("best_baseline", "")
        gain = res.metrics_test.loc["LSTM", "MAE vs best baseline %"]
        if gain >= 0:
            st.success(f"LSTM MAE is **{gain:.1f}% lower** than the best baseline ({best}).")
        else:
            st.warning(f"LSTM MAE is **{-gain:.1f}% higher** than the best baseline ({best}). "
                       "A well-tuned seasonal baseline can be hard to beat. Try a longer look-back, more epochs, "
                       "or more data.")
        l, r = st.columns([3, 2])
        with l:
            st.markdown("**Test set** (never used for training or early stopping)")
            st.dataframe(res.metrics_test.style.format("{:.2f}"), width="stretch")
            st.markdown("**Validation set** (used for early stopping)")
            st.dataframe(res.metrics_val.style.format("{:.2f}"), width="stretch")
        with r:
            st.pyplot(PL.plot_history(res.history))
        st.pyplot(PL.plot_metric_bars(res.metrics_test))
        st.caption(f"Windows: train {res.n_windows['train']:,} · val {res.n_windows['val']:,} · "
                   f"test {res.n_windows['test']:,}. Features: {', '.join(res.feature_names)}. "
                   "WAPE = Σ|error| / Σ actual, a percentage error that stays defined when counts are near zero.")

# --------------------------------------------------------------------------- #
with tab_fc:
    if res is None:
        st.info("Train a model first.")
    else:
        t = res.hourly.index
        mode = st.radio("Pick a forecast", ["Browse by time", "Worst LSTM forecast", "A holiday forecast"],
                        horizontal=True)
        n = len(res.test_origins)
        lstm_err = np.abs(res.test_y - res.test_preds["LSTM"]).mean(axis=1)
        if mode == "Browse by time":
            k = st.slider("Forecast origin (test period)", 0, n - 1, n // 2,
                          format="%d", help="Each step = one hour later")
        elif mode == "Worst LSTM forecast":
            k = int(np.argmax(lstm_err))
        else:
            hol_k = np.where(res.holiday[res.test_origins + 1])[0]
            if len(hol_k) == 0:
                st.info("No holiday forecasts in the test period; showing the middle of it.")
                k = n // 2
            else:
                k = int(hol_k[st.slider("Holiday window", 0, len(hol_k) - 1, 0)])
        st.caption(f"Origin: **{t[res.test_origins[k]]:%A %d %b %Y, %H:%M}** · "
                   f"LSTM MAE for this window: {lstm_err[k]:.0f} veh/h")
        context = st.slider("Hours of history to show", 12, 168, 72, step=12)
        shown = st.multiselect("Models to show", res.models, default=["LSTM", "Seasonal naive (168h)"]
                               if "Seasonal naive (168h)" in res.models else ["LSTM"])
        st.pyplot(PL.plot_forecast(res, k, context, shown))

# --------------------------------------------------------------------------- #
with tab_err:
    if res is None:
        st.info("Train a model first.")
    else:
        c1, c2 = st.columns(2)
        label = c1.selectbox("Break errors down by", list(E.BREAKDOWNS))
        metric = c2.radio("Metric", ["MAE", "RMSE"], horizontal=True)
        col = E.BREAKDOWNS[label]
        gm = E.grouped_metrics(res.long_test, col, res.models)
        st.pyplot(PL.plot_breakdown(gm[metric], metric, f"{metric} by {label.lower()} (test period)"))
        tbl = gm[metric].copy()
        tbl.insert(0, "n forecasts", gm["n"])
        st.dataframe(tbl.round(1), width="stretch")
        if col == "day_type":
            st.caption("Holidays are rare, so that row has few samples and its error is noisy. Read it as a hint.")
        st.download_button("Download test predictions (CSV)",
                           res.long_test.to_csv(index=False).encode(), "test_predictions_long.csv")

# --------------------------------------------------------------------------- #
with tab_notes:
    st.markdown("""
### Experiment design
* **Task:** direct multi-step forecast. At hour *t* the LSTM sees the previous *L* hours and predicts hours *t+1 … t+H* at once.
* **Split:** train → validation → test strictly in time order. Windows are assigned by where their **targets** fall.
* **Scaling:** mean/std computed on training rows only.
* **Inputs:** scaled counts, an "imputed" flag, sin/cos hour & weekday, holiday flag, optional temperature & rain (history only).
* **Baselines:** last value, seasonal naive (24 h, 168 h), mean of the same hour over the previous 4 weeks.
  Each refuses to run if it would need a value that is not yet known at forecast time.

### Preventing future leakage
Chronological split · train-only scaler · causal gap filling (no interpolation) · only calendar features (knowable in advance)
and past observations · automated tests in `tests/test_no_leakage.py`.

### Holidays
Traffic on public holidays looks like a weekend, so a model that only knows "it is Tuesday" will over-predict.
The holiday flag lets the LSTM learn this, but each holiday occurs only a handful of times per year, so the model
sees few examples and its holiday error is higher and noisy. Seasonal baselines are hit twice: the holiday itself
(copying last week's normal day) and the following week (copying last week's holiday). Holiday calendars are
country-specific; add your own dates in the sidebar for non-US data.

### Missing sensor readings
Real counters drop out. Here gaps are flagged, filled with past-only values, and the *model is told* which inputs were filled.
Windows with imputed **targets** are excluded from training and scoring for **every** model, and windows whose input is
more than 25 % imputed are skipped. Long outages (days) cannot be filled meaningfully. Reported errors therefore describe
performance when the sensor is working.

### Limitations
One road segment, hourly resolution, no incidents or events, a single train/test split (no rolling-origin backtest),
weather is not forecast. Use it as an analytics demo, not for live routing.
""")
