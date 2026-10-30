"""Loading, cleaning and gap handling for hourly traffic counts.

Output of every loader is a *raw* frame indexed by timestamp with a ``traffic``
column (plus optional ``temp`` / ``rain``).  ``clean_hourly`` turns it into the
regular hourly frame used downstream:

    traffic   float  - counts, gaps filled CAUSALLY (past data only)
    observed  bool   - True where a real sensor reading exists
    temp, rain (optional weather covariates, forward-filled)
"""
from __future__ import annotations

import gzip
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
UCI_FILE = DATA_DIR / "Metro_Interstate_Traffic_Volume.csv"
SEASON = 168  # hours in a week - used for causal gap filling


# --------------------------------------------------------------------------- #
# Public dataset: UCI "Metro Interstate Traffic Volume" (I-94 westbound, MN)
# --------------------------------------------------------------------------- #
def download_uci(force: bool = False) -> Path:
    """Download the UCI dataset into ``data/`` (needs internet access)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if UCI_FILE.exists() and not force:
        return UCI_FILE
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith((".csv", ".csv.gz")))
        raw = zf.read(name)
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    UCI_FILE.write_bytes(raw)
    return UCI_FILE


def load_uci(path: str | Path | None = None) -> pd.DataFrame:
    """Read the UCI file and return a de-duplicated raw frame."""
    path = Path(path) if path else download_uci()
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["date_time"])
    # Known glitches in this dataset: temp == 0 K and a 9831 mm/h rain value.
    df.loc[df["temp"] < 200, "temp"] = np.nan
    df.loc[df["rain_1h"] > 500, "rain_1h"] = np.nan
    # The file repeats an hour once per weather description -> collapse to 1 row/hour.
    out = df.groupby("ts").agg(
        traffic=("traffic_volume", "mean"),
        temp=("temp", "mean"),
        rain=("rain_1h", "mean"),
    )
    out.attrs["duplicates_removed"] = int(len(df) - len(out))
    return out


# --------------------------------------------------------------------------- #
# Any user CSV
# --------------------------------------------------------------------------- #
def load_csv(source, ts_col: str, y_col: str, temp_col: str | None = None,
             rain_col: str | None = None) -> pd.DataFrame:
    """Generic loader: timestamp + count column (sub-hourly rows are summed per hour)."""
    df = pd.read_csv(source)
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    keep = ts.notna()
    frame = pd.DataFrame({"traffic": pd.to_numeric(df.loc[keep, y_col], errors="coerce").values},
                         index=ts[keep].dt.floor("h"))
    if temp_col:
        frame["temp"] = pd.to_numeric(df.loc[keep, temp_col], errors="coerce").values
    if rain_col:
        frame["rain"] = pd.to_numeric(df.loc[keep, rain_col], errors="coerce").values
    agg = {"traffic": lambda s: s.sum(min_count=1)}
    agg.update({c: "mean" for c in frame.columns if c != "traffic"})
    out = frame.groupby(level=0).agg(agg)
    out.attrs["duplicates_removed"] = int(keep.sum() - len(out))
    return out


# --------------------------------------------------------------------------- #
# Synthetic stand-in (offline demo / unit tests)
# --------------------------------------------------------------------------- #
def make_synthetic(start="2016-01-01", end="2018-09-30 23:00", seed: int = 0) -> pd.DataFrame:
    """Hourly commuter-style counts with weather, holidays and injected sensor gaps."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values

    hol_days = USFederalHolidayCalendar().holidays(idx.min().normalize(), idx.max().normalize())
    is_hol = idx.normalize().isin(hol_days)
    weekday = (dow < 5) & ~is_hol

    g = lambda mu, sd: np.exp(-0.5 * ((h - mu) / sd) ** 2)
    prof_wd = 500 + 3800 * g(7.5, 1.3) + 4300 * g(16.5, 1.6) + 1500 * g(12.5, 3.5)
    prof_we = 350 + 2900 * g(13.0, 3.5)
    base = np.where(weekday, prof_wd, prof_we)
    base = np.where(is_hol, base * 0.85, base)

    season = 1 + 0.08 * np.sin(2 * np.pi * (doy - 100) / 365)
    temp = 283 + 12 * np.sin(2 * np.pi * (doy - 110) / 365) + 4 * np.sin(2 * np.pi * (h - 15) / 24) \
        + rng.normal(0, 2, n)
    rain = np.where(rng.random(n) < 0.08, rng.exponential(1.5, n), 0.0)
    weather_factor = 1 - 0.07 * np.minimum(rain, 5) / 5

    e = np.zeros(n)                                  # AR(1) noise -> realistic autocorrelation
    z = rng.normal(0, 1, n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + z[t]
    traffic = np.clip(base * season * weather_factor * (1 + 0.06 * e), 0, None).round()

    df = pd.DataFrame({"traffic": traffic, "temp": temp, "rain": rain}, index=idx)
    miss = rng.random(n) < 0.006                     # random single-hour dropouts
    for _ in range(4):                               # a few multi-day sensor outages
        s = int(rng.integers(0, n - 150))
        miss[s: s + int(rng.integers(20, 110))] = True
    df.loc[miss, ["traffic"]] = np.nan
    df.index.name = "ts"
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def impute_causal(s: pd.Series, season: int = SEASON) -> pd.Series:
    """Fill gaps using ONLY past information.

    1. same hour one week earlier (if that value exists)
    2. otherwise carry the last observation forward.
    Linear interpolation is deliberately avoided: it looks at the value *after*
    the gap, which would leak the future into the history.
    """
    return s.fillna(s.shift(season)).ffill()


def clean_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw frame -> regular hourly frame with ``observed`` flag and quality report."""
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]
    full_idx = pd.date_range(raw.index.min().floor("h"), raw.index.max().floor("h"), freq="h", name="ts")
    df = raw.reindex(full_idx)

    df.loc[df["traffic"] < 0, "traffic"] = np.nan          # impossible counts
    observed = df["traffic"].notna()
    first_valid = observed.idxmax()                        # drop leading gap (nothing to fill from)
    df, observed = df.loc[first_valid:], observed.loc[first_valid:]

    out = pd.DataFrame(index=df.index)
    out["traffic"] = impute_causal(df["traffic"])
    out["observed"] = observed
    for c in ("temp", "rain"):
        if c in df:
            out[c] = df[c].ffill()                         # causal weather fill
    out.attrs.update(raw.attrs)
    out.attrs["quality"] = quality_report(out)
    return out


def quality_report(hourly: pd.DataFrame) -> dict:
    missing = ~hourly["observed"]
    run_id = (missing != missing.shift()).cumsum()
    gap_lengths = missing.groupby(run_id).sum()[missing.groupby(run_id).first()]
    return {
        "rows": int(len(hourly)),
        "start": hourly.index.min(),
        "end": hourly.index.max(),
        "missing_hours": int(missing.sum()),
        "missing_pct": float(missing.mean() * 100),
        "gaps": int(len(gap_lengths)),
        "longest_gap_hours": int(gap_lengths.max()) if len(gap_lengths) else 0,
        "duplicates_removed": int(hourly.attrs.get("duplicates_removed", 0)),
    }


def load_source(kind: str, **kw) -> pd.DataFrame:
    """kind in {'uci', 'synthetic', 'csv'} -> clean hourly frame."""
    if kind == "uci":
        raw = load_uci(kw.get("path"))
    elif kind == "synthetic":
        raw = make_synthetic()
    elif kind == "csv":
        raw = load_csv(kw["source"], kw["ts_col"], kw["y_col"], kw.get("temp_col"), kw.get("rain_col"))
    else:
        raise ValueError(f"unknown data source {kind!r}")
    return clean_hourly(raw)
