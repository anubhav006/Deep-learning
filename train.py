"""Command-line experiment runner.

Examples
--------
python train.py --source synthetic --epochs 10
python train.py --source uci --lookback 72 --horizon 6
python train.py --source csv --csv my.csv --ts-col date --y-col count
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import data as D
from src import evaluate as E
from src import plots as PL
from src.config import Config
from src.pipeline import run_pipeline, save_results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["uci", "synthetic", "csv"], default="uci")
    ap.add_argument("--csv"); ap.add_argument("--ts-col"); ap.add_argument("--y-col")
    ap.add_argument("--lookback", type=int, default=72)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--units", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()

    cfg = Config(lookback=a.lookback, horizon=a.horizon, epochs=a.epochs, batch_size=a.batch_size,
                 units=tuple(a.units), use_weather=not a.no_weather)
    kw = {"source": a.csv, "ts_col": a.ts_col, "y_col": a.y_col} if a.source == "csv" else {}
    hourly = D.load_source(a.source, **kw)
    print("Data quality:", hourly.attrs["quality"])

    res, model = run_pipeline(hourly, cfg)
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    save_results(res, out / "last_run.pkl")
    model.save(out / "lstm.keras")
    res.metrics_test.to_csv(out / "metrics_test.csv")
    PL.plot_overview(hourly, res.b1, res.b2).savefig(out / "overview.png")
    PL.plot_history(res.history).savefig(out / "training_curve.png")
    PL.plot_forecast(res, len(res.test_origins) // 2).savefig(out / "forecast_example.png")
    for label, col in E.BREAKDOWNS.items():
        tab = E.grouped_metrics(res.long_test, col, res.models)["MAE"]
        PL.plot_breakdown(tab, "MAE", f"MAE by {label.lower()}").savefig(out / f"mae_by_{col}.png")

    print(f"\nWindows: {res.n_windows}")
    print("\nValidation metrics\n", res.metrics_val.to_string())
    print("\nTEST metrics\n", res.metrics_test.to_string())
    print(f"\nArtifacts written to {out.resolve()}")


if __name__ == "__main__":
    main()
