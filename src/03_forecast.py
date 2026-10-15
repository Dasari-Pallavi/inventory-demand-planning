import pandas as pd
import numpy as np

WINDOWS = [7, 14, 28]   # candidate moving-average windows
HOLDOUT = 28            # days used to test accuracy
HORIZON = 14            # days to forecast ahead

df = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])

def wape(actual, pred):
    return np.abs(actual - pred).sum() / max(actual.sum(), 1e-9)

summary, future = [], []
for sku, g in df.groupby("sku"):
    s = g.set_index("date")["units_sold"].asfreq("D").fillna(0)
    actual = s.iloc[-HOLDOUT:]

    # Baseline: "tomorrow = today"
    naive_err = wape(actual.values, s.shift(1).iloc[-HOLDOUT:].values)

    # Try each MA window; the prediction uses only past data (shift(1))
    best_w, best_err = None, None
    for w in WINDOWS:
        pred = s.rolling(w).mean().shift(1).iloc[-HOLDOUT:]
        err = wape(actual.values, pred.values)
        if best_err is None or err < best_err:
            best_w, best_err = w, err

    forecast_daily = s.tail(best_w).mean()
    summary.append([sku, best_w, round(best_err, 4), round(naive_err, 4),
                    round(forecast_daily, 2)])

    start = s.index[-1] + pd.Timedelta(days=1)
    for d in pd.date_range(start, periods=HORIZON):
        future.append([sku, d, round(forecast_daily, 2)])

pd.DataFrame(summary, columns=["sku", "best_window", "wape", "naive_wape", "forecast_daily"]) \
  .to_csv("data/processed/forecast_summary.csv", index=False)
pd.DataFrame(future, columns=["sku", "date", "forecast_units"]) \
  .to_csv("data/processed/forecast_14d.csv", index=False)

res = pd.read_csv("data/processed/forecast_summary.csv")
print("Average WAPE (MA):   ", round(res["wape"].mean(), 3))
print("Average WAPE (naive):", round(res["naive_wape"].mean(), 3))
print("SKUs where MA beats naive:", (res["wape"] < res["naive_wape"]).sum(), "of", len(res))