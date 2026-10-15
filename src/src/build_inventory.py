"""Build SKU-level inventory metrics, moving-average forecast and stock-out risk."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/retail_store_inventory.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Assumptions (change here, then re-run) ----
LEAD_TIME_DAYS = 3       # days between placing and receiving a replenishment order
Z_SERVICE = 1.65         # 95% service level
HORIZON = 14             # days to forecast
TEST_DAYS = 28           # backtest window
WINDOWS = [7, 14, 28]    # moving-average windows to compare
OVERSTOCK_DAYS = 60      # days of cover above this = overstock
LOW_COVER_DAYS = 3       # "low stock day" = inventory below this many days of demand

# ---------- 1. Load and standardise ----------
df = pd.read_csv(RAW)
df.columns = [re.sub(r"[^0-9a-zA-Z]+", "_", c.strip().lower()).strip("_") for c in df.columns]
need = ["date", "store_id", "product_id", "category", "inventory_level", "units_sold", "price"]
missing = [c for c in need if c not in df.columns]
if missing:
    sys.exit(f"Missing columns {missing}. Columns found: {list(df.columns)}")

df["date"] = pd.to_datetime(df["date"])
n0 = len(df)
df = df.drop_duplicates(["date", "store_id", "product_id"]).dropna(subset=need)
df = df[(df.units_sold >= 0) & (df.inventory_level >= 0) & (df.price > 0)].copy()
print(f"Rows: {n0:,} raw -> {len(df):,} after cleaning")

df["sku"] = df["product_id"].astype(str) + "|" + df["store_id"].astype(str)
df["revenue"] = df["units_sold"] * df["price"]
last_date = df["date"].max()

# ---------- 2. Daily demand matrix (rows = dates, columns = SKUs) ----------
daily = (df.pivot_table(index="date", columns="sku", values="units_sold", aggfunc="sum")
           .asfreq("D").fillna(0))

# ---------- 3. SKU summary ----------
s = (df.groupby("sku")
       .agg(product_id=("product_id", "first"), store_id=("store_id", "first"),
            category=("category", "first"), days=("date", "nunique"),
            total_units=("units_sold", "sum"), revenue=("revenue", "sum"),
            avg_daily=("units_sold", "mean"), std_daily=("units_sold", "std"),
            avg_inventory=("inventory_level", "mean"))
       .reset_index())
s["std_daily"] = s["std_daily"].fillna(0)
s["current_inventory"] = s["sku"].map(df.sort_values("date").groupby("sku").inventory_level.last())
s["cv"] = s["std_daily"] / s["avg_daily"]
s["annual_turnover"] = s["avg_daily"] * 365 / s["avg_inventory"]

# Fast / Medium / Slow by average daily units (top / middle / bottom quartile)
q25, q75 = s["avg_daily"].quantile([0.25, 0.75])
s["movement"] = np.where(s.avg_daily >= q75, "Fast",
                  np.where(s.avg_daily <= q25, "Slow", "Medium"))

# ABC by revenue share (A = first 80%, B = next 15%, C = last 5%)
s = s.sort_values("revenue", ascending=False).reset_index(drop=True)
cum = s["revenue"].cumsum() / s["revenue"].sum()
s["abc"] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
s.loc[0, "abc"] = "A"

# Share of days with less than LOW_COVER_DAYS of stock
df["sku_avg_daily"] = df["sku"].map(s.set_index("sku")["avg_daily"])
df["low_stock"] = df["inventory_level"] < LOW_COVER_DAYS * df["sku_avg_daily"]
s["low_stock_days_pct"] = s["sku"].map(df.groupby("sku").low_stock.mean())

# ---------- 4. Backtest moving averages (one-step-ahead, last TEST_DAYS) ----------
test_idx = daily.index[-TEST_DAYS:]
actual = daily.loc[test_idx]
results = []
preds = {f"MA{w}": daily.shift(1).rolling(w).mean() for w in WINDOWS}
preds["Naive (yesterday)"] = daily.shift(1)
preds["Seasonal naive (7d ago)"] = daily.shift(7)
for name, p in preds.items():
    err = (p.loc[test_idx] - actual).abs()
    results.append({"model": name,
                    "MAE": err.stack().mean(),
                    "WAPE_pct": 100 * err.sum().sum() / actual.sum().sum()})
bt = pd.DataFrame(results).sort_values("WAPE_pct").reset_index(drop=True)
print("\nBacktest (lower is better):\n", bt.round(2).to_string(index=False))
best = min(WINDOWS, key=lambda w: bt.loc[bt.model == f"MA{w}", "WAPE_pct"].iloc[0])
print(f"\nBest moving average: MA{best}")

# ---------- 5. Forecast next HORIZON days ----------
s["forecast_daily"] = s["sku"].map(daily.tail(best).mean())
s["forecast_14d"] = s["forecast_daily"] * HORIZON
fdates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON)
fc = pd.DataFrame({"date": np.repeat(fdates, len(s)), "sku": np.tile(s.sku, HORIZON),
                   "forecast_units": np.tile(s.forecast_daily.values, HORIZON)})

# ---------- 6. Stock-out risk ----------
s["lead_time_demand"] = s["forecast_daily"] * LEAD_TIME_DAYS
s["safety_stock"] = Z_SERVICE * s["std_daily"] * np.sqrt(LEAD_TIME_DAYS)
s["reorder_point"] = s["lead_time_demand"] + s["safety_stock"]
s["days_of_cover"] = s["current_inventory"] / s["forecast_daily"].replace(0, np.nan)
s["stockout_risk"] = np.where(s.current_inventory < s.lead_time_demand, "High",
                        np.where(s.current_inventory < s.reorder_point, "Medium", "Low"))
s["overstock"] = np.where(s.days_of_cover > OVERSTOCK_DAYS, "Yes", "No")

# ---------- 7. Total daily demand + forecast (for charts) ----------
tot = pd.DataFrame({"date": daily.index, "actual": daily.sum(axis=1).values,
                    f"ma{best}_fit": preds[f"MA{best}"].sum(axis=1).values})
tot = pd.concat([tot, pd.DataFrame({"date": fdates, "forecast": s.forecast_daily.sum()})])

# ---------- 8. Save ----------
s.to_csv(OUT / "sku_summary.csv", index=False, encoding="utf-8-sig")
fc.to_csv(OUT / "forecast_next14.csv", index=False)
bt.to_csv(OUT / "backtest.csv", index=False)
tot.to_csv(OUT / "daily_total.csv", index=False)
(OUT / "best_window.txt").write_text(str(best))

print("\nSKU-store combinations:", len(s))
print("Movement:", s.movement.value_counts().to_dict())
print("ABC:", s.abc.value_counts().to_dict())
print("Risk:", s.stockout_risk.value_counts().to_dict())
print("Overstock:", s.overstock.value_counts().to_dict())