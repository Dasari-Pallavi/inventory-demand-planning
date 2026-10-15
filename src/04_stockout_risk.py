import pandas as pd
import numpy as np

LEAD_TIME = 3    # days (assumption, since the dataset has no lead time)
Z = 1.65         # 95% service level
REVIEW = 7       # days of cover to order for

df = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"]).sort_values("date")
fc = pd.read_csv("data/processed/forecast_summary.csv")

current_stock = df.groupby("sku")["inventory_level"].last().rename("current_stock")
std_daily = df.groupby("sku")["units_sold"].apply(lambda x: x.tail(90).std()).rename("std_daily")

r = fc.merge(current_stock, on="sku").merge(std_daily, on="sku")
r["days_cover"] = np.where(r["forecast_daily"] > 0, r["current_stock"] / r["forecast_daily"], 999)
r["safety_stock"] = Z * r["std_daily"] * np.sqrt(LEAD_TIME)
r["reorder_point"] = r["forecast_daily"] * LEAD_TIME + r["safety_stock"]

r["risk"] = np.where(r["days_cover"] < LEAD_TIME, "High",
            np.where(r["current_stock"] < r["reorder_point"], "Medium", "Low"))
r["order_qty"] = np.maximum(0, np.ceil(r["reorder_point"] + r["forecast_daily"] * REVIEW
                                       - r["current_stock"]))

# Overstock flag (too much cash tied up)
r["overstock"] = np.where(r["days_cover"] > 60, "Yes", "No")

out = r[["sku", "current_stock", "forecast_daily", "std_daily", "days_cover",
         "safety_stock", "reorder_point", "risk", "order_qty", "overstock"]].round(2)
out.to_csv("data/processed/stockout_risk.csv", index=False)

print(out["risk"].value_counts())
print("Overstocked SKUs:", (out["overstock"] == "Yes").sum())
print(out[out["risk"] == "High"].sort_values("days_cover").head(10))