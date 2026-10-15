import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])
df["revenue"] = df["units_sold"] * df["price"]

sku = df.groupby("sku").agg(
    store_id=("store_id", "first"),
    product_id=("product_id", "first"),
    category=("category", "first"),
    total_units=("units_sold", "sum"),
    avg_daily_units=("units_sold", "mean"),
    avg_inventory=("inventory_level", "mean"),
    avg_price=("price", "mean"),
    revenue=("revenue", "sum"),
).reset_index()

# Inventory turnover and days of cover
sku["turnover"] = sku["total_units"] / sku["avg_inventory"]
sku["days_of_cover"] = sku["avg_inventory"] / sku["avg_daily_units"]

# Velocity: top 25% = Fast, bottom 25% = Slow
q25, q75 = sku["avg_daily_units"].quantile([0.25, 0.75])
sku["velocity"] = np.where(sku["avg_daily_units"] >= q75, "Fast",
                  np.where(sku["avg_daily_units"] <= q25, "Slow", "Medium"))

# ABC by revenue (A = first 80%, B = next 15%, C = last 5%)
sku = sku.sort_values("revenue", ascending=False).reset_index(drop=True)
prior_share = (sku["revenue"].cumsum() - sku["revenue"]) / sku["revenue"].sum()
sku["abc"] = np.where(prior_share < 0.80, "A", np.where(prior_share < 0.95, "B", "C"))

cols = ["sku", "store_id", "product_id", "category", "total_units", "avg_daily_units",
        "avg_inventory", "avg_price", "revenue", "turnover", "days_of_cover",
        "velocity", "abc"]
sku[cols].round(2).to_csv("data/processed/sku_classification.csv", index=False)

print(sku["velocity"].value_counts())
print(sku["abc"].value_counts())
print(sku[["sku", "avg_daily_units", "velocity", "abc"]].head(10))