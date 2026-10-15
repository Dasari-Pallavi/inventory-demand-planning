import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("images").mkdir(exist_ok=True)

# 1. Load
df = pd.read_csv("data/raw/retail_store_inventory.csv")
print("Shape:", df.shape)
print(df.info())
print("Missing values:\n", df.isna().sum())

# 2. Standardise column names: "Holiday/Promotion" -> "holiday_promotion"
df.columns = (df.columns.str.strip().str.lower()
              .str.replace("/", "_").str.replace(" ", "_"))

# 3. Clean
df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates()
df = df.dropna(subset=["date", "store_id", "product_id", "units_sold", "inventory_level"])
df = df[(df["units_sold"] >= 0) & (df["inventory_level"] >= 0)]
df["sku"] = df["store_id"] + "-" + df["product_id"]
df = df.sort_values(["sku", "date"]).reset_index(drop=True)
df.to_csv("data/processed/clean_sales.csv", index=False)
print("Clean shape:", df.shape, "| SKUs:", df["sku"].nunique())

# 4. EDA charts
# 4a. Total daily units sold
daily = df.groupby("date")["units_sold"].sum()
plt.figure(figsize=(10, 4))
daily.plot()
plt.title("Total Units Sold per Day")
plt.xlabel("Date"); plt.ylabel("Units")
plt.savefig("images/01_daily_units_sold.png", dpi=150, bbox_inches="tight"); plt.close()

# 4b. Monthly units by category
monthly = (df.assign(month=df["date"].dt.to_period("M").astype(str))
             .pivot_table(index="month", columns="category", values="units_sold", aggfunc="sum"))
monthly.plot(figsize=(11, 5))
plt.title("Monthly Units Sold by Category")
plt.ylabel("Units"); plt.xticks(rotation=45)
plt.savefig("images/02_monthly_by_category.png", dpi=150, bbox_inches="tight"); plt.close()

# 4c. Distribution of daily units sold
plt.figure(figsize=(7, 4))
df["units_sold"].hist(bins=40)
plt.title("Distribution of Daily Units Sold (per SKU-day)")
plt.xlabel("Units sold"); plt.ylabel("Frequency")
plt.savefig("images/03_units_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

# 4d. Promotion effect (if column exists)
if "holiday_promotion" in df.columns:
    df.groupby("holiday_promotion")["units_sold"].mean().plot(kind="bar", figsize=(5, 4))
    plt.title("Average Units Sold: Promotion vs Normal")
    plt.ylabel("Avg units"); plt.xticks(rotation=0)
    plt.savefig("images/04_promo_effect.png", dpi=150, bbox_inches="tight"); plt.close()

print("EDA charts saved in images/")