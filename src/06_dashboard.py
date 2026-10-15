import pandas as pd
import matplotlib.pyplot as plt

cls = pd.read_csv("data/processed/sku_classification.csv")
risk = pd.read_csv("data/processed/stockout_risk.csv")
sales = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Inventory & Demand Planning Dashboard", fontsize=18, fontweight="bold")

# 1. Velocity
cls["velocity"].value_counts().reindex(["Fast", "Medium", "Slow"]).plot(
    kind="bar", ax=ax[0, 0], color=["#2E75B6", "#9DC3E6", "#BDD7EE"])
ax[0, 0].set_title("SKUs by Velocity"); ax[0, 0].tick_params(axis="x", rotation=0)

# 2. ABC
cls["abc"].value_counts().reindex(["A", "B", "C"]).plot(
    kind="bar", ax=ax[0, 1], color="#548235")
ax[0, 1].set_title("SKUs by ABC Class (revenue)"); ax[0, 1].tick_params(axis="x", rotation=0)

# 3. Risk
risk["risk"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).plot(
    kind="bar", ax=ax[0, 2], color=["#C00000", "#FFC000", "#70AD47"])
ax[0, 2].set_title("Stock-out Risk"); ax[0, 2].tick_params(axis="x", rotation=0)

# 4. Top 10 most urgent SKUs
urgent = risk.sort_values("days_cover").head(10)
ax[1, 0].barh(urgent["sku"], urgent["days_cover"], color="#C00000")
ax[1, 0].invert_yaxis(); ax[1, 0].set_title("10 Lowest Days of Cover")
ax[1, 0].set_xlabel("Days")

# 5. Actual vs moving average for the top-revenue SKU
top_sku = cls.sort_values("revenue", ascending=False)["sku"].iloc[0]
s = sales[sales["sku"] == top_sku].set_index("date")["units_sold"].tail(90)
ax[1, 1].plot(s.index, s.values, label="Actual", alpha=0.6)
ax[1, 1].plot(s.index, s.rolling(7).mean(), label="7-day MA", linewidth=2)
ax[1, 1].set_title(f"Actual vs Moving Average ({top_sku})")
ax[1, 1].legend(); ax[1, 1].tick_params(axis="x", rotation=45)

# 6. Slow movers with overstock
over = risk.merge(cls[["sku", "velocity"]], on="sku")
over = over[(over["overstock"] == "Yes") & (over["velocity"] == "Slow")] \
         .sort_values("days_cover", ascending=False).head(10)
ax[1, 2].barh(over["sku"], over["days_cover"], color="#7F7F7F")
ax[1, 2].invert_yaxis(); ax[1, 2].set_title("Slow Movers with Overstock (days of cover)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("images/dashboard.png", dpi=150)
print("Saved images/dashboard.png")