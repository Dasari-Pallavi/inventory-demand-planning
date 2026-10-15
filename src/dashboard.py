import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inventory Dashboard", layout="wide")
risk = pd.read_csv("outputs/stockout_risk.csv")

st.title("Inventory & Demand Planning Dashboard")
cats = sorted(risk["category"].unique())
sel_cat = st.sidebar.multiselect("Category", cats, default=cats)
sel_vel = st.sidebar.multiselect("Velocity", ["Fast", "Medium", "Slow"], default=["Fast", "Medium", "Slow"])
f = risk[risk["category"].isin(sel_cat) & risk["velocity"].isin(sel_vel)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("SKUs", len(f))
c2.metric("High stock-out risk", int((f["risk"] == "High").sum()))
c3.metric("Overstocked", int((f["risk"] == "Overstock").sum()))
c4.metric("Avg forecast error (WAPE)", f"{f['wape'].mean():.1%}")

left, right = st.columns(2)
left.subheader("Days of cover by SKU")
left.bar_chart(f.set_index("product_id")["days_of_cover"])
right.subheader("Risk status")
right.bar_chart(f["risk"].value_counts())

st.subheader("Reorder list (High / Medium risk)")
st.dataframe(f[f["risk"].isin(["High", "Medium"])]
             .sort_values("days_of_cover")[["product_id", "category", "velocity", "current_stock",
                                            "days_of_cover", "reorder_point", "suggested_order_qty", "risk"]])

st.image("outputs/charts/06_forecast_vs_actual.png", caption="Forecast vs actual (top SKU)")