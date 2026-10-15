# Inventory & Demand Planning Analysis

## Objective
Identify slow- vs fast-moving SKUs, forecast 14-day demand with a
moving-average model, and flag stock-out risk. Delivered as an Excel
model with live formulas plus a dashboard.

## Dataset
Kaggle "Retail Store Inventory Forecasting Dataset": 73,100 daily rows,
5 stores x 20 products. A SKU is defined as Store + Product, giving 100 SKUs.
Download the CSV and save it as `data/raw/retail_store_inventory.csv`
(it is not stored in this repository).

## Tools
Python (pandas, numpy, matplotlib), Excel (openpyxl, live formulas), Git/GitHub.

## Method
1. Cleaning and EDA (`01_clean_eda.py`)
2. Fast/Medium/Slow and ABC classification (`02_classify_skus.py`)
3. Moving-average forecast with a 28-day backtest, measured by WAPE (`03_forecast.py`)
4. Safety stock, reorder point and stock-out risk flag (`04_stockout_risk.py`)
5. Excel model (`05_build_excel.py`) and dashboard image (`06_dashboard.py`)

## Key findings
- 100 SKUs analysed: 25 Fast, 50 Medium, 25 Slow (top and bottom quartile of average daily units).
- 79 SKUs are class A, 16 are class B and 5 are class C.
- Best forecast window: <<7/14/28>> days. Moving-average WAPE is 64.4% vs 87.9% for the
  naive "tomorrow = today" baseline. The moving average beat naive on 99 of 100 SKUs.
- Stock-out risk (lead time 3 days): <<High>> High, <<Medium>> Medium, <<Low>> Low.
- Overstocked SKUs (more than 60 days of cover): <<number>>.
- <<One EDA finding, e.g. effect of promotions on units sold, taken from images/04_promo_effect.png>>

## Assumptions
- **Lead time = 3 days.** Lead time is not in the dataset. I used 3 days because
  the stores restock frequently, and a 7-day assumption flagged 85 of 100 SKUs as High risk.
  It can be changed in the Assumptions sheet of the Excel model.
- Service level 95% (z = 1.65).
- Overstock threshold: more than 60 days of cover.

## Limitations
- Revenue is spread evenly across SKUs (79 of 100 are class A). This is typical of
  this synthetic Kaggle dataset, so the ABC split is less selective than in real retail.
- <<number>> of 100 SKUs hold less than one day of average demand in stock, so most
  are flagged High or Medium risk even with a short lead time.
- The moving average ignores trend and seasonality.
- Next steps: exponential smoothing or Prophet, a holding-cost estimate for
  overstock, and a supplier-specific lead time.

## How to run
```
pip install -r requirements.txt
python src/01_clean_eda.py
python src/02_classify_skus.py
python src/03_forecast.py
python src/04_stockout_risk.py
python src/05_build_excel.py
python src/06_dashboard.py
```

## Results
![Dashboard](images/dashboard.png)
![Excel Dashboard](images/excel_dashboard.png)
![Excel Risk](images/excel_risk.png)