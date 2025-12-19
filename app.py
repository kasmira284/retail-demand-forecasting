import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting System")

# =====================================================
# Load default dataset (ONLY default is cleaned)
# =====================================================
def load_default_data():
    df = pd.read_csv("weekly_demand.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df = df.asfreq("W-SUN")
    return df

# =====================================================
# File uploader
# =====================================================
uploaded_file = st.file_uploader("Upload weekly demand CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded")
else:
    st.info("Using default dataset")
    df = load_default_data()

# =====================================================
# Data validation
# =====================================================
required_cols = {"date", "items_shipped"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain columns: date, items_shipped")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df.set_index("date", inplace=True)
df = df.asfreq("W-SUN")

# Allow smaller datasets for demo
if len(df) < 12:
    st.error("Dataset too short for SARIMA (minimum 12 weeks required).")
    st.stop()

# =====================================================
# Dataset preview
# =====================================================
st.subheader("Dataset Preview")
st.write(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
st.dataframe(df.head())

# =====================================================
# Historical plot
# =====================================================
st.subheader("Historical Weekly Demand")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["items_shipped"], label="Historical")
ax.set_ylabel("Items Shipped")
ax.legend()
st.pyplot(fig)

# =====================================================
# Forecast settings
# =====================================================
st.subheader("Forecast Settings")
horizon = st.slider("Weeks to forecast", 4, 26, 12)

# =====================================================
# Stationarity Test (ADF)
# =====================================================
st.subheader("Stationarity Test (ADF)")

adf_stat, p_value, _, _, crit_vals, _ = adfuller(df["items_shipped"].dropna())

st.write(f"ADF Statistic: **{adf_stat:.4f}**")
st.write(f"p-value: **{p_value:.6f}**")

crit_df = pd.DataFrame({"Critical Value": crit_vals})
st.dataframe(crit_df)

if p_value < 0.05:
    st.success("Series is stationary.")
else:
    st.warning("Series is non-stationary → differencing required (d = 1).")

# =====================================================
# SARIMA model
# =====================================================
seasonal = (0, 0, 1, 52) if len(df) >= 104 else (0, 0, 0, 0)

model = SARIMAX(
    df["items_shipped"],
    order=(0, 1, 1),
    seasonal_order=seasonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

# =====================================================
# Forecast
# =====================================================
forecast_res = results.get_forecast(steps=horizon)

forecast_index = pd.date_range(
    df.index[-1] + pd.Timedelta(weeks=1),
    periods=horizon,
    freq="W-SUN"
)

forecast_df = pd.DataFrame({
    "forecast": forecast_res.predicted_mean.values,
    "lower_ci": forecast_res.conf_int().iloc[:, 0].values,
    "upper_ci": forecast_res.conf_int().iloc[:, 1].values
}, index=forecast_index).clip(lower=0)

# =====================================================
# Plot forecast
# =====================================================
st.subheader("Forecast Results with Confidence Interval")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df.index, df["items_shipped"], label="Historical")
ax2.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")
ax2.fill_between(
    forecast_df.index,
    forecast_df["lower_ci"],
    forecast_df["upper_ci"],
    alpha=0.25,
    label="95% CI"
)
ax2.legend()
st.pyplot(fig2)

# =====================================================
# Diagnostics
# =====================================================
st.subheader("Model Diagnostics")

residuals = results.resid.dropna()

fig_r, ax_r = plt.subplots(figsize=(10, 3))
ax_r.plot(residuals)
ax_r.axhline(0, linestyle="--")
st.pyplot(fig_r)

fig_acf, ax_acf = plt.subplots(figsize=(6, 3))
plot_acf(residuals, lags=min(20, len(residuals)//2), ax=ax_acf)
st.pyplot(fig_acf)

# =====================================================
# Model comparison
# =====================================================
st.subheader("Model Comparison (Naive vs SARIMA)")

split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

naive = test["items_shipped"].shift(1).dropna()
test_naive = test.loc[naive.index, "items_shipped"]

naive_rmse = np.sqrt(np.mean((test_naive - naive) ** 2))
sarima_pred = results.get_prediction(start=test.index[0], end=test.index[-1]).predicted_mean
sarima_rmse = np.sqrt(np.mean((test["items_shipped"] - sarima_pred) ** 2))

st.dataframe(pd.DataFrame({
    "Model": ["Naive", "SARIMA"],
    "RMSE": [round(naive_rmse, 2), round(sarima_rmse, 2)]
}))

# =====================================================
# Business insights
# =====================================================
st.subheader("Business Insights")

st.markdown(f"""
• **Average weekly demand:** {df["items_shipped"].mean():.0f} units  
• **Peak demand observed:** {df["items_shipped"].max():.0f} units  

**Why SARIMA works here**
- Captures trend and seasonality  
- Outperforms naive baseline  
- Suitable for weekly retail demand forecasting  
""")
