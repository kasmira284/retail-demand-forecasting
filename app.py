import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting System")

# -----------------------------------
# Helper: Trim trailing zero-demand weeks
# -----------------------------------
def trim_trailing_zeros(df, target_col="items_shipped"):
    df = df.copy()
    while len(df) > 0 and df[target_col].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df

# -----------------------------------
# Load default dataset
# -----------------------------------
def load_default_data():
    df = pd.read_csv("weekly_demand.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df = df.asfreq("W-SUN")
    df = trim_trailing_zeros(df)
    return df

# -----------------------------------
# File uploader
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload your weekly demand CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded")
else:
    st.info("Using default dataset")
    df = load_default_data()

# -----------------------------------
# Data validation & preparation
# -----------------------------------
required_cols = {"date", "items_shipped"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain columns: date, items_shipped")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df.set_index("date", inplace=True)
df = df.asfreq("W-SUN")

df = trim_trailing_zeros(df)

if len(df) < 20:
    st.error("Dataset too short after cleaning to build a forecast.")
    st.stop()

# -----------------------------------
# Dataset preview
# -----------------------------------
st.subheader("Dataset Preview")
st.write(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
st.dataframe(df.head())

# -----------------------------------
# Historical plot
# -----------------------------------
st.subheader("Historical Weekly Demand")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["items_shipped"], label="Historical Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Items Shipped")
ax.legend()
st.pyplot(fig)

# -----------------------------------
# Forecast settings
# -----------------------------------
st.subheader("Forecast Settings")

horizon = st.slider(
    "Select number of weeks to forecast",
    min_value=4,
    max_value=26,
    value=12
)

# -----------------------------------
# SARIMA model
# -----------------------------------
use_seasonality = len(df) >= 104

if use_seasonality:
    model = SARIMAX(
        df["items_shipped"],
        order=(0, 1, 1),
        seasonal_order=(0, 0, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
else:
    model = SARIMAX(
        df["items_shipped"],
        order=(0, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

results = model.fit(disp=False)

# -----------------------------------
# Forecast + Confidence Intervals
# -----------------------------------
forecast_res = results.get_forecast(steps=horizon)

forecast_index = pd.date_range(
    start=df.index[-1] + pd.Timedelta(weeks=1),
    periods=horizon,
    freq="W-SUN"
)

forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

forecast_df = pd.DataFrame({
    "forecast": forecast_mean.values,
    "lower_ci": conf_int.iloc[:, 0].values,
    "upper_ci": conf_int.iloc[:, 1].values
}, index=forecast_index)

# No negative demand
forecast_df[["forecast", "lower_ci", "upper_ci"]] = forecast_df[
    ["forecast", "lower_ci", "upper_ci"]
].clip(lower=0)

# -----------------------------------
# Plot forecast with confidence intervals
# -----------------------------------
st.subheader("Forecast Results with Confidence Interval")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(df.index, df["items_shipped"], label="Historical", color="blue")
ax2.plot(forecast_df.index, forecast_df["forecast"], label="Forecast", color="orange")

ax2.fill_between(
    forecast_df.index,
    forecast_df["lower_ci"],
    forecast_df["upper_ci"],
    color="orange",
    alpha=0.25,
    label="95% Confidence Interval"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Items Shipped")
ax2.legend()
st.pyplot(fig2)

# -----------------------------------
# Download forecast
# -----------------------------------
download_df = forecast_df.reset_index()
download_df.columns = ["date", "forecast", "lower_ci", "upper_ci"]

st.download_button(
    label="Download Forecast with Confidence Interval (CSV)",
    data=download_df.to_csv(index=False),
    file_name="forecast_with_ci.csv",
    mime="text/csv"
)

# -----------------------------------
# Explanation
# -----------------------------------
st.info(
    "The shaded region represents the 95% confidence interval, indicating "
    "the range within which future demand is expected to fall with high probability."
)

