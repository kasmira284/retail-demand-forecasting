import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting System")

# -------------------------------
# Load default dataset
# -------------------------------
def load_default_data():
    df = pd.read_csv("weekly_demand.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df = df.asfreq("W")
    return df

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your weekly demand CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset")
    df = load_default_data()

# -------------------------------
# Data validation
# -------------------------------
required_cols = {"date", "items_shipped"}
if not required_cols.issubset(df.columns):
    st.error("Dataset must contain columns: date, items_shipped")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df.set_index("date", inplace=True)
df = df.asfreq("W")

# -------------------------------
# Show dataset
# -------------------------------
st.subheader("Dataset Preview")
st.write(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
st.dataframe(df.head())

# -------------------------------
# Plot historical demand
# -------------------------------
st.subheader("Historical Weekly Demand")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["items_shipped"], label="Items Shipped")
ax.set_xlabel("Date")
ax.set_ylabel("Items Shipped")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Forecast settings
# -------------------------------
st.subheader("Forecast Settings")

horizon = st.slider(
    "Select number of weeks to forecast",
    min_value=4,
    max_value=26,
    value=12
)

# -------------------------------
# Train SARIMA model
# -------------------------------
use_seasonality = len(df) >= 120  # safe threshold

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

# -------------------------------
# Forecast
# -------------------------------
forecast = results.get_forecast(steps=horizon)

forecast_index = pd.date_range(
    start=df.index[-1] + pd.Timedelta(weeks=1),
    periods=horizon,
    freq="W"
)

forecast_series = pd.Series(
    forecast.predicted_mean.values,
    index=forecast_index
)

# -------------------------------
# Plot forecast
# -------------------------------
st.subheader("Forecast Results")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df.index, df["items_shipped"], label="Historical")
ax2.plot(forecast_series.index, forecast_series, label="Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Items Shipped")
ax2.legend()
st.pyplot(fig2)

# -------------------------------
# Download forecast
# -------------------------------
forecast_df = forecast_series.reset_index()
forecast_df.columns = ["date", "forecast_items"]

st.download_button(
    label="Download Forecast CSV",
    data=forecast_df.to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv"
)


