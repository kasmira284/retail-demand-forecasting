import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("Retail Demand Forecasting System")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("weekly_demand.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    return df

data = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("Weekly Demand Dataset")
st.write(f"Date range: {data.index.min()} to {data.index.max()}")
st.dataframe(data.head(10))

# --------------------------------------------------
# Historical Demand Plot
# --------------------------------------------------
st.subheader("Historical Weekly Demand")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data.index, data["items_shipped"], label="Historical Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Items Shipped")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# Forecast Settings
# --------------------------------------------------
st.subheader("Forecast Settings")

forecast_horizon = st.slider(
    "Select number of weeks to forecast",
    min_value=4,
    max_value=26,
    value=12,
    step=1
)

st.write(f"Forecasting **{forecast_horizon} weeks ahead**")

# --------------------------------------------------
# SARIMAX MODEL (TRAIN ONCE)
# --------------------------------------------------
@st.cache_resource
def train_sarimax(series):
    model = SARIMAX(
        series,
        order=(0, 1, 1),
        seasonal_order=(0, 0, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    return fitted_model

sarimax_model = train_sarimax(data["items_shipped"])

# --------------------------------------------------
# Forecast Generation
# --------------------------------------------------
forecast = sarimax_model.get_forecast(steps=forecast_horizon)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

forecast_index = pd.date_range(
    start=data.index[-1] + pd.Timedelta(weeks=1),
    periods=forecast_horizon,
    freq="W"
)

forecast_mean.index = forecast_index
forecast_ci.index = forecast_index

# --------------------------------------------------
# Forecast Plot with Confidence Intervals
# --------------------------------------------------
st.subheader("SARIMAX Forecast with Confidence Intervals")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(data.index, data["items_shipped"], label="Historical")
ax2.plot(
    forecast_mean.index,
    forecast_mean,
    linestyle="--",
    label="Forecast"
)

ax2.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="gray",
    alpha=0.3,
    label="Confidence Interval"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Items Shipped")
ax2.legend()

st.pyplot(fig2)

# --------------------------------------------------
# Forecast Table
# --------------------------------------------------
st.subheader("Forecast Values")

forecast_df = pd.DataFrame({
    "Forecasted Items Shipped": forecast_mean.round(0).astype(int),
    "Lower Bound": forecast_ci.iloc[:, 0].round(0).astype(int),
    "Upper Bound": forecast_ci.iloc[:, 1].round(0).astype(int),
})

st.dataframe(forecast_df)

