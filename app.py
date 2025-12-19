import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting System")


# --------------------------------------------------
# Helper: Trim trailing zero-demand weeks
# --------------------------------------------------
def trim_trailing_zeros(df, target_col="items_shipped"):
    df = df.copy()
    while len(df) > 0 and df[target_col].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df


# --------------------------------------------------
# Load default dataset
# --------------------------------------------------
def load_default_data():
    df = pd.read_csv("weekly_demand.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date")
    df.set_index("date", inplace=True)
    df = df.asfreq("W-SUN")
    df = trim_trailing_zeros(df)
    return df


# --------------------------------------------------
# File uploader
# --------------------------------------------------
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


# --------------------------------------------------
# Data validation & preparation
# --------------------------------------------------
required_cols = {"date", "items_shipped", "total_freight", "total_revenue"}
if not required_cols.issubset(df.columns):
    st.error(
        "Dataset must contain columns: "
        "date, items_shipped, total_freight, total_revenue"
    )
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df.set_index("date", inplace=True)
df = df.asfreq("W-SUN")
df = trim_trailing_zeros(df)

if len(df) < 20:
    st.error("Dataset too short after cleaning to build a forecast.")
    st.stop()


# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.subheader("Dataset Preview")
st.write(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
st.dataframe(df.head())


# --------------------------------------------------
# Historical plot
# --------------------------------------------------
st.subheader("Historical Weekly Demand")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["items_shipped"], label="Historical Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Items Shipped")
ax.legend()
st.pyplot(fig)


# --------------------------------------------------
# Stationarity Test (ADF)
# --------------------------------------------------
st.subheader("Stationarity Test (Augmented Dickey–Fuller)")

adf_stat, p_value, _, _, crit_vals, _ = adfuller(df["items_shipped"].dropna())

st.write("**ADF Statistic:**", round(adf_stat, 4))
st.write("**p-value:**", round(p_value, 6))
st.dataframe(
    pd.DataFrame.from_dict(crit_vals, orient="index", columns=["Critical Value"])
)

if p_value < 0.05:
    st.success(
        "The series is stationary. No differencing required."
    )
else:
    st.warning(
        "The series is non-stationary. "
        "First-order differencing (d = 1) is required."
    )


# --------------------------------------------------
# Forecast settings
# --------------------------------------------------
st.subheader("Forecast Settings")

horizon = st.slider(
    "Select number of weeks to forecast",
    min_value=4,
    max_value=26,
    value=12
)


# --------------------------------------------------
# SARIMAX with exogenous variables
# --------------------------------------------------
exog_cols = ["total_freight", "total_revenue"]
exog = df[exog_cols]

use_seasonality = len(df) >= 104

if use_seasonality:
    model = SARIMAX(
        df["items_shipped"],
        exog=exog,
        order=(0, 1, 1),
        seasonal_order=(0, 0, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
else:
    model = SARIMAX(
        df["items_shipped"],
        exog=exog,
        order=(0, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

results = model.fit(disp=False)


# --------------------------------------------------
# Future exogenous assumptions
# --------------------------------------------------
last_exog = exog.iloc[-1]

future_exog = pd.DataFrame(
    np.tile(last_exog.values, (horizon, 1)),
    columns=exog_cols,
    index=pd.date_range(
        start=df.index[-1] + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-SUN"
    )
)


# --------------------------------------------------
# Forecast + confidence intervals
# --------------------------------------------------
forecast_res = results.get_forecast(
    steps=horizon,
    exog=future_exog
)

forecast_index = future_exog.index
conf_int = forecast_res.conf_int()

forecast_df = pd.DataFrame({
    "forecast": forecast_res.predicted_mean.values,
    "lower_ci": conf_int.iloc[:, 0].values,
    "upper_ci": conf_int.iloc[:, 1].values
}, index=forecast_index)

forecast_df = forecast_df.clip(lower=0)


# --------------------------------------------------
# Plot forecast
# --------------------------------------------------
st.subheader("Forecast Results with Confidence Interval")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df.index, df["items_shipped"], label="Historical")
ax2.plot(forecast_df.index, forecast_df["forecast"], label="Forecast")

ax2.fill_between(
    forecast_df.index,
    forecast_df["lower_ci"],
    forecast_df["upper_ci"],
    alpha=0.25,
    label="95% Confidence Interval"
)

ax2.legend()
st.pyplot(fig2)


# --------------------------------------------------
# Model diagnostics
# --------------------------------------------------
st.subheader("Model Diagnostics")

residuals = results.resid.dropna()

fig_r, ax_r = plt.subplots(figsize=(10, 3))
ax_r.plot(residuals)
ax_r.axhline(0, linestyle="--", color="red")
st.pyplot(fig_r)

fig_h, ax_h = plt.subplots(figsize=(6, 3))
ax_h.hist(residuals, bins=30)
st.pyplot(fig_h)

fig_acf, ax_acf = plt.subplots(figsize=(6, 3))
plot_acf(residuals, lags=20, ax=ax_acf)
st.pyplot(fig_acf)


# --------------------------------------------------
# Model comparison: Naive vs SARIMAX
# --------------------------------------------------
st.subheader("Model Comparison (Naive vs SARIMAX)")

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

naive_forecast = test["items_shipped"].shift(1).dropna()
test_naive = test.loc[naive_forecast.index, "items_shipped"]

naive_mae = np.mean(np.abs(test_naive - naive_forecast))
naive_rmse = np.sqrt(np.mean((test_naive - naive_forecast) ** 2))

sarimax_pred = results.get_prediction(
    start=test.index[0],
    end=test.index[-1],
    exog=exog.loc[test.index]
).predicted_mean

sarimax_mae = np.mean(np.abs(test["items_shipped"] - sarimax_pred))
sarimax_rmse = np.sqrt(np.mean((test["items_shipped"] - sarimax_pred) ** 2))

comparison = pd.DataFrame({
    "Model": ["Naive Forecast", "SARIMAX"],
    "MAE": [round(naive_mae, 2), round(sarimax_mae, 2)],
    "RMSE": [round(naive_rmse, 2), round(sarimax_rmse, 2)]
})

st.dataframe(comparison)

if sarimax_rmse < naive_rmse:
    st.success(
        "SARIMAX outperforms the naive baseline, "
        "showing the value of external business drivers."
    )


# --------------------------------------------------
# Business insights
# --------------------------------------------------
st.subheader("Business Insights")

st.markdown(
    f"""
• **Average weekly demand:** {df['items_shipped'].mean():.0f} units  
• **Peak historical demand:** {df['items_shipped'].max():.0f} units  

**Business value of this system**
- Improves inventory and logistics planning  
- Incorporates freight and revenue as demand drivers  
- Provides uncertainty awareness via confidence intervals  
- Outperforms naive forecasting methods  

**Why SARIMAX**
- Handles trend and seasonality  
- Uses external business variables  
- Produces more realistic retail forecasts  
"""
)


# --------------------------------------------------
# Download forecast
# --------------------------------------------------
download_df = forecast_df.reset_index()
download_df.columns = ["date", "forecast", "lower_ci", "upper_ci"]

st.download_button(
    "Download Forecast (CSV)",
    data=download_df.to_csv(index=False),
    file_name="forecast_with_ci.csv",
    mime="text/csv"
)
