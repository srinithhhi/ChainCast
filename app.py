import streamlit as st
import pandas as pd
import glob
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# App settings for wide/dark mode, monospace font, and modern visual look.
st.set_page_config(page_title="Crypto Price Predictor", layout="wide")

# --- Sidebar: All Controls, Explanations, Inputs ---
with st.sidebar:
    st.title("Crypto Price Predictor")
    
    with st.expander("Feature Explanations"):
        st.markdown(
            """
            **Moving Average (MA):**
            - *7-Day:* Average closing price over the last 7 days (short-term trend).
            - *30-Day:* Average closing price over the last 30 days (longer-term trend).
            
            **Volatility:** Standard deviation of closing prices over 7 days (measures fluctuation).
            
            **Daily Return:** Percent change in closing price vs. previous day (momentum/reversal).
            """
        )
    
    st.header("Controls")

# --- Load Model & Features ---
@st.cache_data
def load_model_and_features():
    model = joblib.load("crypto_regression_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, model_features = load_model_and_features()

# --- Load Data ---
@st.cache_data
def load_and_prepare_data():
    file_paths = glob.glob("crypto-coins/*.csv")
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file)
        coin = os.path.basename(file).replace("coin_", "").replace(".csv", "")
        df["Symbol"] = coin
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    merged['Date'] = pd.to_datetime(merged['Date'])
    merged = merged.sort_values(['Symbol', 'Date'])
    merged['Daily_Return'] = merged.groupby('Symbol')['Close'].pct_change()
    merged['MA_7'] = merged.groupby("Symbol")['Close'].transform(lambda x: x.rolling(7).mean())
    merged['MA_30'] = merged.groupby("Symbol")['Close'].transform(lambda x: x.rolling(30).mean())
    merged['Volatility'] = merged.groupby("Symbol")['Close'].transform(lambda x: x.rolling(7).std())
    merged = merged.dropna().reset_index(drop=True)
    return merged

df = load_and_prepare_data()

symbols = df['Symbol'].unique()
with st.sidebar:
    symbol = st.selectbox("Cryptocurrency", symbols)
    date_min = df['Date'].min().to_pydatetime()
    date_max = df['Date'].max().to_pydatetime()
    date_range = st.slider(
        "Date Range",
        min_value=date_min,
        max_value=date_max,
        value=(date_min, date_max),
        format="YYYY-MM-DD"
    )
    st.header("Prediction Input")
    latest_row = df[(df['Symbol'] == symbol) & (df['Date'] <= pd.to_datetime(date_range[1]))].sort_values('Date').iloc[-1]
    user_inputs = {}
    for feature in model_features:
        if feature == "Symbol":
            user_inputs[feature] = symbol
        else:
            val = latest_row.get(feature, 0)
            val = 0 if pd.isnull(val) else float(round(val, 6))
            user_inputs[feature] = st.number_input(feature, value=val, key=feature)
    predict_btn = st.button("Predict Closing Price", use_container_width=True)
    with st.expander("Advanced Options"):
        st.caption("Model is pre-trained. For most users, no further tuning is necessary.")
    st.markdown("---")
    st.markdown(
        """
        **Instructions:**
        - Select cryptocurrency and date range.
        - Review/adjust technical input features.
        - Click 'Predict Closing Price' for forecast.
        - Use expanders for advanced info and diagnostics.
        """
    )

# --- Data Filtering Based On Sidebar Choice ---
df_coin = df[
    (df['Symbol'] == symbol)
    & (df['Date'] >= pd.to_datetime(date_range[0]))
    & (df['Date'] <= pd.to_datetime(date_range[1]))
].copy()
if df_coin.empty:
    st.error("No data available for selected coin and date range.")
    st.stop()

# --- Main Content Area: Charts, Diagnostics ---
st.title(f"{symbol} Analysis Dashboard")
st.subheader("Current Data Preview")
st.dataframe(df_coin.tail(), use_container_width=True)

st.subheader("Closing Price and Moving Averages")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_coin['Date'], y=df_coin['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df_coin['Date'], y=df_coin['MA_7'], mode='lines', name='7-Day MA'))
fig.add_trace(go.Scatter(x=df_coin['Date'], y=df_coin['MA_30'], mode='lines', name='30-Day MA'))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    legend=dict(orientation="h", y=1.07, x=0.5, xanchor="center")
)
st.plotly_chart(fig, use_container_width=True)

# --- Prediction Logic ---
if predict_btn:
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted closing price: ${prediction:,.2f}")
    # Save to history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({**user_inputs, "Predicted Close": round(prediction, 2)})

st.subheader("Feature Importances")
try:
    reg = model.named_steps['regressor']
    importances = reg.feature_importances_
    feature_names = model.named_steps['preprocessing'].get_feature_names_out()
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    fig_imp = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(template="plotly_dark")
    st.plotly_chart(fig_imp, use_container_width=True)
except Exception:
    st.info("Feature importance plot not available for this model.")

st.subheader("Feature Correlation Heatmap")
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA_7', 'MA_30', 'Volatility']
if all(col in df_coin.columns for col in corr_cols):
    fig_corr = px.imshow(
        df_coin[corr_cols].corr(),
        color_continuous_scale="RdBu",
        text_auto=True,
        title="Correlation Matrix"
    )
    fig_corr.update_layout(template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not all correlation columns present to plot heatmap.")

if "history" in st.session_state and st.session_state["history"]:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["history"]), use_container_width=True)
