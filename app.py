import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Custom CSS for a hyper-professional Fintech look
st.markdown("""
<style>
    /* Main Background & Font settings */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }
    
    /* Glassmorphism Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.25rem !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Metric Values & Labels */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #3b82f6, #06b6d4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #2563eb, #0891b2);
        box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Expander & Divider styling */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.5) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Input & Slider styling overrides */
    .stTextInput input, .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    with open("stock_open_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📈 Advanced Stock Opening Price Predictor")
st.markdown("Use this professional dashboard to visualize historical stock data and predict the next day's opening price based on our trained machine learning model.")

current_year = datetime.date.today().year

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", value="")
start_year, end_year = st.sidebar.slider(
    "Select Year Range:",
    min_value=2000,
    max_value=current_year,
    value=(2020, current_year)
)
st.sidebar.markdown("---")
predict_button = st.sidebar.button("Generate Prediction", type="primary")

if predict_button:

    if ticker:
        with st.spinner(f"Fetching data for {ticker}..."):
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31" if end_year < current_year else datetime.date.today().strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_date, end=end_date)

        df = df.sort_index()

        # Feature engineering
        df["Return"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_10"] = df["Close"].rolling(10).mean()
        df["Volatility_5"] = df["Close"].rolling(5).std()

        df["Lag_1"] = df["Close"].shift(1)
        df["Lag_2"] = df["Close"].shift(2)
        df["Lag_3"] = df["Close"].shift(3)

        df = df.dropna()

        latest_data = df.iloc[-1:]

        prediction = model.predict(latest_data)[0]

        st.markdown("---")
        st.subheader("📊 Prediction & Current Status")
        
        # Get actual latest price and volume
        latest_close = df["Close"].iloc[-1]
        latest_open = df["Open"].iloc[-1]
        latest_volume = df["Volume"].iloc[-1]
        
        if isinstance(latest_close, pd.Series): latest_close = latest_close.iloc[0]
        if isinstance(latest_open, pd.Series): latest_open = latest_open.iloc[0]
        if isinstance(latest_volume, pd.Series): latest_volume = latest_volume.iloc[0]
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Close", f"${float(latest_close):.2f}")
        with col2:
            st.metric("Latest Open", f"${float(latest_open):.2f}")
        with col3:
            st.metric("Trading Volume", f"{int(latest_volume):,}")
        with col4:
            delta = float(prediction) - float(latest_close)
            delta_pct = (delta / float(latest_close)) * 100
            st.metric("Predicted Next Open", f"${float(prediction):.2f}", f"{delta:.2f} ({delta_pct:.2f}%)", delta_color="normal")
        
        st.markdown("---")
        st.subheader(f"📈 {ticker} Interactive Price Chart")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'].squeeze(),
            high=df['High'].squeeze(),
            low=df['Low'].squeeze(),
            close=df['Close'].squeeze(),
            name=ticker
        )])
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=True,
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add a raw data expander for transparency
        with st.expander("View Raw Historical Data"):
            st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

    else:
        st.sidebar.error("Please enter a stock ticker to proceed.")
else:
    if not ticker:
        st.info("👈 Please enter a stock ticker in the sidebar and click **Generate Prediction** to start.")