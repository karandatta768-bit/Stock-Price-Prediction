import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("stock_open_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📈 Next-Day Stock Opening Price Prediction")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if st.button("Predict"):

    if ticker:
        df = yf.download(ticker, period="6mo")

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

        st.subheader(f"Predicted Next Opening Price: ${prediction:.2f}")

        st.line_chart(df["Close"])

    else:
        st.warning("Please enter a stock ticker.")