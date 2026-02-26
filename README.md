# 📈 Next-Day Stock Opening Price Prediction

An end-to-end time-series forecasting project that predicts the next trading day's opening stock price using historical market data from Yahoo Finance. The model is deployed as an interactive web application using Streamlit.

---

## 🎯 Project Objective

The goal of this project is to forecast the next day's opening price of a stock using historical price and volume data.  

The solution follows proper time-series modeling principles to prevent data leakage and ensure realistic evaluation.

---

## 📊 Data Source

Historical stock market data retrieved using the **yfinance API**, including:

- Open
- High
- Low
- Close
- Volume

---

## 🛠 Feature Engineering

To capture market behavior, the following features were engineered:

- Lag features (previous closing prices)
- Moving averages (5-day and 10-day)
- Rolling volatility
- Daily returns

These features help model short-term trends, momentum, and volatility patterns.

---

## ⏳ Time-Series Modeling Strategy

Unlike traditional ML models, stock data is time-dependent.

To avoid data leakage:

- Data was sorted chronologically
- No random shuffling was performed
- Train-test split was time-based (80% train / 20% test)
- Target variable was shifted using:

```python
df["Target"] = df["Open"].shift(-1)
