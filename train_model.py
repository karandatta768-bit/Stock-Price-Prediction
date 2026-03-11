import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Downloading data...")
ticker = "AAPL"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

df = df.sort_index()

print("Calculating features...")
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Close"].rolling(10).std()
df["Return"] = df["Close"].pct_change()
df["RSI"] = ta.momentum.RSIIndicator(df["Close"].squeeze()).rsi()
macd = ta.trend.MACD(df["Close"].squeeze())

df["MACD"] = macd.macd()
df["MACD_signal"] = macd.macd_signal()
bb = ta.volatility.BollingerBands(df["Close"].squeeze())

df["BB_high"] = bb.bollinger_hband()
df["BB_low"] = bb.bollinger_lband()

current_close_series = df["Close"].squeeze()
df["Future_Close"] = current_close_series.shift(-1)
df["Signal"] = np.where(df["Future_Close"] > current_close_series, 1, 0)
df = df.dropna()

features = [
    "MA_5", "MA_10", "MA_20", "Volatility", "Return", 
    "RSI", "MACD", "MACD_signal", "BB_high", "BB_low"
]

X = df[features]
y = df["Signal"]
split = int(len(df) * 0.8)

X_train = X[:split]
y_train = y[:split]

print("Training RandomForest model...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Exporting model to stock_model.pkl...")
with open("stock_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Done! stock_model.pkl is ready.")
