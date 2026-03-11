!pip install yfinance
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ticker = "AAPL"

df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

df = df.sort_index()

df.head()
df = df.sort_index()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Close"].rolling(10).std()
!pip install ta

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
    "MA_5",
    "MA_10",
    "MA_20",
    "Volatility",
    "Return",
    "RSI",
    "MACD",
    "MACD_signal",
    "BB_high",
    "BB_low"
]

X = df[features]

y = df["Signal"]
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]
dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))
from xgboost import XGBClassifier
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("XGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, rf_pred))
importance = rf_model.feature_importances_

plt.figure(figsize=(10,6))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.show()
df_test = df.iloc[split:].copy()

df_test["Prediction"] = rf_pred

df_test["Strategy_Return"] = df_test["Return"] * df_test["Prediction"]

df_test["Cumulative_Return"] = (1 + df_test["Strategy_Return"]).cumprod()
plt.figure(figsize=(10,6))

plt.plot(df_test["Cumulative_Return"])

plt.title("Trading Strategy Performance")

plt.show()
latest_data = X.iloc[-1:]

prediction = rf_model.predict(latest_data)

if prediction == 1:
    print("BUY signal")
else:
    print("SELL signal")
plt.figure(figsize=(12,6))

plt.plot(df["Close"])

plt.title("Stock Price")

plt.show()
import pickle
pickle.dump(rf_model, open("stock_model.pkl", "wb"))
from google.colab import files
files.download("stock_open_model.pkl")