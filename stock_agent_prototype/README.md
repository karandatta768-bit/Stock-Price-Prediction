# Stock Agent Prototype

This prototype is separate from your main project and uses your earlier notebook as a reference.

It improves the original approach by:

- keeping the time order intact
- using lag-safe features for next-day prediction
- supporting either Yahoo Finance data or your own CSV file
- generating `BUY`, `SELL`, or `HOLD` signals
- applying a confidence threshold
- adding walk-forward validation
- backtesting with position shifting and transaction costs
- printing a simple explanation for the latest signal

## What It Does

1. Downloads historical price data from Yahoo Finance with `yfinance`
2. Builds technical indicators:
   - moving averages
   - volatility
   - RSI
   - MACD
   - Bollinger Bands
3. Trains a tuned `GradientBoostingClassifier`
4. Evaluates predictions on the latest test segment
5. Simulates a long/cash strategy
6. Returns the latest signal with confidence and feature context

## Install

```bash
pip install yfinance ta scikit-learn pandas numpy
```

## Run

```bash
python stock_agent.py --ticker AAPL --start 2016-01-01 --end 2025-01-01
```

Optional arguments:

```bash
python stock_agent.py --ticker MSFT --threshold 0.60 --cost-bps 10 --save-model
```

Offline or custom data:

```bash
python stock_agent.py --csv "C:\path\to\prices.csv" --export-signals
```

Your CSV should contain:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

## Notes

- `BUY` means the model has enough confidence to be long for the next step.
- `SELL` means bearish confidence is high enough.
- `HOLD` means the model is not confident enough either way.
- The backtest currently uses a simple long/cash setup. It does not place real trades.

## Output

The script prints:

- train/test date ranges
- walk-forward validation summary
- classification metrics
- backtest summary
- latest model signal and confidence
- the feature values that most likely influenced that signal

## Phone Access

You can also run the mobile-friendly web app:

```bash
python app.py
```

Then open the shown address on your phone browser, for example:

```text
http://192.168.1.5:5000
```

Phone access works when:

- your computer and phone are on the same Wi-Fi
- Python is allowed through Windows Firewall
- you keep the app running on your computer
