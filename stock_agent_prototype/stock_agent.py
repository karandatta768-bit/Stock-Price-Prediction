from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

try:
    import ta
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: ta. Install with `pip install ta`."
    ) from exc

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_gap_5_20",
    "volatility_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_high",
    "bb_low",
    "bb_width",
    "volume_change",
]


def configure_local_cache() -> None:
    cache_dir = Path.cwd() / ".yfinance-cache"
    cache_dir.mkdir(exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


@dataclass
class BacktestSummary:
    market_total_return: float
    strategy_total_return: float
    strategy_annualized_return: float
    strategy_annualized_volatility: float
    strategy_sharpe_like: float
    max_drawdown: float
    trades: int


@dataclass
class WalkForwardSummary:
    mean_accuracy: float
    std_accuracy: float
    folds: int


@dataclass
class DiagnosticSummary:
    train_accuracy: float
    test_accuracy: float
    majority_baseline_accuracy: float
    train_positive_rate: float
    test_positive_rate: float
    generalization_gap: float
    diagnosis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone stock signal prototype")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, for example AAPL")
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path with Date, Open, High, Low, Close, Volume columns",
    )
    parser.add_argument("--start", default="2016-01-01", help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Confidence threshold for BUY or SELL signals",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost in basis points per position change",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the latest chronological test segment",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model and metadata in the current folder",
    )
    parser.add_argument(
        "--export-signals",
        action="store_true",
        help="Export the test-period predictions and signals to CSV",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of walk-forward validation splits",
    )
    return parser.parse_args()


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker {ticker!r}. "
            "This can happen if the symbol is invalid or if Yahoo Finance "
            "is unreachable from the current environment."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    return df


def load_csv_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    date_column = next((col for col in df.columns if col.lower() == "date"), None)
    if date_column is None:
        raise ValueError("CSV must contain a Date column.")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()
    return normalize_ohlcv_columns(df)


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    expected = {"open", "high", "low", "close", "volume"}
    for column in df.columns:
        lowered = str(column).strip().lower().replace(" ", "")
        if lowered in expected:
            rename_map[column] = lowered.capitalize()

    normalized = df.rename(columns=rename_map).copy()
    missing = [name for name in ["Open", "High", "Low", "Close", "Volume"] if name not in normalized.columns]
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing)
        )

    return normalized[["Open", "High", "Low", "Close", "Volume"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    data["return_1d"] = close.pct_change(1)
    data["return_5d"] = close.pct_change(5)

    data["ma_5"] = close.rolling(5).mean() / close - 1
    data["ma_10"] = close.rolling(10).mean() / close - 1
    data["ma_20"] = close.rolling(20).mean() / close - 1
    data["ma_gap_5_20"] = data["ma_5"] - data["ma_20"]

    data["volatility_10"] = data["return_1d"].rolling(10).std()
    data["volume_change"] = volume.pct_change()

    data["rsi_14"] = ta.momentum.RSIIndicator(close).rsi() / 100.0

    macd = ta.trend.MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(close)
    data["bb_high"] = bollinger.bollinger_hband() / close - 1
    data["bb_low"] = bollinger.bollinger_lband() / close - 1
    data["bb_width"] = (
        bollinger.bollinger_hband() - bollinger.bollinger_lband()
    ) / close

    data["future_return_1d"] = close.shift(-1) / close - 1
    data["target"] = (data["future_return_1d"] > 0).astype(int)

    data = data.dropna().copy()
    return data


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    splits: int,
) -> WalkForwardSummary:
    if len(X) < 6:
        raise ValueError("Not enough rows for walk-forward validation. Add more historical data.")

    effective_splits = max(2, min(splits, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=effective_splits)

    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        scores.append(accuracy_score(y_test, predictions))

    return WalkForwardSummary(
        mean_accuracy=float(np.mean(scores)),
        std_accuracy=float(np.std(scores)),
        folds=len(scores),
    )


def chronological_split(
    data: pd.DataFrame, feature_columns: Iterable[str], test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    X = data[list(feature_columns)]
    y = data["target"]

    split_idx = int(len(data) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(data) - 1))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    test_frame = data.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, test_frame


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ClassifierMixin:
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def summarize_diagnostics(
    y_train: pd.Series,
    train_pred: np.ndarray,
    y_test: pd.Series,
    test_pred: np.ndarray,
) -> DiagnosticSummary:
    train_accuracy = float(accuracy_score(y_train, train_pred))
    test_accuracy = float(accuracy_score(y_test, test_pred))
    majority_class = int(y_train.mode().iloc[0])
    majority_baseline_accuracy = float((y_test == majority_class).mean())
    train_positive_rate = float(y_train.mean())
    test_positive_rate = float(y_test.mean())
    generalization_gap = float(train_accuracy - test_accuracy)

    if generalization_gap >= 0.18:
        diagnosis = "high variance risk: training accuracy is materially higher than test accuracy"
    elif test_accuracy <= majority_baseline_accuracy + 0.02:
        diagnosis = "weak edge / possible underfitting: test accuracy is close to the majority-class baseline"
    elif abs(train_positive_rate - test_positive_rate) >= 0.12:
        diagnosis = "distribution shift risk: class balance changes noticeably between train and test periods"
    else:
        diagnosis = "no strong overfitting signal on this split, but results remain sample-limited"

    return DiagnosticSummary(
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        majority_baseline_accuracy=majority_baseline_accuracy,
        train_positive_rate=train_positive_rate,
        test_positive_rate=test_positive_rate,
        generalization_gap=generalization_gap,
        diagnosis=diagnosis,
    )


def probability_to_signal(prob_up: float, threshold: float) -> str:
    if prob_up >= threshold:
        return "BUY"
    if prob_up <= 1 - threshold:
        return "SELL"
    return "HOLD"


def backtest_predictions(
    test_frame: pd.DataFrame,
    prob_up: np.ndarray,
    threshold: float,
    cost_bps: float,
) -> BacktestSummary:
    frame = test_frame.copy()
    frame["prob_up"] = prob_up
    frame["signal"] = frame["prob_up"].apply(lambda value: probability_to_signal(value, threshold))

    # Long/cash behavior for the prototype: BUY=1, SELL/HOLD=0.
    frame["position"] = (frame["signal"] == "BUY").astype(int)
    frame["position_prev"] = frame["position"].shift(1).fillna(0)
    frame["trade_flag"] = (frame["position"] != frame["position_prev"]).astype(int)

    cost_rate = cost_bps / 10000.0
    frame["strategy_return"] = (
        frame["position_prev"] * frame["future_return_1d"] - frame["trade_flag"] * cost_rate
    )
    frame["market_return"] = frame["future_return_1d"]

    frame["strategy_curve"] = (1 + frame["strategy_return"]).cumprod()
    frame["market_curve"] = (1 + frame["market_return"]).cumprod()

    periods_per_year = 252
    strategy_mean = frame["strategy_return"].mean()
    strategy_std = frame["strategy_return"].std()
    annualized_return = (1 + strategy_mean) ** periods_per_year - 1 if pd.notna(strategy_mean) else 0.0
    annualized_vol = strategy_std * np.sqrt(periods_per_year) if pd.notna(strategy_std) else 0.0
    sharpe_like = annualized_return / annualized_vol if annualized_vol else 0.0

    running_max = frame["strategy_curve"].cummax()
    drawdown = frame["strategy_curve"] / running_max - 1

    return BacktestSummary(
        market_total_return=float(frame["market_curve"].iloc[-1] - 1),
        strategy_total_return=float(frame["strategy_curve"].iloc[-1] - 1),
        strategy_annualized_return=float(annualized_return),
        strategy_annualized_volatility=float(annualized_vol),
        strategy_sharpe_like=float(sharpe_like),
        max_drawdown=float(drawdown.min()),
        trades=int(frame["trade_flag"].sum()),
    )


def explain_latest_signal(
    X_test: pd.DataFrame,
    model: ClassifierMixin,
    threshold: float,
) -> dict[str, object]:
    latest_row = X_test.iloc[-1]
    latest_prob = model.predict_proba(X_test.iloc[-1:])[0, 1]
    signal = probability_to_signal(float(latest_prob), threshold)

    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    top_features = importances.head(5).index.tolist()
    feature_snapshot = {
        feature: round(float(latest_row[feature]), 6) for feature in top_features
    }

    return {
        "signal": signal,
        "prob_up": round(float(latest_prob), 4),
        "confidence": round(float(max(latest_prob, 1 - latest_prob)), 4),
        "reason": build_signal_reason(signal, float(latest_prob), threshold),
        "top_features": feature_snapshot,
    }


def build_signal_reason(signal: str, prob_up: float, threshold: float) -> str:
    if signal == "BUY":
        return (
            f"The model sees bullish next-day odds above the confidence threshold "
            f"({prob_up:.2%} vs threshold {threshold:.2%})."
        )
    if signal == "SELL":
        return (
            f"The model sees bearish next-day odds above the confidence threshold "
            f"({1 - prob_up:.2%} vs threshold {threshold:.2%})."
        )
    return (
        f"The model is not confident enough to take a directional stance "
        f"({prob_up:.2%} probability of an up move)."
    )


def maybe_save_artifacts(
    ticker: str,
    model: ClassifierMixin,
    threshold: float,
    backtest: BacktestSummary,
    walk_forward: WalkForwardSummary,
    diagnostics: DiagnosticSummary,
) -> None:
    if joblib is None:
        print("Model save skipped because joblib is not installed.")
        return

    base = Path.cwd()
    model_path = base / f"{ticker.lower()}_stock_agent_model.joblib"
    metadata_path = base / f"{ticker.lower()}_stock_agent_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "ticker": ticker,
        "model": "gradient_boosting",
        "features": FEATURE_COLUMNS,
        "threshold": threshold,
        "backtest_summary": asdict(backtest),
        "walk_forward_summary": asdict(walk_forward),
        "diagnostics": asdict(diagnostics),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def export_signal_frame(
    ticker: str,
    test_frame: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    prob_up: np.ndarray,
    threshold: float,
) -> Path:
    export_frame = test_frame.copy()
    export_frame["target"] = y_test.values
    export_frame["prediction"] = y_pred
    export_frame["prob_up"] = prob_up
    export_frame["signal"] = [
        probability_to_signal(float(value), threshold) for value in prob_up
    ]

    path = Path.cwd() / f"{ticker.lower()}_test_signals.csv"
    export_frame.to_csv(path, index=True)
    return path


def analyze_stock(
    ticker: str = "AAPL",
    csv_path: str | None = None,
    start: str = "2016-01-01",
    end: str = "2025-01-01",
    threshold: float = 0.55,
    cost_bps: float = 5.0,
    test_size: float = 0.2,
    cv_splits: int = 5,
    save_model: bool = False,
    export_signals: bool = False,
) -> dict[str, object]:
    configure_local_cache()

    export_name = Path(csv_path).stem if csv_path else ticker.lower()
    source_label = csv_path if csv_path else ticker

    if csv_path:
        raw_data = load_csv_data(csv_path)
    else:
        raw_data = fetch_data(ticker, start, end)

    data = build_features(raw_data)
    walk_forward = walk_forward_validate(
        data[FEATURE_COLUMNS], data["target"], cv_splits
    )
    X_train, X_test, y_train, y_test, test_frame = chronological_split(
        data, FEATURE_COLUMNS, test_size
    )

    model = train_model(X_train, y_train)
    train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    prob_up = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    latest = explain_latest_signal(X_test, model, threshold)
    backtest = backtest_predictions(test_frame, prob_up, threshold, cost_bps)
    diagnostics = summarize_diagnostics(y_train, train_pred, y_test, y_pred)

    export_path = None
    if export_signals:
        export_path = export_signal_frame(
            export_name, test_frame, y_test, y_pred, prob_up, threshold
        )

    if save_model:
        maybe_save_artifacts(export_name, model, threshold, backtest, walk_forward, diagnostics)

    prediction_frame = test_frame[["future_return_1d"]].copy()
    prediction_frame["target"] = y_test.values
    prediction_frame["prediction"] = y_pred
    prediction_frame["prob_up"] = prob_up
    prediction_frame["signal"] = [
        probability_to_signal(float(value), threshold) for value in prob_up
    ]

    return {
        "source": source_label,
        "model": "gradient_boosting",
        "rows": len(data),
        "train_start": X_train.index.min().date().isoformat(),
        "train_end": X_train.index.max().date().isoformat(),
        "test_start": X_test.index.min().date().isoformat(),
        "test_end": X_test.index.max().date().isoformat(),
        "accuracy": float(accuracy),
        "classification_report_text": classification_report(
            y_test, y_pred, digits=4
        ),
        "classification_report": classification_report(
            y_test, y_pred, digits=4, output_dict=True
        ),
        "walk_forward": asdict(walk_forward),
        "diagnostics": asdict(diagnostics),
        "backtest": asdict(backtest),
        "latest": latest,
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": {
            name: float(value)
            for name, value in zip(FEATURE_COLUMNS, model.feature_importances_, strict=False)
        },
        "recent_signals": (
            prediction_frame.tail(12).reset_index().rename(columns={"index": "date"}).to_dict("records")
        ),
        "export_path": str(export_path) if export_path else None,
    }


def main() -> None:
    args = parse_args()
    results = analyze_stock(
        ticker=args.ticker,
        csv_path=args.csv,
        start=args.start,
        end=args.end,
        threshold=args.threshold,
        cost_bps=args.cost_bps,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        save_model=args.save_model,
        export_signals=args.export_signals,
    )

    print_section("Dataset")
    print(f"Source: {results['source']}")
    print(f"Model:  {results['model']}")
    print(f"Rows after feature engineering: {results['rows']}")
    print(f"Train range: {results['train_start']} to {results['train_end']}")
    print(f"Test range:  {results['test_start']} to {results['test_end']}")

    print_section("Walk-Forward Validation")
    print(f"Folds:         {results['walk_forward']['folds']}")
    print(f"Mean accuracy: {results['walk_forward']['mean_accuracy']:.4f}")
    print(f"Std accuracy:  {results['walk_forward']['std_accuracy']:.4f}")

    print_section("Model Evaluation")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(results["classification_report_text"])

    print_section("Diagnostics")
    print(f"Train accuracy:        {results['diagnostics']['train_accuracy']:.4f}")
    print(f"Test accuracy:         {results['diagnostics']['test_accuracy']:.4f}")
    print(f"Majority baseline:     {results['diagnostics']['majority_baseline_accuracy']:.4f}")
    print(f"Train positive rate:   {results['diagnostics']['train_positive_rate']:.4f}")
    print(f"Test positive rate:    {results['diagnostics']['test_positive_rate']:.4f}")
    print(f"Generalization gap:    {results['diagnostics']['generalization_gap']:.4f}")
    print(f"Diagnosis:             {results['diagnostics']['diagnosis']}")

    print_section("Backtest Summary")
    print(f"Market total return:   {results['backtest']['market_total_return']:.2%}")
    print(f"Strategy total return: {results['backtest']['strategy_total_return']:.2%}")
    print(f"Annualized return:     {results['backtest']['strategy_annualized_return']:.2%}")
    print(f"Annualized volatility: {results['backtest']['strategy_annualized_volatility']:.2%}")
    print(f"Sharpe-like ratio:     {results['backtest']['strategy_sharpe_like']:.3f}")
    print(f"Max drawdown:          {results['backtest']['max_drawdown']:.2%}")
    print(f"Trades:                {results['backtest']['trades']}")

    print_section("Latest Signal")
    print(f"Signal:     {results['latest']['signal']}")
    print(f"Prob up:    {results['latest']['prob_up']:.4f}")
    print(f"Confidence: {results['latest']['confidence']:.4f}")
    print(f"Reason:     {results['latest']['reason']}")
    print("Top feature snapshot:")
    for feature, value in results["latest"]["top_features"].items():
        print(f"  - {feature}: {value}")

    if results["export_path"]:
        print(f"\nExported test signals to: {results['export_path']}")


if __name__ == "__main__":
    main()
