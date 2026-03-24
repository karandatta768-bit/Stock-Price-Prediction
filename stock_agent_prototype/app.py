from __future__ import annotations

import socket
from pathlib import Path

from flask import Flask, render_template, request

from stock_agent import analyze_stock


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "sample_prices.csv"

app = Flask(__name__)


def detect_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


@app.route("/", methods=["GET", "POST"])
def index():
    form_data = {
        "mode": "csv",
        "ticker": "AAPL",
        "csv_path": str(DEFAULT_CSV),
        "start": "2020-01-01",
        "end": "2025-01-01",
        "threshold": "0.55",
        "cost_bps": "5",
        "test_size": "0.2",
        "cv_splits": "5",
    }
    result = None
    error = None

    if request.method == "POST":
        form_data.update(request.form.to_dict())
        try:
            result = analyze_stock(
                ticker=form_data["ticker"].strip() or "AAPL",
                csv_path=form_data["csv_path"].strip() if form_data["mode"] == "csv" else None,
                start=form_data["start"],
                end=form_data["end"],
                threshold=float(form_data["threshold"]),
                cost_bps=float(form_data["cost_bps"]),
                test_size=float(form_data["test_size"]),
                cv_splits=int(form_data["cv_splits"]),
                export_signals=True,
            )
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    return render_template(
        "index.html",
        form_data=form_data,
        result=result,
        error=error,
        local_ip=detect_local_ip(),
        default_csv=str(DEFAULT_CSV),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
