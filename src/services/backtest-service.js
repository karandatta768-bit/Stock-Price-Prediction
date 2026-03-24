export class BacktestService {
  summarize(snapshot) {
    const candles = snapshot.candles ?? [];
    if (candles.length < 2) {
      return {
        bars: candles.length,
        totalReturnPct: 0,
        winRatePct: 0,
        maxDrawdownPct: 0,
        signalChanges: 0,
        latestBias: "flat",
        equityCurve: [],
        tradeLog: []
      };
    }

    let equity = 1;
    let peak = 1;
    let wins = 0;
    let losses = 0;
    let signalChanges = 0;
    let previousPosition = 0;
    let latestBias = "flat";
    const equityCurve = [{ index: 0, equity: 1, timestamp: candles[0].timestamp }];
    const tradeLog = [];

    for (let index = 1; index < candles.length; index += 1) {
      const previous = candles[index - 1];
      const current = candles[index];
      const bias = current.close > previous.close ? 1 : current.close < previous.close ? -1 : 0;
      const position = bias > 0 ? 1 : 0;
      if (position !== previousPosition) {
        signalChanges += 1;
        tradeLog.push({
          index,
          timestamp: current.timestamp,
          action: position ? "BUY" : "EXIT",
          price: current.close,
          bias: bias > 0 ? "bullish" : bias < 0 ? "bearish" : "flat"
        });
      }
      previousPosition = position;

      const periodReturn = (current.close - previous.close) / previous.close;
      const strategyReturn = position * periodReturn;
      equity *= 1 + strategyReturn;
      peak = Math.max(peak, equity);

      if (strategyReturn > 0) {
        wins += 1;
      } else if (strategyReturn < 0) {
        losses += 1;
      }

      latestBias = bias > 0 ? "bullish" : bias < 0 ? "bearish" : "flat";
      equityCurve.push({
        index,
        equity: round(equity),
        timestamp: current.timestamp
      });
    }

    const trades = wins + losses;
    const drawdown = peak > 0 ? ((equity - peak) / peak) * 100 : 0;

    return {
      bars: candles.length,
      totalReturnPct: round((equity - 1) * 100),
      winRatePct: round(trades ? (wins / trades) * 100 : 0),
      maxDrawdownPct: round(Math.abs(drawdown)),
      signalChanges,
      latestBias,
      equityCurve,
      tradeLog
    };
  }
}

function round(value) {
  return Number(value.toFixed(2));
}
