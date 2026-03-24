export class IndicatorEngine {
  compute(snapshot) {
    const closes = snapshot.candles.map((candle) => candle.close);
    const volumes = snapshot.candles.map((candle) => candle.volume);
    const shortSma = average(closes.slice(-5));
    const mediumSma = average(closes.slice(-20));
    const longSma = average(closes.slice(-50));
    const momentum = Number((closes.at(-1) - closes.at(-6)).toFixed(2));
    const volumeRatio = Number((volumes.at(-1) / average(volumes.slice(-10))).toFixed(2));
    const volatility = Number((standardDeviation(closes.slice(-20)) / mediumSma).toFixed(4));

    return {
      shortSma: round(shortSma),
      mediumSma: round(mediumSma),
      longSma: round(longSma),
      momentum,
      volumeRatio,
      volatility,
      trend:
        shortSma > mediumSma && mediumSma > longSma
          ? "bullish"
          : shortSma < mediumSma && mediumSma < longSma
            ? "bearish"
            : "neutral"
    };
  }
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values) {
  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function round(value) {
  return Number(value.toFixed(2));
}
