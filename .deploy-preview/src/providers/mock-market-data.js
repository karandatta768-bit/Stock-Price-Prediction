import { MARKETS } from "../domain/constants.js";

function seededNumber(seed, min, max) {
  let value = 0;
  for (const char of seed) {
    value += char.charCodeAt(0);
  }
  const normalized = (Math.sin(value) + 1) / 2;
  return min + normalized * (max - min);
}

function buildCandles(symbol, market) {
  const base = market === MARKETS.US
    ? seededNumber(symbol, 120, 250)
    : seededNumber(symbol, 900, 3200);

  const candles = [];
  let previousClose = base;
  for (let index = 0; index < 60; index += 1) {
    const drift = Math.sin(index / 6) * (market === MARKETS.US ? 1.8 : 14);
    const close = Number((previousClose + drift).toFixed(2));
    const open = Number((previousClose + drift / 2).toFixed(2));
    const high = Number((Math.max(open, close) + Math.abs(drift) * 0.7 + 0.5).toFixed(2));
    const low = Number((Math.min(open, close) - Math.abs(drift) * 0.6 - 0.5).toFixed(2));
    const volume = Math.round(seededNumber(`${symbol}-${index}`, 200000, 1500000));
    candles.push({
      timestamp: new Date(Date.now() - (60 - index) * 60_000).toISOString(),
      open,
      high,
      low,
      close,
      volume
    });
    previousClose = close;
  }
  return candles;
}

export class MockMarketDataGateway {
  async getSnapshot({ symbol, market }) {
    const candles = buildCandles(symbol, market);
    const latest = candles.at(-1);
    return {
      symbol,
      market,
      provider: market === MARKETS.US ? "mock-us-feed" : "mock-india-feed",
      sourceMode: "demo",
      quote: {
        price: latest.close,
        open: latest.open,
        previousClose: Number(candles.at(-2).close.toFixed(2)),
        change: Number((latest.close - candles.at(-2).close).toFixed(2)),
        changePct: Number((((latest.close - candles.at(-2).close) / candles.at(-2).close) * 100).toFixed(2)),
        timestamp: latest.timestamp
      },
      candles,
      status: {
        stale: false,
        marketOpen: true,
        providerHealthy: true
      }
    };
  }
}
