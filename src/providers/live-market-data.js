import { MARKETS } from "../domain/constants.js";

const INDIA_SUFFIX = ".NS";

function mapSymbol(symbol, market) {
  return market === MARKETS.INDIA ? `${symbol.toUpperCase()}${INDIA_SUFFIX}` : symbol.toUpperCase();
}

function buildQuote(meta, result) {
  const price = Number(meta.regularMarketPrice ?? result.indicators.quote[0].close.at(-1) ?? 0);
  const previousClose = Number(meta.chartPreviousClose ?? meta.previousClose ?? price);
  const change = Number((price - previousClose).toFixed(2));
  const changePct = previousClose
    ? Number((((price - previousClose) / previousClose) * 100).toFixed(2))
    : 0;

  return {
    price: Number(price.toFixed(2)),
    open: Number((meta.regularMarketOpen ?? result.indicators.quote[0].open.at(-1) ?? price).toFixed(2)),
    previousClose: Number(previousClose.toFixed(2)),
    change,
    changePct,
    timestamp: new Date((result.timestamp.at(-1) ?? Math.floor(Date.now() / 1000)) * 1000).toISOString()
  };
}

function buildCandles(result) {
  const quote = result.indicators?.quote?.[0];
  const timestamps = result.timestamp ?? [];
  if (!quote || timestamps.length === 0) {
    return [];
  }

  return timestamps.map((timestamp, index) => ({
    timestamp: new Date(timestamp * 1000).toISOString(),
    open: Number((quote.open?.[index] ?? quote.close?.[index] ?? 0).toFixed(2)),
    high: Number((quote.high?.[index] ?? quote.close?.[index] ?? 0).toFixed(2)),
    low: Number((quote.low?.[index] ?? quote.close?.[index] ?? 0).toFixed(2)),
    close: Number((quote.close?.[index] ?? 0).toFixed(2)),
    volume: Number(quote.volume?.[index] ?? 0)
  })).filter((candle) => candle.close > 0);
}

function inferMarketOpen(meta) {
  const state = String(meta.marketState ?? "").toUpperCase();
  return state === "REGULAR" || state === "PRE" || state === "POST";
}

export class LiveMarketDataGateway {
  async getSnapshot({ symbol, market }) {
    const providerSymbol = mapSymbol(symbol, market);
    const url = new URL(`https://query1.finance.yahoo.com/v8/finance/chart/${providerSymbol}`);
    url.searchParams.set("interval", "1d");
    url.searchParams.set("range", "3mo");
    url.searchParams.set("includePrePost", "true");

    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 Codex Trading Agent"
      }
    });

    if (!response.ok) {
      throw new Error(`Live market data request failed with status ${response.status}`);
    }

    const payload = await response.json();
    const result = payload?.chart?.result?.[0];
    const error = payload?.chart?.error;
    if (!result || error) {
      throw new Error(error?.description ?? `No live market data returned for ${providerSymbol}`);
    }

    const candles = buildCandles(result);
    if (candles.length < 2) {
      throw new Error(`Not enough candle data returned for ${providerSymbol}`);
    }

    const meta = result.meta ?? {};
    return {
      symbol,
      market,
      provider: `yahoo-finance:${providerSymbol}`,
      sourceMode: "live",
      quote: buildQuote(meta, result),
      candles: candles.slice(-60),
      status: {
        stale: false,
        marketOpen: inferMarketOpen(meta),
        providerHealthy: true
      }
    };
  }
}

