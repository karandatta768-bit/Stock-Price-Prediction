import { MARKETS } from "./constants.js";

const INDIA_SYMBOLS = new Set(["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"]);

export function inferMarket(symbol) {
  return INDIA_SYMBOLS.has(symbol.toUpperCase()) ? MARKETS.INDIA : MARKETS.US;
}
