import { ACTIONS } from "../domain/constants.js";

export class SignalEngine {
  evaluate({ snapshot, indicators }) {
    const bullishScore = [
      indicators.trend === "bullish" ? 1 : 0,
      indicators.momentum > 0 ? 1 : 0,
      indicators.volumeRatio >= 1 ? 1 : 0,
      snapshot.quote.changePct > 0 ? 1 : 0
    ].reduce((sum, value) => sum + value, 0);

    const bearishScore = [
      indicators.trend === "bearish" ? 1 : 0,
      indicators.momentum < 0 ? 1 : 0,
      snapshot.quote.changePct < 0 ? 1 : 0,
      indicators.volumeRatio >= 1.1 ? 1 : 0
    ].reduce((sum, value) => sum + value, 0);

    let action = ACTIONS.HOLD;
    if (bullishScore >= 3 && indicators.volatility < 0.03) {
      action = ACTIONS.BUY;
    } else if (bearishScore >= 3) {
      action = ACTIONS.SELL;
    } else if (indicators.volatility > 0.035) {
      action = ACTIONS.WAIT;
    }

    return {
      action,
      bullishScore,
      bearishScore,
      setup: inferSetup(action, indicators),
      timeframe: indicators.volatility < 0.02 ? "swing" : "intraday"
    };
  }
}

function inferSetup(action, indicators) {
  if (action === ACTIONS.BUY) {
    return indicators.trend === "bullish" ? "trend continuation breakout" : "recovery setup";
  }
  if (action === ACTIONS.SELL) {
    return "momentum breakdown";
  }
  if (action === ACTIONS.WAIT) {
    return "high-volatility stand aside";
  }
  return "range-bound hold";
}
