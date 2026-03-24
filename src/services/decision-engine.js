import { ACTIONS } from "../domain/constants.js";

export class DecisionEngine {
  decide({ snapshot, indicators, signal, riskProfile }) {
    const price = snapshot.quote.price;
    const confidence = calculateConfidence({ signal, indicators, snapshot, riskProfile });
    const stopLoss = Number((price * (1 - riskProfile.stopLossPct)).toFixed(2));
    const target = Number((price * (1 + riskProfile.takeProfitPct)).toFixed(2));
    const riskPerUnit = Math.max(price - stopLoss, 0.01);
    const budget = riskProfile.maxAutoTradeValue;
    const quantity = Math.max(Math.floor((budget * 0.01) / riskPerUnit), 1);

    return {
      action: signal.action,
      confidence,
      timeframe: signal.timeframe,
      entryPrice: price,
      stopLoss,
      targetPrice:
        signal.action === ACTIONS.SELL
          ? Number((price * (1 - riskProfile.takeProfitPct)).toFixed(2))
          : target,
      suggestedQuantity: quantity,
      setup: signal.setup,
      indicators
    };
  }
}

function calculateConfidence({ signal, indicators, snapshot, riskProfile }) {
  let score = 0.5;
  score += signal.bullishScore * 0.08;
  score -= signal.bearishScore * 0.04;
  score += indicators.trend === "bullish" ? 0.08 : indicators.trend === "bearish" ? -0.06 : 0;
  score += snapshot.status.providerHealthy ? 0.04 : -0.15;
  score -= indicators.volatility > 0.03 ? 0.1 : 0;
  score += riskProfile.confidenceThreshold > 0.66 ? 0.02 : 0;
  return Number(Math.max(0.1, Math.min(score, 0.95)).toFixed(2));
}
