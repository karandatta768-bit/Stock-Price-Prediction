import { MARKETS } from "./constants.js";

export function createRiskProfiles() {
  return [
    {
      market: MARKETS.US,
      maxPositionValue: 15000,
      maxAutoTradeValue: 4000,
      maxDailyLoss: 1200,
      stopLossPct: 0.025,
      takeProfitPct: 0.05,
      maxSymbolExposurePct: 0.18,
      confidenceThreshold: 0.64,
      requireApprovalAboveValue: 4000
    },
    {
      market: MARKETS.INDIA,
      maxPositionValue: 600000,
      maxAutoTradeValue: 120000,
      maxDailyLoss: 30000,
      stopLossPct: 0.02,
      takeProfitPct: 0.045,
      maxSymbolExposurePct: 0.15,
      confidenceThreshold: 0.68,
      requireApprovalAboveValue: 100000
    }
  ];
}
