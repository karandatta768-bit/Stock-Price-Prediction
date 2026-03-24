import { ACTIONS } from "../domain/constants.js";

export class RiskManager {
  constructor(stateStore) {
    this.stateStore = stateStore;
  }

  assess({ userId, market, symbol, snapshot, decision }) {
    const profile = this.stateStore.getRiskProfile(market);
    const tradeValue = Number((decision.suggestedQuantity * decision.entryPrice).toFixed(2));
    const existingExposure = this.stateStore.getSymbolPositionValue(
      userId,
      market,
      symbol,
      decision.entryPrice
    );
    const maxExposure = Number((profile.maxPositionValue * profile.maxSymbolExposurePct).toFixed(2));
    const brokerConnection = this.stateStore.getBrokerConnection(userId, market);

    const reasons = [];
    let executionMode = "auto";
    let allowed = true;

    if (snapshot.status.stale) {
      reasons.push("Live data is stale.");
      allowed = false;
    }
    if (!snapshot.status.marketOpen) {
      reasons.push("Market is closed.");
      executionMode = "informational";
    }
    if (!brokerConnection || brokerConnection.status !== "connected") {
      reasons.push("Broker connection unavailable.");
      executionMode = "informational";
    }
    if (decision.action === ACTIONS.HOLD || decision.action === ACTIONS.WAIT) {
      reasons.push("Current recommendation is informational only.");
      return {
        allowed,
        executionMode: "informational",
        reasons,
        tradeValue
      };
    }
    if (!allowed) {
      return {
        allowed,
        executionMode,
        reasons,
        tradeValue
      };
    }
    if (tradeValue > profile.requireApprovalAboveValue) {
      reasons.push("Trade value exceeds auto-trade threshold.");
      executionMode = "approval_required";
    }
    if (decision.confidence < profile.confidenceThreshold) {
      reasons.push("Confidence is below market threshold.");
      executionMode = "approval_required";
    }
    if (existingExposure + tradeValue > maxExposure) {
      reasons.push("Symbol exposure would exceed market limit.");
      executionMode = "approval_required";
    }
    if (tradeValue > profile.maxPositionValue) {
      reasons.push("Suggested trade exceeds maximum position value.");
      allowed = false;
      executionMode = "blocked";
    }

    return {
      allowed,
      executionMode,
      reasons,
      tradeValue
    };
  }
}
