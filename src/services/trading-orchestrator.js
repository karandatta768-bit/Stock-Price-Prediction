import { inferMarket } from "../domain/market-map.js";
import { ACTIONS, ORDER_STATUS } from "../domain/constants.js";
import { createId } from "../utils/id.js";

export class TradingOrchestrator {
  constructor(dependencies) {
    Object.assign(this, dependencies);
  }

  async analyzeSymbol({ userId, symbol }) {
    const normalizedSymbol = symbol.toUpperCase();
    const market = inferMarket(normalizedSymbol);
    const riskProfile = this.stateStore.getRiskProfile(market);
    const snapshot = await this.marketData.getSnapshot({ symbol: normalizedSymbol, market });
    const indicators = this.indicatorEngine.compute(snapshot);
    const signal = this.signalEngine.evaluate({ snapshot, indicators });
    const decision = this.decisionEngine.decide({ snapshot, indicators, signal, riskProfile });
    const backtestSummary = this.backtestService.summarize(snapshot);
    const riskAssessment = this.riskManager.assess({
      userId,
      market,
      symbol: normalizedSymbol,
      snapshot,
      decision
    });

    const recommendation = {
      id: createId("rec"),
      userId,
      symbol: normalizedSymbol,
      market,
      createdAt: new Date().toISOString(),
      snapshot,
      signal,
      decision,
      backtestSummary,
      riskAssessment
    };
    this.stateStore.saveRecommendation(recommendation);

    const explanationText = this.explanationService.explain({
      snapshot,
      signal,
      decision,
      backtestSummary,
      riskAssessment
    });
    this.stateStore.saveExplanation({
      recommendationId: recommendation.id,
      text: explanationText,
      createdAt: new Date().toISOString()
    });

    const isExecutableAction = decision.action === ACTIONS.BUY || decision.action === ACTIONS.SELL;

    let order = null;
    if (isExecutableAction && riskAssessment.allowed && riskAssessment.executionMode === "auto") {
      order = await this.brokerGateway.placeOrder({
        userId,
        market,
        symbol: normalizedSymbol,
        action: decision.action,
        quantity: decision.suggestedQuantity,
        price: decision.entryPrice,
        metadata: { recommendationId: recommendation.id }
      });
    } else if (isExecutableAction && riskAssessment.allowed && riskAssessment.executionMode === "approval_required") {
      order = await this.brokerGateway.requestApprovalOrder({
        userId,
        market,
        symbol: normalizedSymbol,
        action: decision.action,
        quantity: decision.suggestedQuantity,
        price: decision.entryPrice,
        metadata: { recommendationId: recommendation.id }
      });
    } else if (isExecutableAction && !riskAssessment.allowed) {
      order = this.stateStore.createOrder({
        id: createId("ord"),
        userId,
        market,
        symbol: normalizedSymbol,
        action: decision.action,
        quantity: decision.suggestedQuantity,
        price: decision.entryPrice,
        status: ORDER_STATUS.BLOCKED,
        metadata: { recommendationId: recommendation.id },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      });
    }

    this.stateStore.appendLog({
      type: "analysis",
      recommendationId: recommendation.id,
      orderId: order?.id,
      symbol: normalizedSymbol,
      market,
      timestamp: new Date().toISOString()
    });

    return { recommendation, explanationText, order, backtestSummary };
  }

  async approveTrade({ userId, orderId }) {
    const order = await this.brokerGateway.approveOrder(orderId, userId);
    if (!order) {
      return null;
    }
    this.stateStore.appendLog({
      type: "approval",
      orderId,
      timestamp: new Date().toISOString()
    });
    return order;
  }

  async rejectTrade({ userId, orderId }) {
    const order = await this.brokerGateway.rejectOrder(orderId, userId);
    if (!order) {
      return null;
    }
    this.stateStore.appendLog({
      type: "rejection",
      orderId,
      timestamp: new Date().toISOString()
    });
    return order;
  }

  async getPortfolio(userId) {
    return this.brokerGateway.getPortfolio(userId);
  }
}
