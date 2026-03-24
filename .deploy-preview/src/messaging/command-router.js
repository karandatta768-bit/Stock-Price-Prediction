import { inferMarket } from "../domain/market-map.js";

export class CommandRouter {
  constructor({ stateStore, orchestrator, watchlistMonitor }) {
    this.stateStore = stateStore;
    this.orchestrator = orchestrator;
    this.watchlistMonitor = watchlistMonitor;
  }

  async route({ userId, text }) {
    const [command, arg] = text.trim().split(/\s+/, 2);
    switch ((command ?? "").toLowerCase()) {
      case "analyze":
        return this.handleAnalyze(userId, arg);
      case "watch":
        return this.handleWatch(userId, arg);
      case "unwatch":
        return this.handleUnwatch(userId, arg);
      case "portfolio":
        return this.handlePortfolio(userId);
      case "orders":
        return this.handleOrders(userId);
      case "approve":
        return this.handleApprove(userId, arg);
      case "reject":
        return this.handleReject(userId, arg);
      case "risk":
        return this.handleRisk();
      case "why":
        return this.handleWhy(userId, arg);
      default:
        return {
          message:
            "Unknown command. Use analyze, watch, unwatch, portfolio, orders, approve, reject, risk, or why."
        };
    }
  }

  async handleAnalyze(userId, symbol) {
    if (!symbol) {
      return { message: "Usage: analyze <symbol>" };
    }
    const { recommendation, explanationText, order } = await this.orchestrator.analyzeSymbol({
      userId,
      symbol
    });
    return {
      message: formatRecommendation(recommendation, explanationText, order),
      data: { recommendation, explanationText, order }
    };
  }

  handleWatch(userId, symbol) {
    if (!symbol) {
      return { message: "Usage: watch <symbol>" };
    }
    const normalizedSymbol = symbol.toUpperCase();
    const market = inferMarket(normalizedSymbol);
    const watchlist = this.watchlistMonitor.addSymbol({ userId, symbol: normalizedSymbol, market });
    return {
      message: `Watching ${normalizedSymbol} in ${market}. Total watched symbols: ${watchlist.length}.`,
      data: { watchlist }
    };
  }

  handleUnwatch(userId, symbol) {
    if (!symbol) {
      return { message: "Usage: unwatch <symbol>" };
    }
    const watchlist = this.watchlistMonitor.removeSymbol({ userId, symbol: symbol.toUpperCase() });
    return {
      message: `Removed ${symbol.toUpperCase()} from watchlist. Remaining symbols: ${watchlist.length}.`,
      data: { watchlist }
    };
  }

  async handlePortfolio(userId) {
    const portfolio = await this.orchestrator.getPortfolio(userId);
    const lines = [
      `Buying power: ${portfolio.buyingPower}`,
      ...portfolio.positions.map(
        (position) =>
          `${position.market} ${position.symbol}: qty ${position.quantity} @ ${position.averagePrice}`
      )
    ];
    return { message: lines.join("\n"), data: portfolio };
  }

  handleOrders(userId) {
    const orders = this.stateStore.getOrders(userId);
    if (orders.length === 0) {
      return { message: "No orders yet." };
    }
    return {
      message: orders
        .map(
          (order) =>
            `${order.id}: ${order.market} ${order.symbol} ${order.action} ${order.quantity} @ ${order.price} [${order.status}]`
        )
        .join("\n"),
      data: { orders }
    };
  }

  async handleApprove(userId, orderId) {
    if (!orderId) {
      return { message: "Usage: approve <trade_id>" };
    }
    const order = await this.orchestrator.approveTrade({ userId, orderId });
    return {
      message: order
        ? `Approved ${order.id}. Status is now ${order.status}.`
        : `Trade ${orderId} was not found.`,
      data: { order }
    };
  }

  async handleReject(userId, orderId) {
    if (!orderId) {
      return { message: "Usage: reject <trade_id>" };
    }
    const order = await this.orchestrator.rejectTrade({ userId, orderId });
    return {
      message: order
        ? `Rejected ${order.id}. Status is now ${order.status}.`
        : `Trade ${orderId} was not found.`,
      data: { order }
    };
  }

  handleRisk() {
    return {
      message: this.stateStore.riskProfiles
        .map(
          (profile) =>
            `${profile.market}: max position ${profile.maxPositionValue}, auto-trade up to ${profile.maxAutoTradeValue}, confidence threshold ${profile.confidenceThreshold}`
        )
        .join("\n"),
      data: { riskProfiles: this.stateStore.riskProfiles }
    };
  }

  handleWhy(userId, idOrSymbol) {
    if (!idOrSymbol) {
      return { message: "Usage: why <symbol|trade_id>" };
    }
    const recommendation = this.stateStore.getRecommendation(idOrSymbol, userId);
    if (!recommendation) {
      return { message: `No explanation found for ${idOrSymbol}.` };
    }
    const explanation = this.stateStore.getExplanationForRecommendation(recommendation.id);
    return {
      message: explanation?.text ?? `No explanation found for ${idOrSymbol}.`,
      data: { recommendation, explanation }
    };
  }
}

function formatRecommendation(recommendation, explanationText, order) {
  const { symbol, market, decision, riskAssessment, snapshot } = recommendation;
  const lines = [
    `${symbol} (${market})`,
    `Price: ${snapshot.quote.price} | Change: ${snapshot.quote.changePct}%`,
    `Timeframe: ${decision.timeframe} | Trend: ${decision.indicators.trend}`,
    `Action: ${decision.action.toUpperCase()} | Confidence: ${decision.confidence}`,
    `Entry: ${decision.entryPrice} | Stop: ${decision.stopLoss} | Target: ${decision.targetPrice}`,
    `Suggested size: ${decision.suggestedQuantity}`,
    `Execution: ${riskAssessment.executionMode}`,
    `Approval required: ${riskAssessment.executionMode === "approval_required" ? "yes" : "no"}`,
    `Explanation: ${explanationText}`
  ];

  if (order) {
    lines.push(`Trade status: ${order.status} (${order.id})`);
  }

  return lines.join("\n");
}
