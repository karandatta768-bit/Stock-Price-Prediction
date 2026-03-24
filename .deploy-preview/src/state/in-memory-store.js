import { ORDER_STATUS } from "../domain/constants.js";

export class InMemoryStateStore {
  constructor(seed = {}) {
    this.users = seed.users ?? [];
    this.riskProfiles = seed.riskProfiles ?? [];
    this.watchlists = seed.watchlists ?? [];
    this.brokerConnections = seed.brokerConnections ?? [];
    this.orders = seed.orders ?? [];
    this.recommendations = seed.recommendations ?? [];
    this.explanations = seed.explanations ?? [];
    this.logs = seed.logs ?? [];
    this.positions = seed.positions ?? [
      { userId: "demo-user", market: "US", symbol: "AAPL", quantity: 10, averagePrice: 182.4 },
      { userId: "demo-user", market: "INDIA", symbol: "RELIANCE", quantity: 5, averagePrice: 2840 }
    ];
  }

  getUser(userId) {
    return this.users.find((user) => user.userId === userId);
  }

  getRiskProfile(market) {
    return this.riskProfiles.find((profile) => profile.market === market);
  }

  getBrokerConnection(userId, market) {
    return this.brokerConnections.find(
      (connection) => connection.userId === userId && connection.market === market
    );
  }

  getPositions(userId) {
    return this.positions.filter((position) => position.userId === userId);
  }

  getSymbolPositionValue(userId, market, symbol, price) {
    return this.positions
      .filter((position) => position.userId === userId && position.market === market && position.symbol === symbol)
      .reduce((sum, position) => sum + position.quantity * price, 0);
  }

  addWatchSymbol(userId, symbol, market) {
    const exists = this.watchlists.some(
      (item) => item.userId === userId && item.symbol === symbol && item.market === market
    );
    if (!exists) {
      this.watchlists.push({ userId, symbol, market, createdAt: new Date().toISOString() });
    }
  }

  removeWatchSymbol(userId, symbol) {
    this.watchlists = this.watchlists.filter(
      (item) => !(item.userId === userId && item.symbol === symbol)
    );
  }

  getWatchlist(userId) {
    return this.watchlists.filter((item) => item.userId === userId);
  }

  saveRecommendation(recommendation) {
    this.recommendations.push(recommendation);
  }

  getRecommendation(idOrSymbol, userId) {
    return [...this.recommendations]
      .reverse()
      .find(
        (item) =>
          item.userId === userId &&
          (item.id === idOrSymbol || item.symbol.toUpperCase() === idOrSymbol.toUpperCase())
      );
  }

  saveExplanation(explanation) {
    this.explanations.push(explanation);
  }

  getExplanationForRecommendation(recommendationId) {
    return [...this.explanations]
      .reverse()
      .find((item) => item.recommendationId === recommendationId);
  }

  createOrder(order) {
    this.orders.push(order);
    return order;
  }

  updateOrder(orderId, patch) {
    const order = this.orders.find((item) => item.id === orderId);
    if (!order) {
      return null;
    }
    Object.assign(order, patch);
    return order;
  }

  getOrder(orderId, userId) {
    return this.orders.find((order) => order.id === orderId && order.userId === userId);
  }

  getOrders(userId) {
    return this.orders.filter((order) => order.userId === userId);
  }

  getPendingApprovalOrders(userId) {
    return this.orders.filter(
      (order) => order.userId === userId && order.status === ORDER_STATUS.PENDING_APPROVAL
    );
  }

  appendLog(event) {
    this.logs.push(event);
  }
}
