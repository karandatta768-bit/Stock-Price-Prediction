export class WatchlistMonitor {
  constructor({ stateStore, orchestrator }) {
    this.stateStore = stateStore;
    this.orchestrator = orchestrator;
  }

  addSymbol({ userId, symbol, market }) {
    this.stateStore.addWatchSymbol(userId, symbol, market);
    return this.stateStore.getWatchlist(userId);
  }

  removeSymbol({ userId, symbol }) {
    this.stateStore.removeWatchSymbol(userId, symbol);
    return this.stateStore.getWatchlist(userId);
  }

  async scan(userId) {
    const watchlist = this.stateStore.getWatchlist(userId);
    const alerts = [];
    for (const item of watchlist) {
      const result = await this.orchestrator.analyzeSymbol({ userId, symbol: item.symbol });
      if (["buy", "sell"].includes(result.recommendation.decision.action)) {
        alerts.push(result);
      }
    }
    return alerts;
  }
}
