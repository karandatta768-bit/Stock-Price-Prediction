export class TradingAgentApp {
  constructor({ commandRouter, stateStore, orchestrator, watchlistMonitor, config }) {
    this.commandRouter = commandRouter;
    this.stateStore = stateStore;
    this.orchestrator = orchestrator;
    this.watchlistMonitor = watchlistMonitor;
    this.config = config;
  }

  async handleMessage(message) {
    return this.commandRouter.route(message);
  }
}
