import { TradingAgentApp } from "./messaging/app.js";
import { CommandRouter } from "./messaging/command-router.js";
import { loadRuntimeConfig } from "./config/runtime-config.js";
import { createRiskProfiles } from "./domain/risk-profile.js";
import { InMemoryStateStore } from "./state/in-memory-store.js";
import { FileBackedStateStore } from "./state/file-backed-store.js";
import { MockMarketDataGateway } from "./providers/mock-market-data.js";
import { LiveMarketDataGateway } from "./providers/live-market-data.js";
import { ResilientMarketDataGateway } from "./providers/resilient-market-data.js";
import { MockBrokerGateway } from "./providers/mock-broker.js";
import { IndicatorEngine } from "./services/indicator-engine.js";
import { SignalEngine } from "./services/signal-engine.js";
import { DecisionEngine } from "./services/decision-engine.js";
import { RiskManager } from "./services/risk-manager.js";
import { ExplanationService } from "./services/explanation-service.js";
import { BacktestService } from "./services/backtest-service.js";
import { TradingOrchestrator } from "./services/trading-orchestrator.js";
import { WatchlistMonitor } from "./services/watchlist-monitor.js";

export function buildApp(config = loadRuntimeConfig()) {
  const seed = {
    users: [
      {
        userId: config.app.defaultUserId,
        channelPreferences: ["web"],
        autoTradeEnabled: true
      }
    ],
    riskProfiles: createRiskProfiles(),
    brokerConnections: [
      { userId: config.app.defaultUserId, market: "US", brokerId: "paper-us", status: "connected" },
      { userId: config.app.defaultUserId, market: "INDIA", brokerId: "paper-india", status: "connected" }
    ]
  };
  const stateStore = config.app.persistState
    ? new FileBackedStateStore({ filePath: config.app.stateFile, seed })
    : new InMemoryStateStore(seed);

  const marketData = createMarketDataGateway(config);
  const brokerGateway = new MockBrokerGateway(stateStore);
  const indicatorEngine = new IndicatorEngine();
  const signalEngine = new SignalEngine();
  const decisionEngine = new DecisionEngine();
  const riskManager = new RiskManager(stateStore);
  const explanationService = new ExplanationService();
  const backtestService = new BacktestService();

  const orchestrator = new TradingOrchestrator({
    stateStore,
    marketData,
    brokerGateway,
    indicatorEngine,
    signalEngine,
    decisionEngine,
    riskManager,
    explanationService,
    backtestService
  });

  const watchlistMonitor = new WatchlistMonitor({ stateStore, orchestrator });
  const commandRouter = new CommandRouter({ stateStore, orchestrator, watchlistMonitor });

  return new TradingAgentApp({ commandRouter, stateStore, orchestrator, watchlistMonitor, config });
}

function createMarketDataGateway(config) {
  const mode = config.marketData?.mode ?? "auto";
  if (mode === "mock") {
    return new MockMarketDataGateway();
  }
  if (mode === "live") {
    return new LiveMarketDataGateway();
  }
  return new ResilientMarketDataGateway({
    primary: new LiveMarketDataGateway(),
    fallback: new MockMarketDataGateway()
  });
}
