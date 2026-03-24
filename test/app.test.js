import test from "node:test";
import assert from "node:assert/strict";
import os from "node:os";
import path from "node:path";
import { buildApp } from "../src/bootstrap.js";
import { createDashboardServer } from "../src/web/server.js";
import { TradingOrchestrator } from "../src/services/trading-orchestrator.js";
import { ACTIONS } from "../src/domain/constants.js";

function buildTestApp() {
  return buildApp({
    app: {
      dashboardPort: 0,
      defaultUserId: "demo-user",
      persistState: false,
      stateFile: path.join(os.tmpdir(), "trading-agent-test-state.json")
    },
    marketData: {
      mode: "mock"
    }
  });
}

test("analyze returns a structured recommendation for a US symbol", async () => {
  const app = buildTestApp();
  const result = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "analyze AAPL"
  });

  assert.match(result.message, /AAPL \(US\)/);
  assert.match(result.message, /Action:/);
  assert.match(result.message, /Explanation:/);
  assert.equal(typeof result.data.recommendation.backtestSummary.totalReturnPct, "number");
});

test("analyze returns a structured recommendation for an Indian symbol", async () => {
  const app = buildTestApp();
  const result = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "analyze RELIANCE"
  });

  assert.match(result.message, /RELIANCE \(INDIA\)/);
  assert.match(result.message, /Execution:/);
});

test("watch and unwatch manage the watchlist", async () => {
  const app = buildTestApp();

  const watchResult = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "watch TCS"
  });
  assert.match(watchResult.message, /Watching TCS in INDIA/);

  const unwatchResult = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "unwatch TCS"
  });
  assert.match(unwatchResult.message, /Removed TCS from watchlist/);
});

test("executable trades create tracked order outcomes", async () => {
  const app = buildTestApp();
  const result = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "analyze AAPL"
  });

  assert.match(result.message, /Approval required: (yes|no)/);
  assert.match(result.message, /Trade status:/);
});

test("portfolio and orders reflect stored state", async () => {
  const app = buildTestApp();
  await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "analyze AAPL"
  });

  const portfolio = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "portfolio"
  });
  const orders = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: "orders"
  });

  assert.match(portfolio.message, /Buying power:/);
  assert.match(orders.message, /AAPL/);
});

test("dashboard server exposes analyze api", async () => {
  const app = buildTestApp();
  const server = createDashboardServer(app);

  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  const response = await fetch(`http://127.0.0.1:${port}/api/analyze?symbol=AAPL&userId=demo-user`);
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.recommendation.symbol, "AAPL");
  assert.ok(Array.isArray(payload.recommendation.snapshot.candles));

  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

test("dashboard server exposes watchlist scan api", async () => {
  const app = buildTestApp();
  const server = createDashboardServer(app);

  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();

  await fetch(`http://127.0.0.1:${port}/api/watch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol: "AAPL", userId: "demo-user" })
  });

  const response = await fetch(`http://127.0.0.1:${port}/api/scan-watchlist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ userId: "demo-user" })
  });
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.ok(Array.isArray(payload.alerts));

  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

test("dashboard server exposes discovery cards api", async () => {
  const app = buildTestApp();
  const server = createDashboardServer(app);

  await new Promise((resolve) => server.listen(0, resolve));
  const { port } = server.address();
  const response = await fetch(`http://127.0.0.1:${port}/api/discover`);
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.ok(Array.isArray(payload.topGainers));
  assert.ok(Array.isArray(payload.topLosers));
  assert.ok(Array.isArray(payload.aiPicks));

  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
});

test("informational hold recommendations do not create approval orders", async () => {
  let approvalRequested = false;
  let orderCreated = false;

  const orchestrator = new TradingOrchestrator({
    stateStore: {
      getRiskProfile() {
        return { confidenceThreshold: 0.64 };
      },
      saveRecommendation() {},
      saveExplanation() {},
      appendLog() {},
      createOrder() {
        orderCreated = true;
        return { id: "ord-test" };
      }
    },
    marketData: {
      async getSnapshot() {
        return {
          quote: { price: 100, changePct: 0.1 },
          status: { stale: false, marketOpen: true, providerHealthy: true }
        };
      }
    },
    indicatorEngine: {
      compute() {
        return { trend: "neutral" };
      }
    },
    signalEngine: {
      evaluate() {
        return { action: ACTIONS.HOLD, timeframe: "swing", setup: "range-bound hold" };
      }
    },
    decisionEngine: {
      decide() {
        return {
          action: ACTIONS.HOLD,
          confidence: 0.2,
          timeframe: "swing",
          entryPrice: 100,
          stopLoss: 95,
          targetPrice: 105,
          suggestedQuantity: 100,
          indicators: { trend: "neutral" }
        };
      }
    },
    riskManager: {
      assess() {
        return {
          allowed: true,
          executionMode: "approval_required",
          reasons: ["Confidence is below market threshold."],
          tradeValue: 10000
        };
      }
    },
    explanationService: {
      explain() {
        return "Hold for now.";
      }
    },
    backtestService: {
      summarize() {
        return { totalReturnPct: 0, winRatePct: 0, maxDrawdownPct: 0, signalChanges: 0, bars: 0 };
      }
    },
    brokerGateway: {
      async placeOrder() {
        throw new Error("placeOrder should not be called for hold actions");
      },
      async requestApprovalOrder() {
        approvalRequested = true;
        return { id: "ord-approval" };
      }
    }
  });

  const result = await orchestrator.analyzeSymbol({ userId: "demo-user", symbol: "AAPL" });

  assert.equal(result.order, null);
  assert.equal(approvalRequested, false);
  assert.equal(orderCreated, false);
});
