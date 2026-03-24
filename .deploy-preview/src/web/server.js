import http from "node:http";
import { renderDashboardHtml } from "./dashboard-template.js";

export function createDashboardServer(app) {
  return http.createServer((request, response) => routeDashboardRequest(app, request, response));
}

export async function routeDashboardRequest(app, request, response) {
  try {
    const url = new URL(request.url, `http://${request.headers.host ?? "localhost"}`);
    if (request.method === "GET" && url.pathname === "/") {
      sendHtml(response, renderDashboardHtml());
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/desk") {
      const userId = url.searchParams.get("userId") ?? app.config.app.defaultUserId;
      const portfolio = await app.orchestrator.getPortfolio(userId);
      sendJson(response, {
        portfolio,
        orders: app.stateStore.getOrders(userId),
        watchlist: app.stateStore.getWatchlist(userId),
        riskProfiles: app.stateStore.riskProfiles,
        latestRecommendation: [...app.stateStore.recommendations]
          .reverse()
          .find((item) => item.userId === userId) ?? null
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/discover") {
      const cards = await buildDiscoverCards(app);
      sendJson(response, cards);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/analyze") {
      const symbol = url.searchParams.get("symbol");
      const userId = url.searchParams.get("userId") ?? app.config.app.defaultUserId;
      const result = await app.handleMessage({ userId, text: `analyze ${symbol}` });
      sendJson(response, result.data ?? {});
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/orders") {
      const userId = url.searchParams.get("userId") ?? app.config.app.defaultUserId;
      sendJson(response, { orders: app.stateStore.getOrders(userId) });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/portfolio") {
      const userId = url.searchParams.get("userId") ?? app.config.app.defaultUserId;
      const portfolio = await app.orchestrator.getPortfolio(userId);
      sendJson(response, portfolio);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/watchlist") {
      const userId = url.searchParams.get("userId") ?? app.config.app.defaultUserId;
      sendJson(response, { watchlist: app.stateStore.getWatchlist(userId) });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/watch") {
      const body = await readJson(request);
      const result = await app.handleMessage({
        userId: body.userId ?? app.config.app.defaultUserId,
        text: `watch ${body.symbol}`
      });
      sendJson(response, result.data ?? {});
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/unwatch") {
      const body = await readJson(request);
      const result = await app.handleMessage({
        userId: body.userId ?? app.config.app.defaultUserId,
        text: `unwatch ${body.symbol}`
      });
      sendJson(response, result.data ?? {});
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/scan-watchlist") {
      const body = await readJson(request);
      const userId = body.userId ?? app.config.app.defaultUserId;
      const alerts = await app.watchlistMonitor.scan(userId);
      sendJson(response, { alerts });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/approve") {
      const body = await readJson(request);
      const result = await app.handleMessage({
        userId: body.userId ?? app.config.app.defaultUserId,
        text: `approve ${body.orderId}`
      });
      sendJson(response, result.data ?? {});
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/reject") {
      const body = await readJson(request);
      const result = await app.handleMessage({
        userId: body.userId ?? app.config.app.defaultUserId,
        text: `reject ${body.orderId}`
      });
      sendJson(response, result.data ?? {});
      return;
    }

    response.statusCode = 404;
    sendJson(response, { error: "Not found" });
  } catch (error) {
    response.statusCode = 500;
    sendJson(response, { error: error.message });
  }
}

async function buildDiscoverCards(app) {
  const symbols = ["AAPL", "MSFT", "NVDA", "RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"];
  const snapshots = await Promise.all(
    symbols.map(async (symbol) => {
      const result = await app.orchestrator.analyzeSymbol({
        userId: app.config.app.defaultUserId,
        symbol
      });
      return {
        symbol,
        market: result.recommendation.market,
        price: result.recommendation.snapshot.quote.price,
        changePct: result.recommendation.snapshot.quote.changePct,
        action: result.recommendation.decision.action,
        confidence: result.recommendation.decision.confidence
      };
    })
  );

  const sortedByChange = [...snapshots].sort((left, right) => right.changePct - left.changePct);
  const sortedByConfidence = [...snapshots]
    .filter((item) => item.action === "buy")
    .sort((left, right) => right.confidence - left.confidence);

  return {
    topGainers: sortedByChange.slice(0, 4),
    topLosers: sortedByChange.slice(-4).reverse(),
    aiPicks: sortedByConfidence.slice(0, 4)
  };
}

export function startDashboardServer(app) {
  const server = createDashboardServer(app);
  return new Promise((resolve) => {
    server.once("error", (error) => {
      if (error.code === "EADDRINUSE") {
        server.listen(0, () => resolve(server));
        return;
      }
      throw error;
    });
    server.listen(app.config.app.dashboardPort, () => resolve(server));
  });
}

function sendHtml(response, html) {
  response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
  response.end(html);
}

function sendJson(response, payload) {
  response.writeHead(response.statusCode || 200, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

async function readJson(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf-8")) : {};
}
