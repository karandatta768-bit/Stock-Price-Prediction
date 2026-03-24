import fs from "node:fs";
import path from "node:path";
import { InMemoryStateStore } from "./in-memory-store.js";

export class FileBackedStateStore extends InMemoryStateStore {
  constructor({ filePath, seed = {} }) {
    const resolvedPath = path.resolve(filePath);
    const diskState = readStateFile(resolvedPath);
    super(mergeSeed(seed, diskState));
    this.filePath = resolvedPath;
    this.flush();
  }

  addWatchSymbol(userId, symbol, market) {
    super.addWatchSymbol(userId, symbol, market);
    this.flush();
  }

  removeWatchSymbol(userId, symbol) {
    super.removeWatchSymbol(userId, symbol);
    this.flush();
  }

  saveRecommendation(recommendation) {
    super.saveRecommendation(recommendation);
    this.flush();
  }

  saveExplanation(explanation) {
    super.saveExplanation(explanation);
    this.flush();
  }

  createOrder(order) {
    const created = super.createOrder(order);
    this.flush();
    return created;
  }

  updateOrder(orderId, patch) {
    const updated = super.updateOrder(orderId, patch);
    this.flush();
    return updated;
  }

  appendLog(event) {
    super.appendLog(event);
    this.flush();
  }

  flush() {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.writeFileSync(this.filePath, JSON.stringify(this.serialize(), null, 2), "utf-8");
  }

  serialize() {
    return {
      users: this.users,
      riskProfiles: this.riskProfiles,
      watchlists: this.watchlists,
      brokerConnections: this.brokerConnections,
      orders: this.orders,
      recommendations: this.recommendations,
      explanations: this.explanations,
      logs: this.logs,
      positions: this.positions
    };
  }
}

function readStateFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return {};
  }
}

function mergeSeed(seed, diskState) {
  return {
    ...seed,
    ...diskState,
    users: diskState.users ?? seed.users,
    riskProfiles: diskState.riskProfiles ?? seed.riskProfiles,
    watchlists: diskState.watchlists ?? seed.watchlists,
    brokerConnections: diskState.brokerConnections ?? seed.brokerConnections,
    orders: diskState.orders ?? seed.orders,
    recommendations: diskState.recommendations ?? seed.recommendations,
    explanations: diskState.explanations ?? seed.explanations,
    logs: diskState.logs ?? seed.logs,
    positions: diskState.positions ?? seed.positions
  };
}
