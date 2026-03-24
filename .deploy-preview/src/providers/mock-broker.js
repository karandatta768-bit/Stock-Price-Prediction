import { ORDER_STATUS } from "../domain/constants.js";
import { createId } from "../utils/id.js";

export class MockBrokerGateway {
  constructor(stateStore) {
    this.stateStore = stateStore;
  }

  async getPortfolio(userId) {
    return {
      buyingPower: 250000,
      positions: this.stateStore.getPositions(userId)
    };
  }

  async placeOrder({ userId, market, symbol, action, quantity, price, metadata }) {
    const order = this.stateStore.createOrder({
      id: createId("ord"),
      userId,
      market,
      symbol,
      action,
      quantity,
      price,
      status: ORDER_STATUS.FILLED,
      metadata,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    });
    return order;
  }

  async requestApprovalOrder({ userId, market, symbol, action, quantity, price, metadata }) {
    return this.stateStore.createOrder({
      id: createId("ord"),
      userId,
      market,
      symbol,
      action,
      quantity,
      price,
      status: ORDER_STATUS.PENDING_APPROVAL,
      metadata,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    });
  }

  async approveOrder(orderId, userId) {
    const order = this.stateStore.getOrder(orderId, userId);
    if (!order) {
      return null;
    }
    return this.stateStore.updateOrder(orderId, {
      status: ORDER_STATUS.FILLED,
      updatedAt: new Date().toISOString()
    });
  }

  async rejectOrder(orderId, userId) {
    const order = this.stateStore.getOrder(orderId, userId);
    if (!order) {
      return null;
    }
    return this.stateStore.updateOrder(orderId, {
      status: ORDER_STATUS.REJECTED,
      updatedAt: new Date().toISOString()
    });
  }
}
