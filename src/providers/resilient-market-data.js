export class ResilientMarketDataGateway {
  constructor({ primary, fallback }) {
    this.primary = primary;
    this.fallback = fallback;
  }

  async getSnapshot(request) {
    try {
      return await this.primary.getSnapshot(request);
    } catch (error) {
      const snapshot = await this.fallback.getSnapshot(request);
      return {
        ...snapshot,
        provider: `${snapshot.provider} (fallback)`,
        sourceMode: "demo",
        status: {
          ...snapshot.status,
          providerHealthy: false
        },
        fallbackReason: error.message
      };
    }
  }
}

