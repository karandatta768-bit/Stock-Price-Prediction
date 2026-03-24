export function loadRuntimeConfig(env = process.env) {
  const persistState = env.PERSIST_STATE != null
    ? env.PERSIST_STATE !== "false"
    : !env.VERCEL;

  return {
    app: {
      dashboardPort: Number(env.DASHBOARD_PORT ?? 3000),
      defaultUserId: env.DEFAULT_USER_ID ?? "demo-user",
      persistState,
      stateFile: env.STATE_FILE ?? "./data/state.json"
    },
    marketData: {
      mode: env.MARKET_DATA_MODE ?? "auto"
    }
  };
}
