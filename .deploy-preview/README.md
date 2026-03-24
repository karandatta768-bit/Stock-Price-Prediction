# Multi-Market Trading Agent

An interactive AI trading dashboard built with Node.js that analyzes US and Indian equities, tracks watchlists, simulates broker workflows, and presents recommendations through a recruiter-friendly web UI.

This project was built to demonstrate full-stack product thinking, backend workflow design, UI refinement, and practical ML evaluation in one portfolio piece.

## Why This Project Stands Out

- Interactive web dashboard with polished motion and guided follow-up flows
- Multi-market support for US and Indian equities
- Live market data support with safe demo fallback
- Watchlist, portfolio, approvals, and order-management flows
- Risk-aware recommendation engine with explanations
- Separate ML prototype with walk-forward validation and diagnostics
- Automated tests for the main app behavior

## Main Features

- Analyze a stock and generate a recommendation with confidence, entry, stop, target, and explanation
- Track symbols in a watchlist and scan them for fresh alerts
- Review orders and approve or reject approval-required trades
- View holdings, buying power, and risk profiles
- Explore discovery cards for top gainers, losers, and AI picks
- Display whether data is coming from a live feed or demo fallback

## Tech Stack

- Node.js
- Vanilla HTML, CSS, and browser-side JavaScript
- Server-rendered dashboard over `node:http`
- Python ML prototype using `pandas`, `scikit-learn`, and technical indicators
- Yahoo Finance live data fetch with mock fallback

## Architecture

The app is organized into clear layers:

- `src/web`
  Dashboard template and HTTP server
- `src/messaging`
  Command routing and app orchestration entrypoint
- `src/services`
  Decision, risk, signal, explanation, watchlist, and backtest logic
- `src/providers`
  Market data and broker gateways
- `src/state`
  In-memory and file-backed state stores
- `stock_agent_prototype`
  Separate ML experimentation and evaluation area

## Live Data vs Demo Data

The dashboard now prefers live Yahoo Finance market data and falls back to demo data if live fetch is unavailable.

`MARKET_DATA_MODE` options:

- `auto`  
  Try live data first, then fall back to demo data
- `live`  
  Use live data only
- `mock`  
  Use demo data only

The UI shows a `Live Data` or `Demo Data` badge so visitors can see which mode is active.

## Getting Started

### Install

This project is dependency-light and uses only built-in Node.js APIs for the main app.

```bash
npm install
```

### Run the Dashboard

```bash
npm start
```

Or:

```bash
node src/run-dashboard.js
```

### Run Tests

```bash
node --test --test-isolation=none
```

## Deployment

This project is ready for preview deployment on Vercel.

It includes:

- `api/index.js` for Vercel serverless routing
- `vercel.json` to rewrite dashboard and API requests through the app handler

When deployed, the app keeps the same behavior as local development:

- dashboard at `/`
- API routes under `/api/*`
- live market data with demo fallback

## Environment Variables

- `DASHBOARD_PORT=3000`
- `DEFAULT_USER_ID=demo-user`
- `PERSIST_STATE=true`
- `STATE_FILE=./data/state.json`
- `MARKET_DATA_MODE=auto`

On Vercel, persistent state defaults to `false` unless you explicitly override it.

## Supported Commands

- `analyze <symbol>`
- `watch <symbol>`
- `unwatch <symbol>`
- `portfolio`
- `orders`
- `approve <trade_id>`
- `reject <trade_id>`
- `risk`
- `why <symbol|trade_id>`

## ML Prototype

The repository also includes a separate prototype in [`stock_agent_prototype/README.md`](./stock_agent_prototype/README.md).

That prototype focuses on:

- lag-safe feature engineering
- time-ordered train/test splits
- walk-forward validation
- backtesting
- overfitting and distribution-shift diagnostics

Current bundled sample result for the prototype:

- model: `gradient_boosting`
- test accuracy: `77.59%`
- walk-forward mean accuracy: `62.55%`

## Project Notes

- Broker execution is mocked for safety
- The dashboard is designed as a portfolio/demo product, not a real brokerage application
- Live market data availability depends on network access and upstream provider availability
- The recommendation engine and ML prototype are educational and portfolio-focused, not financial advice

## Recruiter-Friendly Summary

This project demonstrates:

- full-stack product building
- backend workflow design
- UI/UX refinement
- stateful app behavior
- testing discipline
- practical ML evaluation with honest diagnostics

If you are reviewing this as a recruiter or hiring manager, the strongest signal here is not just the interface, but the combination of user experience, backend logic, and careful handling of model limitations.
