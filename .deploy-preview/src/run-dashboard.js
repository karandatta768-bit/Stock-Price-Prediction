import { buildApp } from "./bootstrap.js";
import { startDashboardServer } from "./web/server.js";

const app = buildApp();
const server = await startDashboardServer(app);
console.log(`Dashboard running at http://localhost:${server.address().port}`);
