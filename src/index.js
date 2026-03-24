import { buildApp } from "./bootstrap.js";

const app = buildApp();

const demoCommands = [
  "analyze AAPL",
  "watch RELIANCE",
  "portfolio",
  "orders"
];

for (const command of demoCommands) {
  const response = await app.handleMessage({
    channel: "telegram",
    userId: "demo-user",
    text: command
  });
  console.log(`\n> ${command}\n${response.message}`);
}
