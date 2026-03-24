import { buildApp } from "../src/bootstrap.js";
import { routeDashboardRequest } from "../src/web/server.js";

const app = buildApp();

export default async function handler(request, response) {
  await routeDashboardRequest(app, request, response);
}

