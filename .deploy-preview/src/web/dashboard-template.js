export function renderDashboardHtml() {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Groww-Style AI Trading App</title>
  <style>
    :root {
      --bg: #eef4fb;
      --surface: #ffffff;
      --ink: #142033;
      --muted: #6b778c;
      --line: #e7edf3;
      --green: #00b386;
      --green-soft: #e8fbf4;
      --red: #e04f5f;
      --red-soft: #fff0f2;
      --shadow: 0 18px 40px rgba(32, 52, 84, 0.08);
      --sky: #dff6ff;
      --blue: #2f6fec;
      --gold: #ffd76a;
      --pointer-x: 50%;
      --pointer-y: 30%;
      --reveal-distance: 22px;
    }
    body[data-theme="dark"] {
      --bg: #09111d;
      --surface: #111b2a;
      --ink: #edf5ff;
      --muted: #96a8c2;
      --line: #203149;
      --green-soft: rgba(0,179,134,0.14);
      --red-soft: rgba(224,79,95,0.16);
      --shadow: 0 20px 48px rgba(0, 0, 0, 0.35);
      --sky: #14243b;
      --blue: #78a6ff;
    }
    * { box-sizing: border-box; }
    html {
      scroll-behavior: smooth;
    }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top, rgba(255,255,255,0.92), transparent 42%),
        linear-gradient(180deg, #f6fbff 0%, var(--bg) 42%, #e8eff8 100%);
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      position: relative;
      overflow-x: hidden;
    }
    body[data-theme="dark"] {
      background:
        radial-gradient(circle at top, rgba(120,166,255,0.12), transparent 38%),
        linear-gradient(180deg, #07111f 0%, var(--bg) 45%, #0c1728 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background:
        radial-gradient(circle at var(--pointer-x) var(--pointer-y), rgba(47,111,236,0.14), transparent 18%),
        radial-gradient(circle at 18% 18%, rgba(0,179,134,0.12), transparent 24%),
        radial-gradient(circle at 82% 16%, rgba(255,215,106,0.14), transparent 20%);
      pointer-events: none;
      z-index: 0;
      transition: background-position 120ms ease-out;
    }
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.2) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.2) 1px, transparent 1px);
      background-size: 36px 36px;
      mask-image: linear-gradient(180deg, rgba(0,0,0,0.45), transparent 88%);
      pointer-events: none;
      z-index: 0;
      opacity: 0.45;
    }
    .background-stage {
      position: fixed;
      inset: 0;
      overflow: hidden;
      pointer-events: none;
      z-index: 0;
    }
    .background-stage span {
      position: absolute;
      border-radius: 999px;
      filter: blur(10px);
      opacity: 0.75;
      transform: translate3d(calc(var(--shift-x, 0px) * 1), calc(var(--shift-y, 0px) * 1), 0);
      transition: transform 180ms ease-out;
      animation: drift 16s ease-in-out infinite;
      will-change: transform, opacity;
    }
    .orb-one {
      width: 280px;
      height: 280px;
      top: 8%;
      left: -60px;
      background: radial-gradient(circle, rgba(0,179,134,0.32), rgba(0,179,134,0));
      --shift-x: 16px;
      --shift-y: -10px;
    }
    .orb-two {
      width: 360px;
      height: 360px;
      top: 22%;
      right: -120px;
      background: radial-gradient(circle, rgba(47,111,236,0.22), rgba(47,111,236,0));
      --shift-x: -22px;
      --shift-y: 18px;
      animation-duration: 19s;
      animation-delay: -4s;
    }
    .orb-three {
      width: 220px;
      height: 220px;
      bottom: 10%;
      left: 42%;
      background: radial-gradient(circle, rgba(255,215,106,0.24), rgba(255,215,106,0));
      --shift-x: 10px;
      --shift-y: 16px;
      animation-duration: 14s;
      animation-delay: -7s;
    }
    .orb-four {
      width: 420px;
      height: 420px;
      bottom: -140px;
      right: 8%;
      background: radial-gradient(circle, rgba(255,255,255,0.66), rgba(255,255,255,0));
      --shift-x: -14px;
      --shift-y: -12px;
      animation-duration: 22s;
      animation-delay: -10s;
    }
    .app {
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      min-height: 100vh;
      position: relative;
      z-index: 1;
    }
    .sidebar {
      background: rgba(255, 255, 255, 0.7);
      border-right: 1px solid rgba(231, 237, 243, 0.9);
      backdrop-filter: blur(18px);
      padding: 24px 18px;
      animation: slideInLeft 720ms cubic-bezier(.2,.8,.2,1) both;
    }
    body[data-theme="dark"] .sidebar {
      background: rgba(10, 17, 30, 0.82);
      border-right-color: rgba(32, 49, 73, 0.95);
    }
    .brand {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(135deg, #f0fff8, #eef6ff);
      border: 1px solid var(--line);
    }
    .brand-mark {
      width: 42px;
      height: 42px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--green), #63d8bc);
      color: #fff;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
    }
    .brand h1 {
      margin: 12px 0 6px;
      font-size: 26px;
      line-height: 1;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    .nav {
      margin-top: 22px;
      display: grid;
      gap: 8px;
    }
    .nav button {
      border: none;
      background: transparent;
      color: var(--ink);
      text-align: left;
      padding: 13px 14px;
      border-radius: 14px;
      font: 600 14px "Segoe UI", sans-serif;
      cursor: pointer;
    }
    .nav button.active {
      background: #eef8f4;
      color: #0f7f62;
    }
    .sidebar-note {
      margin-top: 24px;
      padding: 14px;
      border-radius: 16px;
      background: #f8fbff;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    body[data-theme="dark"] .sidebar-note {
      background: rgba(17, 27, 42, 0.88);
    }
    .content {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .topbar, .card {
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(231, 237, 243, 0.92);
      border-radius: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    body[data-theme="dark"] .topbar,
    body[data-theme="dark"] .card {
      background: rgba(17, 27, 42, 0.84);
      border-color: rgba(32, 49, 73, 0.96);
    }
    .topbar {
      padding: 20px 22px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 18px;
    }
    .animate-in {
      opacity: 0;
      transform: translateY(var(--reveal-distance)) scale(0.985);
      animation: riseIn 700ms cubic-bezier(.2,.8,.2,1) forwards;
      animation-delay: var(--delay, 0ms);
      will-change: transform, opacity;
    }
    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .headline {
      font-size: 36px;
      line-height: 1.05;
      margin: 0 0 8px;
      font-weight: 700;
    }
    .controls {
      margin-top: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    input, select, button.action {
      border-radius: 999px;
      padding: 12px 16px;
      border: 1px solid rgba(231, 237, 243, 0.95);
      background: rgba(255, 255, 255, 0.86);
      font: inherit;
    }
    body[data-theme="dark"] input,
    body[data-theme="dark"] select,
    body[data-theme="dark"] button.action.secondary,
    body[data-theme="dark"] .ask-chip,
    body[data-theme="dark"] .item-actions button,
    body[data-theme="dark"] .theme-toggle {
      background: rgba(15, 25, 40, 0.92);
      color: var(--ink);
      border-color: rgba(32, 49, 73, 0.96);
    }
    input, select {
      min-width: 140px;
    }
    button.action {
      background: linear-gradient(135deg, var(--green), #1dd0aa);
      color: #fff;
      border: none;
      cursor: pointer;
      font-weight: 700;
      box-shadow: 0 12px 22px rgba(0, 179, 134, 0.22);
      transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }
    button.action:hover {
      transform: translateY(-2px);
      box-shadow: 0 16px 28px rgba(0, 179, 134, 0.26);
      filter: saturate(1.05);
    }
    button.action:active {
      transform: translateY(0);
    }
    button.action.secondary {
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .assistant-box {
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(135deg, #f6fffb, #f5f9ff);
      border: 1px solid var(--line);
      position: relative;
      overflow: hidden;
    }
    body[data-theme="dark"] .assistant-box {
      background: linear-gradient(135deg, rgba(0,179,134,0.08), rgba(120,166,255,0.12));
    }
    .assistant-box::after {
      content: "";
      position: absolute;
      inset: auto -10% -45% auto;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(47,111,236,0.16), transparent 70%);
      animation: drift 12s ease-in-out infinite;
      pointer-events: none;
    }
    .assistant-box pre {
      margin: 8px 0 0;
      white-space: pre-wrap;
      font: 13px/1.55 "Segoe UI", sans-serif;
      color: var(--ink);
    }
    .quick-asks {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }
    .ask-chip {
      border: 1px solid rgba(47,111,236,0.16);
      background: rgba(255,255,255,0.8);
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 12px;
      font: 600 12px "Segoe UI", sans-serif;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .ask-chip:hover {
      transform: translateY(-2px);
      border-color: rgba(47,111,236,0.3);
      box-shadow: 0 10px 18px rgba(32, 52, 84, 0.08);
    }
    .follow-up-card {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(47,111,236,0.1);
      font-size: 13px;
      line-height: 1.5;
      color: var(--muted);
    }
    .data-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      border: 1px solid rgba(47,111,236,0.14);
      font-size: 12px;
      font-weight: 700;
      color: var(--ink);
    }
    .data-badge.live {
      color: #0f7f62;
      border-color: rgba(0,179,134,0.24);
      background: rgba(232,251,244,0.92);
    }
    .data-badge.demo {
      color: #8a6400;
      border-color: rgba(255,215,106,0.35);
      background: rgba(255, 247, 214, 0.92);
    }
    .data-badge small {
      font-weight: 600;
      color: var(--muted);
    }
    .ticker-shell {
      overflow: hidden;
      padding: 14px 0;
      border-radius: 20px;
      border: 1px solid rgba(231, 237, 243, 0.9);
      background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(240,247,255,0.88));
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    body[data-theme="dark"] .ticker-shell {
      background: linear-gradient(135deg, rgba(17,27,42,0.88), rgba(12,23,40,0.92));
      border-color: rgba(32, 49, 73, 0.92);
    }
    .ticker-track {
      display: flex;
      gap: 14px;
      width: max-content;
      padding-left: 18px;
      animation: tickerMove 26s linear infinite;
      will-change: transform;
    }
    .ticker-track:hover {
      animation-play-state: paused;
    }
    .ticker-pill {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(47,111,236,0.08);
      font-size: 13px;
      white-space: nowrap;
    }
    body[data-theme="dark"] .ticker-pill {
      background: rgba(15, 25, 40, 0.94);
      border-color: rgba(32, 49, 73, 0.92);
    }
    .ticker-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 0 rgba(0,179,134,0.3);
      animation: beacon 1.8s ease-out infinite;
    }
    .headline-glow {
      display: inline-block;
      background: linear-gradient(90deg, #132033 0%, #2f6fec 50%, #132033 100%);
      background-size: 220% 100%;
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      animation: shimmerText 8s linear infinite;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }
    .discover-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
    }
    .stat {
      padding: 18px;
    }
    .stat-value {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
    }
    .panel, .chart-card {
      padding: 20px;
    }
    .chart-card {
      position: relative;
    }
    .section {
      display: none;
      opacity: 0;
      transform: translateY(14px);
    }
    .section.active {
      display: block;
      animation: sectionSwap 380ms ease-out both;
    }
    .chart-head {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 12px;
    }
    .chart-title {
      font-size: 30px;
      font-weight: 700;
      margin-top: 4px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #f2f6fa;
      color: var(--ink);
      border: 1px solid rgba(47,111,236,0.08);
      transition: transform 160ms ease, background-color 160ms ease, color 160ms ease;
    }
    body[data-theme="dark"] .pill {
      background: rgba(15, 25, 40, 0.94);
      color: var(--ink);
      border-color: rgba(32, 49, 73, 0.92);
    }
    .positive {
      color: var(--green);
      background: var(--green-soft);
    }
    .negative {
      color: var(--red);
      background: var(--red-soft);
    }
    .pill.status-filled,
    .pill.status-placed {
      color: var(--green);
      background: var(--green-soft);
      border-color: rgba(0,179,134,0.22);
    }
    .pill.status-pending_approval {
      color: #8a6400;
      background: rgba(255, 215, 106, 0.22);
      border-color: rgba(255, 215, 106, 0.35);
    }
    .pill.status-rejected,
    .pill.status-cancelled,
    .pill.status-blocked {
      color: var(--red);
      background: var(--red-soft);
      border-color: rgba(224,79,95,0.22);
    }
    body[data-theme="dark"] .pill.status-pending_approval {
      color: #ffd76a;
      background: rgba(255, 215, 106, 0.14);
      border-color: rgba(255, 215, 106, 0.26);
    }
    .order-status {
      min-width: 92px;
      justify-content: center;
      text-align: center;
      text-transform: capitalize;
      letter-spacing: 0.01em;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }
    .order-status.status-pending_approval {
      min-width: 156px;
    }
    body[data-theme="dark"] .order-status {
      color: #f4f8ff !important;
      background: #243449 !important;
      border-color: #314760 !important;
      box-shadow: none;
    }
    body[data-theme="dark"] .order-status.status-filled,
    body[data-theme="dark"] .order-status.status-placed {
      color: #86efbf !important;
      background: rgba(0,179,134,0.18) !important;
      border-color: rgba(0,179,134,0.34) !important;
    }
    body[data-theme="dark"] .order-status.status-rejected,
    body[data-theme="dark"] .order-status.status-cancelled,
    body[data-theme="dark"] .order-status.status-blocked {
      color: #ff9ea8 !important;
      background: rgba(224,79,95,0.18) !important;
      border-color: rgba(224,79,95,0.34) !important;
    }
    body[data-theme="dark"] .order-status.status-pending_approval {
      color: #ffe08a !important;
      background: rgba(255,215,106,0.18) !important;
      border-color: rgba(255,215,106,0.34) !important;
    }
    canvas {
      width: 100%;
      height: 340px;
      display: block;
      margin-top: 16px;
      border-radius: 18px;
      background: linear-gradient(180deg, #fbfffe, #f6f9fc);
    }
    .chart-tooltip {
      position: absolute;
      pointer-events: none;
      opacity: 0;
      background: rgba(20,32,51,0.92);
      color: #fff;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.5;
      white-space: nowrap;
      transform: translate(12px, 12px);
    }
    .theme-toggle {
      margin-top: 16px;
      width: 100%;
      border-radius: 999px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      color: var(--ink);
      font: 700 13px "Segoe UI", sans-serif;
      cursor: pointer;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    .mini-card {
      padding: 14px;
      border-radius: 16px;
      background: #fbfdff;
      border: 1px solid var(--line);
      transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .mini-card:hover {
      transform: translateY(-3px);
      box-shadow: 0 12px 24px rgba(32, 52, 84, 0.08);
    }
    body[data-theme="dark"] .mini-card,
    body[data-theme="dark"] .item,
    body[data-theme="dark"] .follow-up-card {
      background: rgba(15, 25, 40, 0.9);
      border-color: rgba(32, 49, 73, 0.92);
    }
    .mini-card strong {
      display: block;
      margin-top: 6px;
      font-size: 20px;
    }
    .list {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .item {
      padding: 16px;
      border-radius: 16px;
      background: #fbfdff;
      border: 1px solid var(--line);
      transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    }
    .item:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 26px rgba(32, 52, 84, 0.08);
      border-color: rgba(47,111,236,0.14);
    }
    .stock-card {
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .stock-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 18px 36px rgba(32, 52, 84, 0.12);
      border-color: rgba(47,111,236,0.18);
    }
    .interactive-sheen {
      position: relative;
      overflow: hidden;
      isolation: isolate;
    }
    .interactive-sheen::before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--pointer-x) var(--pointer-y), rgba(255,255,255,0.36), transparent 24%);
      opacity: 0;
      transition: opacity 160ms ease;
      z-index: -1;
    }
    .interactive-sheen:hover::before {
      opacity: 1;
    }
    .pulse-update {
      animation: pulseUpdate 520ms ease;
    }
    .stock-price {
      margin-top: 8px;
      font-size: 22px;
      font-weight: 700;
    }
    .stock-meta {
      margin-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
    }
    .item-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }
    .item-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .item-actions button {
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      cursor: pointer;
      font: 600 12px "Segoe UI", sans-serif;
      transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }
    body[data-theme="dark"] .item-actions button {
      background: rgba(15, 25, 40, 0.94);
      color: var(--ink);
      border-color: rgba(32, 49, 73, 0.92);
    }
    .item-actions button:hover {
      transform: translateY(-1px);
      border-color: rgba(47,111,236,0.22);
      box-shadow: 0 8px 18px rgba(32, 52, 84, 0.08);
    }
    .hint {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    @keyframes drift {
      0%, 100% { transform: translate3d(calc(var(--shift-x, 0px) * 1), calc(var(--shift-y, 0px) * 1), 0) scale(1); }
      50% { transform: translate3d(calc(var(--shift-x, 0px) * -1), calc(var(--shift-y, 0px) * -1), 0) scale(1.05); }
    }
    @keyframes riseIn {
      from {
        opacity: 0;
        transform: translateY(var(--reveal-distance)) scale(0.985);
      }
      to {
        opacity: 1;
        transform: translateY(0) scale(1);
      }
    }
    @keyframes slideInLeft {
      from {
        opacity: 0;
        transform: translateX(-18px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }
    @keyframes sectionSwap {
      from {
        opacity: 0;
        transform: translateY(14px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    @keyframes pulseUpdate {
      0% { transform: scale(1); box-shadow: 0 0 0 rgba(47,111,236,0); }
      45% { transform: scale(1.03); box-shadow: 0 0 0 10px rgba(47,111,236,0.08); }
      100% { transform: scale(1); box-shadow: 0 0 0 rgba(47,111,236,0); }
    }
    @keyframes tickerMove {
      0% { transform: translateX(0); }
      100% { transform: translateX(-50%); }
    }
    @keyframes beacon {
      0% { box-shadow: 0 0 0 0 rgba(0,179,134,0.28); }
      100% { box-shadow: 0 0 0 12px rgba(0,179,134,0); }
    }
    @keyframes shimmerText {
      0% { background-position: 0% 50%; }
      100% { background-position: 200% 50%; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after {
        animation: none !important;
        transition: none !important;
      }
      .animate-in,
      .section,
      .section.active {
        opacity: 1;
        transform: none;
      }
      .ticker-track {
        animation: none;
        width: auto;
        flex-wrap: wrap;
      }
    }
    @media (max-width: 1120px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: none; border-bottom: 1px solid var(--line); }
      .topbar, .grid, .grid-2, .stats, .discover-grid { grid-template-columns: 1fr; }
      .background-stage span { opacity: 0.55; }
    }
  </style>
</head>
<body>
  <div class="background-stage" aria-hidden="true">
    <span class="orb-one"></span>
    <span class="orb-two"></span>
    <span class="orb-three"></span>
    <span class="orb-four"></span>
  </div>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">G</div>
        <h1>Groww AI</h1>
        <div class="muted">A Groww-style investing app with an AI agent that analyzes, explains, and follows up on your stock decisions.</div>
      </div>
      <nav class="nav">
        <button class="active" data-section="dashboard">Dashboard</button>
        <button data-section="stock">Stock Details</button>
        <button data-section="watchlist">Watchlist</button>
        <button data-section="orders">Orders</button>
        <button data-section="portfolio">Portfolio</button>
        <button data-section="backtest">Backtest</button>
      </nav>
      <div class="sidebar-note">
        Analyze a stock, add it to watchlist, and let the AI tell you whether to buy, hold, sell, or wait.
      </div>
      <button id="themeToggle" class="theme-toggle" type="button">Switch to Dark Mode</button>
    </aside>

    <main class="content">
      <section class="topbar interactive-sheen animate-in" style="--delay: 40ms;">
        <div>
          <div class="eyebrow">Smart Investing</div>
          <h2 class="headline"><span class="headline-glow">Discover what to buy, what to hold, and what to sell.</span></h2>
          <div class="muted">This looks like a modern investing app, but it runs on your trading-agent workflow underneath.</div>
          <div id="dataModeBadge" class="data-badge demo">Demo Data <small id="dataProviderText">Fallback feed active</small></div>
          <div class="controls">
            <input id="symbolInput" value="AAPL" placeholder="Search stock">
            <select id="timeframeSelect">
              <option value="15">15 bars</option>
              <option value="30">30 bars</option>
              <option value="60" selected>60 bars</option>
            </select>
            <button class="action" id="analyzeButton">Analyze</button>
            <button class="action secondary" id="watchButton">Add to Watchlist</button>
            <button class="action secondary" id="scanButton">Scan Watchlist</button>
            <button class="action secondary" id="refreshButton">Refresh</button>
          </div>
        </div>
        <div class="assistant-box">
          <div class="eyebrow">AI Follow Up</div>
          <pre id="explanationBox">Run an analysis to see the AI explanation.</pre>
          <pre id="alertsBox" style="margin-top:14px;">No alerts yet.</pre>
          <div class="quick-asks">
            <button class="ask-chip" data-followup="why">Why this stock?</button>
            <button class="ask-chip" data-followup="risk">Show risk</button>
            <button class="ask-chip" data-followup="watch">Add to watchlist</button>
            <button class="ask-chip" data-followup="scan">Scan ideas</button>
          </div>
          <div id="followUpBox" class="follow-up-card">Use these quick follow-ups to turn the dashboard into a guided conversation for first-time visitors.</div>
        </div>
      </section>

      <section class="ticker-shell animate-in" style="--delay: 80ms;">
        <div id="discoverTicker" class="ticker-track"></div>
      </section>

      <section id="dashboard" class="section active">
        <div class="discover-grid" style="margin-bottom:18px;">
          <div class="card panel interactive-sheen animate-in" style="--delay: 100ms;">
            <div class="eyebrow">Top Gainers</div>
            <div id="topGainersList" class="list"></div>
          </div>
          <div class="card panel interactive-sheen animate-in" style="--delay: 150ms;">
            <div class="eyebrow">Top Losers</div>
            <div id="topLosersList" class="list"></div>
          </div>
          <div class="card panel interactive-sheen animate-in" style="--delay: 200ms;">
            <div class="eyebrow">AI Picks</div>
            <div id="aiPicksList" class="list"></div>
          </div>
        </div>

        <div class="stats">
          <div class="card stat interactive-sheen animate-in" style="--delay: 250ms;"><div class="eyebrow">Action</div><div id="actionStat" class="stat-value">-</div></div>
          <div class="card stat interactive-sheen animate-in" style="--delay: 300ms;"><div class="eyebrow">Confidence</div><div id="confidenceStat" class="stat-value">-</div></div>
          <div class="card stat interactive-sheen animate-in" style="--delay: 350ms;"><div class="eyebrow">Execution</div><div id="executionStat" class="stat-value">-</div></div>
          <div class="card stat interactive-sheen animate-in" style="--delay: 400ms;"><div class="eyebrow">Buying Power</div><div id="buyingPowerStat" class="stat-value">-</div></div>
        </div>

        <div class="grid" style="margin-top:18px;">
          <div class="card chart-card interactive-sheen animate-in" style="--delay: 450ms;">
            <div class="chart-head">
              <div>
                <div class="eyebrow">Stock Chart</div>
                <div id="symbolTitle" class="chart-title">AAPL</div>
                <div id="metaLine" class="muted">Waiting for analysis...</div>
              </div>
              <div id="pricePill" class="pill">No data</div>
            </div>
            <canvas id="priceChart" width="900" height="340"></canvas>
            <div id="priceTooltip" class="chart-tooltip"></div>
            <div class="hint">Hover to inspect candles. Scroll to zoom the visible window.</div>
          </div>

          <div class="card panel interactive-sheen animate-in" style="--delay: 500ms;">
            <div class="eyebrow">Trade Plan</div>
            <div class="mini-grid">
              <div class="mini-card"><div class="muted">Entry</div><strong id="entryStat">-</strong></div>
              <div class="mini-card"><div class="muted">Stop</div><strong id="stopStat">-</strong></div>
              <div class="mini-card"><div class="muted">Target</div><strong id="targetStat">-</strong></div>
              <div class="mini-card"><div class="muted">Size</div><strong id="sizeStat">-</strong></div>
            </div>
            <div class="item" style="margin-top:14px;">
              <div class="eyebrow">Risk Context</div>
              <div id="riskReasons" class="muted">No risk flags.</div>
            </div>
            <div class="item" style="margin-top:14px;">
              <div class="eyebrow">Replay Read</div>
              <div id="backtestRead" class="muted">No replay summary yet.</div>
            </div>
          </div>
        </div>

        <div class="grid-2" style="margin-top:18px;">
          <div class="card panel interactive-sheen animate-in" style="--delay: 550ms;">
            <div class="eyebrow">Performance Snapshot</div>
            <div class="mini-grid">
              <div class="mini-card"><div class="muted">Return</div><strong id="backtestReturn">-</strong></div>
              <div class="mini-card"><div class="muted">Win Rate</div><strong id="backtestWinRate">-</strong></div>
              <div class="mini-card"><div class="muted">Drawdown</div><strong id="backtestDrawdown">-</strong></div>
              <div class="mini-card"><div class="muted">Signal Changes</div><strong id="backtestSignals">-</strong></div>
            </div>
          </div>
          <div class="card panel interactive-sheen animate-in" style="--delay: 600ms;">
            <div class="eyebrow">Quick Summary</div>
            <div class="list">
              <div class="item">AI recommendation updates instantly after each stock analysis.</div>
              <div class="item">Watchlist scan can follow up on multiple stocks for fresh buy or sell ideas.</div>
              <div class="item">Portfolio, orders, and backtest remain available in dedicated sections.</div>
            </div>
          </div>
        </div>
      </section>

      <section id="watchlist" class="section">
        <div class="card panel interactive-sheen">
          <div class="eyebrow">My Watchlist</div>
          <div id="watchlistList" class="list"></div>
        </div>
      </section>

      <section id="stock" class="section">
        <div class="grid">
          <div class="card chart-card interactive-sheen">
            <div class="chart-head">
              <div>
                <div class="eyebrow">Stock Details</div>
                <div id="detailTitle" class="chart-title">Select a stock</div>
                <div id="detailMeta" class="muted">Open a stock from the dashboard cards, watchlist, or search bar.</div>
              </div>
              <div id="detailActionPill" class="pill">Awaiting analysis</div>
            </div>
            <canvas id="detailChart" width="900" height="340"></canvas>
          </div>
          <div class="card panel interactive-sheen">
            <div class="eyebrow">AI Call</div>
            <div class="mini-grid">
              <div class="mini-card"><div class="muted">Recommendation</div><strong id="detailRecommendation">-</strong></div>
              <div class="mini-card"><div class="muted">Confidence</div><strong id="detailConfidence">-</strong></div>
              <div class="mini-card"><div class="muted">Entry</div><strong id="detailEntry">-</strong></div>
              <div class="mini-card"><div class="muted">Target</div><strong id="detailTarget">-</strong></div>
            </div>
            <div class="item" style="margin-top:14px;">
              <div class="eyebrow">Why This Stock</div>
              <div id="detailWhy" class="muted">The AI explanation will appear here.</div>
            </div>
            <div class="item" style="margin-top:14px;">
              <div class="eyebrow">Follow Up Plan</div>
              <div id="detailFollowUp" class="muted">Analyze a stock to generate a follow-up path.</div>
            </div>
          </div>
        </div>
      </section>

      <section id="orders" class="section">
        <div class="card panel interactive-sheen">
          <div class="eyebrow">Orders</div>
          <div id="ordersList" class="list"></div>
        </div>
      </section>

      <section id="portfolio" class="section">
        <div class="grid">
          <div class="card panel interactive-sheen">
            <div class="eyebrow">Holdings</div>
            <div id="portfolioList" class="list"></div>
          </div>
          <div class="card panel interactive-sheen">
            <div class="eyebrow">Risk Profiles</div>
            <div id="riskProfileList" class="list"></div>
          </div>
        </div>
      </section>

      <section id="backtest" class="section">
        <div class="grid">
          <div class="card chart-card interactive-sheen">
            <div class="chart-head">
              <div>
                <div class="eyebrow">Backtest Curve</div>
                <div class="chart-title">Equity trajectory</div>
                <div class="muted">A simple replay based on the same candles used for analysis.</div>
              </div>
            </div>
            <canvas id="equityChart" width="900" height="320"></canvas>
          </div>
          <div class="card panel interactive-sheen">
            <div class="eyebrow">Trade Log</div>
            <div id="tradeLogList" class="list"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const userId = "demo-user";
    const savedTheme = localStorage.getItem("groww-ai-theme") || "light";
    document.body.setAttribute("data-theme", savedTheme);
    let latestRecommendation = null;
    let chartZoom = 60;
    let visibleCandles = [];
    let hoveredCandle = null;

    document.addEventListener("pointermove", (event) => {
      const x = (event.clientX / window.innerWidth) * 100;
      const y = (event.clientY / window.innerHeight) * 100;
      document.documentElement.style.setProperty("--pointer-x", x.toFixed(2) + "%");
      document.documentElement.style.setProperty("--pointer-y", y.toFixed(2) + "%");
      document.documentElement.style.setProperty("--float-x", ((x - 50) / 50).toFixed(2));
      document.documentElement.style.setProperty("--float-y", ((y - 50) / 50).toFixed(2));
    });

    for (const button of document.querySelectorAll(".nav button")) {
      button.addEventListener("click", () => switchSection(button.dataset.section));
    }

    document.getElementById("themeToggle").addEventListener("click", toggleTheme);
    document.getElementById("analyzeButton").addEventListener("click", () => analyze());
    document.getElementById("watchButton").addEventListener("click", () => watch());
    document.getElementById("scanButton").addEventListener("click", () => scanWatchlist());
    document.getElementById("refreshButton").addEventListener("click", () => refreshDesk());
    for (const chip of document.querySelectorAll(".ask-chip")) {
      chip.addEventListener("click", () => handleFollowUpAction(chip.dataset.followup));
    }
    document.getElementById("timeframeSelect").addEventListener("change", () => {
      chartZoom = Number(document.getElementById("timeframeSelect").value);
      if (latestRecommendation) renderAnalysis({ recommendation: latestRecommendation, explanationText: document.getElementById("explanationBox").textContent });
    });

    const priceChart = document.getElementById("priceChart");
    priceChart.addEventListener("mousemove", handleChartHover);
    priceChart.addEventListener("mouseleave", hideTooltip);
    priceChart.addEventListener("wheel", (event) => {
      event.preventDefault();
      chartZoom = event.deltaY > 0 ? Math.min(chartZoom + 5, 60) : Math.max(chartZoom - 5, 10);
      document.getElementById("timeframeSelect").value = nearestTimeframe(chartZoom);
      if (latestRecommendation) renderAnalysis({ recommendation: latestRecommendation, explanationText: document.getElementById("explanationBox").textContent });
    }, { passive: false });

    analyze();
    refreshDesk();
    syncThemeLabel();

    function nearestTimeframe(value) {
      if (value <= 15) return "15";
      if (value <= 30) return "30";
      return "60";
    }

    function toggleTheme() {
      const nextTheme = document.body.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.body.setAttribute("data-theme", nextTheme);
      localStorage.setItem("groww-ai-theme", nextTheme);
      syncThemeLabel();
      if (visibleCandles.length) drawPriceChart(visibleCandles);
      if (latestRecommendation) {
        drawDetailChart(latestRecommendation.snapshot.candles.slice(-chartZoom));
        drawEquityChart(latestRecommendation.backtestSummary.equityCurve || []);
      }
    }

    function syncThemeLabel() {
      const isDark = document.body.getAttribute("data-theme") === "dark";
      document.getElementById("themeToggle").textContent = isDark ? "Switch to Light Mode" : "Switch to Dark Mode";
    }

    function switchSection(sectionId) {
      for (const section of document.querySelectorAll(".section")) {
        section.classList.toggle("active", section.id === sectionId);
      }
      for (const button of document.querySelectorAll(".nav button")) {
        button.classList.toggle("active", button.dataset.section === sectionId);
      }
    }

    function pulseNode(nodeId) {
      const node = document.getElementById(nodeId);
      if (!node) return;
      node.classList.remove("pulse-update");
      void node.offsetWidth;
      node.classList.add("pulse-update");
    }

    function updateText(nodeId, value, options = {}) {
      const node = document.getElementById(nodeId);
      if (!node) return;
      const nextValue = String(value);
      if (node.textContent !== nextValue) {
        node.textContent = nextValue;
        if (options.pulse !== false) pulseNode(nodeId);
      }
    }

    function updateHtml(nodeId, value) {
      const node = document.getElementById(nodeId);
      if (!node) return;
      if (node.innerHTML !== value) {
        node.innerHTML = value;
      }
    }

    async function handleFollowUpAction(action) {
      if (action === "scan") {
        updateText("followUpBox", "Scanning your watchlist for fresh ideas...");
        await scanWatchlist();
        updateText("followUpBox", "Watchlist scan finished. Check the watchlist alerts panel for updated buy or sell prompts.", { pulse: false });
        return;
      }

      if (!latestRecommendation) {
        updateText("followUpBox", "Run an analysis first so I can answer a meaningful follow-up.", { pulse: false });
        return;
      }

      if (action === "why") {
        switchSection("stock");
        updateText("followUpBox", "Opened the stock details view so visitors can see the AI reasoning in plain language.", { pulse: false });
        return;
      }

      if (action === "risk") {
        switchSection("dashboard");
        updateText("followUpBox", "Risk context is highlighted in the trade plan card with entry, stop, target, and caution flags.", { pulse: false });
        pulseNode("riskReasons");
        return;
      }

      if (action === "watch") {
        await watch();
        updateText("followUpBox", latestRecommendation.symbol + " was added to the watchlist so a visitor can keep following the idea.", { pulse: false });
      }
    }

    function renderDataBadge(snapshot) {
      const badge = document.getElementById("dataModeBadge");
      const isLive = snapshot.sourceMode === "live";
      badge.className = "data-badge " + (isLive ? "live" : "demo");
      badge.firstChild.textContent = isLive ? "Live Data " : "Demo Data ";
      updateText("dataProviderText", snapshot.provider || (isLive ? "Yahoo Finance" : "Mock feed"), { pulse: false });
    }

    async function analyze() {
      const symbol = document.getElementById("symbolInput").value.trim();
      if (!symbol) return;
      const response = await fetch("/api/analyze?symbol=" + encodeURIComponent(symbol) + "&userId=" + encodeURIComponent(userId));
      const payload = await response.json();
      renderAnalysis(payload);
      await refreshDesk();
      switchSection("stock");
    }

    async function watch() {
      const symbol = document.getElementById("symbolInput").value.trim();
      if (!symbol) return;
      await fetch("/api/watch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, userId })
      });
      document.getElementById("alertsBox").textContent = symbol.toUpperCase() + " added to watchlist.";
      await refreshDesk();
      switchSection("watchlist");
    }

    async function unwatch(symbol) {
      await fetch("/api/unwatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, userId })
      });
      await refreshDesk();
    }

    async function approve(orderId) {
      await fetch("/api/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ orderId, userId })
      });
      await refreshDesk();
      switchSection("orders");
    }

    async function reject(orderId) {
      await fetch("/api/reject", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ orderId, userId })
      });
      await refreshDesk();
      switchSection("orders");
    }

    async function scanWatchlist() {
      const response = await fetch("/api/scan-watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId })
      });
      const payload = await response.json();
      const alerts = payload.alerts || [];
      document.getElementById("alertsBox").textContent = alerts.length
        ? alerts.map((item) => item.recommendation.symbol + ": " + item.recommendation.decision.action.toUpperCase() + " | confidence " + item.recommendation.decision.confidence).join("\\n")
        : "No fresh buy or sell alerts on the current watchlist scan.";
      await refreshDesk();
      switchSection("watchlist");
    }

    async function refreshDesk() {
      const [payload, discover] = await Promise.all([
        fetch("/api/desk?userId=" + encodeURIComponent(userId)).then((response) => response.json()),
        fetch("/api/discover").then((response) => response.json())
      ]);
      updateText("buyingPowerStat", payload.portfolio.buyingPower ?? "-");
      renderDiscoverList("topGainersList", discover.topGainers || []);
      renderDiscoverList("topLosersList", discover.topLosers || []);
      renderDiscoverList("aiPicksList", discover.aiPicks || []);
      renderDiscoverTicker(discover);
      renderOrders(payload.orders || []);
      renderWatchlist(payload.watchlist || []);
      renderPortfolio(payload.portfolio);
      renderRiskProfiles(payload.riskProfiles || []);
      if (!latestRecommendation && payload.latestRecommendation) {
        latestRecommendation = payload.latestRecommendation;
        renderAnalysis({ recommendation: latestRecommendation, explanationText: document.getElementById("explanationBox").textContent });
      }
    }

    function renderAnalysis(payload) {
      const recommendation = payload.recommendation;
      if (!recommendation) return;
      latestRecommendation = recommendation;
      visibleCandles = recommendation.snapshot.candles.slice(-chartZoom);
      hoveredCandle = null;
      renderDataBadge(recommendation.snapshot);
      updateText("symbolTitle", recommendation.symbol + " (" + recommendation.market + ")");
      updateText("metaLine", "Price " + recommendation.snapshot.quote.price + " | Open " + recommendation.snapshot.quote.open + " | Change " + recommendation.snapshot.quote.changePct + "% | Trend " + recommendation.decision.indicators.trend + " | Timeframe " + recommendation.decision.timeframe, { pulse: false });
      updateText("pricePill", recommendation.snapshot.quote.changePct + "%");
      document.getElementById("pricePill").className = "pill " + (recommendation.snapshot.quote.changePct >= 0 ? "positive" : "negative");
      updateText("actionStat", recommendation.decision.action.toUpperCase());
      updateText("confidenceStat", recommendation.decision.confidence);
      updateText("executionStat", recommendation.riskAssessment.executionMode);
      updateText("entryStat", recommendation.decision.entryPrice);
      updateText("stopStat", recommendation.decision.stopLoss);
      updateText("targetStat", recommendation.decision.targetPrice);
      updateText("sizeStat", recommendation.decision.suggestedQuantity);
      updateText("riskReasons", recommendation.riskAssessment.reasons.length ? recommendation.riskAssessment.reasons.join(" ") : "No extra risk flags.", { pulse: false });
      updateText("backtestReturn", recommendation.backtestSummary.totalReturnPct + "%");
      updateText("backtestWinRate", recommendation.backtestSummary.winRatePct + "%");
      updateText("backtestDrawdown", recommendation.backtestSummary.maxDrawdownPct + "%");
      updateText("backtestSignals", recommendation.backtestSummary.signalChanges);
      updateText("backtestRead", "Replay bias is " + recommendation.backtestSummary.latestBias + " with " + recommendation.backtestSummary.signalChanges + " regime changes across " + recommendation.backtestSummary.bars + " bars.", { pulse: false });
      updateText("explanationBox", payload.explanationText || document.getElementById("explanationBox").textContent, { pulse: false });
      renderTradeLog(recommendation.backtestSummary.tradeLog || []);
      drawPriceChart(visibleCandles);
      drawEquityChart(recommendation.backtestSummary.equityCurve || []);
      renderStockDetails(recommendation, payload.explanationText || "");
    }

    function renderOrders(orders) {
      const node = document.getElementById("ordersList");
      node.innerHTML = orders.length ? orders.map((order) => {
        const actions = order.status === "pending_approval" ? "<div class='item-actions'><button onclick=\\"approve('" + order.id + "')\\">Approve</button><button onclick=\\"reject('" + order.id + "')\\">Reject</button></div>" : "";
        return "<div class='item'><div class='item-head'><div><strong>" + order.symbol + "</strong> " + order.action.toUpperCase() + " " + order.quantity + " @ " + order.price + "</div><div class='pill order-status status-" + order.status + "'>" + formatOrderStatus(order.status) + "</div></div>" + actions + "</div>";
      }).join("") : "<div class='item'>No orders yet.</div>";
    }

    function formatOrderStatus(status) {
      return String(status || "")
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
    }

    function renderWatchlist(items) {
      const node = document.getElementById("watchlistList");
      node.innerHTML = items.length ? items.map((item) =>
        "<div class='item'><div class='item-head'><div><strong>" + item.symbol + "</strong><div class='muted'>" + item.market + " market | added " + new Date(item.createdAt).toLocaleString() + "</div></div><div class='item-actions'><button onclick=\\"document.getElementById('symbolInput').value='" + item.symbol + "'; analyze();\\">Analyze</button><button onclick=\\"unwatch('" + item.symbol + "')\\">Remove</button></div></div></div>"
      ).join("") : "<div class='item'>No watched stocks yet.</div>";
    }

    function renderPortfolio(portfolio) {
      const node = document.getElementById("portfolioList");
      const positions = portfolio.positions || [];
      node.innerHTML = positions.length ? positions.map((position) =>
        "<div class='item'><div class='item-head'><div><strong>" + position.symbol + "</strong><div class='muted'>" + position.market + " market</div></div><div><strong>" + position.quantity + "</strong> @ " + position.averagePrice + "</div></div></div>"
      ).join("") : "<div class='item'>No holdings yet.</div>";
    }

    function renderRiskProfiles(profiles) {
      const node = document.getElementById("riskProfileList");
      node.innerHTML = profiles.map((profile) =>
        "<div class='item'><strong>" + profile.market + "</strong><div class='muted'>Auto-trade up to " + profile.maxAutoTradeValue + " | Max position " + profile.maxPositionValue + " | Confidence threshold " + profile.confidenceThreshold + "</div></div>"
      ).join("");
    }

    function renderTradeLog(entries) {
      const node = document.getElementById("tradeLogList");
      node.innerHTML = entries.length ? entries.slice().reverse().map((entry) =>
        "<div class='item'><div class='item-head'><div><strong>" + entry.action + "</strong><div class='muted'>" + new Date(entry.timestamp).toLocaleString() + "</div></div><div class='pill'>" + entry.bias + "</div></div><div class='muted'>Replay price " + entry.price + "</div></div>"
      ).join("") : "<div class='item'>No replay trade transitions yet.</div>";
    }

    function renderStockDetails(recommendation, explanationText) {
      updateText("detailTitle", recommendation.symbol + " (" + recommendation.market + ")");
      updateText("detailMeta",
        "Live price " + recommendation.snapshot.quote.price +
        " | Open " + recommendation.snapshot.quote.open +
        " | Change " + recommendation.snapshot.quote.changePct + "%" +
        " | Trend " + recommendation.decision.indicators.trend,
      { pulse: false });
      updateText("detailActionPill", recommendation.decision.action.toUpperCase());
      document.getElementById("detailActionPill").className = "pill " + (recommendation.snapshot.quote.changePct >= 0 ? "positive" : "negative");
      updateText("detailRecommendation", recommendation.decision.action.toUpperCase());
      updateText("detailConfidence", recommendation.decision.confidence);
      updateText("detailEntry", recommendation.decision.entryPrice);
      updateText("detailTarget", recommendation.decision.targetPrice);
      updateText("detailWhy", explanationText || "No explanation available.", { pulse: false });
      updateText("detailFollowUp",
        recommendation.decision.action === "buy"
          ? "Buy zone active. Watch for target " + recommendation.decision.targetPrice + " and guard with stop " + recommendation.decision.stopLoss + "."
          : recommendation.decision.action === "sell"
            ? "Weakness detected. Consider exit or reduce exposure while the trend remains soft."
            : recommendation.decision.action === "hold"
              ? "Current setup favors patience. Hold existing position and wait for stronger confirmation."
              : "The AI is asking you to wait. Let volatility cool before taking the next step.",
      { pulse: false });
      drawDetailChart(recommendation.snapshot.candles.slice(-chartZoom));
    }

    function renderDiscoverList(nodeId, items) {
      const node = document.getElementById(nodeId);
      node.innerHTML = items.length ? items.map((item) =>
        "<div class='item stock-card' onclick=\\"openStock('" + item.symbol + "')\\"><div class='item-head'><div><strong>" + item.symbol + "</strong><div class='muted'>" + item.market + "</div></div><div class='pill " + (item.changePct >= 0 ? "positive" : "negative") + "'>" + item.changePct + "%</div></div><div class='stock-price'>" + item.price + "</div><div class='stock-meta'><span class='muted'>" + item.action.toUpperCase() + "</span><span class='muted'>Conf " + item.confidence + "</span></div></div>"
      ).join("") : "<div class='item'>No stocks available.</div>";
    }

    function renderDiscoverTicker(discover) {
      const tickerItems = [
        ...(discover.topGainers || []),
        ...(discover.aiPicks || []),
        ...(discover.topLosers || [])
      ];
      const content = tickerItems.length
        ? tickerItems.map((item) =>
          "<div class='ticker-pill'><span class='ticker-dot'></span><strong>" + item.symbol + "</strong><span>" + item.market + "</span><span>" + item.action.toUpperCase() + "</span><span class='" + (item.changePct >= 0 ? "positive" : "negative") + "'>" + item.changePct + "%</span></div>"
        ).join("")
        : "<div class='ticker-pill'>Waiting for live discovery ideas...</div>";
      updateHtml("discoverTicker", content + content);
    }

    function openStock(symbol) {
      document.getElementById("symbolInput").value = symbol;
      analyze();
    }

    function drawPriceChart(candles) {
      const canvas = document.getElementById("priceChart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!candles.length) return;
      const highs = candles.map((c) => c.high);
      const lows = candles.map((c) => c.low);
      const min = Math.min(...lows);
      const max = Math.max(...highs);
      const padding = 26;

      ctx.strokeStyle = "rgba(20,32,51,0.08)";
      for (let i = 0; i < 5; i++) {
        const y = padding + (i * (canvas.height - padding * 2) / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
      }

      const candleWidth = Math.max(4, ((canvas.width - padding * 2) / candles.length) * 0.58);
      candles.forEach((candle, index) => {
        const x = padding + (index * (canvas.width - padding * 2) / Math.max(candles.length - 1, 1));
        const openY = valueToY(candle.open, min, max, canvas.height, padding);
        const closeY = valueToY(candle.close, min, max, canvas.height, padding);
        const highY = valueToY(candle.high, min, max, canvas.height, padding);
        const lowY = valueToY(candle.low, min, max, canvas.height, padding);
        const color = candle.close >= candle.open ? "#00b386" : "#e04f5f";
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillRect(x - candleWidth / 2, Math.min(openY, closeY), candleWidth, Math.max(3, Math.abs(closeY - openY)));
        if (hoveredCandle === index) {
          ctx.strokeStyle = "#2f6fec";
          ctx.beginPath();
          ctx.moveTo(x, padding);
          ctx.lineTo(x, canvas.height - padding);
          ctx.stroke();
        }
      });
    }

    function drawEquityChart(points) {
      const canvas = document.getElementById("equityChart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!points.length) return;
      const values = points.map((point) => point.equity);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const padding = 26;
      ctx.strokeStyle = "rgba(20,32,51,0.08)";
      for (let i = 0; i < 5; i++) {
        const y = padding + (i * (canvas.height - padding * 2) / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
      }
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = padding + (index * (canvas.width - padding * 2) / Math.max(points.length - 1, 1));
        const y = valueToY(point.equity, min, max, canvas.height, padding);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = "#00b386";
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    function drawDetailChart(candles) {
      const canvas = document.getElementById("detailChart");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!candles.length) return;
      const highs = candles.map((c) => c.high);
      const lows = candles.map((c) => c.low);
      const min = Math.min(...lows);
      const max = Math.max(...highs);
      const padding = 26;

      ctx.strokeStyle = "rgba(20,32,51,0.08)";
      for (let i = 0; i < 5; i++) {
        const y = padding + (i * (canvas.height - padding * 2) / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
      }

      const candleWidth = Math.max(4, ((canvas.width - padding * 2) / candles.length) * 0.58);
      candles.forEach((candle, index) => {
        const x = padding + (index * (canvas.width - padding * 2) / Math.max(candles.length - 1, 1));
        const openY = valueToY(candle.open, min, max, canvas.height, padding);
        const closeY = valueToY(candle.close, min, max, canvas.height, padding);
        const highY = valueToY(candle.high, min, max, canvas.height, padding);
        const lowY = valueToY(candle.low, min, max, canvas.height, padding);
        const color = candle.close >= candle.open ? "#00b386" : "#e04f5f";
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillRect(x - candleWidth / 2, Math.min(openY, closeY), candleWidth, Math.max(3, Math.abs(closeY - openY)));
      });
    }

    function handleChartHover(event) {
      if (!visibleCandles.length) return;
      const canvas = event.currentTarget;
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const padding = 26;
      const ratio = Math.max(0, Math.min(1, (x - padding) / Math.max(canvas.width - padding * 2, 1)));
      hoveredCandle = Math.max(0, Math.min(visibleCandles.length - 1, Math.round(ratio * (visibleCandles.length - 1))));
      drawPriceChart(visibleCandles);
      const candle = visibleCandles[hoveredCandle];
      const tooltip = document.getElementById("priceTooltip");
      tooltip.style.opacity = "1";
      tooltip.style.left = event.offsetX + "px";
      tooltip.style.top = event.offsetY + "px";
      tooltip.innerHTML = new Date(candle.timestamp).toLocaleString() + "<br>O " + candle.open + " | H " + candle.high + "<br>L " + candle.low + " | C " + candle.close;
    }

    function hideTooltip() {
      hoveredCandle = null;
      document.getElementById("priceTooltip").style.opacity = "0";
      if (visibleCandles.length) drawPriceChart(visibleCandles);
    }

    function valueToY(value, min, max, canvasHeight, padding) {
      return canvasHeight - padding - ((value - min) / Math.max(max - min, 0.0001)) * (canvasHeight - padding * 2);
    }
  </script>
</body>
</html>`;
}
