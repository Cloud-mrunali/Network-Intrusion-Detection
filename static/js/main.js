/**
 * main.js — NIDS Dashboard JavaScript
 * Handles prediction form, chart rendering, metrics table, and alert feed.
 */

// ── Global state ──────────────────────────────────────────────────────────────
let totalChecked = 0;
let totalAttacks = 0;
let totalNormal  = 0;
let confValues   = [];
let attackChart  = null;

const attackCounts = {
  normal: 0, DoS: 0, Probe: 0, R2L: 0, U2R: 0, Other: 0,
};

const FAMILY_COLORS = {
  normal: "#22c55e",
  DoS:    "#ef4444",
  Probe:  "#f59e0b",
  R2L:    "#a855f7",
  U2R:    "#ec4899",
  Other:  "#6b7280",
};

const FAMILY_ICONS = {
  normal: "✅",
  DoS:    "🚫",
  Probe:  "🔎",
  R2L:    "🔓",
  U2R:    "⚠️",
  Other:  "❓",
};

// ── Quick presets ─────────────────────────────────────────────────────────────
const PRESETS = {
  normal: {
    protocol_type: "tcp", service: "http", flag: "SF",
    duration: 0, src_bytes: 215, dst_bytes: 45076,
    count: 1, srv_count: 1, same_srv_rate: 1.0,
    dst_host_count: 255, serror_rate: 0.0, logged_in: 1,
  },
  dos: {
    protocol_type: "tcp", service: "http", flag: "S0",
    duration: 0, src_bytes: 0, dst_bytes: 0,
    count: 511, srv_count: 511, same_srv_rate: 1.0,
    dst_host_count: 255, serror_rate: 1.0, logged_in: 0,
  },
  probe: {
    protocol_type: "tcp", service: "ftp", flag: "REJ",
    duration: 0, src_bytes: 0, dst_bytes: 0,
    count: 229, srv_count: 28, same_srv_rate: 0.12,
    dst_host_count: 229, serror_rate: 0.0, logged_in: 0,
  },
  r2l: {
    protocol_type: "tcp", service: "ftp_data", flag: "SF",
    duration: 1, src_bytes: 491, dst_bytes: 0,
    count: 1, srv_count: 1, same_srv_rate: 1.0,
    dst_host_count: 150, serror_rate: 0.0, logged_in: 0,
  },
};

function loadPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  Object.entries(p).forEach(([k, v]) => {
    const el = document.querySelector(`[name="${k}"]`);
    if (el) el.value = v;
  });
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById("modelStatus");
  try {
    const res  = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      badge.textContent = "Model Ready";
      badge.className   = "badge badge-success";
    } else {
      badge.textContent = "Model Not Loaded";
      badge.className   = "badge badge-warning";
    }
  } catch {
    badge.textContent = "Server Error";
    badge.className   = "badge badge-danger";
  }
}

// ── Prediction form ───────────────────────────────────────────────────────────
document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = document.getElementById("submitBtn");
  btn.disabled    = true;
  btn.textContent = "Analysing…";

  const formData = new FormData(e.target);
  const payload  = {};
  formData.forEach((v, k) => {
    payload[k] = isNaN(v) || v === "" ? v : parseFloat(v);
  });

  try {
    const res  = await fetch("/api/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      showToast(data.error, "danger");
      return;
    }

    renderResult(data);
    updateStats(data);
    addAlertItem(data);
    updateChart();

  } catch (err) {
    showToast("Request failed: " + err.message, "danger");
  } finally {
    btn.disabled    = false;
    btn.textContent = "Analyse Connection";
  }
});

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  const box    = document.getElementById("resultBox");
  const icon   = document.getElementById("resultIcon");
  const label  = document.getElementById("resultLabel");
  const family = document.getElementById("resultFamily");
  const conf   = document.getElementById("resultConf");
  const meta   = document.getElementById("resultMeta");

  const fam = data.attack_family || "Other";

  box.className = `result-box result-${fam.toLowerCase()}`;

  icon.textContent   = FAMILY_ICONS[fam]  || "❓";
  label.textContent  = data.prediction    || "Unknown";
  family.textContent = fam;
  label.style.color  = FAMILY_COLORS[fam] || "#94a3b8";
  conf.textContent   = data.confidence
    ? (data.confidence * 100).toFixed(1) + "%"
    : "—";

  meta.textContent = `Detected at ${new Date(data.timestamp).toLocaleTimeString()} · `
    + (data.is_attack ? "⚠️ INTRUSION DETECTED" : "✅ Benign traffic");

  box.classList.remove("hidden");
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(data) {
  totalChecked++;
  if (data.is_attack) totalAttacks++;
  else                totalNormal++;
  if (data.confidence) confValues.push(data.confidence);

  document.getElementById("totalChecked").textContent = totalChecked;
  document.getElementById("totalAttacks").textContent = totalAttacks;
  document.getElementById("totalNormal").textContent  = totalNormal;

  const avg = confValues.length
    ? (confValues.reduce((a, b) => a + b, 0) / confValues.length * 100).toFixed(1) + "%"
    : "—";
  document.getElementById("avgConf").textContent = avg;

  attackCounts[data.attack_family] = (attackCounts[data.attack_family] || 0) + 1;
}

// ── Alerts feed ───────────────────────────────────────────────────────────────
function addAlertItem(data) {
  const list = document.getElementById("alertsList");
  const fam  = data.attack_family || "Other";
  const el   = document.createElement("div");
  el.className = "alert-item";
  el.style.borderLeftColor = FAMILY_COLORS[fam] || "#6b7280";
  el.innerHTML = `
    <div class="alert-dot" style="background:${FAMILY_COLORS[fam]}"></div>
    <div>
      <span class="alert-label">${data.prediction}</span>
      <span class="muted"> — ${fam}</span>
    </div>
    ${data.confidence ? `<span style="color:${FAMILY_COLORS[fam]}">${(data.confidence*100).toFixed(1)}%</span>` : ""}
    <span class="alert-time">${new Date(data.timestamp).toLocaleTimeString()}</span>
  `;
  // Remove placeholder if present
  const placeholder = list.querySelector("p");
  if (placeholder) placeholder.remove();

  list.insertBefore(el, list.firstChild);
  // Cap at 30
  if (list.children.length > 30) list.removeChild(list.lastChild);
}

// ── Chart ─────────────────────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById("attackChart").getContext("2d");
  attackChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: Object.keys(attackCounts),
      datasets: [{
        data:            Object.values(attackCounts),
        backgroundColor: Object.keys(attackCounts).map(k => FAMILY_COLORS[k]),
        borderColor:     "#1e293b",
        borderWidth: 3,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#94a3b8", font: { size: 12 }, padding: 12 },
        },
      },
    },
  });
}

function updateChart() {
  if (!attackChart) return;
  attackChart.data.datasets[0].data = Object.values(attackCounts);
  attackChart.update();
}

// ── Metrics ───────────────────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch("/api/metrics");
    if (!res.ok) throw new Error("metrics not available");
    const data = await res.json();
    renderMetrics(data);
  } catch {
    document.getElementById("metricsBody").innerHTML =
      "<p class='muted'>Train the model first (run train.py) to see metrics.</p>";
  }
}

function renderMetrics(data) {
  const body = document.getElementById("metricsBody");
  let best = "";
  let bestF1 = -1;
  Object.values(data).forEach(m => {
    if (m.macro_f1 > bestF1) { bestF1 = m.macro_f1; best = m.model; }
  });

  let rows = "";
  Object.values(data).forEach(m => {
    const isBest = m.model === best;
    rows += `
      <tr class="${isBest ? "best-row" : ""}">
        <td>${m.model}${isBest ? " ⭐" : ""}</td>
        <td>${(m.accuracy  * 100).toFixed(2)}%</td>
        <td>${(m.macro_f1  * 100).toFixed(2)}%</td>
        <td>${(m.weighted_f1 * 100).toFixed(2)}%</td>
        <td>${m.roc_auc ? (m.roc_auc * 100).toFixed(2) + "%" : "—"}</td>
      </tr>
    `;
  });

  body.innerHTML = `
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th>
          <th>Macro F1</th><th>Weighted F1</th><th>ROC-AUC</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function toggleMetrics() {
  const content = document.getElementById("metricsContent");
  const arrow   = document.getElementById("metricsArrow");
  const isHidden = content.classList.contains("hidden");
  content.classList.toggle("hidden", !isHidden);
  arrow.textContent = isHidden ? "▲" : "▼";
  if (isHidden) loadMetrics();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position:fixed; bottom:20px; right:20px; z-index:999;
    background:${type === "danger" ? "#ef4444" : "#3b82f6"};
    color:#fff; padding:12px 20px; border-radius:8px;
    font-size:13px; box-shadow:0 4px 20px rgba(0,0,0,.3);
    max-width:300px;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  initChart();
  loadMetrics();
  // Load preset: normal by default
  loadPreset("normal");
});
