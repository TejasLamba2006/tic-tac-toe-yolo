"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  datasetInfo: null, // last successful /api/dataset/* response
  configPath: null, // path returned by /api/config/generate
  runId: null,
  ws: null,
};

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// ---------------------------------------------------------------------------
// Dataset tab
// ---------------------------------------------------------------------------
const scanPathBtn = document.getElementById("scan-path-btn");
const datasetPathInput = document.getElementById("dataset-path-input");
const dropzone = document.getElementById("dropzone");
const zipInput = document.getElementById("zip-input");
const dropzoneFilename = document.getElementById("dropzone-filename");
const summaryCard = document.getElementById("dataset-summary-card");
const summaryBody = document.getElementById("dataset-summary-body");
const splitCard = document.getElementById("dataset-split-card");
const splitBtn = document.getElementById("split-dataset-btn");
const splitStatus = document.getElementById("split-dataset-status");

scanPathBtn.addEventListener("click", async () => {
  const path = datasetPathInput.value.trim();
  if (!path) return;
  setBusy(scanPathBtn, true, "Scanning...");
  try {
    const res = await postJSON("/api/dataset/select-path", { path });
    onDatasetInfo(res);
  } catch (err) {
    alert(`Scan failed: ${err.message}`);
  } finally {
    setBusy(scanPathBtn, false, "Scan");
  }
});

dropzone.addEventListener("click", () => zipInput.click());
["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadZip(file);
});
zipInput.addEventListener("change", () => {
  const file = zipInput.files[0];
  if (file) uploadZip(file);
});

async function uploadZip(file) {
  dropzoneFilename.textContent = `Uploading ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const res = await fetch("/api/dataset/upload", { method: "POST", body: formData });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const info = await res.json();
    dropzoneFilename.textContent = `Loaded: ${file.name}`;
    onDatasetInfo(info);
  } catch (err) {
    dropzoneFilename.textContent = "";
    alert(`Upload failed: ${err.message}`);
  }
}

function onDatasetInfo(info) {
  state.datasetInfo = info;
  summaryCard.hidden = false;

  const splitRows = Object.entries(info.splits || {})
    .map(([split, d]) => `<li><strong>${split}</strong>: ${d.count} images (${d.images_dir})</li>`)
    .join("") || "<li>No train/valid/test split detected &mdash; will auto-split at runtime.</li>";

  summaryBody.innerHTML = `
    <p class="hint">Root: <span class="mono-small">${escapeHtml(info.root)}</span></p>
    <ul>${splitRows}</ul>
    <p class="hint">Total images: <strong>${info.total_images}</strong></p>
    <p class="hint">data.yaml: ${info.data_yaml ? `<span class="mono-small">${escapeHtml(info.data_yaml)}</span>` : '<em>not found &mdash; class names will be requested in Configuration</em>'}</p>
    ${info.class_names && info.class_names.length ? `<p class="hint">Classes: ${info.class_names.map(escapeHtml).join(", ")}</p>` : ""}
  `;

  document.getElementById("config-no-dataset-warning").classList.add("hidden");
  document.getElementById("config-form").classList.remove("hidden");
  document.getElementById("class-names-field").hidden = !!info.data_yaml;

  splitCard.classList.toggle("hidden", !!info.has_split_layout);
  splitStatus.textContent = "";

  // Prefill dataset name from folder name.
  const nameInput = document.querySelector('input[name="dataset_name"]');
  if (!nameInput.value) {
    const parts = info.root.replace(/\\/g, "/").split("/");
    nameInput.value = parts[parts.length - 1] || "dataset";
  }
}

splitBtn.addEventListener("click", async () => {
  if (!state.datasetInfo) return;
  const payload = {
    path: state.datasetInfo.root,
    train: Number(document.getElementById("split-train-ratio").value),
    val: Number(document.getElementById("split-val-ratio").value),
    test: Number(document.getElementById("split-test-ratio").value),
    seed: Number(document.getElementById("split-seed").value),
  };
  setBusy(splitBtn, true, "Splitting...");
  splitStatus.textContent = "";
  try {
    const info = await postJSON("/api/dataset/split", payload);
    onDatasetInfo(info);
    splitStatus.textContent = "Dataset split complete.";
  } catch (err) {
    alert(`Split failed: ${err.message}`);
  } finally {
    setBusy(splitBtn, false, "Split Dataset");
  }
});

// ---------------------------------------------------------------------------
// Config tab
// ---------------------------------------------------------------------------
const configForm = document.getElementById("config-form");
const deployToggle = document.getElementById("deploy-stedge-toggle");
const stedgeFields = document.getElementById("stedge-fields");
const configPreviewCard = document.getElementById("config-preview-card");
const configPreview = document.getElementById("config-preview");
const configGenerateStatus = document.getElementById("config-generate-status");

deployToggle.addEventListener("change", () => {
  stedgeFields.classList.toggle("hidden", !deployToggle.checked);
});

configForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.datasetInfo) {
    alert("Scan or upload a dataset first.");
    return;
  }
  const fd = new FormData(configForm);
  const payload = {
    dataset_info: state.datasetInfo,
    dataset_name: fd.get("dataset_name"),
    epochs: Number(fd.get("epochs")),
    batch_size: Number(fd.get("batch_size")),
    imgsz: Number(fd.get("imgsz")),
    device: fd.get("device") || "cpu",
    workers: Number(fd.get("workers")),
    enable_int8: fd.get("enable_int8") === "on",
    class_names: fd.get("class_names") || "",
    deploy_stedge: fd.get("deploy_stedge") === "on",
    stedge_target: fd.get("stedge_target") || "",
    stedge_username: fd.get("stedge_username") || "",
    stedge_password: fd.get("stedge_password") || "",
  };

  configGenerateStatus.textContent = "Generating...";
  try {
    const res = await postJSON("/api/config/generate", payload);
    state.configPath = res.config_path;
    configPreview.textContent = res.preview;
    configPreviewCard.classList.remove("hidden");
    configGenerateStatus.textContent = `Saved to ${res.config_path}`;
    document.getElementById("active-config-path").textContent = res.config_path;
    document.getElementById("config-path-row").hidden = false;
    document.getElementById("start-run-btn").disabled = false;

    const preflightInput = document.getElementById("preflight-config-input");
    const resultsInput = document.getElementById("results-config-input");
    if (!preflightInput.value) preflightInput.value = res.config_path;
    if (!resultsInput.value) resultsInput.value = res.config_path;
  } catch (err) {
    configGenerateStatus.textContent = "";
    alert(`Config generation failed: ${err.message}`);
  }
});

// ---------------------------------------------------------------------------
// Run tab
// ---------------------------------------------------------------------------
const startBtn = document.getElementById("start-run-btn");
const stopBtn = document.getElementById("stop-run-btn");
const clearLogBtn = document.getElementById("clear-log-btn");
const terminalEl = document.getElementById("terminal");
const runStatusBadge = document.getElementById("run-status-badge");
const globalStatusBadge = document.getElementById("global-status");

startBtn.addEventListener("click", async () => {
  if (!state.configPath) return;
  setBusy(startBtn, true, "Starting...");
  try {
    const res = await postJSON("/api/pipeline/run", { config_path: state.configPath });
    state.runId = res.run_id;
    setStatus("running");
    stopBtn.disabled = false;
    connectLogSocket(res.run_id);
    // Auto-switch to Run tab so the user sees output immediately.
    document.querySelector('.nav-item[data-tab="run"]').click();
  } catch (err) {
    alert(`Failed to start pipeline: ${err.message}`);
  } finally {
    setBusy(startBtn, false, "▶ Start Pipeline");
  }
});

stopBtn.addEventListener("click", async () => {
  if (!state.runId) return;
  setBusy(stopBtn, true, "Stopping...");
  try {
    await fetch(`/api/pipeline/stop/${state.runId}`, { method: "POST" });
  } catch (err) {
    console.error(err);
  } finally {
    setBusy(stopBtn, false, "■ Stop");
  }
});

clearLogBtn.addEventListener("click", () => {
  terminalEl.textContent = "";
  lastLineEl = null;
});

function connectLogSocket(runId) {
  if (state.ws) state.ws.close();
  lastLineEl = null;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/logs/${runId}`);
  state.ws = ws;

  ws.onmessage = (evt) => {
    const frame = JSON.parse(evt.data);
    if (frame.type === "line") {
      appendLogLine(frame.line, false);
    } else if (frame.type === "replace") {
      // In-place progress-bar rewrite (tqdm/Ultralytics use a bare "\r"
      // to redraw the current line) -- update rather than append.
      appendLogLine(frame.line, true);
    } else if (frame.type === "event" && frame.event === "done") {
      setStatus(frame.status);
      startBtn.disabled = false;
      stopBtn.disabled = true;
    } else if (frame.type === "event" && frame.event === "error") {
      appendLogLine(`[dashboard] ${frame.message}`, false);
    }
  };
  ws.onclose = () => {
    if (state.ws === ws) state.ws = null;
  };
}

// Tracks the DOM node most recently written, so a "replace" frame can
// update it in place instead of appending a new line.
let lastLineEl = null;

function appendLogLine(line, replace) {
  const atBottom =
    terminalEl.scrollTop + terminalEl.clientHeight >= terminalEl.scrollHeight - 30;
  let el = replace && lastLineEl ? lastLineEl : null;
  if (!el) {
    el = document.createElement("div");
    terminalEl.appendChild(el);
  }
  el.innerHTML = colorizeLogLine(line);
  lastLineEl = el;
  if (atBottom) terminalEl.scrollTop = terminalEl.scrollHeight;
}

// Renders plain pipeline log text with terminal-style coloring on the
// client. The pipeline itself emits plain (uncolored) text here because
// its own ANSI colorizer only activates on a real TTY (see
// pipeline/utils/logging.py) -- stdout is a pipe when run as a
// subprocess, so nothing ships from the server to color. We recognize
// the same "TIMESTAMP | LEVEL | logger.name | message" shape it always
// writes and re-apply level-based coloring here instead.
const LOG_LINE_RE = /^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(.*)$/;

function colorizeLogLine(rawLine) {
  const match = rawLine.match(LOG_LINE_RE);
  if (match) {
    const [, ts, level, name, msg] = match;
    const levelClass = `log-level-${level.toLowerCase()}`;
    return (
      `<span class="log-ts">${escapeHtml(ts)}</span>` +
      ` <span class="log-pipe">|</span> ` +
      `<span class="log-level ${levelClass}">${escapeHtml(level)}</span>` +
      ` <span class="log-pipe">|</span> ` +
      `<span class="log-name">${escapeHtml(name)}</span>` +
      ` <span class="log-pipe">|</span> ` +
      `<span class="${levelClass}-msg">${escapeHtml(msg)}</span>`
    );
  }

  const escaped = escapeHtml(rawLine);
  if (rawLine.startsWith("$ ")) return `<span class="log-cmd">${escaped}</span>`;
  if (rawLine.startsWith("[dashboard]")) return `<span class="log-dashboard">${escaped}</span>`;
  if (/^\s*(Traceback \(most recent call last\)|File "|\.\.\.|\^+\s*$)/.test(rawLine) || /^\s{2,}\S/.test(rawLine)) {
    return `<span class="log-traceback">${escaped}</span>`;
  }
  if (/\b(FAIL|ERROR|Exception)\b/.test(rawLine)) {
    return `<span class="log-level-error-msg">${escaped}</span>`;
  }
  if (/\bSKIP\b/.test(rawLine)) return `<span class="log-level-warning-msg">${escaped}</span>`;
  return escaped;
}

function setStatus(status) {
  const map = {
    idle: ["Idle", "badge-idle"],
    starting: ["Starting", "badge-running"],
    running: ["Running", "badge-running"],
    success: ["Success", "badge-success"],
    failed: ["Failed", "badge-failed"],
    stopped: ["Stopped", "badge-stopped"],
  };
  const [label, cls] = map[status] || map.idle;
  for (const badge of [runStatusBadge, globalStatusBadge]) {
    badge.textContent = label;
    badge.className = `badge ${cls}`;
  }
}

// ---------------------------------------------------------------------------
// Preflight tab
// ---------------------------------------------------------------------------
const preflightConfigInput = document.getElementById("preflight-config-input");
const runPreflightBtn = document.getElementById("run-preflight-btn");
const preflightResultsCard = document.getElementById("preflight-results-card");
const preflightOverallBadge = document.getElementById("preflight-overall-badge");
const preflightOverallText = document.getElementById("preflight-overall-text");
const preflightPython = document.getElementById("preflight-python");
const preflightPackages = document.getElementById("preflight-packages");
const preflightDisk = document.getElementById("preflight-disk");

runPreflightBtn.addEventListener("click", async () => {
  const configPath = preflightConfigInput.value.trim() || state.configPath || "";
  if (!configPath) {
    alert("Enter a config path (or generate one in the Configuration tab first).");
    return;
  }
  preflightConfigInput.value = configPath;
  setBusy(runPreflightBtn, true, "Running checks...");
  try {
    const res = await postJSON("/api/preflight", { config_path: configPath });
    renderPreflight(res);
  } catch (err) {
    alert(`Preflight failed: ${err.message}`);
  } finally {
    setBusy(runPreflightBtn, false, "Run Checks");
  }
});

function renderCheckRow(status, title, detail, hint) {
  const iconChar = status === "ok" ? "✓" : status === "warn" ? "!" : "✕";
  return `
    <div class="check-row">
      <span class="check-icon ${status}">${iconChar}</span>
      <div>
        <div class="check-title">${escapeHtml(title)}</div>
        ${detail ? `<div class="check-detail">${escapeHtml(String(detail))}</div>` : ""}
        ${hint ? `<div class="check-hint">${escapeHtml(String(hint))}</div>` : ""}
      </div>
    </div>`;
}

function renderPreflight(res) {
  preflightResultsCard.classList.remove("hidden");

  preflightOverallBadge.textContent = res.overall_ok ? "Ready" : "Issues found";
  preflightOverallBadge.className = `badge ${res.overall_ok ? "badge-success" : "badge-failed"}`;
  preflightOverallText.textContent = res.overall_ok
    ? "All checks passed — the pipeline should run cleanly."
    : "Some checks failed — fix the items below before running.";

  preflightPython.innerHTML = `Python <strong>${escapeHtml(res.python.version)}</strong> &mdash; <span class="mono-small">${escapeHtml(res.python.executable)}</span>`;

  preflightPackages.innerHTML = res.packages
    .map((p) => {
      const status = p.ok ? "ok" : "fail";
      const detail = p.available
        ? "installed"
        : p.required
          ? "not installed (required for this config)"
          : "not installed (not required for this config)";
      return renderCheckRow(status, `${p.package} — ${p.label}`, detail, p.ok ? "" : p.install_hint);
    })
    .join("");

  const disk = res.disk;
  preflightDisk.innerHTML = renderCheckRow(
    disk.low_space_warning ? "warn" : "ok",
    `Disk space (${disk.artifacts_dir})`,
    `${disk.free_gb} GB free of ${disk.total_gb} GB`,
    disk.low_space_warning ? "Less than 2 GB free — training/export may fail." : ""
  );

}

// ---------------------------------------------------------------------------
// Results tab
// ---------------------------------------------------------------------------
const resultsConfigInput = document.getElementById("results-config-input");
const loadResultsBtn = document.getElementById("load-results-btn");
const resultsEmpty = document.getElementById("results-empty");
const resultsContent = document.getElementById("results-content");
const resultsScorecards = document.getElementById("results-scorecards");
const benchmarkEmpty = document.getElementById("benchmark-empty");
const benchmarkChartWrap = document.getElementById("benchmark-chart-wrap");

const chartInstances = {};

function renderChart(canvasId, config) {
  const canvas = document.getElementById(canvasId);
  // Destroy any existing instance. Chart.js v4 also binds the instance to
  // the canvas as `canvas.chart`; if our map lost track of it (e.g. after a
  // hot reload), a stale instance would otherwise throw
  // "Canvas is already in use" and silently blank the chart.
  const existing = chartInstances[canvasId] || canvas.chart;
  if (existing) {
    try { existing.destroy(); } catch (e) { /* already gone */ }
  }
  chartInstances[canvasId] = new Chart(canvas, config);
  // If the canvas was rendered while its tab was hidden (display:none),
  // it initialised at 0x0 and draws nothing once shown. Force a resize so
  // it picks up the real dimensions.
  if (canvas.clientWidth === 0) {
    try { chartInstances[canvasId].resize(); } catch (e) { /* noop */ }
  }
}

function chartOptions(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: { display: !!title, text: title, color: "#cbd5e1" },
      legend: { labels: { color: "#cbd5e1" } },
    },
    scales: {
      x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
      y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
    },
  };
}

function lineDataset(label, data, color) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color,
    tension: 0.25,
    pointRadius: 2,
    spanGaps: true,
  };
}

function pickMetric(metrics, ...keys) {
  if (!metrics) return null;
  for (const k of keys) {
    if (metrics[k] !== undefined && metrics[k] !== null) return metrics[k];
  }
  return null;
}

function formatMetric(value) {
  return value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : Number(value).toFixed(3);
}

loadResultsBtn.addEventListener("click", async () => {
  const configPath = resultsConfigInput.value.trim() || state.configPath || "";
  if (!configPath) {
    alert("Enter a config path (or generate one in the Configuration tab first).");
    return;
  }
  resultsConfigInput.value = configPath;
  setBusy(loadResultsBtn, true, "Loading...");
  try {
    const res = await fetch(`/api/results?config_path=${encodeURIComponent(configPath)}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderResults(data);
  } catch (err) {
    alert(`Failed to load results: ${err.message}`);
  } finally {
    setBusy(loadResultsBtn, false, "Load Results");
  }
});

function renderResults(data) {
  const hasAnything = Boolean(
    data.has_report ||
    (data.training_curve && data.training_curve.length) ||
    (data.model_sizes && data.model_sizes.length)
  );
  resultsEmpty.classList.toggle("hidden", hasAnything);
  resultsContent.classList.toggle("hidden", !hasAnything);
  if (!hasAnything) return;

  renderScorecards(data);
  renderStageChart(data.stage_timeline);
  renderTrainingCharts(data.training_curve);
  renderSizeChart(data.model_sizes);
  renderBenchmarkChart(data.benchmark_metrics);
}

function renderScorecards(data) {
  const evalM = data.evaluation_metrics || {};
  const trainM = data.training_metrics || {};
  const cards = [
    ["Precision", pickMetric(evalM, "precision") ?? pickMetric(trainM, "precision", "metrics/precision(B)")],
    ["Recall", pickMetric(evalM, "recall") ?? pickMetric(trainM, "recall", "metrics/recall(B)")],
    ["mAP50", pickMetric(evalM, "mAP50", "map50") ?? pickMetric(trainM, "mAP50", "metrics/mAP50(B)")],
    [
      "mAP50-95",
      pickMetric(evalM, "mAP50_95", "mAP50-95", "map50_95") ??
      pickMetric(trainM, "mAP50-95", "metrics/mAP50-95(B)"),
    ],
    ["F1", pickMetric(evalM, "f1", "f1_score") ?? pickMetric(trainM, "f1")],
  ];
  resultsScorecards.innerHTML = cards
    .map(
      ([label, value]) => `
    <div class="scorecard">
      <div class="scorecard-value">${formatMetric(value)}</div>
      <div class="scorecard-label">${escapeHtml(label)}</div>
    </div>`
    )
    .join("");
}

const STAGE_COLORS = {
  success: "#22c55e",
  failed: "#ef4444",
  skipped: "#94a3b8",
  running: "#3b82f6",
};

function renderStageChart(timeline) {
  if (!timeline || !timeline.length) return;
  renderChart("chart-stages", {
    type: "bar",
    data: {
      labels: timeline.map((s) => s.name),
      datasets: [
        {
          label: "Duration (s)",
          data: timeline.map((s) => s.duration_seconds),
          backgroundColor: timeline.map((s) => STAGE_COLORS[s.status] || "#64748b"),
        },
      ],
    },
    options: chartOptions("Pipeline Stage Timeline"),
  });
}

function renderTrainingCharts(curve) {
  if (!curve || !curve.length) return;
  const epochs = curve.map((r) => r.epoch);
  renderChart("chart-loss", {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        lineDataset("Box Loss", curve.map((r) => r.box_loss), "#f87171"),
        lineDataset("Cls Loss", curve.map((r) => r.cls_loss), "#fbbf24"),
        lineDataset("DFL Loss", curve.map((r) => r.dfl_loss), "#60a5fa"),
      ],
    },
    options: chartOptions("Training Loss"),
  });
  renderChart("chart-map", {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        lineDataset("Precision", curve.map((r) => r.precision), "#34d399"),
        lineDataset("Recall", curve.map((r) => r.recall), "#a78bfa"),
        lineDataset("mAP50", curve.map((r) => r.map50), "#38bdf8"),
        lineDataset("mAP50-95", curve.map((r) => r.map50_95), "#f472b6"),
      ],
    },
    options: chartOptions("Precision / Recall / mAP"),
  });
}

function renderSizeChart(sizes) {
  if (!sizes || !sizes.length) return;
  renderChart("chart-size", {
    type: "bar",
    data: {
      labels: sizes.map((s) => s.label),
      datasets: [
        {
          label: "Size (MB)",
          data: sizes.map((s) => +(s.size_bytes / (1024 * 1024)).toFixed(2)),
          backgroundColor: "#3b82f6",
        },
      ],
    },
    options: chartOptions("Model Size Comparison"),
  });
}

function renderBenchmarkChart(metrics) {
  const entries = Object.entries(metrics || {}).filter(([, v]) => typeof v === "number");
  const hasData = entries.length > 0;
  benchmarkEmpty.classList.toggle("hidden", hasData);
  benchmarkChartWrap.classList.toggle("hidden", !hasData);
  if (!hasData) return;
  renderChart("chart-benchmark", {
    type: "bar",
    data: {
      labels: entries.map(([k]) => k),
      datasets: [
        {
          label: "Value",
          data: entries.map(([, v]) => v),
          backgroundColor: "#22c55e",
        },
      ],
    },
    options: chartOptions("Benchmark Metrics"),
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

function setBusy(btn, busy, label) {
  btn.disabled = busy;
  btn.textContent = label;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
