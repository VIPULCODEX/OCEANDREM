const CATEGORY_COLOR = {
  Normal: "#3fcf8e",
  Watch: "#e3c23c",
  Warning: "#e58a3a",
  Severe: "#e2543f",
  Extreme: "#b5203a",
};

const CLUSTER_PALETTE = ["#3fcf8e", "#4fa3e3", "#e3c23c", "#e58a3a", "#c463d9", "#e2543f"];

const PLOTLY_DARK = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#e8f5f0", size: 12 },
  margin: { t: 10, r: 20, l: 50, b: 40 },
};

let DATA = null;
let frameIdx = 0;
let playing = true;
let speed = 1;
let timer = null;

let REAL = null;
let realFrameIdx = 0;
let realPlaying = true;
let realTimer = null;

async function boot() {
  const res = await fetch("data.json");
  DATA = await res.json();

  setupTabs();
  populateSelectors();
  drawClaimBanner();
  drawStatic();
  renderFrame(0);
  startLoop();

  document.getElementById("playBtn").addEventListener("click", togglePlay);
  document.getElementById("speedBtn").addEventListener("click", cycleSpeed);
  document.getElementById("clusterToggle").addEventListener("change", () => renderFrame(frameIdx));
  document.getElementById("daySelect").addEventListener("change", (e) => {
    const idx = DATA.grids.findIndex((g) => g.day === Number(e.target.value));
    if (idx >= 0) renderFrame(idx);
  });
  document.getElementById("depthSelect").addEventListener("change", drawScatter);
  document.getElementById("profileSelect").addEventListener("change", drawProfile);

  bootReal();
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  function showTab(name) {
    buttons.forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    panels.forEach((p) => p.classList.toggle("active", p.dataset.tab === name));
    positionLiquidIndicator();
    // Plotly charts drawn while their tab was hidden render at zero width;
    // a resize event after the panel becomes visible fixes them (all our
    // charts use responsive:true, which listens for this).
    requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
  }

  buttons.forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  document.querySelectorAll("[data-tab-link]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.preventDefault();
      showTab(el.dataset.tabLink);
    });
  });

  positionLiquidIndicator();
  window.addEventListener("resize", positionLiquidIndicator);
}

function positionIndicator(navId, indicatorId, btnSelector) {
  const nav = document.getElementById(navId);
  const indicator = document.getElementById(indicatorId);
  if (!nav || !indicator) return;
  const active = nav.querySelector(btnSelector);
  if (!active) return;

  const newLeft = active.offsetLeft;
  const newWidth = active.offsetWidth;
  const prevLeft = indicator.dataset.left ? parseFloat(indicator.dataset.left) : newLeft;
  const prevWidth = indicator.dataset.width ? parseFloat(indicator.dataset.width) : newWidth;
  indicator.dataset.left = newLeft;
  indicator.dataset.width = newWidth;

  if (Math.abs(prevLeft - newLeft) < 0.5) {
    // Same spot (e.g. just a resize) -- no stretch needed, just settle.
    indicator.style.transition = "left 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), width 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)";
    indicator.style.left = `${newLeft}px`;
    indicator.style.width = `${newWidth}px`;
    return;
  }

  // Real "liquid glass" motion, iOS-style: the pill briefly ELONGATES to
  // bridge its old and new position (like a droplet stretching before it
  // detaches), THEN contracts onto the new tab -- not a plain slide.
  const bridgeLeft = Math.min(prevLeft, newLeft);
  const bridgeRight = Math.max(prevLeft + prevWidth, newLeft + newWidth);

  indicator.style.transition = "left 0.16s ease-out, width 0.16s ease-out";
  indicator.style.left = `${bridgeLeft}px`;
  indicator.style.width = `${bridgeRight - bridgeLeft}px`;

  window.clearTimeout(indicator._settleTimer);
  indicator._settleTimer = window.setTimeout(() => {
    indicator.style.transition = "left 0.38s cubic-bezier(0.34, 1.56, 0.64, 1), width 0.38s cubic-bezier(0.34, 1.56, 0.64, 1)";
    indicator.style.left = `${newLeft}px`;
    indicator.style.width = `${newWidth}px`;
  }, 160);
}

function positionLiquidIndicator() {
  positionIndicator("bottomNav", "liquidIndicator", ".bottom-tab-btn.active");
  positionIndicator("tabNav", "tabUnderline", ".tab-btn.active");
}

function drawClaimBanner() {
  const m = DATA.meta;
  document.getElementById("claimHeadline").textContent =
    `${m.model_name} cuts error by ${m.avg_rmse_improvement_pct}% vs. a naive guess`;
  document.getElementById("claimCorr").textContent = m.avg_correlation.toFixed(2);
  document.getElementById("claimVsRf").textContent =
    `${m.avg_rmse_vs_rf_pct >= 0 ? "" : "-"}${Math.abs(m.avg_rmse_vs_rf_pct)}%`;
  document.getElementById("resultsCorr").textContent = m.avg_correlation.toFixed(2);
}

async function bootReal() {
  try {
    const res = await fetch("data_real.json");
    REAL = await res.json();
  } catch (e) {
    console.warn("real data unavailable", e);
    return;
  }

  const c = REAL.cmems;
  document.getElementById("realSstValue").textContent = `${c.series[c.series.length - 1].sst.toFixed(2)}°C`;
  document.getElementById("realSstDate").textContent = `as of ${c.window_end}`;
  document.getElementById("realAnomalyValue").textContent =
    `${c.today_anomaly >= 0 ? "+" : ""}${c.today_anomaly.toFixed(2)}°C vs. window mean (${c.window_mean_sst.toFixed(2)}°C)`;
  const chip = document.getElementById("realChip");
  chip.textContent = c.today_category;
  chip.className = `chip ${c.today_category}`;

  const sss = REAL.cmems_sss;
  const sstTrace = {
    x: c.series.map((d) => d.date), y: c.series.map((d) => d.sst),
    mode: "lines+markers", line: { color: "#3fcf8e", width: 2 }, marker: { size: 5 },
    name: "SST (°C)",
  };
  const traces = [sstTrace];
  const layout = {
    ...PLOTLY_DARK,
    xaxis: { title: "Date", gridcolor: "#1c4a41" },
    yaxis: { title: "SST (°C)", gridcolor: "#1c4a41", titlefont: { color: "#3fcf8e" } },
    legend: { orientation: "h", y: -0.25 },
  };
  if (sss) {
    traces.push({
      x: sss.series.map((d) => d.date), y: sss.series.map((d) => d.sss),
      mode: "lines+markers", line: { color: "#4fa3e3", width: 2, dash: "dot" }, marker: { size: 5 },
      name: "SSS (PSU)", yaxis: "y2",
    });
    layout.yaxis2 = { title: "SSS (PSU)", overlaying: "y", side: "right", showgrid: false, titlefont: { color: "#4fa3e3" } };
  }
  Plotly.newPlot("realTrendChart", traces, layout, { displayModeBar: false, responsive: true });

  drawCurrents();
  drawChlorophyll();

  renderRealFrame(0);
  document.getElementById("realPlayBtn").addEventListener("click", toggleRealPlay);
  startRealLoop();
}

function drawCurrents() {
  const cur = REAL.cmems_currents;
  if (!cur) return;
  const snap = cur.snapshot;

  document.getElementById("currentsDesc").textContent =
    `Real surface current vectors, ${snap.time.slice(0, 10)} — mean speed ${cur.window_mean_speed.toFixed(2)} m/s ` +
    `(${cur.today_anomaly >= 0 ? "+" : ""}${cur.today_anomaly.toFixed(2)} vs. window mean). Arrow = direction, color = speed.`;

  Plotly.newPlot(
    "currentsChart",
    [{
      x: snap.lon, y: snap.lat, mode: "markers", type: "scatter",
      marker: {
        size: 9, symbol: "arrow", angle: snap.heading, angleref: "up",
        color: snap.speed, colorscale: "Viridis", colorbar: { title: "m/s" },
      },
      hovertemplate: "speed %{marker.color:.2f} m/s<extra></extra>",
    }],
    {
      ...PLOTLY_DARK,
      xaxis: { title: "Longitude", range: [45, 105], gridcolor: "#1c4a41" },
      yaxis: { title: "Latitude", range: [5, 30], gridcolor: "#1c4a41" },
      showlegend: false,
    },
    { displayModeBar: false, responsive: true }
  );
}

function drawChlorophyll() {
  const chl = REAL.cmems_chl;
  if (!chl) return;
  Plotly.newPlot(
    "chlChart",
    [{
      x: chl.series.map((d) => d.date), y: chl.series.map((d) => d.chl),
      mode: "lines+markers", line: { color: "#3fcf8e", width: 2 }, marker: { size: 5 },
      name: "Chlorophyll (mg/m³)",
    }],
    { ...PLOTLY_DARK, xaxis: { title: "Date", gridcolor: "#1c4a41" }, yaxis: { title: "Chlorophyll (mg/m³)", gridcolor: "#1c4a41" } },
    { displayModeBar: false, responsive: true }
  );
}

function renderRealFrame(idx) {
  realFrameIdx = idx;
  const frame = REAL.mosdac_frames[idx];
  Plotly.react(
    "realMapChart",
    [{
      x: frame.lon, y: frame.lat, mode: "markers", type: "scatter",
      marker: { size: 6, color: frame.sst, colorscale: "Thermal", colorbar: { title: "SST °C" } },
      hovertemplate: "SST %{marker.color:.1f}°C<extra></extra>",
    }],
    {
      ...PLOTLY_DARK,
      xaxis: { title: "Longitude", range: [45, 105], gridcolor: "#1c4a41" },
      yaxis: { title: "Latitude", range: [5, 30], gridcolor: "#1c4a41" },
      showlegend: false,
    },
    { displayModeBar: false, responsive: true }
  );
  document.getElementById("realFrameLabel").textContent = `${frame.time} GMT, 25 Aug`;
}

function startRealLoop() {
  clearInterval(realTimer);
  if (!realPlaying || !REAL) return;
  realTimer = setInterval(() => {
    renderRealFrame((realFrameIdx + 1) % REAL.mosdac_frames.length);
  }, 1600);
}

function toggleRealPlay() {
  realPlaying = !realPlaying;
  document.getElementById("realPlayBtn").textContent = realPlaying ? "⏸" : "▶";
  startRealLoop();
}

function populateSelectors() {
  const daySelect = document.getElementById("daySelect");
  DATA.grids.forEach((g) => {
    const opt = document.createElement("option");
    opt.value = g.day;
    opt.textContent = `Day ${g.day}`;
    daySelect.appendChild(opt);
  });

  const depthSelect = document.getElementById("depthSelect");
  DATA.meta.depth_levels.forEach((z, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${z} m`;
    if (z === 100) opt.selected = true;
    depthSelect.appendChild(opt);
  });

  const profileSelect = document.getElementById("profileSelect");
  DATA.argo_test.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${p.lat.toFixed(1)}°N, ${p.lon.toFixed(1)}°E (day ${p.day})`;
    profileSelect.appendChild(opt);
  });

  document.getElementById("kpiFloats").textContent = DATA.argo_test.length;
  document.getElementById("kpiSkill").textContent = `-${DATA.meta.avg_rmse_improvement_pct}%`;
  const peak = DATA.meta.peak_event;
  document.getElementById("kpiPeak").textContent =
    `Season peak: day ${peak.day}, +${peak.anomaly}°C (${peak.category})`;

  const [lo, hi] = DATA.meta.region.lon_range;
  const [latlo, lathi] = DATA.meta.region.lat_range;
  const wv = `https://worldview.earthdata.nasa.gov/?v=${lo - 3},${latlo - 3},${hi + 3},${lathi + 3}` +
    `&l=Reference_Labels_15m(hidden),Coastlines_15m,VIIRS_NOAA20_CorrectedReflectance_TrueColor` +
    `&lg=true`;
  document.getElementById("worldviewFrame").src = wv;
  document.getElementById("worldviewLink").href = wv;
}

function drawStatic() {
  drawModelSummary();

  // Per-depth comparison as LINES, not grouped bars -- five series as bars
  // would be an unreadable wall of color; lines make "which one is lowest
  // at which depth" actually legible.
  const depths = DATA.metrics.map((m) => `${m.depth}m`);
  const series = [
    { key: "rmse_baseline", name: "Naive guess", color: "#5c7d76", dash: "dot" },
    { key: "rmse_rf", name: "Random Forest", color: "#9a7fd1", dash: "dash" },
    { key: "rmse_cnn", name: "CNN", color: "#e58a3a", dash: "dash" },
    { key: "rmse_vit", name: "ViT", color: "#4fa3e3", dash: "dash" },
    { key: "rmse_autoencoder", name: "Autoencoder", color: "#c463d9", dash: "dash" },
    { key: "rmse_lstm", name: "LSTM", color: "#e2543f", dash: "dash" },
    { key: "rmse_model", name: DATA.meta.model_name || "FFNN", color: "#3fcf8e", dash: "solid" },
  ];
  Plotly.newPlot(
    "rmseChart",
    series.map((s) => ({
      x: depths, y: DATA.metrics.map((m) => m[s.key]),
      mode: "lines+markers", name: s.name,
      line: { color: s.color, width: s.key === "rmse_model" ? 3 : 1.5, dash: s.dash },
      marker: { size: 5 },
    })),
    { ...PLOTLY_DARK, legend: { orientation: "h", y: -0.3 }, yaxis: { title: "RMSE (°C)", gridcolor: "#1c4a41" }, xaxis: { title: "Depth" } },
    { displayModeBar: false, responsive: true }
  );

  // Heatwave time series (static line + shaded category bands; moving marker added per-frame)
  const days = DATA.heatwave_series.map((d) => d.day);
  const anomalies = DATA.heatwave_series.map((d) => d.anomaly);
  const yMax = Math.max(2.4, Math.max(...anomalies) + 0.3);
  const bands = [
    { y0: -0.5, y1: 0.5, color: "rgba(63,207,142,0.10)" },
    { y0: 0.5, y1: 1.0, color: "rgba(227,194,60,0.10)" },
    { y0: 1.0, y1: 1.5, color: "rgba(229,138,58,0.10)" },
    { y0: 1.5, y1: 2.0, color: "rgba(226,84,63,0.10)" },
    { y0: 2.0, y1: yMax, color: "rgba(181,32,58,0.12)" },
  ].map((b) => ({
    type: "rect", xref: "x", yref: "y", x0: 0, x1: DATA.meta.n_days,
    y0: b.y0, y1: b.y1, fillcolor: b.color, line: { width: 0 },
  }));

  Plotly.newPlot(
    "heatwaveChart",
    [
      { x: days, y: anomalies, mode: "lines", line: { color: "#3fcf8e", width: 2 }, name: "SST anomaly" },
      { x: [days[0]], y: [anomalies[0]], mode: "markers", marker: { size: 12, color: "#e8f5f0", line: { color: "#3fcf8e", width: 2 } }, name: "Today", showlegend: false },
    ],
    { ...PLOTLY_DARK, shapes: bands, yaxis: { title: "Anomaly (°C)", range: [-0.5, yMax], gridcolor: "#1c4a41" }, xaxis: { title: "Day of season window" }, showlegend: false },
    { displayModeBar: false, responsive: true }
  );

  drawMetricsTable();
  drawScatter();
  drawProfile();
}

function drawModelSummary() {
  const summary = DATA.model_summary;
  if (!summary) return;
  const best = Math.min(...summary.map((m) => m.avg_rmse));
  Plotly.newPlot(
    "modelSummaryChart",
    [{
      x: summary.map((m) => m.name), y: summary.map((m) => m.avg_rmse),
      type: "bar",
      marker: { color: summary.map((m) => (m.avg_rmse === best ? "#3fcf8e" : "#4fa3e3")) },
      text: summary.map((m) => m.avg_rmse.toFixed(3)),
      textposition: "outside",
      hovertemplate: "%{x}<br>mean RMSE %{y:.3f}°C<extra></extra>",
    }],
    { ...PLOTLY_DARK, yaxis: { title: "Mean RMSE (°C), lower = better", gridcolor: "#1c4a41" }, xaxis: { tickangle: -15 }, showlegend: false },
    { displayModeBar: false, responsive: true }
  );
}

function drawMetricsTable() {
  const modelName = DATA.meta.model_name || "Model";
  const baselineName = DATA.meta.baseline_model_name || "Random Forest";
  const rows = DATA.metrics.map((m) => `
    <tr>
      <td>${m.depth} m</td>
      <td>${m.rmse_baseline.toFixed(3)}</td>
      <td>${m.rmse_rf.toFixed(3)}</td>
      <td>${m.rmse_cnn.toFixed(3)}</td>
      <td>${m.rmse_vit.toFixed(3)}</td>
      <td>${m.rmse_autoencoder.toFixed(3)}</td>
      <td>${m.rmse_lstm.toFixed(3)}</td>
      <td><b>${m.rmse_model.toFixed(3)}</b></td>
      <td>${m.correlation.toFixed(3)}</td>
      <td>${m.bias >= 0 ? "+" : ""}${m.bias.toFixed(3)}</td>
    </tr>`).join("");
  document.getElementById("metricsTable").innerHTML = `
    <thead><tr><th>Depth</th><th>Naive guess</th><th>${baselineName}</th><th>CNN</th><th>ViT</th><th>Autoencoder</th><th>LSTM</th><th>${modelName}</th><th>Correlation</th><th>Bias</th></tr></thead>
    <tbody>${rows}</tbody>`;
}

function drawScatter() {
  const depthIdx = Number(document.getElementById("depthSelect").value || 1);
  const z = DATA.meta.depth_levels[depthIdx];
  const actual = DATA.argo_test.map((p) => p.actual[depthIdx]);
  const predicted = DATA.argo_test.map((p) => p.predicted[depthIdx]);
  const lo = Math.min(...actual, ...predicted);
  const hi = Math.max(...actual, ...predicted);

  Plotly.newPlot(
    "scatterChart",
    [
      { x: actual, y: predicted, mode: "markers", type: "scatter", marker: { color: "#3fcf8e", opacity: 0.7, size: 8 }, name: "Argo profiles" },
      { x: [lo, hi], y: [lo, hi], mode: "lines", line: { dash: "dash", color: "#9aa0a6" }, name: "Perfect prediction" },
    ],
    { ...PLOTLY_DARK, xaxis: { title: `Actual @ ${z}m (°C)`, gridcolor: "#1c4a41" }, yaxis: { title: `Predicted @ ${z}m (°C)`, gridcolor: "#1c4a41" }, showlegend: false },
    { displayModeBar: false, responsive: true }
  );
}

function drawProfile() {
  const i = Number(document.getElementById("profileSelect").value || 0);
  const p = DATA.argo_test[i];
  const depths = DATA.meta.depth_levels;

  Plotly.newPlot(
    "profileChart",
    [
      { x: p.actual, y: depths, mode: "lines+markers", name: "Actual (float reading)", line: { color: "#3fcf8e", width: 2 } },
      { x: p.predicted, y: depths, mode: "lines+markers", name: `Predicted (${DATA.meta.model_name || "model"})`, line: { color: "#e58a3a", width: 2, dash: "dash" }, marker: { symbol: "square" } },
    ],
    { ...PLOTLY_DARK, xaxis: { title: "Temperature (°C)", gridcolor: "#1c4a41" }, yaxis: { title: "Depth (m)", autorange: "reversed", gridcolor: "#1c4a41" }, legend: { orientation: "h", y: -0.2 } },
    { displayModeBar: false, responsive: true }
  );
}

function renderFrame(idx) {
  frameIdx = idx;
  const frame = DATA.grids[idx];
  const useCluster = document.getElementById("clusterToggle").checked;

  const mapTraces = [
    {
      x: frame.lon, y: frame.lat, mode: "markers", type: "scatter",
      marker: useCluster
        ? { size: 8, color: frame.cluster.map((c) => CLUSTER_PALETTE[c % CLUSTER_PALETTE.length]) }
        : { size: 8, color: frame.sst, colorscale: "Thermal", colorbar: { title: "SST °C" } },
      name: "Satellite grid",
      hovertemplate: useCluster ? "cluster %{marker.color}<extra></extra>" : "SST %{marker.color:.2f}°C<extra></extra>",
    },
    {
      x: DATA.argo_test.map((p) => p.lon), y: DATA.argo_test.map((p) => p.lat),
      mode: "markers", type: "scatter",
      marker: { size: 9, color: "#0b0b0b", symbol: "x", line: { width: 1, color: "#fff" } },
      name: "Argo floats (test set)",
    },
  ];

  Plotly.react(
    "mapChart", mapTraces,
    {
      ...PLOTLY_DARK,
      xaxis: { title: "Longitude", range: DATA.meta.region.lon_range, gridcolor: "#1c4a41" },
      yaxis: { title: "Latitude", range: DATA.meta.region.lat_range, gridcolor: "#1c4a41" },
      legend: { orientation: "h", y: -0.2 },
    },
    { displayModeBar: false, responsive: true }
  );

  document.getElementById("daySelect").value = frame.day;
  document.getElementById("liveLabel").textContent = `Day ${frame.day} / ${DATA.meta.n_days}`;

  const hw = DATA.heatwave_series[frame.day] || DATA.heatwave_series[DATA.heatwave_series.length - 1];
  document.getElementById("kpiAnomaly").textContent = `${hw.anomaly >= 0 ? "+" : ""}${hw.anomaly.toFixed(2)}°C`;
  document.getElementById("kpiCategory").textContent = "vs. climatology baseline";
  const chip = document.getElementById("kpiChip");
  chip.textContent = hw.category;
  chip.className = `chip ${hw.category}`;

  Plotly.restyle("heatwaveChart", { x: [[frame.day]], y: [[hw.anomaly]] }, [1]);
}

function startLoop() {
  clearInterval(timer);
  if (!playing) return;
  timer = setInterval(() => {
    const next = (frameIdx + 1) % DATA.grids.length;
    renderFrame(next);
  }, 1400 / speed);
}

function togglePlay() {
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  startLoop();
}

function cycleSpeed() {
  speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
  document.getElementById("speedBtn").textContent = `${speed}x`;
  startLoop();
}

boot();
