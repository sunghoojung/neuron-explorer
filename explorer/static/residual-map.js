const API = "/api/residuals";
const DRAFT_KEY = "neuron-explorer-residual-groups-v1";
const NS = "http://www.w3.org/2000/svg";
const COLORS = ["#38c8ff", "#ffb547", "#b798e5", "#78c9a3", "#ff759e", "#8ba8ff", "#e2cf68", "#57b99b"];
function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = String(text);
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function text(value, fallback = "-") {
  return typeof value === "string" && value.length ? value : fallback;
}

function fixed(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "-";
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "-";
}

function captureFromDraft() {
  try {
    const draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    return typeof draft?.captureId === "string" ? draft.captureId : null;
  } catch {
    return null;
  }
}

function groupColor(groups, name) {
  const index = groups.indexOf(name);
  return COLORS[index >= 0 ? index % COLORS.length : 0];
}

async function request(path, options = {}) {
  const init = { method: options.method || "GET", headers: { Accept: "application/json" } };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  if (options.signal) init.signal = options.signal;
  const response = await fetch(`${API}${path}`, init);
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    // The status and route are enough to explain a non-JSON response safely.
  }
  if (!response.ok) {
    throw new Error(text(payload.error?.message || payload.error || payload.detail, `Request failed (${response.status}).`));
  }
  return payload;
}

export function createResidualMap({ host, map, onLayer, onCase } = {}) {
  if (!host) throw new Error("Residual map needs a host element.");
  const state = {
    enabled: false,
    mode: "pca",
    captureId: null,
    captures: [],
    summary: null,
    result: null,
    fitId: null,
    layer: null,
    point: null,
    groupA: null,
    groupB: null,
    split: "all",
    showCentroids: true,
    selectedPairId: null,
    selectedSampleId: null,
    version: 0,
    sampleVersion: 0,
    somVersion: 0,
    cellVersion: 0,
    closed: false,
  };

  host.classList.add("residual-map-host");
  const panel = el("section", "rm-panel");
  panel.setAttribute("aria-label", "Residual group map");
  const header = el("div", "rm-head");
  const heading = el("div", "rm-heading");
  heading.append(el("span", "rm-eyebrow", "RESIDUAL GROUPS"), el("strong", "rm-title", "Residual map"));
  const collapse = el("button", "rm-collapse", "Collapse");
  collapse.type = "button";
  collapse.setAttribute("aria-expanded", "true");
  collapse.setAttribute("aria-label", "Collapse residual map panel");
  header.append(heading, collapse);

  const body = el("div", "rm-body");
  const captureRow = el("div", "rm-control-row rm-capture-row");
  const captureField = field("Saved capture", "rm-capture");
  const captureSelect = el("select", "rm-select");
  captureSelect.setAttribute("aria-label", "Saved capture");
  captureField.append(captureSelect);
  const refreshButton = el("button", "rm-refresh", "Refresh");
  refreshButton.type = "button";
  refreshButton.setAttribute("aria-label", "Refresh residual map data");
  captureRow.append(captureField, refreshButton);

  const modeRow = el("nav", "rm-mode-row");
  modeRow.setAttribute("aria-label", "Map view");
  const pcaButton = modeButton("pca", "PCA projection");
  const somButton = modeButton("som", "SOM fit");
  modeRow.append(pcaButton, somButton);

  const controls = el("div", "rm-control-grid");
  const pointField = field("State point", "rm-point");
  const pointSelect = el("select", "rm-select");
  pointSelect.setAttribute("aria-label", "Residual state point");
  pointField.append(pointSelect);
  const groupAField = field("Group A", "rm-group");
  const groupASelect = el("select", "rm-select");
  groupASelect.setAttribute("aria-label", "First response group");
  groupAField.append(groupASelect);
  const groupBField = field("Group B", "rm-group");
  const groupBSelect = el("select", "rm-select");
  groupBSelect.setAttribute("aria-label", "Second response group");
  groupBField.append(groupBSelect);
  const splitField = field("Show", "rm-split");
  const splitSelect = el("select", "rm-select");
  splitSelect.setAttribute("aria-label", "Dataset split");
  for (const [value, label] of [["all", "All splits"], ["train", "Train"], ["validation", "Validation"], ["test", "Test"]]) {
    const option = el("option", "", label);
    option.value = value;
    splitSelect.append(option);
  }
  splitField.append(splitSelect);
  controls.append(pointField, groupAField, groupBField, splitField);

  const message = el("div", "rm-message", "Choose a saved capture to view its projection.");
  message.setAttribute("role", "status");
  message.setAttribute("aria-live", "polite");
  const layerSection = el("section", "rm-layer-section");
  const layerHead = el("div", "rm-layer-head");
  layerHead.append(el("strong", "", "Layer gaps"), el("span", "rm-layer-note", "Raw paired residual distance"));
  const layerStrip = el("div", "rm-layer-strip");
  layerStrip.setAttribute("role", "group");
  layerStrip.setAttribute("aria-label", "Choose model layer");
  layerSection.append(layerHead, layerStrip);

  const pcaPane = el("div", "rm-pane");
  const chartHeader = el("div", "rm-chart-head");
  const chartTitle = el("strong", "", "PCA projection");
  const chartMeta = el("span", "rm-chart-meta", "Axes fit on training data");
  const centroidLabel = el("label", "rm-toggle");
  const centroidInput = el("input");
  centroidInput.type = "checkbox";
  centroidInput.checked = true;
  centroidInput.setAttribute("aria-label", "Show group centers");
  centroidLabel.append(centroidInput, document.createTextNode("centers"));
  chartHeader.append(el("div", "rm-chart-heading", ""));
  chartHeader.lastChild.append(chartTitle, chartMeta);
  chartHeader.append(centroidLabel);
  const svg = svgEl("svg", { class: "rm-scatter", viewBox: "0 0 520 300", role: "group", "aria-label": "Projected residual states by group" });
  const legend = el("div", "rm-legend");
  const gapMetric = el("div", "rm-gap-metric", "Raw paired gap will appear after loading a capture.");
  const projectionNote = el("div", "rm-note", "PCA coordinates are fit on training families. Raw pair distances are calculated before projection. These views describe geometry, not response quality.");
  pcaPane.append(chartHeader, svg, legend, gapMetric, projectionNote);

  const somPane = el("div", "rm-pane rm-som-pane");
  somPane.hidden = true;
  const fitRow = el("div", "rm-control-row");
  const fitField = field("Saved map", "rm-fit-field");
  const fitSelect = el("select", "rm-select");
  fitSelect.setAttribute("aria-label", "Saved self-organizing map");
  fitField.append(fitSelect);
  fitRow.append(fitField);
  const fitSummary = el("div", "rm-fit-summary", "Choose a saved fit for this capture, layer, and point.");
  const somGrid = el("div", "rm-som-grid");
  somGrid.setAttribute("role", "grid");
  somGrid.setAttribute("aria-label", "Self-organizing map cells");
  const somNote = el("div", "rm-note", "Counts follow the selected split; color bars show each cell's overall group mix. Cell positions are not comparable across layers or separate map fits. This describes geometry only.");
  const cellDetail = el("div", "rm-cell-detail", "Choose a cell to inspect its assigned cases.");
  somPane.append(fitRow, fitSummary, somGrid, somNote, cellDetail);

  const samplePanel = el("section", "rm-sample-panel");
  const sampleHead = el("div", "rm-sample-head");
  sampleHead.append(el("strong", "", "Selected case"), el("span", "rm-sample-status", "Choose a point"));
  const sampleMeta = el("div", "rm-sample-meta");
  const sampleMessages = el("div", "rm-sample-messages");
  samplePanel.append(sampleHead, sampleMeta, sampleMessages);

  const filters = el("details", "rm-filters");
  const filterTitle = el("summary", "", "Compare groups and choose state point");
  filters.append(filterTitle, controls, message);
  body.append(captureRow, modeRow, filters, layerSection, pcaPane, somPane, samplePanel);
  panel.append(header, body);
  host.replaceChildren(panel);

  function field(label, className = "") {
    const wrapper = el("label", `rm-field ${className}`.trim());
    wrapper.append(el("span", "", label));
    return wrapper;
  }

  function modeButton(mode, label) {
    const button = el("button", `rm-mode${state.mode === mode ? " on" : ""}`, label);
    button.type = "button";
    button.dataset.mode = mode;
    button.setAttribute("aria-pressed", String(state.mode === mode));
    return button;
  }

  function setMessage(value, kind = "") {
    message.textContent = value;
    message.className = `rm-message${kind ? ` ${kind}` : ""}`;
    if (kind === "error") filters.open = true;
  }

  function setOptions(select, values, selected, labeler = (value) => value, valueOf = (value) => String(value)) {
    select.replaceChildren();
    for (const value of values) {
      const option = el("option", "", labeler(value));
      option.value = valueOf(value);
      option.selected = option.value === String(selected);
      select.append(option);
    }
    select.disabled = values.length === 0;
  }

  function splitLabel(split) {
    return split === "train" ? "Train" : split === "validation" ? "Validation" : split === "test" ? "Test" : "All";
  }

  function samplesForResult() {
    return Array.isArray(state.result?.projection?.points) ? state.result.projection.points : [];
  }

  function activeGroups() {
    return state.summary?.groups?.map((item) => item.group).filter((name) => typeof name === "string") || [];
  }

  function setCaptureControls() {
    const captures = state.captures;
    setOptions(captureSelect, captures, state.captureId, (item) => `${item.name || item.capture_id.slice(0, 12)} · ${item.capture_id.slice(0, 6)}`, (item) => item.capture_id);
    if (captures.length && !captures.some((item) => item.capture_id === state.captureId)) state.captureId = captures[0].capture_id;
  }

  function syncSummaryControls() {
    const provenance = state.summary?.provenance || {};
    const points = Array.isArray(provenance.points) ? provenance.points : [];
    const layers = Array.isArray(provenance.layers) ? provenance.layers : [];
    const groups = activeGroups();
    if (!points.includes(state.point)) state.point = points.includes("resid_post") ? "resid_post" : points[0] || null;
    if (!layers.includes(state.layer)) state.layer = layers[0] ?? null;
    if (!groups.includes(state.groupA)) state.groupA = groups[0] ?? null;
    if (!groups.includes(state.groupB) || state.groupB === state.groupA) state.groupB = groups.find((group) => group !== state.groupA) ?? null;
    setOptions(pointSelect, points, state.point, (point) => ({ resid_pre: "Before attention", resid_mid: "After attention", resid_post: "After the MLP", attn_write: "Attention write", mlp_write: "MLP write" }[point] || point));
    setOptions(groupASelect, groups, state.groupA);
    setOptions(groupBSelect, groups.filter((group) => group !== state.groupA), state.groupB);
    splitSelect.value = state.split;
  }

  function renderLayerStrip(rows) {
    layerStrip.replaceChildren();
    if (!rows.length) {
      layerStrip.append(el("span", "rm-muted", "No layer gap summary"));
      return;
    }
    for (const row of rows) {
      const layer = Number(row.layer);
      if (!Number.isInteger(layer)) continue;
      const button = el("button", `rm-layer${layer === state.layer ? " on" : ""}`);
      button.type = "button";
      button.dataset.layer = String(layer);
      button.append(el("strong", "", `L${layer}`), el("span", "", `${fixed(row.mean_delta_l2, 2)} L2`));
      const cosine = Number(row.centroid_cosine);
      const n = Number(row.n_pairs);
      button.title = `Layer ${layer}. Raw mean paired L2 gap ${fixed(row.mean_delta_l2)}. Centroid cosine ${fixed(cosine)}. ${Number.isFinite(n) ? n : 0} matched pairs.`;
      button.setAttribute("aria-pressed", String(layer === state.layer));
      layerStrip.append(button);
    }
  }

  function renderGapMetric(result) {
    const selected = result.selected || {};
    const layerRow = (result.layers || []).find((item) => Number(item.layer) === Number(selected.layer)) || {};
    gapMetric.replaceChildren(
      el("strong", "", `Raw gap L${selected.layer ?? state.layer}: ${fixed(layerRow.mean_delta_l2)} L2`),
      el("span", "", `RMS ${fixed(layerRow.mean_delta_rms)} · centroid cosine ${fixed(layerRow.centroid_cosine)} · ${layerRow.n_pairs ?? 0} pairs`),
    );
    gapMetric.title = "L2 and RMS are computed from unprojected residual differences. PCA coordinates are only for display.";
  }

  function renderLegend() {
    legend.replaceChildren();
    for (const group of [state.groupA, state.groupB].filter(Boolean)) {
      const item = el("span", "rm-legend-item");
      const swatch = el("span", "rm-swatch");
      swatch.style.backgroundColor = groupColor(activeGroups(), group);
      item.append(swatch, document.createTextNode(group));
      legend.append(item);
    }
    const train = el("span", "rm-legend-item");
    train.append(el("span", "rm-marker train"), document.createTextNode("Train"));
    const heldout = el("span", "rm-legend-item");
    heldout.append(el("span", "rm-marker heldout"), document.createTextNode("Held out"));
    legend.append(train, heldout);
  }

  function renderScatter(projection) {
    svg.replaceChildren();
    const points = Array.isArray(projection?.points) ? projection.points : [];
    const pairs = Array.isArray(projection?.pairs) ? projection.pairs : [];
    const centroids = state.showCentroids && Array.isArray(projection?.centroids) ? projection.centroids : [];
    if (!points.length) {
      svg.append(svgEl("text", { x: 260, y: 150, class: "rm-svg-empty", "text-anchor": "middle" }));
      svg.lastChild.textContent = "No projected cases in this split";
      return;
    }
    const W = 520, H = 300, left = 32, right = 18, top = 18, bottom = 27;
    const xs = points.map((point) => Number(point.x)).filter(Number.isFinite);
    const ys = points.map((point) => Number(point.y)).filter(Number.isFinite);
    if (!xs.length || !ys.length) return;
    let minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    if (minX === maxX) { minX -= 1; maxX += 1; }
    if (minY === maxY) { minY -= 1; maxY += 1; }
    const px = (value) => left + (Number(value) - minX) / (maxX - minX) * (W - left - right);
    const py = (value) => H - bottom - (Number(value) - minY) / (maxY - minY) * (H - top - bottom);
    svg.append(svgEl("path", { d: `M${left} ${top}V${H - bottom}H${W - right}`, class: "rm-axis" }));
    svg.append(svgEl("text", { x: W / 2, y: H - 5, class: "rm-axis-label", "text-anchor": "middle" }));
    svg.lastChild.textContent = `PC 1 · ${fixed((projection.explained_variance_ratio || [])[0] * 100, 1)}%`;
    svg.append(svgEl("text", { x: 10, y: H / 2, class: "rm-axis-label", "text-anchor": "middle", transform: `rotate(-90 10 ${H / 2})` }));
    svg.lastChild.textContent = `PC 2 · ${fixed((projection.explained_variance_ratio || [])[1] * 100, 1)}%`;
    const byId = new Map(points.map((point) => [point.id, point]));
    for (const pair of pairs) {
      const a = byId.get(pair.a_id), b = byId.get(pair.b_id);
      if (!a || !b) continue;
      svg.append(svgEl("line", {
        x1: px(a.x), y1: py(a.y), x2: px(b.x), y2: py(b.y),
        class: `rm-pair-line${pair.pair_id === state.selectedPairId ? " selected" : ""}`,
      }));
    }
    for (const point of points) {
      const x = px(point.x), y = py(point.y);
      const color = groupColor(activeGroups(), point.group);
      const selected = point.id === state.selectedSampleId || point.pair_id === state.selectedPairId;
      let marker;
      if (point.split === "train") {
        marker = svgEl("circle", { cx: x, cy: y, r: selected ? 6 : 4.5, class: `rm-point${selected ? " selected" : ""}`, fill: color });
      } else {
        marker = svgEl("path", { d: `M${x} ${y - (selected ? 7 : 5)} L${x + (selected ? 6 : 4.5)} ${y + (selected ? 5 : 4)} L${x - (selected ? 6 : 4.5)} ${y + (selected ? 5 : 4)} Z`, class: `rm-point${selected ? " selected" : ""}`, fill: color });
      }
      marker.dataset.sampleId = point.id;
      marker.dataset.pairId = point.pair_id;
      marker.setAttribute("role", "button");
      marker.setAttribute("tabindex", "0");
      marker.setAttribute("aria-label", `${point.group}, ${splitLabel(point.split)}, ${point.id}, state norm ${fixed(point.norm)}`);
      const title = svgEl("title");
      title.textContent = `${point.group} · ${splitLabel(point.split)} · norm ${fixed(point.norm)}`;
      marker.append(title);
      marker.style.setProperty("--rm-point-color", color);
      svg.append(marker);
    }
    for (const centroid of centroids) {
      const x = px(centroid.x), y = py(centroid.y), color = groupColor(activeGroups(), centroid.group);
      const size = 7;
      const diamond = svgEl("path", { d: `M${x} ${y - size} L${x + size} ${y} L${x} ${y + size} L${x - size} ${y} Z`, class: "rm-centroid", fill: color });
      const title = svgEl("title");
      title.textContent = `${centroid.group} center · n=${centroid.n}`;
      diamond.append(title);
      svg.append(diamond);
    }
    chartMeta.textContent = `Train fit · rank ${projection.rank ?? "-"} · ${points.length} cases`;
    renderLegend();
  }

  function renderSample(sample, pointRow = null) {
    sampleHead.querySelector(".rm-sample-status").textContent = sample ? `${sample.group || "Sample"} · ${splitLabel(sample.split)}` : "Choose a point";
    sampleMeta.replaceChildren();
    sampleMessages.replaceChildren();
    if (!sample) return;
    const selectedPoint = pointRow || samplesForResult().find((item) => item.id === sample.id);
    const pair = (state.result?.projection?.pairs || []).find((item) => item.pair_id === sample.pair_id);
    const metadata = [
      ["Pair", sample.pair_id], ["Family", sample.family_id], ["Layer", state.layer],
      ["State point", state.point], ["State norm", selectedPoint?.norm], ["Pair L2", pair?.distance_l2],
    ];
    for (const [label, value] of metadata) {
      if (value === undefined || value === null || value === "") continue;
      const item = el("span", "rm-meta-item");
      const formatted = ["State norm", "Pair L2"].includes(label) ? fixed(value) : String(value);
      item.append(el("small", "", label), el("strong", "", formatted));
      sampleMeta.append(item);
    }
    const details = el("details", "rm-conversation");
    details.append(el("summary", "", "Saved conversation"));
    for (const messageItem of sample.messages || []) {
      const block = el("div", "rm-message-turn");
      block.append(el("small", "", text(messageItem.role, "message")), el("p", "", text(messageItem.content, "")));
      details.append(block);
    }
    if (sample.target || sample.foil) {
      const candidates = el("div", "rm-candidates");
      if (sample.target) candidates.append(el("span", "", `Target: ${sample.target}`));
      if (sample.foil) candidates.append(el("span", "", `Foil: ${sample.foil}`));
      details.append(candidates);
    }
    sampleMessages.append(details);
  }

  async function selectSample(sampleId, pointRow = null, notify = true) {
    if (!state.captureId || !sampleId) return;
    state.selectedSampleId = sampleId;
    if (pointRow?.pair_id) state.selectedPairId = pointRow.pair_id;
    renderScatter(state.result?.projection);
    sampleHead.querySelector(".rm-sample-status").textContent = "Loading saved conversation";
    const version = ++state.sampleVersion;
    const captureId = state.captureId;
    try {
      const query = new URLSearchParams({ sample_id: sampleId });
      const response = await request(`/captures/${encodeURIComponent(captureId)}/sample?${query}`);
      if (version !== state.sampleVersion) return;
      renderSample(response.sample, pointRow);
      if (notify) onCase?.({ capture_id: captureId, sample_id: sampleId, pair_id: response.sample?.pair_id, layer: state.layer, point: state.point });
    } catch (error) {
      if (version !== state.sampleVersion) return;
      renderSample(null);
      sampleHead.querySelector(".rm-sample-status").textContent = `Case details unavailable: ${error.message}`;
    }
  }

  function onScatterActivate(event) {
    const point = event.target.closest?.("[data-sample-id]");
    if (!point) return;
    const pointRow = samplesForResult().find((item) => item.id === point.dataset.sampleId);
    selectSample(point.dataset.sampleId, pointRow);
  }

  function renderFitList(fits) {
    fitRow.replaceChildren();
    fitField.append(fitSelect);
    const filtered = fits.filter((item) => item.capture_id === state.captureId && Number(item.fit?.layer) === Number(state.layer) && item.fit?.point === state.point);
    const matching = filtered.find((item) => item.fit_id === state.fitId) || filtered[0];
    state.fitId = matching?.fit_id || null;
    setOptions(fitSelect, filtered, state.fitId, (item) => `${item.fit?.mode === "paired_delta" ? "Group difference" : "States"} · ${item.fit_id.slice(0, 10)} · rank ${item.basis_rank ?? "-"}`, (item) => item.fit_id);
    fitRow.append(fitField);
    if (!matching) {
      fitSummary.textContent = "No saved fit matches this capture, layer, and state point.";
      somGrid.replaceChildren();
      cellDetail.textContent = "Fit a map in the SOM tab of Residual groups, then return here.";
    }
  }

  function currentCellCount(cell) {
    const counts = cell.split_counts || {};
    return state.split === "all" ? Object.values(counts).reduce((sum, count) => sum + Number(count || 0), 0) : Number(counts[state.split] || 0);
  }

  function renderSom(fit) {
    if (!fit) return;
    const config = fit.fit || {};
    const quant = (fit.quantization || []).find((item) => state.split === "all" || item.split === state.split);
    fitSummary.textContent = `${config.rows || 2} × ${config.cols || 2} · ${config.mode === "paired_delta" ? "group differences" : "captured states"} · rank ${fit.basis_rank ?? "-"} · ${fit.training_units ?? fit.training_samples ?? "-"} training units from ${fit.training_families ?? "-"} families${quant ? ` · ${quant.split} mean L2 ${fixed(quant.mean_l2)}` : ""}`;
    const rows = Math.min(16, Math.max(1, Number(config.rows) || 2));
    const cols = Math.min(16, Math.max(1, Number(config.cols) || 2));
    somGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(34px, 1fr))`;
    somGrid.replaceChildren();
    const cells = Array.isArray(fit.cells) ? fit.cells : [];
    const maxCount = Math.max(1, ...cells.map(currentCellCount));
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const cell = cells.find((item) => Number(item.row) === row && Number(item.col) === col);
        const button = el("button", "rm-som-cell");
        button.type = "button";
        if (!cell) {
          button.disabled = true;
          button.textContent = `${row + 1},${col + 1}\n0`;
        } else {
          const count = currentCellCount(cell);
          button.dataset.cellId = cell.cell_id;
          button.disabled = count === 0;
          button.style.setProperty("--rm-cell-fill", `${Math.min(82, 12 + count / maxCount * 70)}%`);
          button.append(el("span", "rm-cell-label", `${row + 1},${col + 1}`), el("strong", "", count));
          const composition = el("span", "rm-cell-composition");
          const groupEntries = Object.entries(cell.group_counts || {}).filter(([, n]) => Number(n) > 0);
          const groupTotal = groupEntries.reduce((sum, [, n]) => sum + Number(n), 0);
          for (const [group, groupCount] of groupEntries) {
            const segment = el("i", "rm-cell-segment");
            segment.style.width = `${groupTotal ? Number(groupCount) / groupTotal * 100 : 0}%`;
            segment.style.backgroundColor = groupColor(activeGroups(), group);
            segment.title = `${group}: ${groupCount}`;
            composition.append(segment);
          }
          composition.setAttribute("aria-hidden", "true");
          button.append(composition);
          const groupCounts = groupEntries.map(([group, n]) => `${group}: ${n}`).join(", ");
          button.title = `${count} cases${groupCounts ? ` · ${groupCounts}` : ""}`;
          button.setAttribute("aria-label", `Cell row ${row + 1}, column ${col + 1}, ${count} cases. ${groupCounts}`);
        }
        somGrid.append(button);
      }
    }
  }

  async function loadSom() {
    if (!state.captureId || state.layer === null || state.layer === undefined || !state.point) return;
    const version = ++state.somVersion;
    state.cellVersion++;
    const captureId = state.captureId;
    const layer = state.layer;
    const point = state.point;
    fitSummary.textContent = "Loading saved maps…";
    try {
      const response = await request("/fits");
      if (version !== state.somVersion || captureId !== state.captureId || layer !== state.layer || point !== state.point) return;
      renderFitList(response.fits || []);
      if (state.fitId) {
        const fitId = state.fitId;
        const fit = await request(`/fits/${encodeURIComponent(fitId)}`);
        if (version !== state.somVersion || captureId !== state.captureId || fitId !== state.fitId) return;
        renderSom(fit);
      }
    } catch (error) {
      if (version === state.somVersion) fitSummary.textContent = `Could not load saved maps: ${error.message}`;
    }
  }

  async function loadFitCell(cellId) {
    if (!state.fitId) return;
    const version = ++state.cellVersion;
    const fitId = state.fitId;
    const split = state.split;
    cellDetail.textContent = "Loading assigned cases…";
    try {
      const query = new URLSearchParams({ split });
      const response = await request(`/fits/${encodeURIComponent(fitId)}/cells/${encodeURIComponent(cellId)}?${query}`);
      if (version !== state.cellVersion || fitId !== state.fitId || split !== state.split) return;
      cellDetail.replaceChildren();
      const head = el("div", "rm-cell-head", `Cell ${Number(response.cell?.row ?? 0) + 1}, ${Number(response.cell?.col ?? 0) + 1} · ${splitLabel(response.split)}`);
      cellDetail.append(head, el("div", "rm-cell-count", `${response.n ?? 0} matched cases`));
      for (const sample of response.samples || []) {
        const button = el("button", "rm-assigned-case", `${sample.group} · ${sample.id} · distance ${fixed(sample.quantization_error)}`);
        button.type = "button";
        button.dataset.sampleId = sample.id;
        button.dataset.pairId = sample.pair_id || "";
        button.addEventListener("click", () => selectSample(sample.id));
        cellDetail.append(button);
      }
      if (!(response.samples || []).length) cellDetail.append(el("span", "rm-muted", "No cases in this cell for the selected split."));
    } catch (error) {
      if (version === state.cellVersion) cellDetail.textContent = `Cell details unavailable: ${error.message}`;
    }
  }

  async function refresh() {
    if (!state.enabled || state.closed) return;
    state.sampleVersion++;
    const version = ++state.version;
    setMessage("Loading saved captures and residual geometry…");
    refreshButton.disabled = true;
    try {
      const captureResponse = await request("/captures");
      if (version !== state.version) return;
      const previousCaptureId = state.captureId;
      state.captures = Array.isArray(captureResponse.captures) ? captureResponse.captures : [];
      if (!state.captureId || !state.captures.some((item) => item.capture_id === state.captureId)) {
        const draftId = captureFromDraft();
        state.captureId = state.captures.some((item) => item.capture_id === draftId) ? draftId : state.captures[0]?.capture_id || null;
      }
      if (previousCaptureId !== state.captureId) {
        state.sampleVersion++;
        state.selectedPairId = null;
        state.selectedSampleId = null;
        renderSample(null);
      }
      setCaptureControls();
      if (!state.captureId) {
        state.summary = null;
        state.result = null;
        setOptions(pointSelect, [], null);
        setOptions(groupASelect, [], null);
        setOptions(groupBSelect, [], null);
        renderLayerStrip([]);
        svg.replaceChildren();
        legend.replaceChildren();
        gapMetric.textContent = "No saved captures are available yet.";
        setMessage("Capture conversations in the Residual groups workspace to view them here.");
        return;
      }
      const summary = await request(`/captures/${encodeURIComponent(state.captureId)}/summary`);
      if (version !== state.version) return;
      state.summary = summary;
      syncSummaryControls();
      if (!state.groupA || !state.groupB) {
        state.result = null;
        renderLayerStrip([]);
        svg.replaceChildren();
        gapMetric.textContent = "A paired group gap needs two distinct response groups.";
        legend.replaceChildren();
        setMessage("Choose two response groups in the capture to compare their residual states.");
        return;
      }
      const query = new URLSearchParams({
        layer: String(state.layer), point: state.point,
        group_a: state.groupA, group_b: state.groupB, split: state.split,
      });
      const result = await request(`/captures/${encodeURIComponent(state.captureId)}/map?${query}`);
      if (version !== state.version) return;
      state.result = result;
      state.layer = Number(result.selected?.layer ?? state.layer);
      onLayer?.(state.layer);
      filterTitle.textContent = `${state.groupA} / ${state.groupB} · ${state.point} · ${splitLabel(state.split)}`;
      renderLayerStrip(result.layers || []);
      renderGapMetric(result);
      renderScatter(result.projection || {});
      renderLegend();
      chartTitle.textContent = `PCA projection · ${text(result.selected?.point, state.point)}`;
      if (map?.setResidualLayers) map.setResidualLayers(result.layers || [], state.layer, (layer) => handleLayerSelect(layer, true));
      if (state.mode === "som") await loadSom();
      const integrity = summary.integrity_status || result.integrity_status;
      const provenance = summary.provenance || {};
      const model = [provenance.model_id, provenance.revision].filter(Boolean).join(" @ ") || "Saved capture";
      const geometry = [provenance.device, provenance.hidden_size ? `${provenance.hidden_size} dimensions` : null].filter(Boolean).join(" · ");
      setMessage(`${model}${geometry ? ` · ${geometry}` : ""} · integrity ${integrity || "unreported"} · ${result.projection?.note || "Projection axes fit on training data."}`, integrity === "verified" ? "success" : "");
      if (state.selectedSampleId) {
        const selected = samplesForResult().find((sample) => sample.id === state.selectedSampleId)
          || samplesForResult().find((sample) => sample.pair_id === state.selectedPairId);
        if (selected) {
          state.selectedSampleId = selected.id;
          state.selectedPairId = selected.pair_id;
          await selectSample(selected.id, selected, false);
        } else {
          state.selectedSampleId = null;
          state.selectedPairId = null;
          renderSample(null);
        }
      }
    } catch (error) {
      if (version !== state.version) return;
      state.result = null;
      renderLayerStrip([]);
      svg.replaceChildren();
      gapMetric.textContent = "Raw pair metrics are unavailable.";
      setMessage(`Residual map unavailable: ${error.message}`, "error");
    } finally {
      if (version === state.version) {
        refreshButton.disabled = false;
      }
    }
  }

  function setMode(mode) {
    state.mode = mode === "som" ? "som" : "pca";
    pcaButton.classList.toggle("on", state.mode === "pca");
    pcaButton.setAttribute("aria-pressed", String(state.mode === "pca"));
    somButton.classList.toggle("on", state.mode === "som");
    somButton.setAttribute("aria-pressed", String(state.mode === "som"));
    pcaPane.hidden = state.mode !== "pca";
    somPane.hidden = state.mode !== "som";
    if (state.mode === "som") loadSom();
  }

  function handleLayerSelect(layer, notify = true) {
    const numeric = Number(layer);
    if (!Number.isInteger(numeric) || numeric < 0) return;
    if (Array.isArray(state.summary?.provenance?.layers) && !state.summary.provenance.layers.includes(numeric)) return;
    state.layer = numeric;
    if (notify) onLayer?.(numeric);
    if (state.enabled) refresh();
  }

  captureSelect.addEventListener("change", () => {
    state.sampleVersion++;
    state.somVersion++;
    state.cellVersion++;
    state.captureId = captureSelect.value || null;
    state.summary = null;
    state.result = null;
    state.layer = null;
    state.point = null;
    state.groupA = null;
    state.groupB = null;
    state.selectedPairId = null;
    state.selectedSampleId = null;
    renderSample(null);
    refresh();
  });
  refreshButton.addEventListener("click", refresh);
  pointSelect.addEventListener("change", () => { state.point = pointSelect.value; refresh(); });
  groupASelect.addEventListener("change", () => { state.groupA = groupASelect.value; if (state.groupA === state.groupB) state.groupB = activeGroups().find((name) => name !== state.groupA) || null; refresh(); });
  groupBSelect.addEventListener("change", () => { state.groupB = groupBSelect.value; refresh(); });
  splitSelect.addEventListener("change", () => { state.split = splitSelect.value; refresh(); });
  modeRow.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (button) setMode(button.dataset.mode);
  });
  layerStrip.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-layer]");
    if (button) handleLayerSelect(button.dataset.layer);
  });
  svg.addEventListener("click", onScatterActivate);
  svg.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const point = event.target.closest?.("[data-sample-id]");
    if (!point) return;
    event.preventDefault();
    onScatterActivate(event);
  });
  centroidInput.addEventListener("change", () => {
    state.showCentroids = centroidInput.checked;
    renderScatter(state.result?.projection);
  });
  fitSelect.addEventListener("change", async () => {
    state.fitId = fitSelect.value || null;
    const version = ++state.somVersion;
    state.cellVersion++;
    if (!state.fitId) return;
    const fitId = state.fitId;
    fitSummary.textContent = "Loading saved map…";
    try {
      const fit = await request(`/fits/${encodeURIComponent(fitId)}`);
      if (version !== state.somVersion || fitId !== state.fitId) return;
      renderSom(fit);
    } catch (error) {
      if (version === state.somVersion) fitSummary.textContent = `Could not load saved map: ${error.message}`;
    }
  });
  somGrid.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-cell-id]");
    if (button) loadFitCell(button.dataset.cellId);
  });
  collapse.addEventListener("click", () => {
    state.closed = !state.closed;
    panel.classList.toggle("collapsed", state.closed);
    collapse.textContent = state.closed ? "Expand" : "Collapse";
    collapse.setAttribute("aria-expanded", String(!state.closed));
    collapse.setAttribute("aria-label", state.closed ? "Expand residual map panel" : "Collapse residual map panel");
  });

  return {
    setEnabled(enabled) {
      state.enabled = Boolean(enabled);
      host.hidden = !state.enabled;
      panel.classList.toggle("enabled", state.enabled);
      if (map?.setResidualMode) map.setResidualMode(state.enabled);
      if (state.enabled) refresh();
      else state.version++;
    },
    setLayer(layer) {
      handleLayerSelect(layer, true);
    },
    refresh,
  };
}
