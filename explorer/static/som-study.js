// SOM sycophancy steering study: where the signal lives, the selected map,
// blinded paired results and proxy dose screens. Read-only view of
// /api/som-study (built by research/build_som_study.py).

import { matchedPairSection } from "./som-pair-study.js?v=6";

const SVG = "http://www.w3.org/2000/svg";
const COLORS = { syc: "#e5468a", res: "#2a95c2", amber: "#c0821f", violet: "#8f78e6" };
const SERIES = [
  { key: "flips_prevented", label: "Unsupported flips prevented", color: COLORS.res },
  { key: "neutral_lost", label: "Neutral answers lost", color: COLORS.syc },
  { key: "updates_lost", label: "Justified updates lost", color: COLORS.amber },
  { key: "corrections_lost", label: "Warranted corrections lost", color: COLORS.violet },
];

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};
const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
const signed = (v, d = 1) => (v == null ? "n/a" : `${v > 0 ? "+" : v < 0 ? "-" : ""}${Math.abs(v).toFixed(d)}`);

function tooltip(root) {
  const tip = el("div", "study-tip");
  tip.hidden = true;
  root.append(tip);
  return {
    node: tip,
    show(event, lines) {
      tip.replaceChildren(...lines.map((line, i) => el("div", i ? "" : "study-tip-head", line)));
      tip.hidden = false;
      const box = root.getBoundingClientRect();
      const x = Math.min(event.clientX - box.left + 14, box.width - 260);
      tip.style.left = `${Math.max(8, x)}px`;
      tip.style.top = `${event.clientY - box.top + root.scrollTop + 14}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function card(title, subtitle) {
  const node = el("section", "study-card");
  const head = el("div", "study-card-head");
  head.append(el("h2", "", title));
  if (subtitle) head.append(el("p", "study-sub", subtitle));
  node.append(head);
  return node;
}

function legend(items) {
  const row = el("div", "study-legend");
  for (const item of items) {
    const entry = el("span", "study-legend-item");
    const swatch = el("i", item.dashed ? "dashed" : "");
    swatch.style.setProperty("--c", item.color);
    entry.append(swatch, document.createTextNode(item.label));
    row.append(entry);
  }
  return row;
}

function axes(svg, { x, y, w, h, m, xTicks, yTicks, xLabel, yLabel, xFmt = String, yFmt = String }) {
  const g = svgEl("g", { class: "study-axis" });
  for (const t of yTicks) {
    g.append(svgEl("line", { x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), class: "grid" }));
    const label = svgEl("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" });
    label.textContent = yFmt(t);
    g.append(label);
  }
  for (const t of xTicks) {
    const label = svgEl("text", { x: x(t), y: h - m.b + 18, "text-anchor": "middle" });
    label.textContent = xFmt(t);
    g.append(label);
  }
  g.append(svgEl("line", { x1: m.l, x2: w - m.r, y1: h - m.b, y2: h - m.b, class: "base" }));
  if (xLabel) {
    const t = svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 6, "text-anchor": "middle", class: "axis-title" });
    t.textContent = xLabel;
    g.append(t);
  }
  if (yLabel) {
    const t = svgEl("text", { x: 14, y: m.t - 10, class: "axis-title" });
    t.textContent = yLabel;
    g.append(t);
  }
  svg.append(g);
}

function layerChart(study, tip) {
  const node = card(
    "Where sycophancy is readable",
    "Probe AUC for predicting caving from the final prompt token, by layer. The signal starts around layer 15; candidates 01-03 used layer 13.",
  );
  const w = 1120, h = 300, m = { l: 48, r: 20, t: 30, b: 42 };
  const layers = study.layers;
  const x = (l) => m.l + (l / 27) * (w - m.l - m.r);
  const y = (v) => m.t + (1 - (Math.max(0.4, Math.min(1, v)) - 0.4) / 0.6) * (h - m.t - m.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img",
    "aria-label": "Probe AUC by layer for discovery cross-validation and tuning transfer" });
  axes(svg, { x, y, w, h, m, xTicks: [0, 5, 10, 13, 15, 20, 22, 25, 27], yTicks: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    xLabel: "layer", yLabel: "AUC", yFmt: (t) => t.toFixed(1) });
  const band = layers.map((r) => `${x(r.layer)},${y(r.tuning_ci[1])}`).join(" ") + " " +
    [...layers].reverse().map((r) => `${x(r.layer)},${y(r.tuning_ci[0])}`).join(" ");
  svg.append(svgEl("polygon", { points: band, fill: COLORS.res, opacity: 0.14 }));
  const line = (key, color, dash) => svg.append(svgEl("polyline", {
    points: layers.map((r) => `${x(r.layer)},${y(r[key])}`).join(" "),
    fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "", "stroke-linejoin": "round",
  }));
  line("discovery_null_q95", "#6b6b6b", "4 4");
  line("discovery_auc", COLORS.amber);
  line("tuning_auc", COLORS.res);
  for (const [layer, text] of [[13, "L13: candidates 01-03"], [22, "L22: new SOM direction"]]) {
    svg.append(svgEl("line", { x1: x(layer), x2: x(layer), y1: m.t, y2: h - m.b, class: "marker" }));
    const label = svgEl("text", { x: x(layer) + 5, y: m.t + 12, class: "marker-label" });
    label.textContent = text;
    svg.append(label);
  }
  const hover = svgEl("line", { y1: m.t, y2: h - m.b, class: "crosshair", visibility: "hidden" });
  svg.append(hover);
  const hit = svgEl("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, fill: "transparent" });
  hit.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * w;
    const layer = Math.max(0, Math.min(27, Math.round(((px - m.l) / (w - m.l - m.r)) * 27)));
    const r = layers[layer];
    hover.setAttribute("x1", x(layer)); hover.setAttribute("x2", x(layer)); hover.setAttribute("visibility", "visible");
    tip.show(event, [`Layer ${layer}`, `Discovery CV AUC ${fmt(r.discovery_auc)} (null 95th pct ${fmt(r.discovery_null_q95)})`,
      `Tuning transfer AUC ${fmt(r.tuning_auc)} [${fmt(r.tuning_ci[0])}, ${fmt(r.tuning_ci[1])}]`]);
  });
  hit.addEventListener("mouseleave", () => { hover.setAttribute("visibility", "hidden"); tip.hide(); });
  svg.append(hit);
  node.append(legend([
    { label: "Tuning transfer (95% CI band)", color: COLORS.res },
    { label: "Discovery cross-validation", color: COLORS.amber },
    { label: "Permutation null, 95th pct", color: "#6b6b6b", dashed: true },
  ]), svg);
  return node;
}

function cellTotals(cell, splits) {
  const t = { sycophancy: 0, independent_resistance: 0 };
  for (const c of cell.counts) if (splits.includes(c.split)) t[c.label] += c.n;
  return t;
}

function somMap(study, tip) {
  const map = study.map;
  const node = card(
    `Layer-${map.layer} SOM, ${map.grid}x${map.grid}, seed ${map.seed}`,
    "Fit on sycophantic states only. Direction = selected prototype minus the resisting centroid.",
  );
  const body = el("div", "study-map-body");
  const w = 340, h = 290, r = 68;
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-hex", role: "img",
    "aria-label": "Self-organizing map cells with sycophantic and resisting counts" });
  const hex = (cx, cy) => Array.from({ length: 6 }, (_, i) => {
    const a = ((60 * i - 90) * Math.PI) / 180;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(" ");
  const step = r * Math.sqrt(3);
  for (const cell of map.cells) {
    const cx = 95 + cell.col * step + (cell.row % 2 ? step / 2 : 0);
    const cy = 80 + cell.row * r * 1.5;
    const fit = cellTotals(cell, ["FIT"]);
    const held = cellTotals(cell, ["SELECT", "CHECK"]);
    const g = svgEl("g", { class: `study-cell${cell.selected ? " selected" : ""}`, tabindex: 0 });
    g.append(svgEl("polygon", { points: hex(cx, cy) }));
    const total = fit.sycophancy + fit.independent_resistance;
    const barW = 84, share = total ? fit.sycophancy / total : 0;
    if (total) {
      g.append(svgEl("rect", { x: cx - barW / 2, y: cy + 14, width: Math.max(2, barW * share - 1), height: 7, rx: 2, fill: COLORS.syc }));
      g.append(svgEl("rect", { x: cx - barW / 2 + barW * share + 1, y: cy + 14, width: Math.max(2, barW * (1 - share) - 1), height: 7, rx: 2, fill: COLORS.res }));
    }
    const n = svgEl("text", { x: cx, y: cy - 4, "text-anchor": "middle", class: "cell-n" });
    n.textContent = fit.sycophancy;
    const sub = svgEl("text", { x: cx, y: cy + 38, "text-anchor": "middle", class: "cell-sub" });
    sub.textContent = cell.selected ? "selected" : `cell ${cell.id}`;
    g.append(n, sub);
    const lines = [`Cell ${cell.id}${cell.selected ? " (selected prototype)" : ""}`,
      `FIT: ${fit.sycophancy} sycophantic, ${fit.independent_resistance} resisting`,
      `Held-out: ${held.sycophancy} sycophantic, ${held.independent_resistance} resisting`];
    g.addEventListener("mousemove", (event) => tip.show(event, lines));
    g.addEventListener("mouseleave", () => tip.hide());
    g.addEventListener("focus", (event) => tip.show({ clientX: event.target.getBoundingClientRect().right, clientY: event.target.getBoundingClientRect().top }, lines));
    g.addEventListener("blur", () => tip.hide());
    svg.append(g);
  }
  const selected = map.cells.find((c) => c.selected);
  const allFit = map.cells.reduce((a, c) => {
    const t = cellTotals(c, ["FIT"]);
    return { s: a.s + t.sycophancy, r: a.r + t.independent_resistance };
  }, { s: 0, r: 0 });
  const selFit = cellTotals(selected, ["FIT"]);
  const side = el("div", "study-map-side");
  side.append(
    legend([{ label: "Sycophantic", color: COLORS.syc }, { label: "Resisting", color: COLORS.res }]),
    el("p", "study-note", `Cell number = sycophantic FIT states. The selected cell holds ${selFit.sycophancy} of ${allFit.s} sycophantic and ${selFit.independent_resistance} of ${allFit.r} resisting FIT states, so cells do not separate the two groups. The projection onto the direction does (right).`),
  );
  const stats = el("dl", "study-stats");
  for (const [k, v] of [
    ["SELECT AUC", fmt(map.select_auc)],
    ["CHECK AUC (held out)", `${fmt(map.check_auc)} [${fmt(map.check_ci[0])}, ${fmt(map.check_ci[1])}]`],
    ["Tuning AUC (report only)", `${fmt(map.tuning_auc)} [${fmt(map.tuning_ci[0])}, ${fmt(map.tuning_ci[1])}]`],
    ["Old layer-13 directions", `gaps ${map.old_layer13.gaps_sd.join(" / ")} SD, holdout AUC ${map.old_layer13.holdout_auc.join(" / ")}`],
  ]) { stats.append(el("dt", "", k), el("dd", "", v)); }
  side.append(stats);
  body.append(svg, side);
  node.append(body);
  return node;
}

function projectionChart(study, tip) {
  const hist = study.map.projection_histogram;
  const node = card("Projection onto the selected direction", "Held-out discovery states (SELECT + CHECK). Higher = closer to the sycophantic prototype.");
  const w = 620, h = 300, m = { l: 20, r: 16, t: 26, b: 42 };
  const edges = hist.edges;
  const syc = hist.counts["HELDOUT|sycophancy"], res = hist.counts["HELDOUT|independent_resistance"];
  const maxN = Math.max(...syc, ...res, 1);
  const mid = m.t + (h - m.t - m.b) / 2;
  const x = (v) => m.l + ((v - edges[0]) / (edges[edges.length - 1] - edges[0])) * (w - m.l - m.r);
  const yUp = (n) => (n / maxN) * ((h - m.t - m.b) / 2 - 4);
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img",
    "aria-label": "Mirrored histogram of projection scores for sycophantic and resisting held-out states" });
  const g = svgEl("g", { class: "study-axis" });
  g.append(svgEl("line", { x1: m.l, x2: w - m.r, y1: mid, y2: mid, class: "base" }));
  for (const t of [edges[0], 0, edges[edges.length - 1]]) {
    const label = svgEl("text", { x: x(t), y: h - m.b + 18, "text-anchor": "middle" });
    label.textContent = t.toFixed(0);
    g.append(label);
  }
  const xt = svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 6, "text-anchor": "middle", class: "axis-title" });
  xt.textContent = "projection";
  const up = svgEl("text", { x: m.l, y: m.t - 8, class: "axis-title" });
  up.textContent = "sycophantic above, resisting below";
  g.append(xt, up);
  svg.append(g);
  for (let i = 0; i < syc.length; i += 1) {
    const x0 = x(edges[i]) + 1, bw = Math.max(1, x(edges[i + 1]) - x(edges[i]) - 2);
    const lines = [`Projection ${edges[i].toFixed(1)} to ${edges[i + 1].toFixed(1)}`, `${syc[i]} sycophantic, ${res[i]} resisting`];
    for (const [n, color, dir] of [[syc[i], COLORS.syc, -1], [res[i], COLORS.res, 1]]) {
      if (!n) continue;
      const hgt = yUp(n);
      const bar = svgEl("rect", { x: x0, y: dir < 0 ? mid - hgt - 1 : mid + 1, width: bw, height: hgt, rx: 2, fill: color });
      bar.addEventListener("mousemove", (event) => tip.show(event, lines));
      bar.addEventListener("mouseleave", () => tip.hide());
      svg.append(bar);
    }
  }
  node.append(legend([{ label: "Sycophantic", color: COLORS.syc }, { label: "Resisting", color: COLORS.res }]), svg);
  return node;
}

function blindedTable(study, tip) {
  const node = card("Blinded paired candidates 01-03", "Unsupported-adoption change, SOM minus baseline, with family-bootstrap 95% CI. Below zero is better.");
  const w = 1120, rowH = 46, m = { l: 300, r: 120, t: 28, b: 36 };
  const rows = study.blinded;
  const h = m.t + rows.length * rowH + m.b;
  const lo = -10, hi = 15;
  const x = (v) => m.l + ((v - lo) / (hi - lo)) * (w - m.l - m.r);
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img",
    "aria-label": "Forest plot of blinded unsupported adoption changes by candidate" });
  const g = svgEl("g", { class: "study-axis" });
  for (const t of [-10, -5, 0, 5, 10, 15]) {
    g.append(svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, class: t === 0 ? "zero" : "grid" }));
    const label = svgEl("text", { x: x(t), y: h - m.b + 18, "text-anchor": "middle" });
    label.textContent = `${t > 0 ? "+" : ""}${t}`;
    g.append(label);
  }
  const target = svgEl("rect", { x: x(lo), y: m.t, width: x(-5) - x(lo), height: h - m.t - m.b, class: "target-zone" });
  g.append(target);
  const tl = svgEl("text", { x: x(-7.5), y: m.t - 10, "text-anchor": "middle", class: "target-label" });
  tl.textContent = "target zone";
  g.append(tl);
  const xt = svgEl("text", { x: x(2.5), y: h - 4, "text-anchor": "middle", class: "axis-title" });
  xt.textContent = "percentage points";
  g.append(xt);
  svg.append(g);
  rows.forEach((row, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    const name = svgEl("text", { x: 12, y: cy - 3, class: "row-label" });
    name.textContent = `${row.id}  ${row.label}`;
    const op = svgEl("text", { x: 12, y: cy + 13, class: "row-sub" });
    op.textContent = row.operator;
    svg.append(name, op);
    if (row.ci) svg.append(svgEl("line", { x1: x(Math.max(lo, row.ci[0])), x2: x(Math.min(hi, row.ci[1])), y1: cy, y2: cy, stroke: "#9a9a9a", "stroke-width": 2 }));
    const dot = svgEl("circle", { cx: x(row.primary_pp), cy, r: 5, fill: row.primary_pp > 0 ? COLORS.syc : COLORS.res, stroke: "#090b0d", "stroke-width": 2 });
    const lines = [`Candidate ${row.id}: ${row.label}`, `Adoption ${signed(row.primary_pp, 2)} pp${row.ci ? ` [${signed(row.ci[0], 2)}, ${signed(row.ci[1], 2)}]` : ""}`,
      `Neutral preserved ${signed(row.neutral_pp, 2)} pp`];
    dot.addEventListener("mousemove", (event) => tip.show(event, lines));
    dot.addEventListener("mouseleave", () => tip.hide());
    const value = svgEl("text", { x: w - m.r + 12, y: cy + 4, class: "row-value" });
    value.textContent = `${signed(row.primary_pp, 2)} pp`;
    svg.append(dot, value);
  });
  node.append(svg);
  return node;
}

function screenDeltas(screen, totals) {
  const base = screen.points.find((p) => p.strength === 0).counts;
  return screen.points.map((p) => ({
    strength: p.strength,
    repetitive: p.repetitive,
    raw: p.counts,
    flips_prevented: base.unsupported_flips - p.counts.unsupported_flips,
    neutral_lost: base.neutral_preserved - p.counts.neutral_preserved,
    updates_lost: base.justified_updates - p.counts.justified_updates,
    corrections_lost: base.warranted_corrections - p.counts.warranted_corrections,
    totals,
  }));
}

function screenChart(screen, totals, tip, yMax) {
  const node = el("div", "study-small");
  node.append(el("h3", "", screen.label));
  const data = screenDeltas(screen, totals);
  const w = 540, h = 260, m = { l: 36, r: 14, t: 28, b: 40 };
  const maxS = Math.max(...data.map((d) => d.strength));
  const x = (s) => m.l + (s / maxS) * (w - m.l - m.r);
  const y = (v) => m.t + (1 - (v + 2) / (yMax + 2)) * (h - m.t - m.b);
  const svg = svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img",
    "aria-label": `${screen.label}: change in counts versus push strength` });
  axes(svg, { x, y, w, h, m, xTicks: data.map((d) => d.strength), yTicks: [0, 5, 10, 15, 20].filter((t) => t <= yMax),
    xLabel: "push strength", yLabel: "count vs baseline", xFmt: (t) => String(t) });
  for (const s of SERIES) {
    svg.append(svgEl("polyline", { points: data.map((d) => `${x(d.strength)},${y(d[s.key])}`).join(" "),
      fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }));
    for (const d of data) svg.append(svgEl("circle", { cx: x(d.strength), cy: y(d[s.key]), r: 3.5, fill: s.color, stroke: "#090b0d", "stroke-width": 2 }));
  }
  for (const d of data) {
    const hit = svgEl("rect", { x: x(d.strength) - 14, y: m.t, width: 28, height: h - m.t - m.b, fill: "transparent" });
    const lines = [`Strength ${d.strength}`, `Unsupported flips ${d.raw.unsupported_flips}/90 (${d.flips_prevented} prevented)`,
      `Neutral preserved ${d.raw.neutral_preserved}/127`, `Justified updates ${d.raw.justified_updates}/25`,
      `Warranted corrections ${d.raw.warranted_corrections}/13`, `Repetitive outputs ${(100 * d.repetitive).toFixed(1)}%`];
    hit.addEventListener("mousemove", (event) => tip.show(event, lines));
    hit.addEventListener("mouseleave", () => tip.hide());
    svg.append(hit);
  }
  node.append(svg);
  return node;
}

function screens(study, tip) {
  const node = card("Push steering dose screens", study.screen_note);
  const all = study.screens.flatMap((s) => screenDeltas(s, study.screen_totals));
  const yMax = Math.max(10, ...all.flatMap((d) => SERIES.map((s) => d[s.key])));
  node.append(legend(SERIES.map((s) => ({ label: s.label, color: s.color }))));
  const grid = el("div", "study-smalls");
  for (const screen of study.screens) grid.append(screenChart(screen, study.screen_totals, tip, Math.ceil(yMax / 5) * 5));
  node.append(grid);
  const details = el("details", "study-table");
  details.append(el("summary", "", "Table view"));
  const table = el("table");
  const head = el("tr");
  for (const c of ["screen", "strength", "unsupported flips /90", "resisted", "misleading /11", "insufficient /39", "neutral /127", "updates /25", "corrections /13", "repetitive"]) head.append(el("th", "", c));
  table.append(head);
  for (const screen of study.screens) for (const p of screen.points) {
    const tr = el("tr");
    for (const v of [screen.id, p.strength, p.counts.unsupported_flips, p.counts.unsupported_resisted, p.counts.misleading_flips,
      p.counts.insufficient_flips, p.counts.neutral_preserved, p.counts.justified_updates, p.counts.warranted_corrections,
      `${(100 * p.repetitive).toFixed(1)}%`]) tr.append(el("td", "", String(v)));
    table.append(tr);
  }
  details.append(table);
  node.append(details);
  return node;
}

export function createStudyView(root) {
  let loaded = false;
  async function activate() {
    if (loaded) return;
    loaded = true;
    root.replaceChildren(el("div", "study-loading", "Loading study…"));
    let study, pair = null, pairError = null;
    try {
      const [response, pairResponse] = await Promise.all([fetch("/api/som-study"), fetch("/api/som-study/matched-pair")]);
      study = await response.json();
      if (!response.ok) throw new Error(study?.error?.message || response.status);
      const pairBody = await pairResponse.json();
      if (pairResponse.ok) pair = pairBody;
      else pairError = pairBody?.error?.message || String(pairResponse.status);
    } catch (error) {
      loaded = false;
      root.replaceChildren(el("div", "study-loading error", `Could not load study: ${error.message}`));
      return;
    }
    const shell = el("div", "study-shell");
    const tip = tooltip(root);
    const head = el("div", "study-heading");
    const titles = el("div");
    titles.append(el("div", "study-eyebrow", "som steering study"), el("h1", "", "Sycophancy SOM: Llama 3.2 3B Instruct"));
    head.append(titles, el("span", "study-chip", "development only · final test unopened"));
    const caveat = el("p", "study-caveat");
    caveat.append(el("b", "", "Status: "), document.createTextNode(study.headline), el("span", "study-target", ` Target: ${study.target}`));
    const top = el("div", "study-grid two");
    top.append(somMap(study, tip), projectionChart(study, tip));
    shell.append(head);
    if (pair) {
      shell.append(...matchedPairSection(pair, tip, { el, svgEl, fmt, signed, card, legend, axes }));
    } else {
      shell.append(el("p", "study-loading", `Matched-pair study not available: ${pairError}`));
    }
    const earlier = el("div", "study-divider earlier");
    earlier.append(el("div", "study-eyebrow", "earlier work, 19 to 27 Oct 2026"), el("h2", "", "Cross-conversation directions: signal, map, blinded candidates and dose screens"));
    shell.append(earlier, caveat, layerChart(study, tip), top, blindedTable(study, tip), screens(study, tip),
      el("p", "study-foot", `${study.model} · ${study.revision.slice(0, 12)} · ${Object.keys(study.source_sha256).length + (pair ? Object.keys(pair.source_sha256).length : 0)} hashed source files`));
    root.replaceChildren(shell, tip.node);
  }
  return { activate };
}
