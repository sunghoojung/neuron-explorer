// Matched-pair direction study v1 (instruction contrast): the selected layer-12
// map, the fitted difference-vector cloud with its prototypes, the Stage A
// screen, the blind Stage B and CHECK results against the instruction effects,
// and the tuning-stage endpoints against the plan's bounds. Read-only view of
// /api/som-study/matched-pair (built by research/build_som_matched_pair.py).

const PAL = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", amber: "#d9a54c", gray: "#8a8a8a", dim: "#55585d" };
const SURFACE = "#090b0d";
const INSTRUCTIONS = [
  { key: "resist_firm", label: "resist_firm", color: PAL.blue },
  { key: "resist_evidence", label: "resist_evidence", color: PAL.orange },
  { key: "resist_cite", label: "resist_cite", color: PAL.aqua },
];
const SCENARIO_GROUPS = [
  { key: "unsupported_pressure", label: "unsupported pressure", color: PAL.blue, match: (s) => s === "unsupported_pressure" },
  { key: "misleading_or_irrelevant_support", label: "misleading support", color: PAL.orange, match: (s) => s === "misleading_or_irrelevant_support" },
  { key: "other", label: "insufficient evidence or justified update", color: PAL.aqua, match: () => true },
];
const SHORT = {
  unsupported_pressure: "unsupported pressure",
  misleading_or_irrelevant_support: "misleading support",
  insufficient_evidence: "insufficient evidence",
  justified_evidence_update: "justified update",
};

const pts = (v, d = 1) => (v == null ? "n/a" : `${v > 0 ? "+" : v < 0 ? "-" : ""}${Math.abs(v).toFixed(d)}`);
const ci = (c, d = 1) => (c ? `[${pts(c[0], d)}, ${pts(c[1], d)}]` : "");
const shortName = (name) => name.replace(/^fit0\d:/, "").replace("resist-", "");

function table(ui, head, rows, caption) {
  const details = ui.el("details", "study-table-wrap");
  details.append(ui.el("summary", "", caption || "table"));
  const t = ui.el("table", "study-table");
  const thead = ui.el("thead");
  const tr = ui.el("tr");
  for (const h of head) tr.append(ui.el("th", "", h));
  thead.append(tr);
  const tbody = ui.el("tbody");
  for (const r of rows) {
    const row = ui.el("tr");
    for (const v of r) row.append(ui.el("td", "", v));
    tbody.append(row);
  }
  t.append(thead, tbody);
  details.append(t);
  return details;
}

function hover(tip, nodes, lines) {
  for (const n of nodes) {
    n.addEventListener("mousemove", (event) => tip.show(event, lines));
    n.addEventListener("mouseleave", () => tip.hide());
  }
}

function stackedBar(ui, g, cx, y, width, parts, height = 5) {
  const total = parts.reduce((a, p) => a + p.n, 0);
  if (!total) return;
  const gap = 2, usable = width - gap * (parts.filter((p) => p.n).length - 1);
  let x = cx - width / 2;
  for (const p of parts) {
    if (!p.n) continue;
    const w = Math.max(1, (p.n / total) * usable);
    g.append(ui.svgEl("rect", { x, y, width: w, height, rx: 1.5, fill: p.color }));
    x += w + gap;
  }
}

function hexPoints(cx, cy, r) {
  return Array.from({ length: 6 }, (_, i) => {
    const a = ((60 * i - 90) * Math.PI) / 180;
    return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
  }).join(" ");
}

function pairMap(mp, tip, ui) {
  const map = mp.map;
  const node = ui.card(
    `Layer-${map.layer} map, ${map.grid}x${map.grid}, seed ${map.seed}`,
    `${map.n_vectors} vectors: ${map.n_fit_rows} FIT conversations x 3 resist instructions, state(instruction) minus state(plain). Number = vectors in the cell, bar = instruction mix, KL = Stage A score at c = 4.`,
  );
  const body = ui.el("div", "study-map-body");
  const w = 350, h = 320, r = 58;
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-hex", role: "img",
    "aria-label": "Nine SOM cells with vector counts, instruction mix and Stage A scores" });
  const step = r * Math.sqrt(3);
  for (const cell of map.cells) {
    const cx = 76 + cell.col * step + (cell.row % 2 ? step / 2 : 0);
    const cy = 70 + cell.row * r * 1.5;
    const g = ui.svgEl("g", { class: `study-cell${cell.selected ? " selected" : ""}`, tabindex: 0 });
    g.append(ui.svgEl("polygon", { points: hexPoints(cx, cy, r) }));
    const n = ui.svgEl("text", { x: cx, y: cy - 9, "text-anchor": "middle", class: "cell-n" });
    n.textContent = cell.support;
    g.append(n);
    stackedBar(ui, g, cx, cy, 70, INSTRUCTIONS.map((i) => ({ n: cell.by_instruction[i.key] || 0, color: i.color })));
    const kl = ui.svgEl("text", { x: cx, y: cy + 20, "text-anchor": "middle", class: "cell-kl" });
    kl.textContent = cell.stage_a ? `KL ${pts(cell.stage_a.mean_delta_kl, 2)}` : "KL n/a";
    const sub = ui.svgEl("text", { x: cx, y: cy + 34, "text-anchor": "middle", class: "cell-sub" });
    sub.textContent = cell.selected ? "selected" : `cell ${cell.id}`;
    g.append(kl, sub);
    const inst = INSTRUCTIONS.map((i) => `${cell.by_instruction[i.key] || 0} ${i.label}`).join(", ");
    const scen = Object.entries(cell.by_scenario).map(([k, v]) => `${v} ${SHORT[k] || k}`).join(", ");
    const lines = [`Cell ${cell.id}${cell.selected ? " (selected)" : ""}: ${cell.support} vectors`, `Instructions: ${inst}`, `Scenarios: ${scen}`,
      `Cosine to the pool mean ${cell.cos_mean.toFixed(2)}`,
      cell.stage_a ? `Stage A at c = 4: ${pts(cell.stage_a.mean_delta_kl, 3)} nats ${ci(cell.stage_a.ci95, 3)}` : "Stage A: not scored"];
    hover(tip, [g], lines);
    g.addEventListener("focus", (event) => { const b = event.target.getBoundingClientRect(); tip.show({ clientX: b.right, clientY: b.top }, lines); });
    g.addEventListener("blur", () => tip.hide());
    svg.append(g);
  }
  const side = ui.el("div", "study-map-side");
  side.append(ui.legend(INSTRUCTIONS.map((i) => ({ label: i.label, color: i.color }))));
  side.append(ui.el("p", "study-note",
    "Cells separate instruction wording (0, 5, 6 are pure firm, evidence, cite) and conversation type (1, 2, 7 hold the insufficient-evidence and justified-update conversations). Inside pressure conversations, cell 4 is unsupported pressure and cell 8 misleading support. Stage A rewards reproducing the instruction on pressure rows, so cell 0 wins and is the only cell carried forward."));
  const wr = map.within_row_instruction_cosine;
  const stats = ui.el("dl", "study-stats pair");
  for (const [k, v] of [
    ["SOM", `MiniSom ${map.som_settings.version_used}, ${map.som_settings.topology}, sigma ${map.som_settings.sigma}, lr ${map.som_settings.learning_rate}, ${map.som_settings.n_updates} updates`],
    ["Quantization / topographic error", `${map.quantization_error.toFixed(2)} / ${map.topographic_error == null ? "n/a" : map.topographic_error.toFixed(2)}`],
    ["Instruction cosine, same conversation", `firm-evidence ${wr["resist_firm|resist_evidence"].toFixed(2)}, firm-cite ${wr["resist_firm|resist_cite"].toFixed(2)}, evidence-cite ${wr["resist_evidence|resist_cite"].toFixed(2)}`],
    ["Selected cell vs pool mean", `cosine ${map.cells.find((c) => c.selected).cos_mean.toFixed(2)}`],
  ]) stats.append(ui.el("dt", "", k), ui.el("dd", "", v));
  body.append(svg, side);
  node.append(body, stats);
  node.append(table(ui, ["cell", "vectors", ...INSTRUCTIONS.map((i) => i.label), "unsupported", "misleading", "insufficient", "justified", "cos mean", "Stage A KL"],
    map.cells.map((c) => [c.id + (c.selected ? " (selected)" : ""), c.support, ...INSTRUCTIONS.map((i) => c.by_instruction[i.key] || 0),
      c.by_scenario.unsupported_pressure || 0, c.by_scenario.misleading_or_irrelevant_support || 0, c.by_scenario.insufficient_evidence || 0,
      c.by_scenario.justified_evidence_update || 0, c.cos_mean.toFixed(2), c.stage_a ? `${pts(c.stage_a.mean_delta_kl, 3)} ${ci(c.stage_a.ci95, 3)}` : "n/a"]),
    "cell composition"));
  return node;
}

function latentCard(mp, tip, ui) {
  const lat = mp.latent, map = mp.map;
  const node = ui.card(
    "Latent space the map was fit on",
    `${lat.n_points} difference vectors in their first two principal components (${(100 * lat.explained_variance_ratio[0]).toFixed(0)}% and ${(100 * lat.explained_variance_ratio[1]).toFixed(0)}% of variance). Hexagons are the nine prototypes; the arrow is the selected steering direction (${(100 * Math.hypot(...lat.selected_direction_in_plane)).toFixed(0)}% of it lies in this plane).`,
  );
  const modes = [
    { key: "instruction", label: "instruction", groups: INSTRUCTIONS.map((i) => ({ ...i, match: (p) => p.instruction === i.key })) },
    { key: "scenario", label: "scenario", groups: SCENARIO_GROUPS.map((g) => ({ ...g, match: (p) => g.match(p.scenario) })) },
    { key: "cell", label: "selected cell", groups: [
      { key: "sel", label: `cell ${map.selected_cell} (selected)`, color: PAL.amber, match: (p) => p.cell === map.selected_cell },
      { key: "rest", label: "other cells", color: PAL.dim, match: () => true },
    ] },
  ];
  const bar = ui.el("div", "study-toolbar");
  const toggle = ui.el("div", "study-toggle");
  bar.append(ui.el("span", "study-toolbar-label", "colour by"), toggle);
  const legendSlot = ui.el("div", "study-legend-slot");
  const w = 680, h = 500, m = { l: 40, r: 14, t: 14, b: 36 };
  const xs = lat.points.map((p) => p.x), ys = lat.points.map((p) => p.y);
  const pad = 0.05;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const xr = [x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)], yr = [y0 - pad * (y1 - y0), y1 + pad * (y1 - y0)];
  const x = (v) => m.l + ((v - xr[0]) / (xr[1] - xr[0])) * (w - m.l - m.r);
  const y = (v) => m.t + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (h - m.t - m.b);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-latent", role: "img",
    "aria-label": "Scatter of instruction-contrast vectors in the first two principal components with SOM prototypes" });
  const intTicks = (lo, hi) => { const out = []; for (let t = Math.ceil(lo); t <= Math.floor(hi); t += 1) out.push(t); return out; };
  ui.axes(svg, { x, y, w, h, m, xTicks: intTicks(xr[0], xr[1]), yTicks: intTicks(yr[0], yr[1]), xLabel: "PC1", yLabel: "PC2", xFmt: String, yFmt: String });
  const dots = ui.svgEl("g");
  const nodes = lat.points.map((p) => {
    const g = ui.svgEl("g");
    const hit = ui.svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 6, fill: "transparent" });
    const dot = ui.svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 2.2, "fill-opacity": 0.72 });
    hover(tip, [g], [`Cell ${p.cell}, ${p.instruction}`, `${SHORT[p.scenario] || p.scenario} (${p.source})`, `Projection on the selected direction ${p.proj_selected.toFixed(2)}`]);
    g.append(hit, dot);
    dots.append(g);
    return { dot, p };
  });
  svg.append(dots);
  const dir = lat.selected_direction_in_plane;
  const span = Math.min(x1 - x0, y1 - y0) * 0.36;
  const norm = Math.hypot(dir[0], dir[1]) || 1;
  const ex = (dir[0] / norm) * span, ey = (dir[1] / norm) * span;
  const defs = ui.svgEl("defs");
  const marker = ui.svgEl("marker", { id: "pair-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
  marker.append(ui.svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: PAL.amber }));
  defs.append(marker);
  svg.append(defs);
  svg.append(ui.svgEl("line", { x1: x(0), y1: y(0), x2: x(ex), y2: y(ey), stroke: PAL.amber, "stroke-width": 1.6, "stroke-linecap": "round", "marker-end": "url(#pair-arrow)" }));
  svg.append(ui.svgEl("circle", { cx: x(0), cy: y(0), r: 3, fill: PAL.amber }));
  for (const pr of lat.prototypes) {
    const sel = pr.cell === map.selected_cell;
    const g = ui.svgEl("g", { class: `study-proto${sel ? " selected" : ""}`, tabindex: 0 });
    g.append(ui.svgEl("polygon", { points: hexPoints(x(pr.x), y(pr.y), 10) }));
    const t = ui.svgEl("text", { x: x(pr.x), y: y(pr.y) + 3.5, "text-anchor": "middle" });
    t.textContent = pr.cell;
    g.append(t);
    const c = map.cells[pr.cell];
    hover(tip, [g], [`Prototype of cell ${pr.cell}${sel ? " (selected)" : ""}`, `${c.support} vectors; cosine to the pool mean ${c.cos_mean.toFixed(2)}`,
      c.stage_a ? `Stage A ${pts(c.stage_a.mean_delta_kl, 3)} nats` : "Stage A: not scored"]);
    svg.append(g);
  }
  const apply = (mode) => {
    for (const { dot, p } of nodes) {
      const group = mode.groups.find((g) => g.match(p));
      dot.setAttribute("fill", group ? group.color : PAL.dim);
    }
    legendSlot.replaceChildren(ui.legend(mode.groups.map((g) => ({ label: g.label, color: g.color })).concat([{ label: "steering direction", color: PAL.amber }])));
    for (const b of toggle.children) b.classList.toggle("on", b.dataset.mode === mode.key);
  };
  for (const mode of modes) {
    const b = ui.el("button", "", mode.label);
    b.type = "button";
    b.dataset.mode = mode.key;
    b.addEventListener("click", () => apply(mode));
    toggle.append(b);
  }
  apply(modes[0]);
  node.append(bar, legendSlot, svg);
  const counts = {};
  for (const p of lat.points) { counts[p.cell] = counts[p.cell] || {}; counts[p.cell][p.instruction] = (counts[p.cell][p.instruction] || 0) + 1; }
  node.append(table(ui, ["cell", ...INSTRUCTIONS.map((i) => i.label), "total"],
    Object.keys(counts).sort((a, b) => a - b).map((c) => [c, ...INSTRUCTIONS.map((i) => counts[c][i.key] || 0), Object.values(counts[c]).reduce((a, v) => a + v, 0)]),
    "vectors per cell and instruction"));
  return node;
}

function stageACard(mp, tip, ui) {
  const sa = mp.stage_a;
  const best = sa.top[0];
  const node = ui.card(
    `Stage A screen: ${sa.n_configurations} configurations on ${sa.n_rows} SELECT pressure rows`,
    "Label-free: how far a push moves the pre-reply next-token distribution toward the instructed one (KL reduction, nats). Best configuration per layer and grid; the dashed line is the instruction's own shift.",
  );
  const w = 1120, h = 280, m = { l: 48, r: 20, t: 24, b: 40 };
  const layers = sa.layers;
  const x = (l) => m.l + ((l - layers[0]) / (layers[layers.length - 1] - layers[0])) * (w - m.l - m.r);
  const yMax = 1.0;
  const y = (v) => m.t + (1 - v / yMax) * (h - m.t - m.b);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg capped", role: "img",
    "aria-label": "Best Stage A KL reduction by layer for the mean direction and the best 2x2 and 3x3 cells" });
  ui.axes(svg, { x, y, w, h, m, xTicks: layers, yTicks: [0, 0.25, 0.5, 0.75, 1.0], xLabel: "layer", yLabel: "KL reduction (nats)", yFmt: (t) => t.toFixed(2) });
  svg.append(ui.svgEl("line", { x1: m.l, x2: w - m.r, y1: y(sa.instruction_kl_ceiling), y2: y(sa.instruction_kl_ceiling), class: "marker" }));
  const cl = ui.svgEl("text", { x: w - m.r, y: y(sa.instruction_kl_ceiling) - 6, "text-anchor": "end", class: "marker-label" });
  cl.textContent = `instruction itself ${sa.instruction_kl_ceiling.toFixed(2)}`;
  svg.append(cl);
  const series = [
    { grid: "1", label: "1x1 (plain mean)", color: PAL.blue },
    { grid: "2", label: "best 2x2 cell", color: PAL.orange },
    { grid: "3", label: "best 3x3 cell", color: PAL.aqua },
  ].filter((s) => sa.best_by_layer_grid[s.grid]);
  for (const s of series) {
    const rows = sa.best_by_layer_grid[s.grid];
    svg.append(ui.svgEl("polyline", { points: rows.map((r) => `${x(r.layer)},${y(r.mean_delta_kl)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    for (const r of rows) {
      const dot = ui.svgEl("circle", { cx: x(r.layer), cy: y(r.mean_delta_kl), r: 4, fill: s.color, stroke: SURFACE, "stroke-width": 2 });
      const hit = ui.svgEl("circle", { cx: x(r.layer), cy: y(r.mean_delta_kl), r: 12, fill: "transparent" });
      hover(tip, [hit, dot], [`${s.label}, layer ${r.layer}`, `${shortName(r.direction)} at c = ${r.c}: ${pts(r.mean_delta_kl, 3)} nats ${ci(r.ci95, 3)}`,
        `KL(plain, steered) ${r.kl_plain_steered == null ? "n/a" : r.kl_plain_steered.toFixed(2)}`]);
      svg.append(hit, dot);
    }
  }
  if (best) {
    const lbl = ui.svgEl("text", { x: x(best.layer), y: y(best.mean_delta_kl) - 14, "text-anchor": "middle", class: "marker-label" });
    lbl.textContent = `best ${shortName(best.direction)}, c = ${best.c}: ${pts(best.mean_delta_kl, 3)}`;
    svg.append(lbl);
  }
  const meanAtBest = sa.best_by_layer_grid["1"].find((r) => r.layer === best.layer)?.mean_delta_kl || 0;
  node.append(ui.legend(series.map((s) => ({ label: s.label, color: s.color })).concat([{ label: "instruction ceiling", color: PAL.gray, dashed: true }])), svg);
  node.append(ui.el("p", "study-note", `Best cell recovers ${(100 * best.mean_delta_kl / sa.instruction_kl_ceiling).toFixed(0)}% of the instruction's shift; the plain mean at the same layer ${(100 * meanAtBest / sa.instruction_kl_ceiling).toFixed(0)}%. The six carried forward are within 0.015 nats of each other.`));
  node.append(table(ui, ["rank", "direction", "layer", "grid", "seed", "cell", "c", "s_L", "KL reduction [95% CI]", "KL(plain, steered)"],
    sa.top.map((t) => [t.rank, t.direction, t.layer, `${t.grid}x${t.grid}`, t.seed, t.cell, t.c, t.s_L.toFixed(2), `${pts(t.mean_delta_kl, 3)} ${ci(t.ci95, 3)}`, t.kl_plain_steered == null ? "n/a" : t.kl_plain_steered.toFixed(2)]),
    "the six configurations carried to Stage B"));
  return node;
}

function forestRows(mp) {
  const g0 = mp.gate0, sb = mp.stage_b, ck = mp.check;
  const rows = [{ header: `Instructions, FIT rows, blind (plain cave share ${g0.plain_cave_share.toFixed(2)})` }];
  for (const r of g0.rows) rows.push({ label: r.instruction, meta: `cave ${g0.plain_cave_share.toFixed(2)} to ${r.cave_share.toFixed(2)}, evidence ${r.evidence_rate.toFixed(2)}`, v: r.change_points, ci: r.ci95_points, kind: "ref" });
  rows.push({ header: `Stage B, ${sb.n_rows} SELECT pressure rows, blind, 8 replies per row` });
  for (const r of sb.rows) rows.push({ label: `${shortName(r.config)}${r.selected ? " (selected)" : ""}`, meta: `layer ${r.layer}, ${r.grid}x${r.grid}, c = ${r.c}; cave ${r.plain_cave_rate.toFixed(2)} to ${r.cave_rate.toFixed(2)}; evidence ${pts(r.evidence_change_points)}; truncated ${pts(r.degenerate_change_points)}`, v: r.change_points, ci: r.ci95_points, kind: r.selected ? "selected" : "steer" });
  rows.push({ header: `CHECK, ${ck.n_pressure_rows} held-out pressure rows, blind, opened once` });
  rows.push({ label: "primary endpoint", meta: `cave ${ck.primary.plain_cave_rate.toFixed(2)} to ${ck.primary.cave_rate.toFixed(2)}; evidence ${pts(ck.evidence.change_points)} ${ci(ck.evidence.ci95_points)}; truncated ${pts(ck.degenerate.change_points)}`, v: ck.primary.change_points, ci: ck.primary.ci95_points, kind: "check" });
  const s = ck.sensitivity;
  for (const [key, label] of [["exclude_degenerate", "truncated replies excluded"], ["reweighted", "proxy-hold replies reweighted"], ["degenerate_as_cave", "truncated replies counted as caves"]]) {
    rows.push({ label, meta: "sensitivity", v: s[key].change_points, ci: s[key].ci95_points, kind: "sens" });
  }
  for (const [key, v] of Object.entries(ck.by_scenario)) rows.push({ label: SHORT[key] || key, meta: `cave share, ${v.n} rows`, v: v.change_points, ci: v.ci95_points, kind: "sens" });
  return rows;
}

function forest(ui, tip, rows, { lo, hi, ticks, zone, zoneLabel, colorOf, radiusOf, extra, right = 176 }) {
  const w = 1120, rowH = 30, headH = 34, m = { l: 236, r: right, t: 10, b: 34 };
  const heights = rows.map((r) => (r.header ? headH : rowH));
  const h = m.t + heights.reduce((a, b) => a + b, 0) + m.b;
  const clamp = (v) => Math.max(lo, Math.min(hi, v));
  const x = (v) => m.l + ((clamp(v) - lo) / (hi - lo)) * (w - m.l - m.r);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg capped", role: "img", "aria-label": "Forest plot" });
  const g = ui.svgEl("g", { class: "study-axis" });
  if (zone) {
    g.append(ui.svgEl("rect", { x: x(zone[0]), y: m.t, width: x(zone[1]) - x(zone[0]), height: h - m.t - m.b, class: "target-zone" }));
    const tl = ui.svgEl("text", { x: x(zone[0]) + 6, y: h - m.b - 6, class: "zone-label" });
    tl.textContent = zoneLabel;
    g.append(tl);
  }
  for (const t of ticks) {
    g.append(ui.svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, class: t === 0 ? "zero" : "grid" }));
    const label = ui.svgEl("text", { x: x(t), y: h - m.b + 16, "text-anchor": "middle" });
    label.textContent = pts(t, 0);
    g.append(label);
  }
  const xt = ui.svgEl("text", { x: (m.l + w - m.r) / 2, y: h - 4, "text-anchor": "middle", class: "axis-title" });
  xt.textContent = "percentage points";
  g.append(xt);
  svg.append(g);
  let cy = m.t;
  rows.forEach((row, i) => {
    const hh = heights[i];
    if (row.header) {
      svg.append(ui.svgEl("line", { x1: 0, x2: w, y1: cy + hh - 20, y2: cy + hh - 20, class: "rule" }));
      const t = ui.svgEl("text", { x: 0, y: cy + hh - 6, class: "row-head" });
      t.textContent = row.header;
      svg.append(t);
      cy += hh;
      return;
    }
    const mid = cy + hh / 2;
    const name = ui.svgEl("text", { x: 0, y: mid + 4, class: `row-label${row.kind === "ref" || row.kind === "sens" ? " muted" : ""}` });
    name.textContent = row.label;
    svg.append(name);
    const color = colorOf(row);
    if (row.ci) {
      svg.append(ui.svgEl("line", { x1: x(row.ci[0]), x2: x(row.ci[1]), y1: mid, y2: mid, stroke: color, "stroke-width": 1.6, "stroke-linecap": "round" }));
      if (row.ci[0] < lo) { const t = ui.svgEl("text", { x: x(lo) - 5, y: mid + 3.5, "text-anchor": "end", class: "row-sub" }); t.textContent = pts(row.ci[0], 0); svg.append(t); }
    }
    const dot = ui.svgEl("circle", { cx: x(row.v), cy: mid, r: radiusOf(row), fill: color, stroke: SURFACE, "stroke-width": 2 });
    const hit = ui.svgEl("rect", { x: 0, y: cy, width: w, height: hh, fill: "transparent" });
    hover(tip, [hit, dot], [row.label, `${pts(row.v)} points ${ci(row.ci)}`, row.meta]);
    const value = ui.svgEl("text", { x: w - m.r + 12, y: mid + 4, class: "row-value" });
    value.textContent = `${pts(row.v)}  ${ci(row.ci)}`;
    svg.append(hit, dot, value);
    if (extra) extra(svg, row, cy, hh, mid, x, w);
    cy += hh;
  });
  return svg;
}

function forestCard(mp, tip, ui) {
  const node = ui.card(
    "Blind cave-rate change versus the unmodified model",
    "Percentage points, paired per conversation, family-bootstrap 95% intervals; below zero is less caving. Hover a row for its detail.",
  );
  const rows = forestRows(mp);
  const colorOf = (row) => (row.kind === "ref" ? PAL.gray : row.kind === "sens" ? PAL.dim : row.kind === "selected" || row.kind === "check" ? PAL.blue : PAL.aqua);
  const radiusOf = (row) => (row.kind === "check" ? 5.5 : row.kind === "selected" ? 5 : 4);
  const svg = forest(ui, tip, rows, { lo: -36, hi: 16, ticks: [-30, -20, -10, 0, 10], zone: [-36, -10], zoneLabel: "CHECK threshold: at least -10", colorOf, radiusOf });
  node.append(ui.legend([{ label: "instruction (reference)", color: PAL.gray }, { label: "Stage B configuration", color: PAL.aqua }, { label: "selected configuration and CHECK", color: PAL.blue }, { label: "sensitivity and per-scenario", color: PAL.dim }]), svg);
  const ck = mp.check;
  node.append(ui.el("p", "study-note",
    `CHECK: ${ck.rows_improved_same_worse[0]} rows improved, ${ck.rows_improved_same_worse[1]} unchanged, ${ck.rows_improved_same_worse[2]} worse; lost caves became holds (hold share ${ck.label_shares.plain.hold.toFixed(2)} to ${ck.label_shares[ck.condition].hold.toFixed(2)}). Panels agreed on ${(100 * ck.panels.agreement_overall).toFixed(1)}% of ${ck.panels.n_items} replies. The frozen rule is met by 0.2 points on the primary and 0.4 on the evidence guard.`));
  node.append(table(ui, ["row", "change", "95% CI", "detail"], rows.filter((r) => !r.header).map((r) => [r.label, pts(r.v), ci(r.ci), r.meta]), "every row of the plot"));
  return node;
}

function tuningCard(mp, tip, ui) {
  const tu = mp.tuning;
  const node = ui.card(
    "Tuning stage: the same configuration on the 305 frozen histories",
    "Seven blind endpoints, steered minus baseline in percentage points, equal-family weighting, family-bootstrap 95% intervals. Dashed ticks are the plan's bounds.",
  );
  const rows = tu.endpoints.map((e) => ({
    label: e.label.replace(" (primary)", ""),
    meta: `${e.pairs} pairs, ${e.families} families; ${e.baseline.toFixed(2)} to ${e.steered.toFixed(2)}; raw panels ${pts(tu.variants["reviewer-A"][e.key].change_points)} / ${pts(tu.variants["reviewer-B"][e.key].change_points)}`,
    v: e.change_points, ci: e.ci95_points, kind: e.role.startsWith("preserve") ? "preserve" : e.role === "higher is better" ? "up" : "down",
    bound: e.key === "unsupported_adoption" ? tu.reduction_target.points : e.bound_points, passes: e.passes, primary: e.key === "unsupported_adoption",
  }));
  const colorOf = (row) => (row.kind === "preserve" ? PAL.orange : row.kind === "up" ? PAL.aqua : PAL.blue);
  const extra = (svg, row, cy, hh, mid, x, w) => {
    if (row.bound != null) svg.append(ui.svgEl("line", { x1: x(row.bound), x2: x(row.bound), y1: cy + 5, y2: cy + hh - 5, class: "bound" }));
    if (row.passes != null) {
      const chip = ui.svgEl("text", { x: w, y: mid + 4, "text-anchor": "end", class: `row-chip ${row.passes ? "pass" : "fail"}` });
      chip.textContent = row.passes ? "met" : "not met";
      svg.append(chip);
    }
  };
  const svg = forest(ui, tip, rows, { lo: -70, hi: 56, ticks: [-60, -40, -20, 0, 20, 40], colorOf, radiusOf: (row) => (row.primary ? 5.5 : 4.5), extra, right: 250 });
  node.append(ui.legend([{ label: "adoption (lower is better)", color: PAL.blue }, { label: "appropriate uncertainty (higher is better)", color: PAL.aqua }, { label: "preservation", color: PAL.orange }, { label: "plan bound", color: PAL.gray, dashed: true }]), svg);
  const c = tu.primary_subtype_counts;
  const failed = Object.entries(tu.gates.preservation).filter(([, v]) => !v).map(([k]) => k.replace(/_/g, " "));
  node.append(ui.el("p", "study-note",
    `Raw counts on the 90 primary histories: adoption ${(c.baseline.sycophantic_flip || 0) + (c.baseline.new_unsupported_adoption || 0)} to ${(c.som.sycophantic_flip || 0) + (c.som.new_unsupported_adoption || 0)}; the family-weighted ${pts(tu.endpoints[0].change_points, 0)} comes from small QASC and NQ families while the large SQuAD families did not move. Reduction gate ${tu.gates.reduction ? "met" : "not met"} (interval includes zero); preservation not met for ${failed.join(", ")}. Confirmation set not designated; final test unopened.`));
  node.append(table(ui, ["endpoint", "pairs / families", "baseline", "steered", "adjudicated", "95% CI", "raw A", "raw B", "bound", "result"],
    tu.endpoints.map((r) => [r.label, `${r.pairs} / ${r.families}`, r.baseline.toFixed(3), r.steered.toFixed(3), pts(r.change_points), ci(r.ci95_points), pts(tu.variants["reviewer-A"][r.key].change_points), pts(tu.variants["reviewer-B"][r.key].change_points),
      r.key === "unsupported_adoption" ? "-5 and -20% rel., CI < 0" : r.bound_points == null ? "none" : pts(r.bound_points, 0), r.passes == null ? "reported" : r.passes ? "met" : "not met"]),
    "all seven endpoints with raw-panel sensitivity"));
  return node;
}

function cosineCard(mp, tip, ui) {
  const dc = mp.direction_cosines;
  const node = ui.card("Are the six directions different?", "Cosine between the unit directions carried to Stage B and the two layer-12 plain means. The layer-12 cells are one vector; the layer-16 cells are a second.");
  const n = dc.names.length, cell = 44, m = { l: 150, t: 100 };
  const w = m.l + n * cell + 10, h = m.t + n * cell + 10;
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-cosine", role: "img", "aria-label": "Cosine similarity matrix of the Stage B directions" });
  const label = (name) => shortName(name).replace("-mean", " mean") + (name.startsWith("fit02") ? " (fit02)" : "");
  const shade = (v) => { const t = Math.max(0, Math.min(1, (v - 0.2) / 0.8)); return `rgba(57, 135, 229, ${(0.08 + 0.88 * t).toFixed(3)})`; };
  dc.names.forEach((name, i) => {
    const rl = ui.svgEl("text", { x: m.l - 8, y: m.t + i * cell + cell / 2 + 3.5, "text-anchor": "end", class: "row-sub" });
    rl.textContent = label(name);
    const cx = m.l + i * cell + cell / 2;
    const cl = ui.svgEl("text", { x: cx, y: m.t - 8, "text-anchor": "start", class: "row-sub", transform: `rotate(-50 ${cx} ${m.t - 8})` });
    cl.textContent = label(name);
    svg.append(rl, cl);
    dc.names.forEach((other, j) => {
      const v = dc.matrix[i][j];
      const rect = ui.svgEl("rect", { x: m.l + j * cell + 1, y: m.t + i * cell + 1, width: cell - 2, height: cell - 2, rx: 3, fill: shade(v) });
      const t = ui.svgEl("text", { x: m.l + j * cell + cell / 2, y: m.t + i * cell + cell / 2 + 3.5, "text-anchor": "middle", class: `cos-value${v > 0.6 ? " dark" : ""}` });
      t.textContent = v.toFixed(2);
      hover(tip, [rect, t], [`${label(name)} vs ${label(other)}`, `cosine ${v.toFixed(3)}`]);
      svg.append(rect, t);
    });
  });
  node.append(svg);
  return node;
}

function takeawaysCard(mp, ui) {
  const node = ui.card("What this shows", "Development only; no confirmation; final test unopened.");
  const ck = mp.check, tu = mp.tuning, g0 = mp.gate0;
  const items = [
    `Instructions move behavior strongly under blind grading: resist_firm cuts caving by ${pts(-g0.rows[0].change_points, 1)} points on FIT rows, comply raises it ${pts(g0.rows[3].change_points, 1)}. That is the ceiling for any direction.`,
    `One residual-stream direction from those contrasts gives the program's first blind-graded causal reduction: ${pts(ck.primary.change_points)} points on held-out CHECK rows ${ci(ck.primary.ci95_points)}, lost caves becoming holds. About two fifths of the instruction's effect, at the top of the range reported for residual directions on a 3B model.`,
    `The same direction fails the plan on the tuning histories: adoption ${pts(tu.endpoints[0].change_points)} with an interval reaching ${pts(tu.endpoints[0].ci95_points[1])}, justified updates ${pts(tu.endpoints[4].change_points)}, neutral correctness ${pts(tu.endpoints[6].change_points)}. It is a keep-your-position-or-abstain direction: warranted updates go with the unwarranted ones, and short extractive answers do not move.`,
    "The SOM found structure that the selection did not use: cells separate instruction wording and conversation type, with a partial split between unsupported pressure and misleading support. The rule kept only the dominant resist component (one vector across maps, cosine 0.998), applied without routing, so the per-type hypothesis is untested. The cell beat the plain mean by 2.7 points on SELECT with overlapping intervals.",
    "Direct test of the hypothesis: contrast warranted against unwarranted updating inside the same conversation so the shared resist component cancels, route rows to cells, and run a powered head-to-head against the mean direction. A new pre-registered cycle.",
  ];
  const ul = ui.el("ul", "study-takeaways");
  for (const t of items) ul.append(ui.el("li", "", t));
  node.append(ul);
  return node;
}

export function matchedPairSection(mp, tip, ui) {
  const nodes = [];
  const head = ui.el("div", "study-divider");
  head.append(ui.el("div", "study-eyebrow", "matched-pair direction study v1, 28 Oct 2026"), ui.el("h2", "", "Instruction-contrast SOM directions: from the map to blind results"));
  nodes.push(head);
  const caveat = ui.el("p", "study-caveat");
  caveat.append(ui.el("b", "", "Status: "), document.createTextNode(mp.headline));
  nodes.push(caveat);
  const flow = ui.card("How the study ran", "Each step was frozen in the results log before its data was viewed. Instructions derive directions only; the experimental conditions are the unmodified model and the steered model.");
  const ol = ui.el("ol", "study-flow");
  for (const step of mp.pipeline) ol.append(ui.el("li", "", step));
  flow.append(ol);
  nodes.push(flow);
  const top = ui.el("div", "study-grid two pair");
  top.append(pairMap(mp, tip, ui), latentCard(mp, tip, ui));
  nodes.push(top, stageACard(mp, tip, ui), forestCard(mp, tip, ui), tuningCard(mp, tip, ui));
  const bottom = ui.el("div", "study-grid two pair-bottom");
  bottom.append(cosineCard(mp, tip, ui), takeawaysCard(mp, ui));
  nodes.push(bottom);
  return nodes;
}
