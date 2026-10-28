// Matched-pair direction study v1 (instruction contrast): the selected layer-12
// map, the fitted difference-vector cloud with its prototypes, the Stage A
// screen, the blind Stage B and CHECK results against the instruction effects,
// and the tuning-stage endpoints against the plan's bounds. Read-only view of
// /api/som-study/matched-pair (built by research/build_som_matched_pair.py).

const PAL = { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", amber: "#c0821f", gray: "#8a8a8a", dim: "#4a4d52" };
const STATUS = { good: "#0ca30c", critical: "#d03b3b" };
const INSTRUCTIONS = [
  { key: "resist_firm", label: "resist_firm", color: PAL.blue },
  { key: "resist_evidence", label: "resist_evidence", color: PAL.orange },
  { key: "resist_cite", label: "resist_cite", color: PAL.aqua },
];
const SCENARIO_GROUPS = [
  { key: "unsupported_pressure", label: "Unsupported pressure", color: PAL.blue, match: (s) => s === "unsupported_pressure" },
  { key: "misleading_or_irrelevant_support", label: "Misleading support", color: PAL.orange, match: (s) => s === "misleading_or_irrelevant_support" },
  { key: "other", label: "Insufficient evidence or justified update", color: PAL.aqua, match: () => true },
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
  details.append(ui.el("summary", "", caption || "Table view"));
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

function stackedBar(ui, g, cx, y, width, parts, height = 6) {
  // parts: [{n, color}]; 2px surface gaps between segments.
  const total = parts.reduce((a, p) => a + p.n, 0);
  if (!total) return;
  const gap = 2, usable = width - gap * (parts.filter((p) => p.n).length - 1);
  let x = cx - width / 2;
  for (const p of parts) {
    if (!p.n) continue;
    const w = Math.max(1, (p.n / total) * usable);
    g.append(ui.svgEl("rect", { x, y, width: w, height, rx: 2, fill: p.color }));
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
    `Layer-${map.layer} SOM on instruction-contrast vectors, ${map.grid}x${map.grid}, seed ${map.seed}`,
    `${map.n_vectors} vectors = ${map.n_fit_rows} FIT conversations x 3 resist instructions, each state(instruction) minus state(plain). Cell number = vectors in the cell; bar = instruction mix; KL = Stage A score at c = 4.`,
  );
  const body = ui.el("div", "study-map-body");
  const w = 360, h = 330, r = 60;
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-hex", role: "img",
    "aria-label": "Nine SOM cells with vector counts, instruction mix and Stage A scores" });
  const step = r * Math.sqrt(3);
  for (const cell of map.cells) {
    const cx = 80 + cell.col * step + (cell.row % 2 ? step / 2 : 0);
    const cy = 72 + cell.row * r * 1.5;
    const g = ui.svgEl("g", { class: `study-cell${cell.selected ? " selected" : ""}`, tabindex: 0 });
    g.append(ui.svgEl("polygon", { points: hexPoints(cx, cy, r) }));
    const n = ui.svgEl("text", { x: cx, y: cy - 8, "text-anchor": "middle", class: "cell-n" });
    n.textContent = cell.support;
    g.append(n);
    stackedBar(ui, g, cx, cy + 2, 76, INSTRUCTIONS.map((i) => ({ n: cell.by_instruction[i.key] || 0, color: i.color })));
    const kl = ui.svgEl("text", { x: cx, y: cy + 22, "text-anchor": "middle", class: "cell-kl" });
    kl.textContent = cell.stage_a ? `KL ${pts(cell.stage_a.mean_delta_kl, 2)}` : "KL n/a";
    const sub = ui.svgEl("text", { x: cx, y: cy + 36, "text-anchor": "middle", class: "cell-sub" });
    sub.textContent = cell.selected ? "selected" : `cell ${cell.id}`;
    g.append(kl, sub);
    const inst = INSTRUCTIONS.map((i) => `${cell.by_instruction[i.key] || 0} ${i.label}`).join(", ");
    const scen = Object.entries(cell.by_scenario).map(([k, v]) => `${v} ${SHORT[k] || k}`).join(", ");
    const lines = [`Cell ${cell.id}${cell.selected ? " (selected for Stage B and CHECK)" : ""}: ${cell.support} vectors`,
      `Instructions: ${inst}`, `Scenarios: ${scen}`,
      `Cosine to the pool mean ${cell.cos_mean.toFixed(2)}; s_L ${cell.s_L.toFixed(2)}`,
      cell.stage_a ? `Stage A at c = 4: ${pts(cell.stage_a.mean_delta_kl, 3)} nats ${ci(cell.stage_a.ci95, 3)}` : "Stage A: not scored"];
    const show = (event) => tip.show(event, lines);
    g.addEventListener("mousemove", show);
    g.addEventListener("mouseleave", () => tip.hide());
    g.addEventListener("focus", (event) => { const b = event.target.getBoundingClientRect(); tip.show({ clientX: b.right, clientY: b.top }, lines); });
    g.addEventListener("blur", () => tip.hide());
    svg.append(g);
  }
  const side = ui.el("div", "study-map-side");
  side.append(ui.legend(INSTRUCTIONS.map((i) => ({ label: i.label, color: i.color }))));
  const wr = map.within_row_instruction_cosine;
  side.append(ui.el("p", "study-note",
    `The map sorts the cloud by instruction wording (cells 0, 5 and 6 are pure firm, evidence and cite) and by conversation type (cells 1, 2 and 7 hold the insufficient-evidence and justified-update conversations, where "resist" means something else). Within pressure conversations cell 4 is almost all unsupported pressure and cell 8 all misleading support. Stage A rewards the cell that best reproduces the instruction on pressure rows, so the firm-on-pressure cell wins and is the only one carried forward.`));
  const stats = ui.el("dl", "study-stats pair");
  for (const [k, v] of [
    ["SOM settings", `MiniSom ${map.som_settings.version_used}, ${map.som_settings.topology}, sigma ${map.som_settings.sigma}, lr ${map.som_settings.learning_rate}, ${map.som_settings.n_updates} updates`],
    ["Quantization error", map.quantization_error.toFixed(2)],
    ["Topographic error", map.topographic_error == null ? "n/a" : map.topographic_error.toFixed(2)],
    ["Instruction cosine, same conversation", `firm vs evidence ${wr["resist_firm|resist_evidence"].toFixed(2)}, firm vs cite ${wr["resist_firm|resist_cite"].toFixed(2)}, evidence vs cite ${wr["resist_evidence|resist_cite"].toFixed(2)}`],
    ["Selected cell vs pool mean", `cosine ${map.cells.find((c) => c.selected).cos_mean.toFixed(2)}`],
  ]) stats.append(ui.el("dt", "", k), ui.el("dd", "", v));
  body.append(svg, side);
  node.append(body, stats);
  node.append(table(ui, ["cell", "vectors", ...INSTRUCTIONS.map((i) => i.label), "unsupported", "misleading", "insufficient", "justified", "cos mean", "Stage A KL (c = 4)"],
    map.cells.map((c) => [c.id + (c.selected ? " (selected)" : ""), c.support, ...INSTRUCTIONS.map((i) => c.by_instruction[i.key] || 0),
      c.by_scenario.unsupported_pressure || 0, c.by_scenario.misleading_or_irrelevant_support || 0, c.by_scenario.insufficient_evidence || 0,
      c.by_scenario.justified_evidence_update || 0, c.cos_mean.toFixed(2), c.stage_a ? `${pts(c.stage_a.mean_delta_kl, 3)} ${ci(c.stage_a.ci95, 3)}` : "n/a"]),
    "Table view: cell composition"));
  return node;
}

function latentCard(mp, tip, ui) {
  const lat = mp.latent, map = mp.map;
  const node = ui.card(
    "The latent space the map was fit on",
    `All ${lat.n_points} difference vectors projected onto their first two principal components (${(100 * lat.explained_variance_ratio[0]).toFixed(0)}% and ${(100 * lat.explained_variance_ratio[1]).toFixed(0)}% of variance). Numbered hexagons are the nine cell prototypes; the arrow is the selected steering direction in this plane.`,
  );
  const modes = [
    { key: "instruction", label: "colour by instruction", groups: INSTRUCTIONS.map((i) => ({ ...i, match: (p) => p.instruction === i.key })) },
    { key: "scenario", label: "colour by scenario", groups: SCENARIO_GROUPS.map((g) => ({ ...g, match: (p) => g.match(p.scenario) })) },
    { key: "cell", label: "selected cell vs rest", groups: [
      { key: "sel", label: `cell ${map.selected_cell} (selected)`, color: PAL.amber, match: (p) => p.cell === map.selected_cell },
      { key: "rest", label: "other cells", color: PAL.gray, match: () => true },
    ] },
  ];
  const toggle = ui.el("div", "study-toggle");
  const legendSlot = ui.el("div");
  const w = 680, h = 460, m = { l: 44, r: 16, t: 20, b: 40 };
  const xs = lat.points.map((p) => p.x), ys = lat.points.map((p) => p.y);
  const pad = 0.04;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const xr = [x0 - pad * (x1 - x0), x1 + pad * (x1 - x0)], yr = [y0 - pad * (y1 - y0), y1 + pad * (y1 - y0)];
  const x = (v) => m.l + ((v - xr[0]) / (xr[1] - xr[0])) * (w - m.l - m.r);
  const y = (v) => m.t + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (h - m.t - m.b);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-latent", role: "img",
    "aria-label": "Scatter of instruction-contrast vectors in the first two principal components with SOM prototypes" });
  const ticks = (lo, hi) => { const s = Math.pow(10, Math.floor(Math.log10(hi - lo))) * ((hi - lo) / Math.pow(10, Math.floor(Math.log10(hi - lo))) > 5 ? 2 : 1); const out = []; for (let t = Math.ceil(lo / s) * s; t <= hi; t += s) out.push(Number(t.toFixed(6))); return out; };
  ui.axes(svg, { x, y, w, h, m, xTicks: ticks(xr[0], xr[1]), yTicks: ticks(yr[0], yr[1]), xLabel: "PC1", yLabel: "PC2", xFmt: (t) => t.toFixed(0), yFmt: (t) => t.toFixed(0) });
  const dots = ui.svgEl("g", { class: "study-dots" });
  const nodes = lat.points.map((p) => {
    const g = ui.svgEl("g");
    const hit = ui.svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 7, fill: "transparent" });
    const dot = ui.svgEl("circle", { cx: x(p.x), cy: y(p.y), r: 2.6, "fill-opacity": 0.85 });
    const lines = [`Cell ${p.cell}, ${p.instruction}`, `${SHORT[p.scenario] || p.scenario} (${p.source})`,
      `Projection on the selected direction ${p.proj_selected.toFixed(2)}`];
    g.addEventListener("mousemove", (event) => tip.show(event, lines));
    g.addEventListener("mouseleave", () => tip.hide());
    g.append(hit, dot);
    dots.append(g);
    return { dot, p };
  });
  svg.append(dots);
  // Selected steering direction as an arrow from the centroid (PCA origin).
  const dir = lat.selected_direction_in_plane;
  const span = Math.min(x1 - x0, y1 - y0) * 0.38;
  const norm = Math.hypot(dir[0], dir[1]) || 1;
  const ex = (dir[0] / norm) * span, ey = (dir[1] / norm) * span;
  svg.append(ui.svgEl("line", { x1: x(0), y1: y(0), x2: x(ex), y2: y(ey), stroke: "#e6e6e6", "stroke-width": 2, "stroke-linecap": "round", "marker-end": "url(#pair-arrow)" }));
  const defs = ui.svgEl("defs");
  const marker = ui.svgEl("marker", { id: "pair-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  marker.append(ui.svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#e6e6e6" }));
  defs.append(marker);
  svg.append(defs);
  for (const pr of lat.prototypes) {
    const sel = pr.cell === map.selected_cell;
    const g = ui.svgEl("g", { class: `study-proto${sel ? " selected" : ""}`, tabindex: 0 });
    g.append(ui.svgEl("polygon", { points: hexPoints(x(pr.x), y(pr.y), 11) }));
    const t = ui.svgEl("text", { x: x(pr.x), y: y(pr.y) + 4, "text-anchor": "middle" });
    t.textContent = pr.cell;
    g.append(t);
    const c = map.cells[pr.cell];
    const lines = [`Prototype of cell ${pr.cell}${sel ? " (selected)" : ""}`, `${c.support} vectors; cosine to the pool mean ${c.cos_mean.toFixed(2)}`,
      c.stage_a ? `Stage A ${pts(c.stage_a.mean_delta_kl, 3)} nats` : "Stage A: not scored"];
    g.addEventListener("mousemove", (event) => tip.show(event, lines));
    g.addEventListener("mouseleave", () => tip.hide());
    svg.append(g);
  }
  const apply = (mode) => {
    for (const { dot, p } of nodes) {
      const group = mode.groups.find((g) => g.match(p));
      dot.setAttribute("fill", group ? group.color : PAL.gray);
    }
    legendSlot.replaceChildren(ui.legend(mode.groups.map((g) => ({ label: g.label, color: g.color }))));
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
  const arrowNote = ui.el("p", "study-note arrow-note", `Arrow: the selected steering direction from the cloud's centroid; ${(100 * norm).toFixed(0)}% of that unit vector lies in this plane.`);
  node.append(toggle, legendSlot, svg, arrowNote);
  node.append(ui.el("p", "study-note",
    `Mean direction in this plane: (${lat.mean_direction_in_plane.map((v) => v.toFixed(2)).join(", ")}); selected direction: (${dir.map((v) => v.toFixed(2)).join(", ")}). Refit check: prototypes match the frozen directions to cosine ${lat.refit.prototype_cosine_min.toFixed(4)} or better.`));
  const counts = {};
  for (const p of lat.points) { counts[p.cell] = counts[p.cell] || {}; counts[p.cell][p.instruction] = (counts[p.cell][p.instruction] || 0) + 1; }
  node.append(table(ui, ["cell", ...INSTRUCTIONS.map((i) => i.label), "total"],
    Object.keys(counts).sort((a, b) => a - b).map((c) => [c, ...INSTRUCTIONS.map((i) => counts[c][i.key] || 0), Object.values(counts[c]).reduce((a, v) => a + v, 0)]),
    "Table view: vectors per cell and instruction"));
  return node;
}

function stageACard(mp, tip, ui) {
  const sa = mp.stage_a;
  const node = ui.card(
    `Stage A screen: ${sa.n_configurations} configurations on ${sa.n_rows} SELECT pressure rows`,
    "Label-free score per cell and strength: how far a push moves the pre-reply next-token distribution toward the instructed one, as a reduction in KL (nats). The dashed line is the instruction's own KL from plain, the most a direction could recover. Best configuration per layer and grid is shown.",
  );
  const w = 1120, h = 300, m = { l: 52, r: 24, t: 28, b: 42 };
  const layers = sa.layers;
  const x = (l) => m.l + ((l - layers[0]) / (layers[layers.length - 1] - layers[0])) * (w - m.l - m.r);
  const yMax = Math.max(1, sa.instruction_kl_ceiling * 1.08);
  const y = (v) => m.t + (1 - v / yMax) * (h - m.t - m.b);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img",
    "aria-label": "Best Stage A KL reduction by layer for the mean direction and the best 2x2 and 3x3 cells" });
  ui.axes(svg, { x, y, w, h, m, xTicks: layers, yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1.0], xLabel: "layer", yLabel: "KL reduction (nats)", yFmt: (t) => t.toFixed(1) });
  svg.append(ui.svgEl("line", { x1: m.l, x2: w - m.r, y1: y(sa.instruction_kl_ceiling), y2: y(sa.instruction_kl_ceiling), class: "marker" }));
  const cl = ui.svgEl("text", { x: w - m.r, y: y(sa.instruction_kl_ceiling) - 6, "text-anchor": "end", class: "marker-label" });
  cl.textContent = `instruction itself: ${sa.instruction_kl_ceiling.toFixed(2)} nats`;
  svg.append(cl);
  const series = [
    { grid: "1", label: "1x1 (plain mean of the vectors)", color: PAL.blue },
    { grid: "2", label: "best 2x2 cell", color: PAL.orange },
    { grid: "3", label: "best 3x3 cell", color: PAL.aqua },
  ].filter((s) => sa.best_by_layer_grid[s.grid]);
  for (const s of series) {
    const rows = sa.best_by_layer_grid[s.grid];
    svg.append(ui.svgEl("polyline", { points: rows.map((r) => `${x(r.layer)},${y(r.mean_delta_kl)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    for (const r of rows) {
      const dot = ui.svgEl("circle", { cx: x(r.layer), cy: y(r.mean_delta_kl), r: 4.5, fill: s.color, stroke: "#090b0d", "stroke-width": 2 });
      const hit = ui.svgEl("circle", { cx: x(r.layer), cy: y(r.mean_delta_kl), r: 12, fill: "transparent" });
      const lines = [`${s.label}, layer ${r.layer}`, `${shortName(r.direction)} at c = ${r.c}: ${pts(r.mean_delta_kl, 3)} nats ${ci(r.ci95, 3)}`,
        `KL(plain, steered) ${r.kl_plain_steered == null ? "n/a" : r.kl_plain_steered.toFixed(2)} (coherence cost)`];
      for (const n of [hit, dot]) { n.addEventListener("mousemove", (event) => tip.show(event, lines)); n.addEventListener("mouseleave", () => tip.hide()); }
      svg.append(hit, dot);
    }
  }
  const best = sa.top[0];
  if (best) {
    const lbl = ui.svgEl("text", { x: x(best.layer) + 10, y: y(best.mean_delta_kl) - 10, class: "marker-label" });
    lbl.textContent = `best: ${shortName(best.direction)}, c = ${best.c}: ${pts(best.mean_delta_kl, 3)}`;
    svg.append(lbl);
  }
  node.append(ui.legend(series.map((s) => ({ label: s.label, color: s.color })).concat([{ label: "instruction ceiling", color: PAL.gray, dashed: true }])), svg);
  node.append(ui.el("p", "study-note",
    `The best cell recovers ${(100 * best.mean_delta_kl / sa.instruction_kl_ceiling).toFixed(0)}% of the instruction's shift at layer ${best.layer}; the plain mean at the same layer recovers ${(100 * (sa.best_by_layer_grid["1"].find((r) => r.layer === best.layer)?.mean_delta_kl || 0) / sa.instruction_kl_ceiling).toFixed(0)}%. All six carried forward sit within 0.015 nats of each other at layers 12 and 16. ${sa.n_incoherent_excluded} configurations were excluded by the coherence rule.`));
  node.append(table(ui, ["rank", "direction", "layer", "grid", "seed", "cell", "c", "s_L", "KL reduction [95% CI]", "KL(plain, steered)"],
    sa.top.map((t) => [t.rank, t.direction, t.layer, `${t.grid}x${t.grid}`, t.seed, t.cell, t.c, t.s_L.toFixed(2), `${pts(t.mean_delta_kl, 3)} ${ci(t.ci95, 3)}`, t.kl_plain_steered == null ? "n/a" : t.kl_plain_steered.toFixed(2)]),
    "Table view: the six configurations carried to Stage B"));
  return node;
}

function forestRows(mp) {
  const g0 = mp.gate0, sb = mp.stage_b, ck = mp.check;
  const rows = [{ header: `Instructions on FIT rows (blind, ceiling): plain cave share ${g0.plain_cave_share.toFixed(2)}` }];
  for (const r of g0.rows) rows.push({ label: r.instruction, sub: `cave share ${r.cave_share.toFixed(2)}, evidence ${r.evidence_rate.toFixed(2)}`, v: r.change_points, ci: r.ci95_points, kind: "ref" });
  rows.push({ header: `Stage B on ${sb.n_rows} SELECT pressure rows (blind, 8 replies per row)` });
  for (const r of sb.rows) rows.push({ label: `${shortName(r.config)} (${r.config.split(":")[0]})`, sub: `layer ${r.layer}, ${r.grid}x${r.grid}, c = ${r.c}; cave ${r.plain_cave_rate.toFixed(2)} to ${r.cave_rate.toFixed(2)}; evidence ${pts(r.evidence_change_points)}, truncated ${pts(r.degenerate_change_points)}`, v: r.change_points, ci: r.ci95_points, kind: r.selected ? "selected" : "steer" });
  rows.push({ header: `CHECK on ${ck.n_pressure_rows} held-out pressure rows (blind, opened once)` });
  rows.push({ label: "primary endpoint", sub: `cave ${ck.primary.plain_cave_rate.toFixed(2)} to ${ck.primary.cave_rate.toFixed(2)}; evidence ${pts(ck.evidence.change_points)} ${ci(ck.evidence.ci95_points)}; truncated ${pts(ck.degenerate.change_points)}`, v: ck.primary.change_points, ci: ck.primary.ci95_points, kind: "check" });
  const s = ck.sensitivity;
  for (const [key, label] of [["exclude_degenerate", "truncated replies excluded"], ["reweighted", "unsampled proxy-hold reweighted"], ["degenerate_as_cave", "truncated replies counted as caves"]]) {
    rows.push({ label, sub: "sensitivity", v: s[key].change_points, ci: s[key].ci95_points, kind: "sens" });
  }
  for (const [key, v] of Object.entries(ck.by_scenario)) rows.push({ label: SHORT[key] || key, sub: `cave share, ${v.n} rows`, v: v.change_points, ci: v.ci95_points, kind: "sens" });
  return rows;
}

function forestCard(mp, tip, ui) {
  const node = ui.card(
    "Blind cave-rate change versus the unmodified model",
    "Percentage points, paired per conversation, family-bootstrap 95% intervals. Below zero is less caving. The shaded zone is the pre-registered CHECK threshold (at least 10 points with the interval excluding zero).",
  );
  const rows = forestRows(mp);
  const w = 1120, rowH = 34, headH = 30, m = { l: 330, r: 110, t: 24, b: 36 };
  const heights = rows.map((r) => (r.header ? headH : rowH));
  const h = m.t + heights.reduce((a, b) => a + b, 0) + m.b;
  const lo = -36, hi = 16;
  const x = (v) => m.l + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (w - m.l - m.r);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img", "aria-label": "Forest plot of blind cave-rate changes for instructions, Stage B configurations and CHECK" });
  const g = ui.svgEl("g", { class: "study-axis" });
  g.append(ui.svgEl("rect", { x: x(lo), y: m.t, width: x(-10) - x(lo), height: h - m.t - m.b, class: "target-zone" }));
  for (const t of [-30, -20, -10, 0, 10]) {
    g.append(ui.svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, class: t === 0 ? "zero" : "grid" }));
    const label = ui.svgEl("text", { x: x(t), y: h - m.b + 18, "text-anchor": "middle" });
    label.textContent = pts(t, 0);
    g.append(label);
  }
  const tl = ui.svgEl("text", { x: x(-23), y: m.t - 8, "text-anchor": "middle", class: "target-label" });
  tl.textContent = "CHECK threshold zone";
  const xt = ui.svgEl("text", { x: x(3), y: h - 4, "text-anchor": "middle", class: "axis-title" });
  xt.textContent = "percentage points";
  g.append(tl, xt);
  svg.append(g);
  let cy = m.t;
  rows.forEach((row, i) => {
    const hh = heights[i];
    if (row.header) {
      const t = ui.svgEl("text", { x: 12, y: cy + hh - 9, class: "row-head" });
      t.textContent = row.header;
      svg.append(t);
      cy += hh;
      return;
    }
    const mid = cy + hh / 2;
    const name = ui.svgEl("text", { x: 12, y: mid - 2, class: `row-label${row.kind === "ref" || row.kind === "sens" ? " muted" : ""}` });
    name.textContent = row.label;
    const sub = ui.svgEl("text", { x: 12, y: mid + 11, class: "row-sub" });
    sub.textContent = row.sub;
    svg.append(name, sub);
    const color = row.kind === "ref" ? PAL.gray : row.kind === "sens" ? PAL.dim : row.kind === "selected" || row.kind === "check" ? PAL.blue : PAL.aqua;
    if (row.ci) svg.append(ui.svgEl("line", { x1: x(row.ci[0]), x2: x(row.ci[1]), y1: mid, y2: mid, stroke: color, "stroke-width": 2, "stroke-linecap": "round", opacity: row.kind === "sens" ? 0.9 : 1 }));
    const rr = row.kind === "check" ? 6.5 : row.kind === "selected" ? 5.5 : 4.5;
    const dot = ui.svgEl("circle", { cx: x(row.v), cy: mid, r: rr, fill: color, stroke: "#090b0d", "stroke-width": 2 });
    const hit = ui.svgEl("rect", { x: m.l, y: cy, width: w - m.l - m.r, height: hh, fill: "transparent" });
    const lines = [row.label, `${pts(row.v)} points ${ci(row.ci)}`, row.sub];
    for (const n of [hit, dot]) { n.addEventListener("mousemove", (event) => tip.show(event, lines)); n.addEventListener("mouseleave", () => tip.hide()); }
    const value = ui.svgEl("text", { x: w - m.r + 10, y: mid + 4, class: "row-value" });
    value.textContent = `${pts(row.v)} ${ci(row.ci)}`;
    svg.append(hit, dot, value);
    cy += hh;
  });
  node.append(ui.legend([{ label: "instruction (reference)", color: PAL.gray }, { label: "Stage B configuration", color: PAL.aqua }, { label: "selected configuration and CHECK", color: PAL.blue }, { label: "sensitivity and per-scenario", color: PAL.dim }]), svg);
  const ck = mp.check;
  node.append(ui.el("p", "study-note",
    `CHECK: ${ck.rows_improved_same_worse[0]} rows improved, ${ck.rows_improved_same_worse[1]} unchanged, ${ck.rows_improved_same_worse[2]} worse; label shares over all pressure replies moved from cave ${ck.label_shares.plain.cave.toFixed(2)} / hold ${ck.label_shares.plain.hold.toFixed(2)} to cave ${ck.label_shares[ck.condition].cave.toFixed(2)} / hold ${ck.label_shares[ck.condition].hold.toFixed(2)}, so lost caves became holds. Panels agreed on ${(100 * ck.panels.agreement_overall).toFixed(1)}% of ${ck.panels.n_items} replies; ${ck.panels.n_disputes} disputes were adjudicated masked. The frozen rule is met by 0.2 points on the primary and 0.4 points on the evidence guard; truncated output rose ${pts(ck.degenerate.change_points)} points.`));
  node.append(table(ui, ["row", "change (points)", "95% CI", "detail"], rows.filter((r) => !r.header).map((r) => [r.label, pts(r.v), ci(r.ci), r.sub]), "Table view: every row of the plot"));
  return node;
}

function tuningCard(mp, tip, ui) {
  const tu = mp.tuning;
  const node = ui.card(
    "Tuning stage: the same configuration on the 305 frozen histories, seven blind endpoints",
    "Steered minus baseline in percentage points, equal-family weighting, family-bootstrap 95% intervals, adjudicated labels. Ticks mark the plan's bounds: reduction of at least 5 points (and 20% relative) with the interval below zero; preservation lower bounds above -5 (updates, corrections) and -3 (neutral).",
  );
  const rows = tu.endpoints;
  const w = 1120, rowH = 40, m = { l: 300, r: 268, t: 24, b: 36 };
  const h = m.t + rows.length * rowH + m.b;
  const lo = -70, hi = 56;
  const clamp = (v) => Math.max(lo, Math.min(hi, v));
  const x = (v) => m.l + ((clamp(v) - lo) / (hi - lo)) * (w - m.l - m.r);
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg", role: "img", "aria-label": "Forest plot of the seven tuning-stage endpoints with the plan's bounds" });
  const g = ui.svgEl("g", { class: "study-axis" });
  for (const t of [-60, -40, -20, 0, 20, 40]) {
    g.append(ui.svgEl("line", { x1: x(t), x2: x(t), y1: m.t, y2: h - m.b, class: t === 0 ? "zero" : "grid" }));
    const label = ui.svgEl("text", { x: x(t), y: h - m.b + 18, "text-anchor": "middle" });
    label.textContent = pts(t, 0);
    g.append(label);
  }
  const xt = ui.svgEl("text", { x: x(0), y: h - 4, "text-anchor": "middle", class: "axis-title" });
  xt.textContent = "percentage points, steered minus baseline";
  g.append(xt);
  svg.append(g);
  rows.forEach((row, i) => {
    const cy = m.t + i * rowH, mid = cy + rowH / 2;
    const name = ui.svgEl("text", { x: 12, y: mid - 2, class: "row-label" });
    name.textContent = row.label;
    const sub = ui.svgEl("text", { x: 12, y: mid + 11, class: "row-sub" });
    sub.textContent = `${row.pairs} pairs, ${row.families} families; ${row.baseline.toFixed(2)} to ${row.steered.toFixed(2)}`;
    svg.append(name, sub);
    const bound = row.key === "unsupported_adoption" ? tu.reduction_target.points : row.bound_points;
    if (bound != null) {
      svg.append(ui.svgEl("line", { x1: x(bound), x2: x(bound), y1: cy + 6, y2: cy + rowH - 6, stroke: "#9a9a9a", "stroke-width": 1.5, "stroke-dasharray": "3 3" }));
    }
    const color = row.role.startsWith("preserve") ? PAL.orange : row.role === "higher is better" ? PAL.aqua : PAL.blue;
    const lo2 = row.ci95_points[0], hi2 = row.ci95_points[1];
    svg.append(ui.svgEl("line", { x1: x(lo2), x2: x(hi2), y1: mid, y2: mid, stroke: color, "stroke-width": 2, "stroke-linecap": "round" }));
    if (lo2 < lo) { const t = ui.svgEl("text", { x: x(lo) - 6, y: mid + 4, "text-anchor": "end", class: "row-sub" }); t.textContent = pts(lo2, 0); svg.append(t); }
    const dot = ui.svgEl("circle", { cx: x(row.change_points), cy: mid, r: 5.5, fill: color, stroke: "#090b0d", "stroke-width": 2 });
    const hit = ui.svgEl("rect", { x: m.l, y: cy, width: w - m.l - m.r, height: rowH, fill: "transparent" });
    const va = mp.tuning.variants, k = row.key;
    const lines = [row.label, `Adjudicated ${pts(row.change_points)} ${ci(row.ci95_points)}`,
      `Raw panel A ${pts(va["reviewer-A"][k].change_points)}, raw panel B ${pts(va["reviewer-B"][k].change_points)}`,
      bound != null ? `Bound ${pts(bound, 0)}: ${row.passes ? "met" : "not met"}` : "Reported, not a gate"];
    for (const n of [hit, dot]) { n.addEventListener("mousemove", (event) => tip.show(event, lines)); n.addEventListener("mouseleave", () => tip.hide()); }
    const value = ui.svgEl("text", { x: w - m.r + 10, y: mid + 4, class: "row-value" });
    value.textContent = `${pts(row.change_points)} ${ci(row.ci95_points)}`;
    svg.append(hit, dot, value);
    if (row.passes != null) {
      const chip = ui.svgEl("text", { x: w - 10, y: mid + 4, "text-anchor": "end", class: `row-chip ${row.passes ? "pass" : "fail"}` });
      chip.textContent = row.passes ? "✓ met" : "✕ not met";
      svg.append(chip);
    }
  });
  node.append(ui.legend([{ label: "adoption endpoints (lower is better)", color: PAL.blue }, { label: "appropriate uncertainty (higher is better)", color: PAL.aqua }, { label: "preservation endpoints", color: PAL.orange }, { label: "plan bound", color: PAL.gray, dashed: true }]), svg);
  const c = tu.primary_subtype_counts;
  node.append(ui.el("p", "study-note",
    `Raw counts on the 90 primary histories: unsupported adoption ${(c.baseline.sycophantic_flip || 0) + (c.baseline.new_unsupported_adoption || 0)} baseline to ${(c.som.sycophantic_flip || 0) + (c.som.new_unsupported_adoption || 0)} steered, resistance ${c.baseline.resistance || 0} to ${c.som.resistance || 0}. The family-weighted -14 comes from small QASC and NQ families; the three large SQuAD families did not move (49 of 67 SQuAD pressure replies are token-identical under steering). Relative reduction ${(100 * tu.reduction_relative).toFixed(0)}%. Gates: reduction ${tu.gates.reduction ? "met" : "not met"}; preservation ${Object.entries(tu.gates.preservation).filter(([, v]) => !v).map(([k]) => k.replace(/_/g, " ")).join(", ") || "all"} not met. Confirmation set not designated; final test unopened.`));
  node.append(table(ui, ["endpoint", "pairs / families", "baseline", "steered", "adjudicated", "95% CI", "raw A", "raw B", "bound", "result"],
    rows.map((r) => [r.label, `${r.pairs} / ${r.families}`, r.baseline.toFixed(3), r.steered.toFixed(3), pts(r.change_points), ci(r.ci95_points), pts(tu.variants["reviewer-A"][r.key].change_points), pts(tu.variants["reviewer-B"][r.key].change_points),
      r.key === "unsupported_adoption" ? "-5 and -20% rel., CI < 0" : r.bound_points == null ? "none" : pts(r.bound_points, 0), r.passes == null ? "reported" : r.passes ? "met" : "not met"]),
    "Table view: all seven endpoints with raw-panel sensitivity"));
  return node;
}

function cosineCard(mp, tip, ui) {
  const dc = mp.direction_cosines;
  const node = ui.card("Are the six directions different?", "Cosine between the unit directions carried to Stage B and the two layer-12 plain means. The three layer-12 SOM cells are one vector; the layer-16 cells are a second.");
  const n = dc.names.length, cell = 46, m = { l: 200, t: 110 };
  const w = m.l + n * cell + 20, h = m.t + n * cell + 16;
  const svg = ui.svgEl("svg", { viewBox: `0 0 ${w} ${h}`, class: "study-svg study-cosine", role: "img", "aria-label": "Cosine similarity matrix of the Stage B directions" });
  const label = (name) => shortName(name).replace("-mean", " mean") + (name.startsWith("fit02") ? " (fit02)" : "");
  const shade = (v) => { const t = Math.max(0, Math.min(1, (v - 0.2) / 0.8)); return `rgba(57, 135, 229, ${(0.08 + 0.9 * t).toFixed(3)})`; };
  dc.names.forEach((name, i) => {
    const rl = ui.svgEl("text", { x: m.l - 8, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "end", class: "row-sub" });
    rl.textContent = label(name);
    const cl = ui.svgEl("text", { x: m.l + i * cell + cell / 2, y: m.t - 8, "text-anchor": "start", class: "row-sub", transform: `rotate(-55 ${m.l + i * cell + cell / 2} ${m.t - 8})` });
    cl.textContent = label(name);
    svg.append(rl, cl);
    dc.names.forEach((other, j) => {
      const v = dc.matrix[i][j];
      const rect = ui.svgEl("rect", { x: m.l + j * cell + 1, y: m.t + i * cell + 1, width: cell - 2, height: cell - 2, rx: 3, fill: shade(v) });
      const t = ui.svgEl("text", { x: m.l + j * cell + cell / 2, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "middle", class: `cos-value${v > 0.6 ? " dark" : ""}` });
      t.textContent = v.toFixed(2);
      const lines = [`${label(name)} vs ${label(other)}`, `cosine ${v.toFixed(3)}`];
      for (const node2 of [rect, t]) { node2.addEventListener("mousemove", (event) => tip.show(event, lines)); node2.addEventListener("mouseleave", () => tip.hide()); }
      svg.append(rect, t);
    });
  });
  node.append(svg);
  return node;
}

function takeawaysCard(mp, ui) {
  const node = ui.card("What this shows", "Read with the status line above: development only, no confirmation, final test unopened.");
  const ck = mp.check, tu = mp.tuning, g0 = mp.gate0;
  const items = [
    `Instructions move behavior strongly and verifiably: under blind grading, resist_firm cuts caving by ${pts(-g0.rows[0].change_points, 1)} points on FIT rows and comply raises it ${pts(g0.rows[3].change_points, 1)}. That is the ceiling the directions are measured against.`,
    `A single residual-stream direction derived from those contrasts gives the study's first blind-graded causal reduction: ${pts(ck.primary.change_points)} points on held-out CHECK rows ${ci(ck.primary.ci95_points)}, with lost caves becoming holds. It recovers about two fifths of the instruction's effect, at the top of the range reported for residual directions on a 3B model.`,
    `The same direction fails the plan on the tuning histories: adoption ${pts(tu.endpoints[0].change_points)} with an interval that reaches ${pts(tu.endpoints[0].ci95_points[1])}, justified updates ${pts(tu.endpoints[4].change_points)}, neutral correctness ${pts(tu.endpoints[6].change_points)}. It is a keep-your-position-or-abstain direction: it removes warranted updates along with unwarranted ones and does nothing on short extractive answers.`,
    "The SOM found structure, but not the structure that was used: its cells separate instruction wording and conversation type, with a partial split between unsupported pressure and misleading support inside pressure rows. The selection rule kept only the dominant resist component (one vector across maps, cosine 0.998), applied to every row without routing, so the per-type hypothesis was not tested. The cell beat the plain mean by 2.7 points on SELECT with overlapping intervals.",
    "Next test that would address the hypothesis directly: contrast warranted against unwarranted updating inside the same conversation so the shared resist component cancels, route rows to cells, and run a powered head-to-head against the mean direction. That is a new pre-registered cycle.",
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
