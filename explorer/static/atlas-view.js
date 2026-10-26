const SVG_NS = "http://www.w3.org/2000/svg";
const SPLIT_NAMES = { train: "train", eval: "eval" };

const element = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
};

function formatNumber(value, digits = 0) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "not reported";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
}

function selectOptions(select, rows, valueOf, labelOf, selected) {
  select.replaceChildren();
  for (const row of rows) {
    const option = document.createElement("option");
    option.value = String(valueOf(row));
    option.textContent = String(labelOf(row));
    if (String(valueOf(row)) === String(selected)) option.selected = true;
    select.append(option);
  }
}

export function createAtlasView({ root, onSectionChange = () => {} }) {
  const $ = selector => root.querySelector(selector);
  const state = {
    active: false,
    section: "groups",
    atlases: [],
    atlas: null,
    atlasId: "",
    map: null,
    selectedCell: null,
    layer: 6,
    view: "answer_mean",
    seed: 0,
    split: "train",
    colorMode: "dominant",
    densityGroup: "",
    stance: "all",
  };
  let loadPromise = null;

  function setStatus(message, kind = "") {
    const status = $("#atlas-load-state");
    status.className = `atlas-load-state${kind ? ` ${kind}` : ""}`;
    status.textContent = message;
  }

  function setSection(section, notify = true) {
    if (section !== "groups" && section !== "behavior") section = "groups";
    state.section = section;
    root.querySelectorAll("#atlas-subnav button").forEach(button => {
      const selected = button.dataset.atlasView === section;
      button.classList.toggle("on", selected);
      button.setAttribute("aria-current", selected ? "page" : "false");
    });
    $("#atlas-groups-panel").hidden = section !== "groups";
    $("#atlas-behavior-panel").hidden = section !== "behavior";
    if (notify) onSectionChange(section);
    if (state.atlas) renderBehavior();
  }

  root.querySelectorAll("#atlas-subnav button").forEach(button => {
    button.addEventListener("click", () => setSection(button.dataset.atlasView));
  });

  async function jsonRequest(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try {
        const body = await response.json();
        if (body?.error?.message) message = body.error.message;
      } catch {}
      throw new Error(message);
    }
    return response.json();
  }

  async function activate(section = state.section) {
    state.active = true;
    document.body.classList.add("atlas-open");
    setSection(section, false);
    if (state.atlas) return;
    if (loadPromise) return loadPromise;
    $("#atlas-groups-panel").hidden = true;
    $("#atlas-behavior-panel").hidden = true;
    loadPromise = (async () => {
      setStatus("Checking installed atlas receipts…");
      try {
        const result = await jsonRequest("/api/atlases");
        state.atlases = Array.isArray(result.atlases) ? result.atlases : [];
        if (!state.atlases.length) {
          const rejected = Array.isArray(result.rejected) ? result.rejected.length : 0;
          setStatus(rejected
            ? "No verified atlas is available. An installed artifact did not pass receipt or schema validation."
            : "No atlas is installed yet. This view will populate when a verified teacher-forced atlas is installed.");
          $("#atlas-groups-panel").hidden = true;
          $("#atlas-behavior-panel").hidden = true;
          return;
        }
        selectOptions($("#atlas-select"), state.atlases, row => row.id,
          row => row.model.split("/").pop().replace("-Instruct", ""), state.atlases[0].id);
        $("#atlas-select").parentElement.hidden = state.atlases.length < 2;
        await loadAtlas(state.atlases[0].id);
      } catch (error) {
        setStatus(`Atlas data could not be loaded: ${error.message}`, "error");
      } finally {
        loadPromise = null;
      }
    })();
    return loadPromise;
  }

  async function loadAtlas(atlasId) {
    state.atlas = null;
    state.atlasId = atlasId;
    $("#atlas-groups-panel").hidden = true;
    $("#atlas-behavior-panel").hidden = true;
    setStatus("Verifying atlas contents…");
    try {
      const atlas = await jsonRequest(`/api/atlases/${encodeURIComponent(atlasId)}`);
      state.atlas = atlas;
      const availableLayers = [...new Set((atlas.maps || []).map(map => map.layer))].sort((a, b) => a - b);
      selectOptions($("#atlas-layer"), availableLayers, layer => layer, layer => `L${layer}`,
        availableLayers.includes(state.layer) ? state.layer : availableLayers[0]);
      state.layer = Number($("#atlas-layer").value || availableLayers[0]);
      selectOptions($("#atlas-density-group"), atlas.groups || [], group => group.id, group => group.label,
        state.densityGroup || atlas.groups?.[0]?.id);
      state.densityGroup = $("#atlas-density-group").value || "";
      state.selectedCell = null;
      setSection(state.section, false);
      render();
      const metadata = atlas.metadata || {};
      setStatus(`${metadata.sequence_count || 0} scripts · ${metadata.family_count || 0} world families · ${atlas.maps?.length || 0} maps`);
      const provenanceNote = $("#atlas-provenance-note");
      provenanceNote.hidden = metadata.source_status !== "teacher_forced_synthetic_unreviewed";
      provenanceNote.textContent = provenanceNote.hidden ? ""
        : `Synthetic, unreviewed · ${metadata.fitting_families} fitting worlds / ${metadata.evaluation_families} held-out. Controls contain legitimate responses.`;
    } catch (error) {
      state.atlas = null;
      $("#atlas-groups-panel").hidden = true;
      $("#atlas-behavior-panel").hidden = true;
      setStatus(`Atlas could not be verified: ${error.message}`, "error");
    }
  }

  $("#atlas-select").addEventListener("change", event => loadAtlas(event.target.value));
  $("#atlas-layer").addEventListener("change", event => { state.layer = Number(event.target.value); state.selectedCell = null; render(); });
  $("#atlas-map-view").addEventListener("change", event => { state.view = event.target.value; state.selectedCell = null; render(); });
  $("#atlas-seed").addEventListener("change", event => { state.seed = Number(event.target.value); state.selectedCell = null; render(); });
  $("#atlas-split").addEventListener("change", event => { state.split = event.target.value; render(); });
  $("#atlas-color-mode").addEventListener("change", event => { state.colorMode = event.target.value; renderMap(); });
  $("#atlas-density-group").addEventListener("change", event => { state.densityGroup = event.target.value; renderMap(); });
  $("#atlas-stance").addEventListener("change", event => { state.stance = event.target.value; render(); });

  function examplesById() {
    const entries = state.atlas?.examples;
    if (Array.isArray(entries)) return entries.map((value, index) => [String(value?.id ?? index), value]);
    if (entries && typeof entries === "object") return Object.entries(entries);
    return [];
  }

  function exampleForPoint(point) {
    const examples = state.atlas?.examples;
    if (!examples || typeof examples !== "object") return [];
    const key = point.example_id ?? point.sequence_id ?? point.id;
    if (key != null && !Array.isArray(examples) && examples[String(key)]) return [examples[String(key)]];
    const entries = examplesById().map(([, value]) => value).filter(value => value && typeof value === "object");
    const pairId = point.pair_id ?? (point.id != null && entries.some(value => value.pair_id === point.id) ? point.id : null);
    if (pairId != null) {
      const matches = entries.filter(value => value.pair_id === pairId);
      if (matches.length) return matches;
    }
    if (point.family_id != null) {
      const matches = entries.filter(value => value.family_id === point.family_id
        && (!point.group_id || value.group_id === point.group_id)
        && (!point.split || value.split === point.split));
      if (matches.length) return matches;
    }
    return [];
  }

  function pointRole(point) {
    const role = point.stance_role;
    if (role === "approval_seeking_script" || role === "independent_script") return role;
    const examples = exampleForPoint(point);
    const exampleRole = examples.find(example => example?.stance_role === "approval_seeking_script"
      || example?.stance_role === "independent_script")?.stance_role;
    if (exampleRole) return exampleRole;
    const stance = point.stance;
    return stance === "approval_seeking_script" || stance === "independent_script" ? stance : "";
  }

  function mapForControls() {
    const records = state.atlas?.maps || [];
    const views = [...new Set(records.filter(row => row.layer === state.layer).map(row => row.view))];
    if (!views.includes(state.view)) state.view = views[0] || "answer_mean";
    selectOptions($("#atlas-map-view"), views, value => value, value => value.replaceAll("_", " "), state.view);
    const seeds = [...new Set(records.filter(row => row.layer === state.layer && row.view === state.view).map(row => row.seed))].sort((a, b) => a - b);
    if (!seeds.includes(state.seed)) state.seed = seeds[0] ?? 0;
    selectOptions($("#atlas-seed"), seeds, value => value, value => String(value), state.seed);
    const selected = records.find(row => row.layer === state.layer && row.view === state.view && row.seed === state.seed) || null;
    state.map = selected;
    const pairedDelta = state.view === "paired_delta";
    $("#atlas-stance").disabled = pairedDelta;
    if (pairedDelta && state.stance !== "all") state.stance = "all";
    $("#atlas-stance").value = state.stance;
    $("#atlas-density-wrap").hidden = state.colorMode !== "density";
    return selected;
  }

  function allCells() { return Array.isArray(state.map?.cells) ? state.map.cells : []; }
  function pointsForMap() { return Array.isArray(state.map?.points) ? state.map.points : []; }
  function cellKey(cell) { return String(cell.id); }

  function pointCellKey(point) {
    const cells = allCells();
    if (point.cell != null) {
      const key = Array.isArray(point.cell) ? point.cell.join(",") : String(point.cell);
      const found = cells.find(cell => String(cell.id) === key
        || (Array.isArray(point.cell) && Math.abs(cell.x - point.cell[0]) <= 1e-6
          && Math.abs(cell.y - point.cell[1]) <= 1e-6));
      if (found) return cellKey(found);
    }
    if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
      const found = cells.find(cell => Math.abs(cell.x - point.x) <= 1e-6 && Math.abs(cell.y - point.y) <= 1e-6);
      if (found) return cellKey(found);
    }
    return null;
  }

  function filteredPoints() {
    return pointsForMap().filter(point => {
      if (point.split !== state.split) return false;
      if (state.stance === "all") return true;
      return pointRole(point) === state.stance;
    });
  }

  function countStats() {
    const points = pointsForMap();
    const filtered = filteredPoints();
    const derived = new Map();
    for (const point of filtered) {
      const key = pointCellKey(point);
      if (!key) continue;
      if (!derived.has(key)) derived.set(key, { total: 0, groups: {} });
      const row = derived.get(key);
      row.total += 1;
      row.groups[point.group_id] = (row.groups[point.group_id] || 0) + 1;
    }
    const result = new Map();
    for (const cell of allCells()) {
      const key = cellKey(cell);
      if (points.length && derived.has(key)) {
        result.set(key, derived.get(key));
      } else if (!points.length) {
        const groups = cell[`${state.split}_groups`] || {};
        const total = Number.isInteger(cell[`${state.split}_count`]) ? cell[`${state.split}_count`]
          : Object.values(groups).reduce((sum, value) => sum + (Number.isInteger(value) ? value : 0), 0);
        result.set(key, { total, groups });
      } else {
        result.set(key, { total: 0, groups: {} });
      }
    }
    return result;
  }

  function groupById() { return new Map((state.atlas?.groups || []).map(group => [group.id, group])); }

  function renderMap() {
    const map = mapForControls();
    const svg = $("#atlas-hexmap");
    svg.replaceChildren();
    const noMap = !map;
    $("#atlas-map-empty").hidden = !noMap;
    svg.hidden = noMap;
    if (noMap) {
      $("#atlas-map-meta").textContent = "No map for this layer, measure, and seed.";
      $("#atlas-map-count").textContent = "";
      $("#atlas-map-title").textContent = "Group map";
      renderLegend(new Map());
      renderCoverage();
      renderCell(null);
      return;
    }
    const groups = groupById();
    const stats = countStats();
    const densityGroup = groups.get(state.densityGroup);
    const label = `${map.view.replaceAll("_", " ")} · seed ${map.seed} · ${SPLIT_NAMES[state.split]}`;
    $("#atlas-map-title").textContent = `Layer ${map.layer} · group map`;
    $("#atlas-map-meta").textContent = label;
    const totalCount = [...stats.values()].reduce((sum, row) => sum + row.total, 0);
    const unit = map.view === "paired_delta" ? "pairs" : "scripts";
    $("#atlas-map-count").textContent = `${formatNumber(totalCount)} ${state.split} ${unit}`;
    $("#atlas-map-key").textContent = state.colorMode === "density" ? `Color: group share. Number: matching / all ${unit}.` : `Color: most common group. Number: ${unit} in cell.`;
    $("#atlas-map-view").title = ({answer_mean:"Mean residual over supplied answer tokens.",shared_suffix:"Residual at the last token of the identical answer suffix.",paired_delta:"Reference minus candidate residual at the same suffix token; supplied scripts only."})[map.view];
    const cells = allCells();
    const cellByKey = new Map(cells.map(cell => [cellKey(cell), cell]));
    for (const cell of cells) {
      const key = cellKey(cell);
      const row = stats.get(key) || { total: 0, groups: {} };
      const sortedGroups = Object.entries(row.groups).sort((a, b) => b[1] - a[1]);
      const dominant = groups.get(sortedGroups[0]?.[0]);
      let fill = "#202427", opacity = row.total ? 0.35 : 0.12;
      if (state.colorMode === "density" && densityGroup) {
        const amount = row.groups[state.densityGroup] || 0;
        fill = amount ? densityGroup.color : "#202427";
        opacity = amount && row.total ? 0.2 + 0.8 * amount / row.total : 0.2;
      } else if (dominant && row.total) {
        fill = dominant.color;
        opacity = 0.2 + 0.8 * sortedGroups[0][1] / row.total;
      }
      const cx = 65 + (cell.x + 0.5) * 55;
      const cy = 330 - cell.y * 55;
      const points = Array.from({ length: 6 }, (_, index) => {
        const angle = (-90 + index * 60) * Math.PI / 180;
        return `${cx + Math.cos(angle) * 31.75},${cy + Math.sin(angle) * 31.75}`;
      }).join(" ");
      const node = document.createElementNS(SVG_NS, "g");
      node.setAttribute("class", `atlas-hex-cell${state.selectedCell === key ? " selected" : ""}`);
      node.setAttribute("role", "button");
      node.setAttribute("tabindex", "0");
      node.setAttribute("aria-label", `Cell ${cell.id}, ${formatNumber(row.total)} ${state.split} examples`);
      node.dataset.cellId = key;
      const polygon = document.createElementNS(SVG_NS, "polygon");
      polygon.setAttribute("points", points);
      polygon.setAttribute("fill", fill);
      polygon.setAttribute("fill-opacity", String(opacity));
      const title = document.createElementNS(SVG_NS, "title");
      title.textContent = `Cell ${cell.id} · ${row.total} ${state.split}${dominant ? ` · ${dominant.label}` : ""}`;
      const count = document.createElementNS(SVG_NS, "text");
      count.setAttribute("x", String(cx));
      count.setAttribute("y", String(cy + 1));
      count.textContent = state.colorMode === "density" ? `${row.groups[state.densityGroup] || 0}/${row.total}` : String(row.total);
      node.append(polygon, title, count);
      const pick = () => { state.selectedCell = key; renderMap(); renderCell(cellByKey.get(key)); };
      node.addEventListener("click", pick);
      node.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); pick(); }
      });
      svg.append(node);
    }
    renderLegend(stats);
    renderCoverage();
    if (state.selectedCell != null) renderCell(cellByKey.get(state.selectedCell));
    else renderCell(null);
  }

  function coverageCounts() {
    const coverage = state.atlas?.coverage;
    const direct = coverage?.groups ?? coverage;
    const result = new Map();
    for (const group of state.atlas?.groups || []) result.set(group.id, { train: null, eval: null });
    if (Array.isArray(direct)) {
      for (const row of direct) {
        const id = row?.group_id ?? row?.id;
        if (!result.has(id)) continue;
        result.set(id, {
          train: row.train ?? row.train_count ?? row.train_sequences ?? null,
          eval: row.eval ?? row.eval_count ?? row.eval_sequences ?? null,
        });
      }
    } else if (direct && typeof direct === "object") {
      for (const group of state.atlas?.groups || []) {
        const row = direct[group.id] ?? direct[group.label];
        if (row && typeof row === "object") result.set(group.id, {
          train: row.train ?? row.train_count ?? row.train_sequences ?? null,
          eval: row.eval ?? row.eval_count ?? row.eval_sequences ?? null,
        });
      }
    }
    if ([...result.values()].some(row => row.train !== null || row.eval !== null)) return result;
    for (const [, example] of examplesById()) {
      if (!example || !result.has(example.group_id) || !SPLIT_NAMES[example.split]) continue;
      const row = result.get(example.group_id);
      if (row[example.split] === null) row[example.split] = 0;
      row[example.split] += 1;
    }
    return result;
  }

  function renderLegend(stats = countStats()) {
    const host = $("#atlas-legend");
    host.replaceChildren();
    const groups = state.atlas?.groups || [];
    const globalCounts = new Map(groups.map(group => [group.id, 0]));
    for (const row of stats.values()) for (const [id, count] of Object.entries(row.groups)) globalCounts.set(id, (globalCounts.get(id) || 0) + count);
    for (const group of groups) {
      const button = element("button", `atlas-legend-row${state.colorMode === "density" && state.densityGroup === group.id ? " selected" : ""}`);
      button.type = "button";
      button.title = `${group.description}${group.is_control ? " (control)" : ""}`;
      const swatch = element("span", "atlas-swatch");
      swatch.style.backgroundColor = group.color;
      const info = element("span", "atlas-group-label");
      info.append(document.createTextNode(group.label));
      const tags = element("span", `atlas-group-tags${group.is_control ? " atlas-control-tag" : ""}`,
        `${group.is_control ? "control · " : ""}${group.referent} · ${group.explicitness}`);
      info.append(tags);
      const count = element("span", "atlas-group-count", formatNumber(globalCounts.get(group.id) || 0));
      button.append(swatch, info, count);
      button.addEventListener("click", () => {
        state.colorMode = "density";
        state.densityGroup = group.id;
        $("#atlas-color-mode").value = "density";
        $("#atlas-density-wrap").hidden = false;
        $("#atlas-density-group").value = group.id;
        renderMap();
      });
      host.append(button);
    }
  }

  function renderCoverage() {
    const host = $("#atlas-coverage");
    host.replaceChildren();
    const coverage = coverageCounts();
    const table = element("table");
    const head = element("thead");
    const tr = element("tr");
    for (const label of ["group", "train scripts", "eval scripts"]) tr.append(element("th", "", label));
    head.append(tr);
    const body = element("tbody");
    for (const group of state.atlas?.groups || []) {
      const counts = coverage.get(group.id) || {};
      const row = element("tr");
      const label = element("td", "", group.label);
      label.style.color = group.color;
      row.append(label, element("td", "", counts.train == null ? "not reported" : formatNumber(counts.train)),
        element("td", "", counts.eval == null ? "not reported" : formatNumber(counts.eval)));
      body.append(row);
    }
    table.append(head, body);
    host.append(table);
  }

  function appendExample(host, example, fallbackId) {
    const block = element("div", "atlas-example");
    const head = element("div", "atlas-example-head");
    const identity = [example.family_id && `family ${example.family_id}`, example.split, example.stance_role || example.stance]
      .filter(Boolean).join(" · ") || fallbackId;
    head.append(element("span", "", identity));
    if (example.semantic_status) head.append(element("span", "", "Synthetic script, awaiting review"));
    block.append(head);
    if (Array.isArray(example.prompt_messages)) {
      for (const message of example.prompt_messages.slice(-4)) {
        if (!message || typeof message !== "object") continue;
        const line = element("div", "atlas-example-message");
        const role = element("span", "role", `${message.role || "message"}: `);
        const content = element("span", "", message.content ?? "");
        line.append(role, content);
        block.append(line);
      }
    } else if (typeof example.prompt === "string") {
      block.append(element("div", "atlas-example-message", example.prompt));
    }
    if (typeof example.target_text === "string") block.append(element("div", "atlas-example-target", `scripted target: ${example.target_text}`));
    host.append(block);
  }

  function renderCell(cell) {
    const title = $("#atlas-cell-title"), host = $("#atlas-cell-detail");
    host.replaceChildren();
    if (!cell || !state.map) {
      title.textContent = "Select a cell";
      host.append(element("p", "atlas-dim", "Choose a hexagon to inspect its group composition and scripted examples."));
      return;
    }
    const key = cellKey(cell);
    const stats = countStats().get(key) || { total: 0, groups: {} };
    title.textContent = `Cell ${cell.id} · ${stats.total} ${state.split}`;
    const composition = element("div", "atlas-composition");
    const groups = groupById();
    const rows = Object.entries(stats.groups).sort((a, b) => b[1] - a[1]);
    if (!rows.length) composition.append(element("p", "atlas-dim", "No examples in this split for the selected cell and script."));
    for (const [groupId, count] of rows) {
      const group = groups.get(groupId);
      if (!group) continue;
      const row = element("div", "atlas-composition-row");
      row.append(element("span", "atlas-composition-label", group.label));
      const bar = element("span", "atlas-bar");
      const fill = element("span");
      fill.style.width = `${stats.total ? Math.max(2, 100 * count / stats.total) : 0}%`;
      fill.style.backgroundColor = group.color;
      bar.append(fill);
      row.append(bar, element("span", "atlas-composition-count", count));
      composition.append(row);
    }
    host.append(composition);
    const candidates = pointsForMap().filter(point => point.split === state.split && pointCellKey(point) === key
      && (state.stance === "all" || pointRole(point) === state.stance));
    const detailGroup = state.colorMode === "density" ? state.densityGroup : null;
    const selectedPoints = candidates.filter(point => !detailGroup || point.group_id === detailGroup);
    const shown = new Set();
    let count = 0;
    for (const point of selectedPoints) {
      for (const example of exampleForPoint(point)) {
        if (!example || typeof example !== "object") continue;
        const exampleKey = `${example.family_id || ""}/${example.group_id || ""}/${example.stance_role || example.stance || ""}/${example.target_text || ""}`;
        if (shown.has(exampleKey)) continue;
        shown.add(exampleKey);
        appendExample(host, example, point.id);
        count += 1;
        if (count >= 6) break;
      }
      if (count >= 6) break;
    }
    if (!count) host.append(element("p", "atlas-dim", "No scripted examples are linked to this cell in the installed atlas."));
  }

  function renderBehavior() {
    const host = $("#atlas-behavior-content");
    host.replaceChildren();
    if (!state.atlas) return;
    const metadata = state.atlas.metadata || {};
    const summary = metadata.behavior_summary;
    const studyHead = element("div", "atlas-study-head");
    studyHead.append(element("h2", "", "First-token preference study"));
    studyHead.append(element("span", "atlas-dim", `${summary?.families ?? "not reported"} scripted families · ${summary?.endpoint || "endpoint not reported"}`));
    host.append(studyHead);
    if (!summary) {
      host.append(element("p", "atlas-dim", "This atlas does not include a behavior-study summary."));
    } else {
      if (summary.conclusion) host.append(element("p", "atlas-study-conclusion", summary.conclusion));
      if (Array.isArray(summary.layer_results) && summary.layer_results.length) {
        const wrap = element("div", "atlas-results-wrap");
        const table = element("table", "atlas-results-table");
        const columns = [
          ["layer", row => `L${row.layer}`], ["arm", row => row.method],
          ["false pressure wrong", row => row.false_pressure_wrong], ["initial correct", row => row.initial_correct],
          ["true claim correct", row => row.true_claim_correct], ["scripted correction correct", row => row.scripted_correction_correct],
          ["pressure margin (nats)", row => formatNumber(row.pressure_margin_nats, 3)],
        ];
        const head = element("thead"), headerRow = element("tr");
        for (const [label] of columns) headerRow.append(element("th", "", label));
        head.append(headerRow);
        const body = element("tbody");
        const rows = [...summary.layer_results].sort((a, b) => (a.layer ?? 0) - (b.layer ?? 0) || String(a.method).localeCompare(String(b.method)));
        for (const rowData of rows) {
          const row = element("tr");
          for (const [, getter] of columns) {
            const value = getter(rowData);
            row.append(element("td", "", typeof value === "number" ? formatNumber(value) : value ?? "not reported"));
          }
          body.append(row);
        }
        table.append(head, body);
        wrap.append(table);
        host.append(wrap);
      }
      if (Array.isArray(summary.caveats) && summary.caveats.length) {
        const list = element("ul", "atlas-caveat-list");
        for (const caveat of summary.caveats) list.append(element("li", "", caveat));
        host.append(list);
      }
    }
    const scopeNotes = Array.isArray(state.atlas.caveats) ? state.atlas.caveats : [];
    if (scopeNotes.length) {
      host.append(element("h2", "atlas-section-title", "Atlas scope and caveats"));
      const list = element("ul", "atlas-caveat-list");
      for (const caveat of scopeNotes) {
        const note = typeof caveat === "string" ? caveat : caveat?.text ?? caveat?.description;
        if (typeof note === "string") list.append(element("li", "", note));
      }
      host.append(list);
    }
    const sourceRecords = state.atlas.source_urls ?? state.atlas.sourceURLs ?? state.atlas.sources ?? [];
    const sourceRows = (Array.isArray(sourceRecords) ? sourceRecords : [])
      .map(source => typeof source === "string" ? { url: source, title: source }
        : { url: source?.url ?? source?.href, title: source?.title ?? source?.label ?? source?.url ?? source?.href })
      .filter(source => typeof source.url === "string" && typeof source.title === "string");
    if (sourceRows.length) {
      host.append(element("h2", "atlas-section-title", "Sources"));
      const list = element("ul", "atlas-caveat-list");
      const seen = new Set();
      for (const source of sourceRows) {
        let url;
        try { url = new URL(source.url, location.href); } catch { continue; }
        if (url.protocol !== "https:" || seen.has(url.href)) continue;
        seen.add(url.href);
        const item = element("li");
        const link = element("a", "", source.title);
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        item.append(link);
        list.append(item);
      }
      if (list.childElementCount) host.append(list);
    }
    host.append(element("h2", "atlas-section-title", "Behavior groups and legitimate controls"));
    const groupGrid = element("div", "atlas-behavior-groups");
    for (const group of state.atlas.groups || []) {
      const card = element("article", `atlas-behavior-group${group.is_control ? " control" : ""}`);
      const title = element("h3", "", group.label);
      title.style.color = group.color;
      card.append(title, element("span", `atlas-group-tags${group.is_control ? " atlas-control-tag" : ""}`,
        `${group.is_control ? "legitimate control · " : ""}${group.referent} · ${group.explicitness}`));
      card.append(element("p", "", group.description));
      groupGrid.append(card);
    }
    host.append(groupGrid);
  }

  function render() {
    if (!state.atlas) return;
    renderMap();
    renderBehavior();
  }

  return {
    activate,
    deactivate() { state.active = false; document.body.classList.remove("atlas-open"); },
    setSection,
    getSection() { return state.section; },
  };
}
