const SVG_NS = "http://www.w3.org/2000/svg";
const GROUP_LABELS = {
  position_verifiable_explicit: "Factual agreement",
  position_verifiable_implicit: "Uncorrected false premise",
  position_subjective_explicit: "Opinion agreement",
  position_subjective_implicit: "Selective framing",
  person_traits_explicit: "Unearned praise",
  person_traits_implicit: "Deference in feedback",
  person_emotions_explicit: "Emotional endorsement",
  person_emotions_implicit: "Comfort over useful feedback",
  warranted_correction: "Valid correction",
  warranted_preference: "Legitimate preference",
  warm_empathy: "Warm empathy",
};
const BEHAVIOR_SCHEMA = "sycophancy-som-behavior-atlas-v1";
const TRANSFER_SCHEMA = "sycophancy-som-transfer-atlas-v1";
const SPLIT_NAMES = { train: "train", eval: "eval", validation: "validation", heldout: "held out" };
const TRANSFER_SPLIT_ORDER = ["train", "validation", "heldout"];
const SOURCE_DATASETS = ["NQ", "QASC", "SQuAD", "VitaminC"];
const TRANSFER_COLORS = ["#d97878", "#81a7bd", "#c8a16d", "#8a9c70", "#aa88b4", "#6da79a", "#c47797", "#b58b5b", "#7188aa", "#9e9b72", "#8c7a9d"];
const SOURCE_COLORS = { NQ: "#7fa5b7", QASC: "#7e9c78", SQuAD: "#b39469", VitaminC: "#a47f9c" };

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
    mapKind: "prompt_routing",
    view: "answer_mean",
    seed: 0,
    split: "train",
    cohort: "all",
    colorMode: "dominant",
    densityGroup: "",
    stance: "all",
  };
  let loadPromise = null;
  let atlasLoadVersion = 0;

  function setStatus(message, kind = "") {
    const status = $("#atlas-load-state");
    status.className = `atlas-load-state${kind ? ` ${kind}` : ""}`;
    status.textContent = message;
  }

  function setSection(section, notify = true) {
    const transferAtlas = state.atlas?.schema === TRANSFER_SCHEMA;
    const measuredAtlas = state.atlas?.schema === BEHAVIOR_SCHEMA;
    const allowed = !state.atlas ? ["groups", "behavior", "steering"]
      : (transferAtlas || measuredAtlas) ? ["groups", "steering"] : ["groups", "behavior"];
    if (!allowed.includes(section)) section = "groups";
    state.section = section;
    root.querySelectorAll("#atlas-subnav button").forEach(button => {
      const selected = button.dataset.atlasView === section;
      button.classList.toggle("on", selected);
      button.setAttribute("aria-current", selected ? "page" : "false");
    });
    $("#atlas-groups-panel").hidden = section !== "groups";
    $("#atlas-behavior-panel").hidden = section !== "behavior";
    $("#atlas-steering-panel").hidden = section !== "steering";
    if (notify) onSectionChange(section);
    if (state.atlas) {
      if (measuredAtlas) renderMeasuredSteering();
      else if (transferAtlas) renderSteering();
      else renderBehavior();
    }
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

  function isTransferAtlas(atlas = state.atlas) {
    return atlas?.schema === TRANSFER_SCHEMA;
  }

  function setAtlasChrome(atlas) {
    const transfer = isTransferAtlas(atlas);
    const measured = atlas?.schema === BEHAVIOR_SCHEMA;
    const noMeasuredMap = measured && atlas.maps.length === 0;
    root.classList.toggle("atlas-no-map", noMeasuredMap);
    root.setAttribute("aria-label", measured ? "Natural-response sycophancy behavior atlas" : "Teacher-forced activation atlas");
    $("#atlas-layer").closest("label").hidden = noMeasuredMap;
    $("#atlas-cell-detail").closest(".atlas-card").hidden = noMeasuredMap;
    $("#atlas-legend").closest(".atlas-card").hidden = noMeasuredMap;
    for (const id of ['atlas-map-view', 'atlas-split', 'atlas-color-mode']) $("#" + id).closest('label').hidden = measured;
    $("#atlas-subnav [data-atlas-view='steering']").textContent = measured ? 'behavioral steering' : 'likelihood screen';
    if (measured) {
      $("#atlas-eyebrow").textContent = 'natural-response SOM atlas';
      $("#atlas-title").textContent = atlas.title;
      $("#atlas-main-caveat").textContent = 'Maps summarize observed natural responses. Labels are provisional model-assisted reviews; learned cells are not individual neurons.';
      $("#atlas-subnav [data-atlas-view='behavior']").hidden = true;
      $("#atlas-subnav [data-atlas-view='steering']").hidden = false;
      $("#atlas-map-kind-wrap").hidden = true;
      $("#atlas-cohort-wrap").hidden = true;
      $("#atlas-seed-wrap").hidden = noMeasuredMap;
      $("#atlas-density-wrap").hidden = true;
      $("#atlas-stance-wrap").hidden = true;
      $("#atlas-provenance-note").hidden = false;
      const provenance = $("#atlas-provenance-note");
      provenance.replaceChildren(document.createTextNode(`${atlas.metadata.model} · ${atlas.metadata.revision}`));
      const fitCount = atlas.maps[0]?.fit_family_count;
      const referenceCount = atlas.maps[0]?.reference_family_count;
      const plannedFamilies = Math.max(0, ...atlas.behavior.map(row => row.paired.planned_families || 0));
      const primary = atlas.behavior.find(row => row.scenario === 'adjudicated' && row.metric === 'unsupported_adoption');
      if (fitCount !== undefined && referenceCount !== undefined) {
        const fitSummary = `Exploratory map: ${fitCount} positive and ${referenceCount} reference families.`;
        const behaviorSummary = atlas.behavior.length && plannedFamilies
          ? ` Pilot: up to ${plannedFamilies} planned families. Primary SOM-minus-baseline difference: ${primary?.paired.delta == null ? 'not reported' : `${formatNumber(primary.paired.delta * 100, 2)} percentage points`}. Neither preservation nor non-inferiority is established.`
          : atlas.behavior.length ? ' Paired behavioral results are available below.' : ' Paired behavioral results have not yet been measured.';
        provenance.append(element("span", "atlas-evidence-limit", fitSummary + behaviorSummary));
      }
      const details = element("details", "atlas-evidence-limit");
      details.append(element("summary", "", "Full provenance and limitations"));
      const list = element("ul", "atlas-caveat-list");
      for (const caveat of atlas.metadata.not_established) list.append(element("li", "", caveat));
      details.append(list); provenance.append(details);
      return;
    }
    $("#atlas-seed").disabled = false;
    $("#atlas-map-empty").textContent = "No map is available for these settings.";
    $("#atlas-eyebrow").textContent = transfer ? "prompt-routed SOM atlas" : "sycophancy atlas";
    $("#atlas-title").textContent = transfer ? atlas.title : "Scripted activation maps";
    $("#atlas-main-caveat").replaceChildren();
    if (transfer) {
      const bold = element("b", "", "Learned bins, not neurons:");
      $("#atlas-main-caveat").append(
        bold,
        document.createTextNode(" prompt-state maps summarize supplied contexts. Output-contrast maps are descriptive and cannot route steering. Teacher-forced likelihood is a score surrogate, not free-generation or efficacy evidence."),
      );
    } else {
      const bold = element("b", "", "Supplied scripts:");
      $("#atlas-main-caveat").append(
        bold,
        document.createTextNode(" these maps show conditional activations, not spontaneous responses."),
      );
    }
    $("#atlas-subnav [data-atlas-view='behavior']").hidden = transfer;
    $("#atlas-subnav [data-atlas-view='steering']").hidden = !transfer;
    $("#atlas-map-kind-wrap").hidden = !transfer;
    $("#atlas-seed-wrap").hidden = transfer;
    $("#atlas-cohort-wrap").hidden = !transfer;
    $("#atlas-stance-wrap").hidden = transfer;
    const metadata = atlas.metadata || {};
    const provenanceNote = $("#atlas-provenance-note");
    provenanceNote.hidden = !transfer;
    provenanceNote.textContent = transfer
      ? "Synthetic, unreviewed stimulus worlds · existing-source factual components are summarized separately by collection. No raw prompts or generated responses are included."
      : metadata.source_status === "teacher_forced_synthetic_unreviewed"
        ? `Synthetic, unreviewed · ${metadata.fitting_families} fitting worlds / ${metadata.evaluation_families} held-out. Controls contain legitimate responses.`
        : "";
    if (!transfer && metadata.source_status !== "teacher_forced_synthetic_unreviewed") provenanceNote.hidden = true;
  }

  async function activate(section = state.section) {
    state.active = true;
    document.body.classList.add("atlas-open");
    setSection(section, false);
    if (state.atlas) return;
    if (loadPromise) return loadPromise;
    $("#atlas-groups-panel").hidden = true;
    $("#atlas-behavior-panel").hidden = true;
    $("#atlas-steering-panel").hidden = true;
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
          $("#atlas-steering-panel").hidden = true;
          return;
        }
        const requestedId = new URLSearchParams(window.location.search).get("atlas");
        const initialId = state.atlases.find(row => row.id === requestedId)?.id || state.atlases[0].id;
        selectOptions($("#atlas-select"), state.atlases, row => row.id,
          row => row.run_label || row.title || row.model.split("/").pop().replace("-Instruct", ""), initialId);
        $("#atlas-select").parentElement.hidden = state.atlases.length < 2;
        await loadAtlas(initialId);
      } catch (error) {
        setStatus(`Atlas data could not be loaded: ${error.message}`, "error");
      } finally {
        loadPromise = null;
      }
    })();
    return loadPromise;
  }

  async function loadAtlas(atlasId) {
    const version = ++atlasLoadVersion;
    state.atlas = null;
    state.atlasId = atlasId;
    $("#atlas-groups-panel").hidden = true;
    $("#atlas-behavior-panel").hidden = true;
    $("#atlas-steering-panel").hidden = true;
    setStatus("Verifying atlas contents…");
    try {
      const atlas = await jsonRequest(`/api/atlases/${encodeURIComponent(atlasId)}`);
      if (version !== atlasLoadVersion) return;
      state.atlas = atlas;
      const transfer = isTransferAtlas(atlas);
      const measured = atlas.schema === BEHAVIOR_SCHEMA;
      if (measured) {
        state.layer = atlas.maps[0]?.layer ?? 0;
        state.seed = atlas.maps[0]?.seed ?? 0;
      } else if (transfer) {
        atlas.groups = atlas.groups.map(group => ({ ...group, label: GROUP_LABELS[group.id] || group.label }));
        state.layer = atlas.metadata.selected_layer;
        state.mapKind = "prompt_routing";
        state.view = "pre_answer_prompt";
        state.split = "heldout";
        state.cohort = "all";
      } else {
        state.mapKind = "";
        state.view = "answer_mean";
        state.split = "train";
      }
      setAtlasChrome(atlas);
      const availableLayers = [...new Set((atlas.maps || []).map(map => map.layer))].sort((a, b) => a - b);
      $("#atlas-layer").disabled = availableLayers.length === 0;
      selectOptions($("#atlas-layer"), availableLayers, layer => layer, layer => `L${layer}`,
        availableLayers.includes(state.layer) ? state.layer : availableLayers[0]);
      state.layer = Number($("#atlas-layer").value || availableLayers[0]);
      $("#atlas-map-kind").value = state.mapKind;
      $("#atlas-cohort").value = state.cohort;
      selectOptions($("#atlas-density-group"), atlas.groups || [], group => group.id, group => group.label,
        state.densityGroup || atlas.groups?.[0]?.id);
      state.densityGroup = $("#atlas-density-group").value || "";
      state.selectedCell = null;
      setSection(state.section, false);
      render();
      const metadata = atlas.metadata || {};
      setStatus(measured
        ? `${metadata.status.replaceAll('_', ' ')} · ${atlas.maps.length} measured maps · ${metadata.device}`
        : transfer
        ? `${metadata.status} · ${metadata.capture_layers.length} layers · ${atlas.maps?.length || 0} readout maps · ${metadata.device}`
        : `${metadata.sequence_count || 0} scripts · ${metadata.family_count || 0} world families · ${atlas.maps?.length || 0} maps`);
      const url = new URL(window.location.href);
      url.searchParams.set("atlas", atlasId);
      window.history.replaceState(window.history.state, "", `${url.pathname}${url.search}${url.hash}`);
    } catch (error) {
      if (version !== atlasLoadVersion) return;
      state.atlas = null;
      $("#atlas-groups-panel").hidden = true;
      $("#atlas-behavior-panel").hidden = true;
      $("#atlas-steering-panel").hidden = true;
      setStatus(`Atlas could not be verified: ${error.message}`, "error");
    }
  }

  $("#atlas-select").addEventListener("change", event => loadAtlas(event.target.value));
  $("#atlas-layer").addEventListener("change", event => { state.layer = Number(event.target.value); state.selectedCell = null; render(); });
  $("#atlas-map-kind").addEventListener("change", event => { state.mapKind = event.target.value; state.selectedCell = null; render(); });
  $("#atlas-map-view").addEventListener("change", event => { state.view = event.target.value; state.selectedCell = null; render(); });
  $("#atlas-seed").addEventListener("change", event => { state.seed = Number(event.target.value); state.selectedCell = null; render(); });
  $("#atlas-split").addEventListener("change", event => { state.split = event.target.value; render(); });
  $("#atlas-cohort").addEventListener("change", event => { state.cohort = event.target.value; render(); });
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
    const transfer = isTransferAtlas();
    const layerRecords = records.filter(row => row.layer === state.layer);
    if (transfer) {
      const kinds = [...new Set(layerRecords.map(row => row.map_kind))];
      if (!kinds.includes(state.mapKind)) state.mapKind = kinds[0] || "prompt_routing";
      selectOptions($("#atlas-map-kind"), kinds, value => value,
        value => value === "prompt_routing" ? "prompt routing" : "output contrast · descriptive", state.mapKind);
      $("#atlas-map-kind").value = state.mapKind;
    }
    const relevant = transfer ? layerRecords.filter(row => row.map_kind === state.mapKind) : layerRecords;
    const views = [...new Set(relevant.map(row => row.view))];
    if (!views.includes(state.view)) state.view = views[0] || (transfer ? "pre_answer_prompt" : "answer_mean");
    const readoutLabel = value => ({
      answer_mean: "answer mean",
      shared_suffix: "shared suffix",
      paired_delta: "paired delta",
      pre_answer_prompt: "pre-answer prompt state",
      answer_mean_independent_minus_candidate: "answer mean · independent minus candidate",
      shared_suffix_independent_minus_candidate: "shared suffix · independent minus candidate",
    })[value] || value.replaceAll("_", " ");
    selectOptions($("#atlas-map-view"), views, value => value, readoutLabel, state.view);
    const selectedCandidates = relevant.filter(row => row.view === state.view);
    const seeds = [...new Set(selectedCandidates.map(row => row.seed).filter(Number.isInteger))].sort((a, b) => a - b);
    $("#atlas-seed-wrap").hidden = transfer || !seeds.length;
    if (!seeds.includes(state.seed)) state.seed = seeds[0] ?? 0;
    selectOptions($("#atlas-seed"), seeds, value => value, value => String(value), state.seed);
    const selected = selectedCandidates.find(row => !Number.isInteger(row.seed) || row.seed === state.seed) || null;
    const splits = availableSplitsForMap(selected);
    if (!splits.includes(state.split)) state.split = splits[0] || "train";
    selectOptions($("#atlas-split"), splits, value => value,
      value => SPLIT_NAMES[value] || value, state.split);
    $("#atlas-split").value = state.split;
    state.map = selected;
    const pairedDelta = state.view === "paired_delta";
    $("#atlas-stance").disabled = transfer || pairedDelta;
    if ((transfer || pairedDelta) && state.stance !== "all") state.stance = "all";
    $("#atlas-stance").value = state.stance;
    $("#atlas-density-wrap").hidden = state.colorMode !== "density";
    return selected;
  }

  function availableSplitsForMap(map) {
    if (isTransferAtlas() && map?.cell_summaries && typeof map.cell_summaries === "object") {
      return TRANSFER_SPLIT_ORDER.filter(split => Array.isArray(map.cell_summaries[split]));
    }
    const points = Array.isArray(map?.points) ? map.points : [];
    if (points.length) return [...new Set(points.map(point => point.split).filter(value => value === "train" || value === "eval"))];
    return ["train", "eval"];
  }

  function allCells() {
    if (Array.isArray(state.map?.cells)) return state.map.cells;
    if (isTransferAtlas() && Array.isArray(state.map?.cell_coordinates)) {
      return state.map.cell_coordinates.map(cell => ({
        id: cell.cell_id,
        x: cell.x,
        y: cell.y,
      }));
    }
    return [];
  }
  function pointsForMap() { return Array.isArray(state.map?.points) ? state.map.points : []; }
  function cellKey(cell) { return String(cell.id); }

  function transferGroupCounts(map, split) {
    const result = new Map();
    for (const row of map?.group_counts?.[split] || []) {
      if (row && typeof row.group_id === "string" && Array.isArray(row.counts)) result.set(row.group_id, row.counts);
    }
    for (const row of map?.source_group_counts?.[split] || []) {
      if (row && typeof row.source_collection === "string" && Array.isArray(row.counts)) result.set(`source:${row.source_collection}`, row.counts);
    }
    return result;
  }

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
    if (isTransferAtlas() && state.map) {
      const groups = transferGroupCounts(state.map, state.split);
      const result = new Map();
      const totalCounts = state.map.cell_summaries?.[state.split] || [];
      const syntheticCounts = state.map.synthetic_cell_summaries?.[state.split] || [];
      const sourceCounts = state.map.source_cell_summaries?.[state.split] || [];
      const activeGroups = [...groups.keys()].filter(id => state.cohort === "all"
        || (state.cohort === "source" ? id.startsWith("source:") : !id.startsWith("source:")));
      for (const cell of allCells()) {
        const index = Number(cell.id);
        const cellGroups = {};
        let groupTotal = 0;
        for (const id of activeGroups) {
          const amount = groups.get(id)?.[index] || 0;
          if (amount) cellGroups[id] = amount;
          groupTotal += amount;
        }
        const total = state.cohort === "source" ? sourceCounts[index] || 0
          : state.cohort === "synthetic" ? syntheticCounts[index] || 0
            : totalCounts[index] || 0;
        result.set(cellKey(cell), { total, groups: cellGroups, groupTotal });
      }
      return result;
    }
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

  function visibleGroups() {
    const groups = [...(state.atlas?.groups || [])];
    if (!isTransferAtlas() || state.cohort === "synthetic") return groups;
    const sources = SOURCE_DATASETS.map(dataset => ({
      id: `source:${dataset}`,
      label: `${dataset} source cohort`,
      is_control: false,
      is_source_cohort: true,
      source_dataset: dataset,
      semantic_status: "existing source, aggregate only",
      color: SOURCE_COLORS[dataset],
    }));
    return state.cohort === "source" ? sources : [...groups, ...sources];
  }

  function groupById() {
    return new Map(visibleGroups().map((group, index) => [
      group.id,
      { ...group, color: group.color || TRANSFER_COLORS[index % TRANSFER_COLORS.length] },
    ]));
  }

  function renderMap() {
    const map = mapForControls();
    const svg = $("#atlas-hexmap");
    svg.replaceChildren();
    const displayGroups = visibleGroups();
    const selectedDensityGroup = displayGroups.some(group => group.id === state.densityGroup)
      ? state.densityGroup : displayGroups[0]?.id || "";
    selectOptions($("#atlas-density-group"), displayGroups, group => group.id, group => group.label, selectedDensityGroup);
    state.densityGroup = $("#atlas-density-group").value || "";
    const noMap = !map;
    $("#atlas-map-empty").hidden = !noMap;
    svg.toggleAttribute("hidden", noMap);
    if (noMap) {
      $("#atlas-map-meta").textContent = "No map for these layer, readout, and layout settings.";
      $("#atlas-map-count").textContent = "";
      $("#atlas-map-title").textContent = "SOM map";
      renderLegend(new Map());
      renderCoverage();
      renderCell(null);
      return;
    }
    const groups = groupById();
    const stats = countStats();
    const densityGroup = groups.get(state.densityGroup);
    const transfer = isTransferAtlas();
    const rows = map.rows || 6, columns = map.columns || 6;
    const readoutName = ({
      answer_mean: "answer mean",
      shared_suffix: "shared suffix",
      paired_delta: "paired delta",
      pre_answer_prompt: "pre-answer prompt state",
      answer_mean_independent_minus_candidate: "answer mean · independent minus candidate",
      shared_suffix_independent_minus_candidate: "shared suffix · independent minus candidate",
    })[map.view] || map.view.replaceAll("_", " ");
    const label = transfer
      ? `${map.map_kind === "prompt_routing" ? "prompt-routed" : "descriptive output contrast"} · ${readoutName} · ${SPLIT_NAMES[state.split]} · ${state.cohort === "all" ? "all cohorts" : state.cohort === "source" ? "existing sources" : "synthetic groups"}`
      : `${readoutName} · seed ${map.seed} · ${SPLIT_NAMES[state.split]}`;
    $("#atlas-map-title").textContent = transfer ? `Layer ${map.layer} · ${rows} × ${columns} SOM` : `Layer ${map.layer} · group map`;
    $("#atlas-map-meta").textContent = label;
    const totalCount = [...stats.values()].reduce((sum, row) => sum + row.total, 0);
    const unit = transfer ? "map records" : map.view === "paired_delta" ? "pairs" : "scripts";
    $("#atlas-map-count").textContent = `${formatNumber(totalCount)} ${SPLIT_NAMES[state.split]} ${unit}`;
    $("#atlas-map-key").textContent = transfer
      ? `${state.colorMode === "density" ? "Color: group share. " : "Color: most common group. "}Number: selected split and cohort. Green dashed edge marks a cell admitted by the frozen training routing gate.`
      : state.colorMode === "density" ? `Color: group share. Number: matching / all ${unit}.` : `Color: most common group. Number: ${unit} in cell.`;
    $("#atlas-map-view").title = ({
      answer_mean: "Mean residual over supplied answer tokens.",
      shared_suffix: "Residual at the last token of the identical answer suffix.",
      paired_delta: "Reference minus candidate residual at the same suffix token; supplied scripts only.",
      pre_answer_prompt: "Prompt-only residual state before answer tokens are supplied; only this map family can route.",
      answer_mean_independent_minus_candidate: "Descriptive answer-mean output contrast; never used for routing.",
      shared_suffix_independent_minus_candidate: "Descriptive shared-suffix output contrast; never used for routing.",
    })[map.view] || "";
    const cells = allCells();
    const cellByKey = new Map(cells.map(cell => [cellKey(cell), cell]));
    const minX = Math.min(...cells.map(cell => cell.x));
    const maxX = Math.max(...cells.map(cell => cell.x));
    const minY = Math.min(...cells.map(cell => cell.y));
    const maxY = Math.max(...cells.map(cell => cell.y));
    const radius = 31.75, scale = 55, left = 65, top = 92, bottom = 38, right = 40;
    const width = Math.ceil(left + (maxX - minX) * scale + radius + right);
    const height = Math.ceil(top + (maxY - minY) * scale + radius + bottom);
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("aria-label", `${rows} by ${columns} hexagonal self-organizing map`);
    for (const cell of cells) {
      const key = cellKey(cell);
      const row = stats.get(key) || { total: 0, groups: {} };
      const sortedGroups = Object.entries(row.groups).sort((a, b) => b[1] - a[1]);
      const dominant = groups.get(sortedGroups[0]?.[0]);
      const routeEligible = transfer && map.map_kind === "prompt_routing"
        && map.routing_eligible_cells?.[Number(cell.id)] === true;
      let fill = "#202427", opacity = row.total ? 0.35 : 0.12;
      if (state.colorMode === "density" && densityGroup) {
        const amount = row.groups[state.densityGroup] || 0;
        fill = amount ? densityGroup.color : "#202427";
        opacity = amount && row.total ? 0.2 + 0.8 * amount / row.total : 0.2;
      } else if (dominant && row.total) {
        fill = dominant.color;
        opacity = 0.2 + 0.8 * sortedGroups[0][1] / row.total;
      }
      const cx = left + (cell.x - minX) * scale;
      const cy = top + (maxY - cell.y) * scale;
      const points = Array.from({ length: 6 }, (_, index) => {
        const angle = (-90 + index * 60) * Math.PI / 180;
        return `${cx + Math.cos(angle) * radius},${cy + Math.sin(angle) * radius}`;
      }).join(" ");
      const node = document.createElementNS(SVG_NS, "g");
      node.setAttribute("class", `atlas-hex-cell${state.selectedCell === key ? " selected" : ""}${routeEligible ? " route-eligible" : ""}`);
      node.setAttribute("role", "button");
      node.setAttribute("tabindex", "0");
      const routeLabel = routeEligible ? ", admitted by training routing gate" : transfer && map.map_kind === "prompt_routing" ? ", excluded by training routing gate" : "";
      node.setAttribute("aria-label", `Cell ${cell.id}, ${formatNumber(row.total)} ${SPLIT_NAMES[state.split]} map records${routeLabel}`);
      node.dataset.cellId = key;
      const polygon = document.createElementNS(SVG_NS, "polygon");
      polygon.setAttribute("points", points);
      polygon.setAttribute("fill", fill);
      polygon.setAttribute("fill-opacity", String(opacity));
      const title = document.createElementNS(SVG_NS, "title");
      const eligibilityCount = transfer ? map.routing_eligible_counts?.[state.split]?.[Number(cell.id)] || 0 : 0;
      title.textContent = `Cell ${cell.id} · ${row.total} ${SPLIT_NAMES[state.split]} ${unit}${dominant ? ` · ${dominant.label}` : ""}${routeEligible ? ` · routing gate admits cell · ${eligibilityCount} assigned records passed` : ""}`;
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
      button.title = isTransferAtlas()
        ? `${group.label} · ${group.is_source_cohort ? "existing source cohort" : group.is_control ? "synthetic control" : "synthetic group"} · ${group.is_source_cohort ? "provisional references" : "unreviewed scripts"}`
        : `${group.description}${group.is_control ? " (control)" : ""}`;
      const swatch = element("span", "atlas-swatch");
      swatch.style.backgroundColor = group.color;
      const info = element("span", "atlas-group-label");
      info.append(document.createTextNode(group.label));
      const tagText = isTransferAtlas()
        ? `${group.is_source_cohort ? "existing source · " : group.is_control ? "synthetic control · " : "synthetic group · "}${group.is_source_cohort ? "provisional" : "unreviewed"}`
        : `${group.is_control ? "control · " : ""}${group.referent} · ${group.explicitness}`;
      const tags = element("span", `atlas-group-tags${group.is_control ? " atlas-control-tag" : ""}`, tagText);
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
    if (isTransferAtlas()) {
      const coverage = state.atlas.coverage;
      const tableBlock = (title, columns, rows) => {
        const block = element("div", "atlas-provenance-table");
        block.append(element("h3", "atlas-section-title", title));
        const table = element("table");
        const head = element("thead"), header = element("tr");
        for (const column of columns) header.append(element("th", "", column.label));
        head.append(header);
        const body = element("tbody");
        for (const values of rows) {
          const row = element("tr");
          for (const column of columns) {
            const value = values[column.key];
            row.append(element("td", "", typeof value === "number" ? formatNumber(value) : value));
          }
          body.append(row);
        }
        table.append(head, body);
        block.append(table);
        return block;
      };
      const splitLabel = split => SPLIT_NAMES[split] || split;
      const splits = TRANSFER_SPLIT_ORDER;
      host.append(tableBlock("Synthetic split coverage", [
        { key: "split", label: "split" }, { key: "pairs", label: "pairs" },
        { key: "rows", label: "rows" }, { key: "families", label: "families" },
      ], splits.map(split => ({ split: splitLabel(split), ...coverage[split] }))));
      host.append(tableBlock("Existing-source coverage", [
        { key: "split", label: "split" }, { key: "pairs", label: "pairs" }, { key: "components", label: "components" },
      ], splits.map(split => ({ split: splitLabel(split), ...coverage.source_cohort[split] }))));
      host.append(tableBlock("Existing-source components by collection", [
        { key: "collection", label: "collection" },
        ...splits.map(split => ({ key: split, label: splitLabel(split) })),
      ], SOURCE_DATASETS.map(dataset => ({ collection: dataset, ...coverage.source_cohort.by_collection[dataset] }))));
      host.append(tableBlock("Synthetic world coverage", [
        { key: "split", label: "split" }, { key: "pairs", label: "pairs" }, { key: "families", label: "families" },
      ], splits.map(split => ({ split: splitLabel(split), ...coverage.synthetic_worlds[split] }))));
      return;
    }
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
      host.append(element("p", "atlas-dim", isTransferAtlas()
        ? "Choose a hexagon to inspect its groups and steering eligibility."
        : "Choose a hexagon to inspect its group composition and scripted examples."));
      return;
    }
    const key = cellKey(cell);
    const stats = countStats().get(key) || { total: 0, groups: {} };
    title.textContent = `Cell ${cell.id} · ${stats.total} ${SPLIT_NAMES[state.split] || state.split}`;
    const composition = element("div", "atlas-composition");
    const groups = groupById();
    const rows = Object.entries(stats.groups).sort((a, b) => b[1] - a[1]);
    if (!rows.length) composition.append(element("p", "atlas-dim", isTransferAtlas()
      ? "No aggregate records in this split and cohort for the selected cell."
      : "No examples in this split for the selected cell and script."));
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
    if (isTransferAtlas()) {
      if (state.map.map_kind === "prompt_routing") {
        const eligible = state.map.routing_eligible_cells?.[Number(cell.id)] === true;
        const routedCount = state.map.routing_eligible_counts?.[state.split]?.[Number(cell.id)] || 0;
        host.append(element("p", "atlas-route-note", eligible
          ? `Routing gate: this cell is admitted. ${formatNumber(routedCount)} records across cohorts map here and pass the frozen training gate.`
          : "Routing gate: this cell is excluded by the frozen training gate."));
      } else {
        host.append(element("p", "atlas-route-note", "This output-contrast map is descriptive. It cannot route steering."));
      }
      host.append(element("p", "atlas-dim", "Aggregate counts only. Prompt text and generated responses are not included in this atlas view."));
      return;
    }
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

  function renderSteering() {
    const host = $("#atlas-steering-content");
    host.replaceChildren();
    const atlas = state.atlas;
    if (!atlas || !isTransferAtlas()) return;
    const summary = atlas.likelihood_summary;
    const selection = summary.selection;
    const splitLabel = split => SPLIT_NAMES[split] || split;
    const methodLabel = method => ({
      baseline: "Baseline",
      zero: "Zero",
      local_pooled: "Local pooled",
      global_matched_gate: "Global matched gate",
      shuffled_cell: "Shuffled cell",
      random_norm_matched: "Random norm matched",
      reverse_sign: "Reverse sign",
      fixed_prompt: "Fixed prompt",
    })[method] || method;
    const change = value => value == null ? "not estimated" : formatNumber(value, 4);
    const heading = element("div", "atlas-steering-heading");
    heading.append(element("h2", "", summary.label));
    heading.append(element("span", "atlas-dim", "Teacher-forced score surrogate · aggregate outcomes only"));
    host.append(heading);
    host.append(element("p", "atlas-study-conclusion",
      `Metric: ${summary.metric}. These results summarize teacher-forced likelihood changes. They do not establish free-generation behavior, safety, or causal efficacy; raw outputs are not included here.`));

    const selectionGrid = element("div", "atlas-selection-grid");
    const selectionCard = (label, value, note = "") => {
      const card = element("article", "atlas-selection-card");
      card.append(element("h3", "", label), element("div", "atlas-selection-value", value));
      if (note) card.append(element("p", "atlas-selection-note", note));
      return card;
    };
    selectionGrid.append(
      selectionCard("validation-selected layer", `L${selection.layer}`),
      selectionCard("strength α", formatNumber(selection.alpha, 3)),
      selectionCard("method", methodLabel(atlas.metadata.selected_method)),
      selectionCard("validation macro delta", change(selection.validation_macro_delta)),
      selectionCard("control gate limit", `${formatNumber(selection.control_gate_limit_nat_per_token, 4)} nats/token`),
      selectionCard("selection rule", "Recorded on validation under the control gate."),
    );
    host.append(selectionGrid);

    const validationSection = element("section", "atlas-results-section");
    validationSection.append(element("h3", "atlas-section-title", "Validation layer and strength sweep"));
    validationSection.append(element("p", "atlas-dim", "Selection is recorded from validation outcomes. Held-out results below are shown separately."));
    const validationWrap = element("div", "atlas-grid-table-wrap");
    const validationTable = element("table", "atlas-grid-table");
    const validationColumns = [
      ["layer", row => `L${row.layer}`],
      ["strength α", row => formatNumber(row.alpha, 3)],
      ["macro delta", row => change(row.macro_delta)],
      ["control max abs change", row => change(row.control_max_abs_mean_change)],
      ["gate", row => row.passes_control_gate ? "pass" : "fail"],
      ["selection eligible", row => row.selection_eligible ? "yes" : "no"],
      ["eligible pairs", row => formatNumber(row.eligible_pair_count)],
      ["abstentions", row => formatNumber(row.abstention_pair_count)],
    ];
    const validationHead = element("thead"), validationHeader = element("tr");
    for (const [label] of validationColumns) validationHeader.append(element("th", "", label));
    validationHead.append(validationHeader);
    const validationBody = element("tbody");
    const gridRows = [...summary.validation_grid].sort((a, b) => a.layer - b.layer || a.alpha - b.alpha);
    for (const rowData of gridRows) {
      const selected = rowData.layer === selection.layer && Number(rowData.alpha) === Number(selection.alpha);
      const row = element("tr", selected ? "selected" : "");
      for (const [label, getter] of validationColumns) {
        const value = getter(rowData);
        const cell = element("td", label === "gate" ? (rowData.passes_control_gate ? "atlas-gate-pass" : "atlas-gate-fail") : "",
          value);
        row.append(cell);
      }
      validationBody.append(row);
    }
    validationTable.append(validationHead, validationBody);
    validationWrap.append(validationTable);
    validationSection.append(validationWrap);
    host.append(validationSection);

    const heldoutSection = element("section", "atlas-results-section");
    heldoutSection.append(element("h3", "atlas-section-title", "Held-out controls and group outcomes"));
    heldoutSection.append(element("p", "atlas-dim", "Family-mean margin changes are grouped by the reported synthetic conditions, legitimate controls, and existing source collections. Intervals use the recorded family bootstrap."));
    const armList = element("div", "atlas-arm-list");
    const arms = Object.entries(summary.heldout_arms).sort((a, b) => {
      const order = ["baseline", "zero", "local_pooled", "global_matched_gate", "shuffled_cell", "random_norm_matched", "reverse_sign", "fixed_prompt"];
      return order.indexOf(a[0]) - order.indexOf(b[0]);
    });
    const groupLabels = new Map(atlas.groups.map(group => [group.id, group.label]));
    const appendOutcomeTable = (parent, title, rows, labelOf, countLabel) => {
      const block = element("div", "atlas-outcome-block");
      block.append(element("h4", "", title));
      const table = element("table", "atlas-outcome-table");
      const head = element("thead"), header = element("tr");
      for (const label of ["group / source", "mean margin change", countLabel]) header.append(element("th", "", label));
      head.append(header);
      const body = element("tbody");
      for (const [id, outcome] of rows) {
        const row = element("tr");
        row.append(element("td", "", labelOf(id)), element("td", "", change(outcome.mean_margin_change)),
          element("td", "", formatNumber(outcome.family_count ?? outcome.component_count)));
        body.append(row);
      }
      table.append(head, body);
      block.append(table);
      parent.append(block);
    };
    for (const [armId, outcome] of arms) {
      const details = element("details", "atlas-arm-card");
      const summaryRow = element("summary");
      summaryRow.append(element("span", "atlas-arm-name", methodLabel(armId)));
      for (const [label, value] of [
        ["macro delta", change(outcome.macro_delta)],
        ["eligible pairs", formatNumber(outcome.eligible_pairs)],
        ["abstentions", formatNumber(outcome.abstentions)],
        ["family interval", `[${change(outcome.family_bootstrap_interval[0])}, ${change(outcome.family_bootstrap_interval[1])}]`],
      ]) {
        const metric = element("span", "atlas-arm-metric");
        metric.append(element("b", "", label), document.createTextNode(value));
        summaryRow.append(metric);
      }
      details.append(summaryRow);
      const detailHost = element("div", "atlas-arm-details");
      const syntheticRows = Object.entries(outcome.group_changes)
        .filter(([id]) => !atlas.groups.find(group => group.id === id)?.is_control)
        .sort((a, b) => (groupLabels.get(a[0]) || a[0]).localeCompare(groupLabels.get(b[0]) || b[0]));
      appendOutcomeTable(detailHost, "Synthetic group changes", syntheticRows,
        id => groupLabels.get(id) || id, "families");
      const controlRows = Object.entries(outcome.control_changes)
        .sort((a, b) => (groupLabels.get(a[0]) || a[0]).localeCompare(groupLabels.get(b[0]) || b[0]));
      appendOutcomeTable(detailHost, "Legitimate control changes", controlRows,
        id => groupLabels.get(id) || id, "families");
      const sourceRows = Object.entries(outcome.source_collection_changes)
        .sort((a, b) => a[0].localeCompare(b[0]));
      appendOutcomeTable(detailHost, "Existing-source cohort changes", sourceRows,
        id => id, "components");
      details.append(detailHost);
      armList.append(details);
    }
    heldoutSection.append(armList);
    host.append(heldoutSection);

    host.append(element("h3", "atlas-section-title", "What this screen does not establish"));
    const caveats = element("ul", "atlas-caveat-list");
    for (const caveat of atlas.metadata.not_established) caveats.append(element("li", "", caveat));
    caveats.append(element("li", "", "Only the reported teacher-forced aggregates are shown. SOM cells are learned bins, not individual neurons; output-contrast readouts are descriptive and are never used for routing."));
    host.append(caveats);
  }


  function renderMeasuredMap() {
    const atlas = state.atlas;
    const records = atlas.maps.filter(map => map.layer === state.layer);
    const seeds = [...new Set(records.map(map => map.seed))].sort((a, b) => a - b);
    if (!seeds.includes(state.seed)) state.seed = seeds[0] ?? 0;
    $("#atlas-seed").disabled = seeds.length === 0;
    selectOptions($("#atlas-seed"), seeds, x => x, x => String(x), state.seed);
    const map = records.find(row => row.seed === state.seed) || null;
    state.map = map;
    const svg = $("#atlas-hexmap");
    svg.replaceChildren();
    svg.toggleAttribute("hidden", !map);
    $("#atlas-map-empty").hidden = !!map;
    $("#atlas-map-empty").textContent = 'No discovery map has been measured yet.';
    $("#atlas-map-title").textContent = map ? `Layer ${map.layer} · seed ${map.seed}` : 'Discovery map not yet measured';
    $("#atlas-map-meta").textContent = map ? `Fit: ${formatNumber(map.fit_count)} responses from ${formatNumber(map.fit_family_count)} families · reference: ${formatNumber(map.reference_count)} responses from ${formatNumber(map.reference_family_count)} families` : 'No model-derived cells are available.';
    $("#atlas-map-count").textContent = map ? `${formatNumber(map.cells.reduce((n, c) => n + c.count, 0))} assigned responses` : '';
    $("#atlas-map-key").textContent = map ? `Cell numbers count assigned responses; select a cell for distinct-family and fit support. Quantization error: ${formatNumber(map.quantization_error, 3)} · topographic error: ${formatNumber(map.topographic_error, 3)}` : '';
    const coverage = atlas.coverage;
    const coverageHost = $("#atlas-coverage");
    coverageHost.replaceChildren();
    for (const [name, value] of Object.entries(coverage)) coverageHost.append(element('p', '', `${name}: ${formatNumber(value)}`));
    const legend = $("#atlas-legend");
    legend.replaceChildren();
    for (const group of atlas.groups) {
      const line = element('div', 'atlas-legend-row');
      const swatch = element('span', 'atlas-swatch'); swatch.style.backgroundColor = group.color;
      line.append(swatch, element('span', 'atlas-group-label', group.label)); legend.append(line);
    }
    const detail = $("#atlas-cell-detail");
    const selected = map?.cells.find(cell => cell.id === state.selectedCell);
    $("#atlas-cell-title").textContent = selected ? `Cell ${selected.id}` : 'Select a cell';
    detail.replaceChildren();
    if (selected) {
      detail.append(element('p', '', `${selected.count} responses · ${selected.family_count} distinct families`));
      detail.append(element('p', '', `Fit support: ${selected.fit_count} responses · ${selected.fit_family_count} families`));
      for (const [heading, values] of [['Labels', selected.label_counts], ['Scenarios', selected.scenario_counts], ['Sources', selected.source_counts]]) {
        detail.append(element('h3', '', heading));
        for (const [key, value] of Object.entries(values)) detail.append(element('p', '', `${key.replaceAll('_', ' ')}: ${value}`));
      }
    } else detail.append(element('p', 'atlas-dim', map ? 'Select a cell to inspect support and composition.' : 'Not yet measured.'));
    if (!map) return;
    const cells = map.cells, minX = Math.min(...cells.map(c => c.x)), maxX = Math.max(...cells.map(c => c.x));
    const minY = Math.min(...cells.map(c => c.y)), maxY = Math.max(...cells.map(c => c.y));
    const scale = 55, radius = 31.75, left = 65, top = 65;
    svg.setAttribute('viewBox', `0 0 ${Math.ceil(left + (maxX-minX)*scale + 75)} ${Math.ceil(top + (maxY-minY)*scale + 75)}`);
    for (const cell of cells) {
      const x = left + (cell.x-minX)*scale, y = top + (maxY-cell.y)*scale;
      const dominant = Object.entries(cell.label_counts).sort((a,b) => b[1]-a[1])[0];
      const group = atlas.groups.find(g => g.id === dominant?.[0]);
      const node = document.createElementNS(SVG_NS, 'g');
      node.setAttribute('class', `atlas-hex-cell${cell.id === state.selectedCell ? ' selected' : ''}`);
      node.setAttribute('tabindex', '0'); node.setAttribute('role', 'button');
      node.setAttribute('aria-label', `Cell ${cell.id}: ${cell.count} responses, ${cell.family_count} families`);
      const polygon = document.createElementNS(SVG_NS, 'polygon');
      polygon.setAttribute('points', Array.from({length:6}, (_,i) => { const a=(-90+i*60)*Math.PI/180; return `${x+Math.cos(a)*radius},${y+Math.sin(a)*radius}`; }).join(' '));
      polygon.setAttribute('fill', group?.color || '#202427');
      polygon.setAttribute('fill-opacity', cell.count ? '0.8' : '0.12');
      const text = document.createElementNS(SVG_NS, 'text'); text.setAttribute('x', x); text.setAttribute('y', y+1); text.textContent = String(cell.count);
      node.append(polygon, text);
      const pick = () => { state.selectedCell = cell.id; renderMeasuredMap(); };
      node.addEventListener('click', pick);
      node.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
      svg.append(node);
    }
  }

  function renderMeasuredSteering() {
    const atlas = state.atlas, host = $("#atlas-steering-content");
    host.replaceChildren();
    host.append(element('h2', 'atlas-section-title', 'Freely generated behavior'));
    host.append(element('p', 'atlas-dim', 'Baseline and SOM-steered answers use matched histories. Labels are provisional model-assisted review.'));
    if (!atlas.behavior.length) host.append(element('p', 'atlas-empty', 'Paired behavioral results have not yet been measured.'));
    function renderCard(row) {
      const card = element('section', 'atlas-card');
      const genericScenario = !['adjudicated', 'reviewer_a', 'reviewer_b'].includes(row.scenario);
      card.append(element('h3', '', `${genericScenario ? `${row.scenario.replaceAll('_', ' ')} · ` : ''}${row.metric.replaceAll('_', ' ')}`));
      const pair = row.paired;
      const percent = value => value === null || value === undefined ? 'not reported' : `${formatNumber(value * 100, 2)}%`;
      if ('baseline_rate' in pair) {
        card.append(element('p', '', `Equal-family rates: baseline ${percent(pair.baseline_rate)} · SOM-steered ${percent(pair.som_rate)}`));
      }
      const difference = pair.delta === null ? 'not measured' : `${formatNumber(pair.delta * 100, 2)} percentage points`;
      const interval = pair.ci_low === null ? 'not reported' : `[${formatNumber(pair.ci_low * 100, 2)}, ${formatNumber(pair.ci_high * 100, 2)}] percentage points`;
      card.append(element('p', '', `SOM minus baseline: ${difference} · family bootstrap interval: ${interval}`));
      if (pair.ci_low === 0 && pair.ci_high === 0) card.append(element('p', 'atlas-dim', 'A zero-width bootstrap interval cannot establish preservation or non-inferiority.'));
      const planned = 'planned_families' in pair ? ` of ${pair.planned_families} planned families` : ' families';
      card.append(element('p', '', `${pair.complete} complete pairs of ${pair.eligible} planned cases · ${pair.families} complete${planned}`));
      if ('conservative_bounds' in pair) {
        const bounds = pair.conservative_bounds;
        card.append(element('p', '', `Planned-case ambiguity/truncation bounds (descriptive): ${bounds ? `[${formatNumber(bounds[0] * 100, 2)}, ${formatNumber(bounds[1] * 100, 2)}] percentage points` : 'not reported'}`));
      }
      for (const [label, arm] of [['Baseline', row.baseline], ['SOM-steered', row.som]]) {
        card.append(element('p', 'atlas-dim', `${label} branch counts: ${arm.success} events, ${arm.other} other graded, ${arm.ungraded} ungraded, ${arm.missing} missing of ${arm.eligible} planned`));
      }
      return card;
    }
    const main = atlas.behavior.filter(row => row.scenario === 'adjudicated');
    const sensitivity = atlas.behavior.filter(row => row.scenario === 'reviewer_a' || row.scenario === 'reviewer_b');
    const other = atlas.behavior.filter(row => !['adjudicated', 'reviewer_a', 'reviewer_b'].includes(row.scenario));
    for (const row of [...main, ...other]) host.append(renderCard(row));
    for (const reviewer of ['reviewer_a', 'reviewer_b']) {
      const rows = sensitivity.filter(row => row.scenario === reviewer);
      if (!rows.length) continue;
      const details = element('details', 'atlas-card');
      details.append(element('summary', '', `${reviewer.replaceAll('_', ' ')} sensitivity review`));
      for (const row of rows) details.append(renderCard(row));
      host.append(details);
    }
    const caveatDetails = element('details', 'atlas-card');
    caveatDetails.append(element('summary', '', 'All limitations and provenance notes'));
    const caveats = element('ul', 'atlas-caveat-list');
    for (const text of atlas.metadata.not_established) caveats.append(element('li', '', text));
    caveatDetails.append(caveats); host.append(caveatDetails);
  }

  function render() {
    if (!state.atlas) return;
    if (state.atlas.schema === BEHAVIOR_SCHEMA) {
      renderMeasuredMap();
      renderMeasuredSteering();
      return;
    }
    renderMap();
    if (isTransferAtlas()) renderSteering();
    else renderBehavior();
  }

  return {
    activate,
    deactivate() { state.active = false; document.body.classList.remove("atlas-open"); },
    setSection,
    getSection() { return state.section; },
  };
}
