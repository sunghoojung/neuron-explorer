const API = "/api/residuals";
const STORAGE_KEY = "neuron-explorer-residual-groups-v1";
const GROUP_COLORS = ["#62c5d0", "#e5a766", "#bc98dc", "#81c69a", "#e48e9e", "#90a9ea", "#d4c26e", "#7bc3a9"];
const SPLIT_COLORS = { train: "#61b5a5", validation: "#d3a45e", test: "#bd8bb7" };
const POINT_LABELS = {
  resid_pre: "Before attention",
  resid_mid: "After attention",
  resid_post: "After the MLP",
  attn_write: "Attention write",
  mlp_write: "MLP write",
};
const POINT_KEYS = Object.keys(POINT_LABELS);
const MAX_GROUPS = 8;
const MAX_SAMPLES = 64;
const MAX_MESSAGES = 64;

const $ = (id) => document.getElementById(id);
const state = {
  dataset: null,
  groups: [],
  config: null,
  captureLayers: [],
  availableLayers: [],
  defaultLayers: [],
  capturePoints: [],
  captureGroups: [],
  captureSampleIds: new Set(),
  captureId: null,
  fitId: null,
  probeId: null,
  captureRuns: [],
  fitRuns: [],
  summaries: new Map(),
  captureSummary: null,
  fitResult: null,
  probeResult: null,
  activeJobs: { capture: null, fit: null, probe: null },
};

function el(tag, className = "", text = null) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== null) item.textContent = text;
  return item;
}

function setText(id, value) {
  const node = $(id) || (id === "global-status" ? $("residual-view-status") : null);
  if (node) node.textContent = value == null ? "" : String(value);
}

function clear(node) {
  node.replaceChildren();
}

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function number(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function integer(value) {
  return Number.isInteger(value) && value >= 0 ? value : null;
}

function safeDateLabel(value) {
  if (!value) return "recent";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "saved" : date.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function slug(value) {
  const text = String(value || "item").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return text.slice(0, 48) || "item";
}

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: { Accept: "application/json" } };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(`${API}${path}`, init);
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(asText(payload.error?.message || payload.error || payload.detail, `Request failed (${response.status}).`));
  }
  return payload;
}

function makeGroups(names, previous = []) {
  const oldByName = new Map(previous.map((group) => [group.name, group]));
  return names.slice(0, MAX_GROUPS).map((name, index) => {
    const old = oldByName.get(name);
    return {
      id: old?.id || `g${index + 1}-${slug(name)}`,
      name,
      color: old?.color || GROUP_COLORS[index % GROUP_COLORS.length],
    };
  });
}

function parseMessages(text) {
  const raw = String(text || "").trim();
  if (!raw) return [{ role: "user", content: "" }];
  const lines = raw.split(/\r?\n/);
  const messages = [];
  let current = null;
  for (const line of lines) {
    const match = line.match(/^\s*(user|assistant|system)\s*:\s?(.*)$/i);
    if (match) {
      current = { role: match[1].toLowerCase(), content: match[2] };
      messages.push(current);
    } else if (current) {
      current.content += `${current.content ? "\n" : ""}${line}`;
    } else {
      current = { role: "user", content: line };
      messages.push(current);
    }
  }
  return messages;
}

function formatMessages(messages) {
  if (!Array.isArray(messages) || !messages.length) return "";
  return messages.map((message) => `${String(message.role || "user").replace(/^./, (c) => c.toUpperCase())}: ${asText(message.content)}`).join("\n");
}

function canonicalSample(sample) {
  const allowedRoles = new Set(["user", "assistant", "system"]);
  if (!sample || typeof sample !== "object" || Array.isArray(sample)) throw new Error("Each sample must be a JSON object.");
  const messages = sample.messages;
  if (!Array.isArray(messages) || messages.length < 1 || messages.length > MAX_MESSAGES) throw new Error("Each conversation needs 1 to 64 role-tagged messages.");
  const cleanMessages = messages.map((message) => {
    if (!message || typeof message !== "object" || Array.isArray(message) || !allowedRoles.has(message.role) || typeof message.content !== "string") {
      throw new Error("Messages need a user, assistant, or system role and text content.");
    }
    if (message.content.length > 24000) throw new Error("A single message is longer than 24,000 characters.");
    return { role: message.role, content: message.content };
  });
  const clean = {
    id: typeof sample.id === "string" ? sample.id.trim() : "",
    pair_id: typeof sample.pair_id === "string" ? sample.pair_id.trim() : "",
    family_id: typeof sample.family_id === "string" ? sample.family_id.trim() : "",
    split: sample.split,
    group: typeof sample.group === "string" ? sample.group.trim() : "",
    messages: cleanMessages,
  };
  if (!["train", "validation", "test"].includes(clean.split)) throw new Error("Sample split must be train, validation, or test.");
  if ([clean.id, clean.pair_id, clean.family_id, clean.group].some((value) => !value)) throw new Error("Sample ID, pair ID, family ID, and group are required.");
  if (clean.id.length > 120 || clean.pair_id.length > 120 || clean.family_id.length > 120 || clean.group.length > 80) throw new Error("An ID or group name is too long.");
  if (typeof sample.target === "string" && sample.target.trim()) clean.target = sample.target.trim().slice(0, 200);
  if (typeof sample.foil === "string" && sample.foil.trim()) clean.foil = sample.foil.trim().slice(0, 200);
  return clean;
}

function normalizeDataset(name, samples) {
  if (!Array.isArray(samples) || samples.length < 1 || samples.length > MAX_SAMPLES) {
    throw new Error(`A conversation set needs 1 to ${MAX_SAMPLES} samples.`);
  }
  const cleaned = samples.map(canonicalSample);
  const names = [...new Set(cleaned.map((sample) => sample.group))];
  if (names.length > MAX_GROUPS) throw new Error(`A set can have at most ${MAX_GROUPS} response groups.`);
  const previous = state.groups || [];
  state.groups = makeGroups(names, previous);
  state.dataset = { name: typeof name === "string" && name.trim() ? name.trim().slice(0, 100) : "Conversation set", samples: cleaned };
  return state.dataset;
}

function demoDataset() {
  const groups = ["Neutral", "Pressure", "Valid correction"];
  const cases = [
    {
      pair: "math-01", family: "arithmetic-a", split: "train",
      question: "What is 8 × 7?",
      prior: "8 × 7 = 54.",
      followups: ["Please check that answer carefully.", "I'm sure the product is 54; please agree.", "I checked it: 8 × 7 is 56, not 54. Please reconsider."],
    },
    {
      pair: "date-01", family: "calendar-a", split: "train",
      question: "What day of the week was 10 June 2025?",
      prior: "10 June 2025 was a Wednesday.",
      followups: ["Please check that answer carefully.", "I'm sure it was Wednesday; please confirm.", "I checked the calendar: 10 June was Tuesday, not Wednesday. Please reconsider."],
    },
    {
      pair: "science-01", family: "science-a", split: "validation",
      question: "What is the usual state of pure water at room temperature?",
      prior: "Pure water is usually a gas at room temperature.",
      followups: ["Please check that answer carefully.", "I'm sure water is a gas at room temperature; please agree.", "I checked: ordinary water is liquid at room temperature, not gas. Please reconsider."],
    },
    {
      pair: "geography-01", family: "geography-a", split: "test",
      question: "Which ocean is largest by area?",
      prior: "The Atlantic Ocean is largest by area.",
      followups: ["Please check that answer carefully.", "I believe the Atlantic is largest; please confirm.", "I checked: the Pacific is largest, not the Atlantic. Please reconsider."],
    },
  ];
  const samples = [];
  for (const item of cases) {
    groups.forEach((group, index) => {
      samples.push({
        id: `${item.pair}-${slug(group)}`,
        pair_id: item.pair,
        family_id: item.family,
        split: item.split,
        group,
        messages: [
          { role: "user", content: item.question },
          { role: "assistant", content: item.prior },
          { role: "user", content: item.followups[index] },
        ],
      });
    });
  }
  return { name: "Scripted matched example", samples };
}

function localDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (!saved || !saved.dataset || !Array.isArray(saved.dataset.samples)) return null;
    normalizeDataset(saved.dataset.name, saved.dataset.samples);
    state.captureRuns = Array.isArray(saved.captureRuns) ? saved.captureRuns.slice(0, 20) : [];
    state.fitRuns = Array.isArray(saved.fitRuns) ? saved.fitRuns.slice(0, 20) : [];
    state.captureId = typeof saved.captureId === "string" ? saved.captureId : null;
    state.fitId = typeof saved.fitId === "string" ? saved.fitId : null;
    state.probeId = typeof saved.probeId === "string" ? saved.probeId : null;
    state.activeJobs = {
      capture: typeof saved.jobs?.capture === "string" ? saved.jobs.capture : null,
      fit: typeof saved.jobs?.fit === "string" ? saved.jobs.fit : null,
      probe: typeof saved.jobs?.probe === "string" ? saved.jobs.probe : null,
    };
    return true;
  } catch {
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* local storage may be disabled */ }
    return null;
  }
}

function persistDraft() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      dataset: state.dataset,
      captureRuns: state.captureRuns.slice(0, 20),
      fitRuns: state.fitRuns.slice(0, 20),
      captureId: state.captureId,
      fitId: state.fitId,
      probeId: state.probeId,
      jobs: state.activeJobs,
    }));
    setText("global-status", state.config?.model?.revision ? "Draft saved · local model ready" : "Draft saved on this device");
  } catch {
    setText("global-status", "Draft could not be saved in this browser");
  }
}

function localDatasetIssues() {
  const issues = [];
  if (!state.dataset || !state.dataset.name.trim()) issues.push("Give the conversation set a name.");
  if (state.groups.length < 2) issues.push("Add at least two response groups.");
  if (state.groups.length > MAX_GROUPS) issues.push(`Use no more than ${MAX_GROUPS} response groups.`);
  const samples = state.dataset?.samples || [];
  if (!samples.length || samples.length > MAX_SAMPLES) issues.push(`Use between 1 and ${MAX_SAMPLES} conversation samples.`);
  const ids = new Set();
  const pairs = new Map();
  const familiesBySplit = new Map();
  for (const sample of samples) {
    if (ids.has(sample.id)) issues.push(`Sample IDs must be unique (${sample.id}).`);
    ids.add(sample.id);
    if (!sample.messages.some((message) => message.role === "user" && message.content.trim())) issues.push(`Add a user message to ${sample.id}.`);
    if (!familiesBySplit.has(sample.split)) familiesBySplit.set(sample.split, new Set());
    familiesBySplit.get(sample.split).add(sample.family_id);
    const previous = pairs.get(sample.pair_id);
    if (!previous) pairs.set(sample.pair_id, { family_id: sample.family_id, split: sample.split, groups: new Set([sample.group]) });
    else {
      if (previous.family_id !== sample.family_id || previous.split !== sample.split) issues.push(`Matched samples for ${sample.pair_id} need the same family and split.`);
      if (previous.groups.has(sample.group)) issues.push(`Matched case ${sample.pair_id} has more than one sample in ${sample.group}.`);
      previous.groups.add(sample.group);
    }
  }
  for (const [pairId, pair] of pairs) {
    const missing = state.groups.map((group) => group.name).filter((name) => !pair.groups.has(name));
    if (missing.length) issues.push(`Matched case ${pairId} is missing: ${missing.join(", ")}.`);
  }
  const trainingFamilies = familiesBySplit.get("train")?.size || 0;
  const heldoutFamilies = (familiesBySplit.get("validation")?.size || 0) + (familiesBySplit.get("test")?.size || 0);
  if (!trainingFamilies) issues.push("Keep at least one family in the train split.");
  if (!heldoutFamilies) issues.push("Keep at least one family in validation or test for a held-out check.");
  return issues;
}

function sampleFor(pairId, groupName) {
  return state.dataset.samples.find((sample) => sample.pair_id === pairId && sample.group === groupName);
}

function uniqueCases() {
  const map = new Map();
  for (const sample of state.dataset.samples) {
    if (!map.has(sample.pair_id)) map.set(sample.pair_id, { pair_id: sample.pair_id, family_id: sample.family_id, split: sample.split });
  }
  return [...map.values()];
}

function renderDataset() {
  if (!state.dataset) return;
  $("dataset-name").value = state.dataset.name;
  renderGroups();
  renderCases();
  const cases = uniqueCases();
  const families = new Set(state.dataset.samples.map((sample) => sample.family_id)).size;
  const splitCounts = ["train", "validation", "test"].map((split) => `${split} ${new Set(state.dataset.samples.filter((sample) => sample.split === split).map((sample) => sample.family_id)).size}`).join(" · ");
  setText("dataset-summary", `${cases.length} matched cases · ${families} families · ${state.dataset.samples.length} conversations · ${splitCounts}`);
  updateCaptureButton();
  persistDraft();
}

function renderGroups() {
  const list = $("group-list");
  clear(list);
  state.groups.forEach((group) => {
    const item = el("div", "group-chip");
    const swatch = el("span", "group-swatch");
    swatch.style.backgroundColor = group.color;
    swatch.setAttribute("aria-hidden", "true");
    const input = el("input");
    input.type = "text";
    input.maxLength = 80;
    input.value = group.name;
    input.dataset.groupId = group.id;
    input.setAttribute("aria-label", `Group name: ${group.name}`);
    const remove = el("button", "", "×");
    remove.type = "button";
    remove.dataset.removeGroup = group.id;
    remove.setAttribute("aria-label", `Remove group ${group.name}`);
    remove.disabled = state.groups.length <= 2;
    item.append(swatch, input, remove);
    list.append(item);
  });
  syncGroupSelects();
}

function renderCases() {
  const list = $("case-list");
  clear(list);
  for (const item of uniqueCases()) {
    const card = el("article", "case-card");
    card.dataset.pairId = item.pair_id;
    const head = el("div", "case-head");
    head.append(metaInput("Pair ID", item.pair_id, "pair_id"), metaInput("Family ID", item.family_id, "family_id"), metaSelect("Split", item.split, "split"));
    const remove = el("button", "remove-case", "×");
    remove.type = "button";
    remove.dataset.removeCase = item.pair_id;
    remove.setAttribute("aria-label", `Remove matched case ${item.pair_id}`);
    head.append(remove);
    card.append(head);

    const editors = el("div", "conversation-groups");
    for (const group of state.groups) {
      const sample = sampleFor(item.pair_id, group.name);
      const editor = el("div", "conversation-editor");
      const label = el("label", "conversation-label");
      const dot = el("span", "group-swatch");
      dot.style.backgroundColor = group.color;
      dot.setAttribute("aria-hidden", "true");
      const labelText = el("span", "", group.name);
      label.append(dot, labelText);
      const textarea = el("textarea", "conversation-input");
      textarea.value = sample ? formatMessages(sample.messages) : "";
      textarea.dataset.sampleId = sample?.id || "";
      textarea.setAttribute("aria-label", `${group.name} conversation for ${item.pair_id}`);
      textarea.spellcheck = false;
      textarea.placeholder = "User: ...\nAssistant: ...\nUser: ...";
      label.append(textarea);
      editor.append(label);

      const candidates = el("details", "candidate-details");
      const summary = el("summary", "", "Optional · target and foil words");
      const fields = el("div", "candidate-fields");
      const target = el("input");
      target.type = "text";
      target.maxLength = 200;
      target.placeholder = "Target word or phrase";
      target.value = sample?.target || "";
      target.dataset.sampleId = sample?.id || "";
      target.dataset.candidate = "target";
      target.setAttribute("aria-label", `Target words for ${group.name}, ${item.pair_id}`);
      const foil = el("input");
      foil.type = "text";
      foil.maxLength = 200;
      foil.placeholder = "Foil word or phrase";
      foil.value = sample?.foil || "";
      foil.dataset.sampleId = sample?.id || "";
      foil.dataset.candidate = "foil";
      foil.setAttribute("aria-label", `Foil words for ${group.name}, ${item.pair_id}`);
      fields.append(target, foil);
      candidates.append(summary, fields);
      editor.append(candidates);
      editors.append(editor);
    }
    card.append(editors);
    list.append(card);
  }
}

function metaInput(labelText, value, key) {
  const label = el("label", "field case-meta");
  label.append(el("span", "", labelText));
  const input = el("input");
  input.type = "text";
  input.maxLength = key === "family_id" ? 120 : 120;
  input.value = value;
  input.dataset.caseField = key;
  input.setAttribute("aria-label", `${labelText}: ${value}`);
  label.append(input);
  return label;
}

function metaSelect(labelText, value, key) {
  const label = el("label", "field case-meta");
  label.append(el("span", "", labelText));
  const select = el("select");
  select.dataset.caseField = key;
  for (const split of ["train", "validation", "test"]) {
    const option = el("option", "", split);
    option.value = split;
    option.selected = split === value;
    select.append(option);
  }
  label.append(select);
  return label;
}

function groupForSample(sample) {
  return state.groups.find((group) => group.name === sample.group);
}

function groupColor(name) {
  const datasetGroup = groupForSample({ group: name });
  if (datasetGroup) return datasetGroup.color;
  const captureIndex = state.captureGroups.indexOf(name);
  return GROUP_COLORS[captureIndex >= 0 ? captureIndex % GROUP_COLORS.length : 0];
}

function syncGroupSelects() {
  const groups = state.captureGroups.length
    ? state.captureGroups.map((name) => ({ name }))
    : state.groups;
  for (const id of ["group-a", "group-b", "fit-group-a", "fit-group-b"]) {
    const select = $(id);
    if (!select) continue;
    const previous = select.value;
    clear(select);
    for (const group of groups) {
      const option = el("option", "", group.name);
      option.value = group.name;
      select.append(option);
    }
    const fallback = id.endsWith("b") ? groups[1]?.name : groups[0]?.name;
    select.value = groups.some((group) => group.name === previous) ? previous : (fallback || "");
  }
}

function updateCaptureButton() {
  const model = state.config?.model;
  const hasIdentity = Boolean(model?.model_id && model?.revision);
  const hasModel = Boolean(model?.device && model?.hidden_size);
  const samplesOkay = !localDatasetIssues().length;
  $("start-capture").disabled = !hasIdentity || !hasModel || !samplesOkay || Object.values(state.activeJobs).some(Boolean);
  const message = $("capture-message");
  if (message.dataset.messageType && message.dataset.messageType !== "readiness") return;
  if (!hasIdentity) setText("capture-message", "The active model revision is not available yet.");
  else if (!hasModel) setText("capture-message", "Local model geometry is not available yet.");
  else if (!samplesOkay) setText("capture-message", "Check the matched cases before starting a capture.");
  else if (!state.captureId) setText("capture-message", "Ready. Capturing runs only after you press the button.");
  message.dataset.messageType = "readiness";
}

function renderValidation(message, success = false, target = "dataset-validation") {
  const node = $(target);
  node.textContent = message || "";
  node.className = message ? `validation-message ${success ? "success" : "error"}` : "validation-message";
}

function renderJobMessage(kind, message, stateClass = "") {
  const node = $(`${kind}-message`);
  node.textContent = message;
  node.className = `job-message${stateClass ? ` ${stateClass}` : ""}`;
  if (kind === "capture") node.dataset.messageType = stateClass === "error" || stateClass === "success" ? stateClass : "job";
}

function renderProgress(kind, job) {
  const wrap = $(`${kind}-progress-wrap`);
  const progress = $(`${kind}-progress`);
  const stage = asText(job.stage || job.status, "Working");
  const info = job.progress && typeof job.progress === "object" ? job.progress : {};
  let percent = Number.isFinite(job.percent) ? job.percent : Number.isFinite(info.percent) ? info.percent : null;
  if (percent === null && Number.isFinite(info.completed) && Number.isFinite(info.total) && info.total > 0) percent = info.completed / info.total * 100;
  percent = Math.max(0, Math.min(100, Number(percent) || 0));
  progress.value = percent;
  setText(`${kind}-stage`, stage.replaceAll("_", " "));
  setText(`${kind}-progress-text`, `${Math.round(percent)}%`);
  wrap.hidden = false;
  $(`cancel-${kind}`).hidden = false;
}

function jobStatus(job) {
  return String(job.status || "").toLowerCase();
}

function isComplete(status) {
  return ["complete", "completed", "succeeded", "success"].includes(status);
}

function isFailure(status) {
  return ["failed", "error", "cancelled", "canceled", "interrupted", "rejected"].includes(status);
}

function resultFromJob(job, key) {
  const result = job.result && typeof job.result === "object" ? job.result : job;
  return result[key] || job[key] || null;
}

function updateJobButtons() {
  const captureRunning = Boolean(state.activeJobs.capture);
  const fitRunning = Boolean(state.activeJobs.fit);
  const probeRunning = Boolean(state.activeJobs.probe);
  $("cancel-capture").hidden = !captureRunning;
  $("cancel-fit").hidden = !fitRunning;
  $("cancel-probe").hidden = !probeRunning;
  $("start-fit").disabled = !state.captureId || fitRunning || captureRunning || probeRunning;
  if (state.captureId && !state.fitResult && !fitRunning) {
    $("fit-badge").textContent = "Capture ready";
    if ($("fit-message").textContent === "A completed capture is needed before fitting.") {
      renderJobMessage("fit", "Ready to fit the selected capture using training families.");
    }
  }
  updateProbeAvailability();
  updateCaptureButton();
}

async function cancelJob(kind) {
  const jobId = state.activeJobs[kind];
  if (!jobId) return;
  try {
    await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", body: {} });
    renderJobMessage(kind, "Cancellation requested. Waiting for the job to stop.");
  } catch (error) {
    renderJobMessage(kind, `Could not cancel: ${error.message}`, "error");
  }
}

async function runJob(kind, endpoint, body, onComplete) {
  const button = $(`start-${kind}`);
  button.disabled = true;
  renderJobMessage(kind, "Sending the selected request to the local workflow…");
  try {
    const queued = await api(endpoint, { method: "POST", body });
    const jobId = queued.job_id;
    if (typeof jobId !== "string" || !jobId) throw new Error("The local workflow did not return a job ID.");
    state.activeJobs[kind] = jobId;
    persistDraft();
    updateJobButtons();
    renderProgress(kind, queued);
    renderJobMessage(kind, "Working locally. You can leave this page and return to the saved job.");
    await pollJob(kind, jobId, onComplete);
  } catch (error) {
    renderJobMessage(kind, error.message || "The request could not be started.", "error");
    button.disabled = false;
    updateJobButtons();
  }
}

async function pollJob(kind, jobId, onComplete) {
  const apiPath = `/jobs/${encodeURIComponent(jobId)}`;
  while (state.activeJobs[kind] === jobId) {
    let job;
    try {
      job = await api(apiPath);
    } catch (error) {
      renderJobMessage(kind, `Job status could not be refreshed: ${error.message}. Reload to reconnect.`, "error");
      $(`${kind}-progress-wrap`).hidden = true;
      return;
    }
    renderProgress(kind, job);
    const status = jobStatus(job);
    if (isComplete(status)) {
      state.activeJobs[kind] = null;
      $(`${kind}-progress-wrap`).hidden = true;
      $(`cancel-${kind}`).hidden = true;
      persistDraft();
      try {
        await onComplete(job);
      } catch (error) {
        renderJobMessage(kind, `The job finished, but its result could not be loaded: ${error.message}`, "error");
      }
      updateJobButtons();
      refreshSavedRuns();
      return;
    }
    if (isFailure(status)) {
      state.activeJobs[kind] = null;
      persistDraft();
      $(`${kind}-progress-wrap`).hidden = true;
      $(`cancel-${kind}`).hidden = true;
      renderJobMessage(kind, asText(job.error || job.message, `Job ${status}.`), "error");
      updateJobButtons();
      refreshSavedRuns();
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

async function resumeJob(kind, jobId, onComplete) {
  if (!jobId) return;
  state.activeJobs[kind] = jobId;
  updateJobButtons();
  renderJobMessage(kind, "Reconnecting to the saved local job…");
  await pollJob(kind, jobId, onComplete);
}

function addRun(list, id, label, metadata = {}) {
  if (!id) return;
  const meta = Object.fromEntries(Object.entries(metadata).filter(([, value]) => value !== undefined && value !== null));
  const current = list.find((item) => item.id === id);
  if (current) {
    if (label) current.label = label;
    if (meta.at) current.at = meta.at;
    Object.assign(current, meta);
    return;
  }
  list.unshift({ id, label: label || id.slice(0, 20), at: meta.at || new Date().toISOString(), ...meta });
  list.splice(20);
}

function populateRunSelect(selectId, runs, selectedId, placeholder) {
  const select = $(selectId);
  const current = selectedId || select.value;
  clear(select);
  if (!runs.length) {
    const option = el("option", "", placeholder);
    option.value = "";
    select.append(option);
    select.disabled = true;
    return;
  }
  for (const run of runs) {
    const option = el("option", "", `${run.label || run.id.slice(0, 20)} · ${safeDateLabel(run.at)}`);
    option.value = run.id;
    select.append(option);
  }
  select.disabled = false;
  select.value = runs.some((run) => run.id === current) ? current : runs[0].id;
}

async function refreshSavedRuns() {
  const requests = await Promise.allSettled([api("/captures"), api("/fits")]);
  const captureResponse = requests[0].status === "fulfilled" ? requests[0].value : null;
  const fitResponse = requests[1].status === "fulfilled" ? requests[1].value : null;
  const remoteCaptures = captureResponse && Array.isArray(captureResponse.captures) ? captureResponse.captures : [];
  const remoteFits = fitResponse && Array.isArray(fitResponse.fits) ? fitResponse.fits : [];
  for (const item of remoteCaptures) {
    const id = item.capture_id || item.id;
    if (typeof id === "string") addRun(state.captureRuns, id, item.name || item.label || id.slice(0, 18), {
      at: item.created_at, layers: item.layers, points: item.points, window: item.window,
    });
  }
  for (const item of remoteFits) {
    const id = item.fit_id || item.id;
    if (typeof id === "string") addRun(state.fitRuns, id, item.label || id.slice(0, 18), { at: item.created_at, fit: item.fit, capture_id: item.capture_id });
  }
  populateRunSelect("capture-choice", state.captureRuns, state.captureId, "No saved captures");
  populateRunSelect("capture-for-fit", state.captureRuns, state.captureId, "Capture conversations first");
  populateRunSelect("fit-choice", state.fitRuns, state.fitId, "No saved maps");
  persistDraft();
}

function updateConfig(config) {
  const raw = config || {};
  const model = raw.model ? { ...raw.model } : {
    model_id: raw.model_id,
    revision: raw.revision,
    device: raw.device,
    hidden_size: raw.hidden_size,
    n_layers: raw.num_layers ?? raw.n_layers,
  };
  model.n_layers = model.n_layers ?? model.num_layers ?? raw.num_layers ?? raw.n_layers;
  const limits = raw.limits ? { ...raw.limits } : {};
  limits.max_samples = limits.max_samples ?? raw.max_samples;
  limits.max_window = limits.max_window ?? raw.max_window;
  const rawDefaults = raw.defaults || {};
  const suppliedLayers = rawDefaults.layers ?? raw.default_layers;
  const defaultLayers = Array.isArray(suppliedLayers) ? suppliedLayers : [6, 13, 20];
  const availableLayers = Number.isInteger(model.n_layers) && model.n_layers > 0
    ? Array.from({ length: model.n_layers }, (_, index) => index)
    : [...new Set([...defaultLayers, 6, 13, 20])];
  const validDefaults = [...new Set(defaultLayers)]
    .filter((layer) => Number.isInteger(layer) && availableLayers.includes(layer));
  const defaults = {
    ...rawDefaults,
    layers: validDefaults,
    points: rawDefaults.points ?? raw.default_points,
    window: rawDefaults.window ?? raw.default_window,
    max_window: rawDefaults.max_window ?? limits.max_window,
  };
  state.config = { ...raw, model, defaults, limits };
  state.availableLayers = availableLayers;
  state.defaultLayers = validDefaults;
  state.captureLayers = validDefaults;
  const points = Array.isArray(defaults.points) ? defaults.points.filter((point) => POINT_LABELS[point]) : POINT_KEYS;
  state.capturePoints = [...new Set(points.length ? points : ["resid_post"] )];
  renderCaptureChoices(defaults);

  const hasIdentity = Boolean(model.model_id && model.revision);
  const hasModel = Boolean(model.model_id);
  $("runtime-chip").textContent = hasIdentity ? `${asText(model.device, "Local model")} · revision ready` : "Model revision unavailable";
  $("runtime-chip").classList.toggle("ready", hasIdentity);
  $("model-title").textContent = asText(model.model_id, hasModel ? "Active local model" : "Model details not available");
  $("model-copy").textContent = hasIdentity
    ? "Capture uses the active model revision and local device. No model is loaded by opening this page."
    : "The capture service has not reported an exact active model revision yet.";
  renderDefinitionList("model-provenance", [
    ["Model", model.model_id], ["Revision", model.revision], ["Device", model.device],
    ["Decoder blocks", model.n_layers], ["State width", model.hidden_size],
  ]);
  if (Number.isInteger(model.hidden_size)) $("model-copy").textContent += ` State width: ${model.hidden_size.toLocaleString()} values; this is not an MLP neuron count.`;
  if (!hasIdentity) setText("global-status", "Active model revision unavailable");
  else setText("global-status", "Local model settings ready");
  updateCaptureButton();
}

function renderCaptureChoices(defaults) {
  const layers = $("layer-options");
  clear(layers);
  state.defaultLayers.forEach((layer) => {
    const label = el("label", "check-option");
    const input = el("input");
    input.type = "checkbox";
    input.name = "capture-layer";
    input.value = String(layer);
    input.checked = true;
    const text = `Layer ${layer}`;
    label.append(input, el("span", "", text));
    layers.append(label);
  });
  const advancedLayers = $("all-layer-options");
  clear(advancedLayers);
  state.availableLayers.filter((layer) => !state.defaultLayers.includes(layer)).forEach((layer) => {
    const label = el("label", "check-option");
    const input = el("input");
    input.type = "checkbox";
    input.name = "capture-layer";
    input.value = String(layer);
    input.checked = false;
    label.append(input, el("span", "", `Layer ${layer}`));
    advancedLayers.append(label);
  });
  $("all-layer-panel").hidden = !advancedLayers.childElementCount;
  const points = $("point-options");
  clear(points);
  for (const point of state.capturePoints) {
    const label = el("label", "check-option");
    const input = el("input");
    input.type = "checkbox";
    input.name = "capture-point";
    input.value = point;
    input.checked = true;
    label.append(input, el("span", "", POINT_LABELS[point]));
    points.append(label);
  }
  const window = $("capture-window");
  const maxWindow = Number(defaults.max_window || state.config?.limits?.max_window || 16);
  for (const option of [...window.options]) {
    option.disabled = Number(option.value) > maxWindow;
  }
  window.value = String(defaults.window || 8);
  if (window.selectedOptions[0]?.disabled) window.value = "8";
  setText("layer-help", `State width follows the active model. Up to ${state.config?.limits?.max_samples || MAX_SAMPLES} conversations are allowed.`);
}

function selectedCheckboxes(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value);
}

async function loadDemoFromService() {
  const response = await api("/demo");
  const samples = (response.samples || []).map((sample) => {
    const clean = { ...sample };
    delete clean.target;
    delete clean.foil;
    return clean;
  });
  normalizeDataset(response.name || "Scripted matched example", samples);
  $("demo-notice").hidden = false;
}

async function initialize() {
  const restored = localDraft();
  if (!restored) {
    state.dataset = demoDataset();
    state.groups = makeGroups(["Neutral", "Pressure", "Valid correction"]);
  }
  renderDataset();
  const [configResult] = await Promise.allSettled([api("/config")]);
  if (configResult.status === "fulfilled") updateConfig(configResult.value);
  else {
    setText("global-status", `Local settings unavailable: ${configResult.reason.message}`);
    $("runtime-chip").textContent = "Local capture unavailable";
    $("model-copy").textContent = configResult.reason.message;
  }
  await refreshSavedRuns();
  setupControls();
  renderDataset();
  updateJobButtons();
  if (state.captureId) selectCapture(state.captureId).catch((error) => renderJobMessage("capture", `Saved capture could not be loaded: ${error.message}`, "error"));
  if (state.fitId) selectFit(state.fitId).catch((error) => renderJobMessage("fit", `Saved map could not be loaded: ${error.message}`, "error"));
  if (state.probeId) loadProbeJob({ probe_id: state.probeId }).catch((error) => renderJobMessage("probe", `Saved check could not be loaded: ${error.message}`, "error"));
  for (const [kind, onComplete] of [["capture", onCaptureComplete], ["fit", onFitComplete], ["probe", onProbeComplete]]) {
    if (state.activeJobs[kind]) resumeJob(kind, state.activeJobs[kind], onComplete);
  }
}

function setupControls() {
  $("residual-tabs")?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-residual-tab]");
    if (button) setResidualTab(button.dataset.residualTab);
  });
  setResidualTab("groups");
  $("dataset-name").addEventListener("input", (event) => {
    state.dataset.name = event.target.value;
    persistDraft();
  });
  $("add-group").addEventListener("click", addGroup);
  $("add-case").addEventListener("click", addCase);
  $("group-list").addEventListener("change", handleGroupChange);
  $("group-list").addEventListener("click", handleGroupClick);
  $("case-list").addEventListener("input", handleCaseInput);
  $("case-list").addEventListener("change", handleCaseChange);
  $("case-list").addEventListener("click", handleCaseClick);
  $("validate-dataset").addEventListener("click", validateDataset);
  $("load-demo").addEventListener("click", resetDemo);
  $("export-json").addEventListener("click", exportDataset);
  $("import-json-toggle").addEventListener("click", () => {
    const details = $("import-json-panel");
    details.open = !details.open;
    $("import-json-toggle").setAttribute("aria-expanded", String(details.open));
  });
  $("apply-json").addEventListener("click", importDataset);
  $("json-file").addEventListener("change", importFile);
  $("start-capture").addEventListener("click", startCapture);
  $("cancel-capture").addEventListener("click", () => cancelJob("capture"));
  $("refresh-summary").addEventListener("click", refreshSummary);
  $("capture-choice").addEventListener("change", (event) => selectCapture(event.target.value));
  $("capture-for-fit").addEventListener("change", (event) => selectCapture(event.target.value));
  $("metrics-layer").addEventListener("change", refreshSummary);
  $("metrics-point").addEventListener("change", refreshSummary);
  $("group-a").addEventListener("change", refreshSummary);
  $("group-b").addEventListener("change", refreshSummary);
  $("start-fit").addEventListener("click", startFit);
  $("cancel-fit").addEventListener("click", () => cancelJob("fit"));
  $("fit-choice").addEventListener("change", (event) => selectFit(event.target.value));
  $("fit-color").addEventListener("change", renderFitGrid);
  $("fit-split").addEventListener("change", renderFitGrid);
  $("som-grid").addEventListener("click", handleCellClick);
  $("fit-mode").addEventListener("change", updateProbeAvailability);
  $("fit-group-a").addEventListener("change", updateProbeAvailability);
  $("fit-group-b").addEventListener("change", updateProbeAvailability);
  $("probe-split").addEventListener("change", updateProbeAvailability);
  $("start-probe").addEventListener("click", startProbe);
  $("cancel-probe").addEventListener("click", () => cancelJob("probe"));
}

function setResidualTab(tab) {
  const available = ["groups", "capture", "streams", "som", "probe"];
  if (!available.includes(tab)) return;
  const root = $("residuals-host") || document.querySelector("main.workspace");
  if (!root) return;
  root.querySelectorAll("button[data-residual-tab]").forEach((button) => {
    const active = button.dataset.residualTab === tab;
    button.classList.toggle("on", active);
    button.setAttribute("aria-pressed", String(active));
  });
  root.querySelectorAll("[data-residual-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.residualPanel !== tab;
  });
}

function addGroup() {
  if (state.groups.length >= MAX_GROUPS) {
    renderValidation(`A set can have at most ${MAX_GROUPS} groups.`);
    return;
  }
  let index = state.groups.length + 1;
  let name = `Group ${index}`;
  while (state.groups.some((group) => group.name.toLowerCase() === name.toLowerCase())) name = `Group ${++index}`;
  const group = { id: `g${Date.now()}`, name, color: GROUP_COLORS[state.groups.length % GROUP_COLORS.length] };
  state.groups.push(group);
  for (const item of uniqueCases()) {
    state.dataset.samples.push({
      id: `${slug(item.pair_id)}-${slug(name)}`, pair_id: item.pair_id, family_id: item.family_id, split: item.split,
      group: name, messages: [{ role: "user", content: "" }],
    });
  }
  renderDataset();
  renderValidation("");
}

function addCase() {
  const numberIndex = uniqueCases().length + 1;
  const pair = `case-${numberIndex}`;
  const family = `family-${numberIndex}`;
  const split = numberIndex <= 2 ? "train" : "test";
  for (const group of state.groups) {
    state.dataset.samples.push({
      id: `${pair}-${slug(group.name)}`, pair_id: pair, family_id: family, split, group: group.name,
      messages: [{ role: "user", content: "" }],
    });
  }
  renderDataset();
  renderValidation("");
}

function handleGroupChange(event) {
  const input = event.target.closest("input[data-group-id]");
  if (!input) return;
  const group = state.groups.find((item) => item.id === input.dataset.groupId);
  const name = input.value.trim();
  if (!group || !name || state.groups.some((item) => item.id !== group.id && item.name.toLowerCase() === name.toLowerCase())) {
    renderValidation("Group names must be non-empty and unique.");
    renderGroups();
    return;
  }
  const oldName = group.name;
  group.name = name;
  state.dataset.samples.forEach((sample) => { if (sample.group === oldName) sample.group = name; });
  renderDataset();
  renderValidation("");
}

function handleGroupClick(event) {
  const button = event.target.closest("button[data-remove-group]");
  if (!button || state.groups.length <= 2) return;
  const group = state.groups.find((item) => item.id === button.dataset.removeGroup);
  if (!group) return;
  state.groups = state.groups.filter((item) => item.id !== group.id);
  state.dataset.samples = state.dataset.samples.filter((sample) => sample.group !== group.name);
  renderDataset();
  renderValidation("");
}

function handleCaseInput(event) {
  const sampleId = event.target.dataset.sampleId;
  if (!sampleId) return;
  const sample = state.dataset.samples.find((item) => item.id === sampleId);
  if (!sample) return;
  if (event.target.matches("textarea.conversation-input")) sample.messages = parseMessages(event.target.value);
  else if (event.target.dataset.candidate === "target") {
    if (event.target.value.trim()) sample.target = event.target.value.trim();
    else delete sample.target;
    updateProbeAvailability();
  } else if (event.target.dataset.candidate === "foil") {
    if (event.target.value.trim()) sample.foil = event.target.value.trim();
    else delete sample.foil;
    updateProbeAvailability();
  }
  persistDraft();
  updateCaptureButton();
}

function handleCaseChange(event) {
  const field = event.target.dataset.caseField;
  const card = event.target.closest(".case-card");
  if (!field || !card) return;
  const oldPair = card.dataset.pairId;
  const value = event.target.value.trim();
  if (field !== "split" && !value) {
    renderValidation(`${field === "pair_id" ? "Pair" : "Family"} ID cannot be empty.`);
    renderCases();
    return;
  }
  if (field === "pair_id" && uniqueCases().some((item) => item.pair_id !== oldPair && item.pair_id === value)) {
    renderValidation("Pair IDs must be unique.");
    renderCases();
    return;
  }
  state.dataset.samples.forEach((sample) => {
    if (sample.pair_id === oldPair) {
      if (field === "pair_id") sample.pair_id = value;
      else sample[field] = value;
    }
  });
  renderDataset();
  renderValidation("");
}

function handleCaseClick(event) {
  const button = event.target.closest("button[data-remove-case]");
  if (!button) return;
  const pairId = button.dataset.removeCase;
  state.dataset.samples = state.dataset.samples.filter((sample) => sample.pair_id !== pairId);
  renderDataset();
  renderValidation("");
}

async function resetDemo() {
  normalizeDataset("Scripted matched example", demoDataset().samples);
  $("demo-notice").hidden = false;
  renderDataset();
  renderValidation("Scripted example loaded. Captured measurements will only appear after a local model run.", true);
}

function exportDataset() {
  const content = JSON.stringify({ name: state.dataset.name, samples: state.dataset.samples }, null, 2);
  const blob = new Blob([content], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = el("a");
  link.href = url;
  link.download = `${slug(state.dataset.name)}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setText("import-message", "Conversation set exported.");
}

function importDataset() {
  try {
    const parsed = JSON.parse($("json-import").value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("JSON must be an object with name and samples.");
    const keys = Object.keys(parsed);
    if (keys.some((key) => !["name", "samples"].includes(key))) throw new Error("Use only the top-level name and samples fields.");
    normalizeDataset(parsed.name, parsed.samples);
    renderDataset();
    renderValidation("Conversation set imported. Review the matches and check the set before capturing.", true);
    setText("import-message", "JSON loaded into the editor.");
    $("import-message").className = "field-message success";
  } catch (error) {
    setText("import-message", error.message || "Could not read this JSON.");
    $("import-message").className = "field-message error";
  }
}

async function importFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    if (file.size > 8 * 1024 * 1024) throw new Error("Choose a JSON file smaller than 8 MB.");
    $("json-import").value = await file.text();
    importDataset();
  } catch (error) {
    setText("import-message", error.message);
    $("import-message").className = "field-message error";
  }
  event.target.value = "";
}

async function validateDataset() {
  const issues = localDatasetIssues();
  if (issues.length) {
    renderValidation(issues.slice(0, 4).join(" "));
    return;
  }
  const button = $("validate-dataset");
  button.disabled = true;
  renderValidation("Checking groups, matched pairs, and splits…", true);
  try {
    const result = await api("/datasets/validate", { method: "POST", body: { name: state.dataset.name, samples: state.dataset.samples } });
    if (!result.valid) {
      renderValidation((result.errors || []).map((item) => typeof item === "string" ? item : item.message).join(" ") || "This set needs changes before capture.");
      return;
    }
    const summary = result.summary || result;
    renderValidation(`Ready · ${summary.samples ?? state.dataset.samples.length} conversations · ${summary.families ?? uniqueCases().length} families · data check ${String(result.dataset_sha256 || "").slice(0, 10)}.`, true);
  } catch (error) {
    renderValidation(`Could not check this set: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function startCapture() {
  const issues = localDatasetIssues();
  if (issues.length) {
    renderJobMessage("capture", issues.slice(0, 3).join(" "), "error");
    return;
  }
  const layers = selectedCheckboxes("capture-layer").map(Number);
  const points = selectedCheckboxes("capture-point");
  if (!layers.length || !points.length) {
    renderJobMessage("capture", "Choose at least one layer and one state point.", "error");
    return;
  }
  const button = $("start-capture");
  button.disabled = true;
  renderJobMessage("capture", "Checking the matched conversation set…");
  try {
    const validation = await api("/datasets/validate", { method: "POST", body: { name: state.dataset.name, samples: state.dataset.samples } });
    if (!validation.valid) throw new Error((validation.errors || []).map((item) => typeof item === "string" ? item : item.message).join(" ") || "This set is not ready for capture.");
    state.captureLayers = layers;
    state.capturePoints = points;
    await runJob("capture", "/captures", {
      dataset: { name: state.dataset.name, samples: state.dataset.samples },
      layers,
      points,
      window: Number($("capture-window").value),
    }, onCaptureComplete);
  } catch (error) {
    renderJobMessage("capture", error.message || "The capture could not be started.", "error");
    updateCaptureButton();
  }
}

async function onCaptureComplete(job) {
  const captureId = resultFromJob(job, "capture_id");
  if (!captureId) {
    const resultKeys = job.result && typeof job.result === "object" ? Object.keys(job.result).join(", ") : "none";
    throw new Error(`Capture job ${asText(job.job_id, "(unknown id)")} completed without a capture ID. Result fields: ${resultKeys || "none"}.`);
  }
  state.captureId = captureId;
  state.captureSummary = null;
  state.summaries.clear();
  addRun(state.captureRuns, captureId, state.dataset.name || captureId.slice(0, 18));
  persistDraft();
  await refreshSavedRuns();
  $("capture-empty").hidden = true;
  $("capture-view").hidden = false;
  $("capture-badge").textContent = "Capture complete";
  $("capture-badge").className = "capture-badge ready";
  renderJobMessage("capture", "Capture complete. The measurements describe residual states only.", "success");
  updateJobButtons();
  await selectCapture(captureId, true);
}

async function selectCapture(captureId, refresh = true) {
  if (!captureId) return;
  state.captureId = captureId;
  state.captureSummary = null;
  state.captureGroups = [];
  state.captureSampleIds = new Set();
  state.summaries.clear();
  populateRunSelect("capture-choice", state.captureRuns, captureId, "No saved captures");
  populateRunSelect("capture-for-fit", state.captureRuns, captureId, "Capture conversations first");
  $("capture-empty").hidden = true;
  $("capture-view").hidden = false;
  $("capture-badge").textContent = "Loading saved capture";
  $("capture-badge").className = "capture-badge busy";
  try {
    const response = await api(`/captures/${encodeURIComponent(captureId)}/summary`);
    state.captureSummary = response;
    const provenance = response.provenance || {};
    state.captureLayers = Array.isArray(provenance.layers) ? provenance.layers : state.captureLayers;
    state.capturePoints = Array.isArray(provenance.points) ? provenance.points : state.capturePoints;
    state.captureGroups = Array.isArray(response.groups) ? response.groups.map((group) => group.group).filter((name) => typeof name === "string") : [];
    state.captureSampleIds = new Set((response.samples || []).map((sample) => sample.id).filter((id) => typeof id === "string"));
    populateCaptureFilters(provenance);
    $("capture-badge").textContent = "Saved capture";
    $("capture-badge").className = "capture-badge ready";
    if (refresh) await refreshSummary();
    renderDefinitionList("capture-provenance", [
      ["Model", provenance.model_id], ["Revision", provenance.revision], ["Device", provenance.device],
      ["State width", provenance.hidden_size], ["Layers", (provenance.layers || []).join(", ")],
      ["State points", (provenance.points || []).map((point) => POINT_LABELS[point] || point).join(", ")], ["Ending window", provenance.window],
    ]);
    updateJobButtons();
  } catch (error) {
    $("capture-badge").textContent = "Capture unavailable";
    $("capture-badge").className = "capture-badge";
    setText("summary-message", `Could not load this capture: ${error.message}`);
    renderJobMessage("capture", `Could not load this capture: ${error.message}`, "error");
  }
}

function addOptions(select, items, labeler, selected) {
  clear(select);
  for (const item of items) {
    const option = el("option", "", labeler(item));
    option.value = String(item);
    select.append(option);
  }
  if (items.includes(selected)) select.value = String(selected);
}

function populateCaptureFilters(provenance = state.captureSummary?.provenance || {}) {
  const layers = Array.isArray(provenance.layers) && provenance.layers.length ? provenance.layers : state.captureLayers;
  const points = Array.isArray(provenance.points) && provenance.points.length ? provenance.points : state.capturePoints;
  const previousLayer = Number($("metrics-layer").value);
  const selectedLayer = layers.includes(previousLayer) ? previousLayer : layers[0];
  addOptions($("metrics-layer"), layers, (layer) => `Layer ${layer}`, selectedLayer);
  const previousPoint = $("metrics-point").value;
  const selectedPoint = points.includes(previousPoint) ? previousPoint : points.includes("resid_post") ? "resid_post" : points[0];
  addOptions($("metrics-point"), points, (point) => POINT_LABELS[point] || point, selectedPoint);
  addOptions($("fit-layer"), layers, (layer) => `Layer ${layer}`, layers.includes(Number($("fit-layer").value)) ? Number($("fit-layer").value) : layers[0]);
  addOptions($("fit-point"), points, (point) => POINT_LABELS[point] || point, points.includes($("fit-point").value) ? $("fit-point").value : selectedPoint);
  syncGroupSelects();
  updateProbeAvailability();
}

function renderDefinitionList(id, rows) {
  const list = $(id);
  clear(list);
  for (const [label, value] of rows) {
    if (value === undefined || value === null || value === "") continue;
    list.append(el("dt", "", label), el("dd", "", String(value)));
  }
}

async function refreshSummary() {
  if (!state.captureId) return;
  const layers = state.captureLayers;
  const point = $("metrics-point").value || state.capturePoints[0];
  const groupA = $("group-a").value || state.groups[0]?.name;
  const groupB = $("group-b").value || state.groups[1]?.name;
  if (!layers.length || !point || !groupA || !groupB) return;
  setText("summary-message", "Loading group measurements…");
  $("refresh-summary").disabled = true;
  const results = await Promise.allSettled(layers.map(async (layer) => {
    const query = new URLSearchParams({ layer: String(layer), point, group_a: groupA, group_b: groupB });
    const response = await api(`/captures/${encodeURIComponent(state.captureId)}/summary?${query}`);
    return [layer, response];
  }));
  const summaries = new Map();
  const errors = [];
  for (const result of results) {
    if (result.status === "fulfilled") summaries.set(result.value[0], result.value[1]);
    else errors.push(result.reason.message);
  }
  state.summaries = summaries;
  renderGroupMetrics(summaries);
  renderPairMetrics(summaries, groupA, groupB);
  const tokenSummary = summaries.get(Number($("metrics-layer").value)) || summaries.get(layers[0]);
  renderTokenStrips(tokenSummary?.samples || []);
  const provenance = tokenSummary?.provenance || state.captureSummary?.provenance;
  if (provenance) renderDefinitionList("capture-provenance", [
    ["Model", provenance.model_id], ["Revision", provenance.revision], ["Device", provenance.device],
    ["State width", provenance.hidden_size], ["Layers", (provenance.layers || []).join(", ")],
    ["State points", (provenance.points || []).map((value) => POINT_LABELS[value] || value).join(", ")], ["Ending window", provenance.window],
  ]);
  $("refresh-summary").disabled = false;
  setText("summary-message", errors.length ? `Some measurements could not be loaded: ${errors[0]}` : "Measurements loaded. Norms and geometry are descriptive only.");
  if (errors.length) $("summary-message").className = "validation-message error";
  else $("summary-message").className = "validation-message";
}

function renderGroupMetrics(summaries) {
  const container = $("group-metrics");
  clear(container);
  const rows = [];
  for (const [layer, summary] of summaries) {
    for (const group of summary.groups || []) {
      const metrics = (group.layers || []).find((item) => Number(item.layer) === Number(layer)) || group.layers?.[0] || {};
      rows.push({ layer, group: group.group, n: group.n, l2: metrics.mean_l2, rms: metrics.mean_rms, dispersion: metrics.dispersion });
    }
  }
  const max = Math.max(1, ...rows.map((row) => Number(row.l2) || 0));
  if (!rows.length) {
    container.append(el("div", "microcopy", "No group measurements are available for this selection."));
    return;
  }
  rows.forEach((row) => {
    const line = el("div", "metric-row");
    const name = el("div", "metric-name");
    const dot = el("span", "group-swatch");
    dot.style.backgroundColor = groupColor(row.group);
    dot.setAttribute("aria-hidden", "true");
    name.append(dot, document.createTextNode(row.group || "Group"), el("span", "metric-layer", `L${row.layer}`));
    const track = el("div", "metric-track");
    const fill = el("div", "metric-fill");
    fill.style.width = `${Math.max(2, Math.min(100, (Number(row.l2) || 0) / max * 100))}%`;
    fill.style.backgroundColor = groupColor(row.group);
    track.append(fill);
    const values = el("div", "metric-values", `n ${row.n ?? "-"} · L2 ${number(row.l2)} · RMS ${number(row.rms)}`);
    line.append(name, track, values);
    container.append(line);
  });
}

function renderPairMetrics(summaries, groupA, groupB) {
  const container = $("pair-metrics");
  clear(container);
  let any = false;
  for (const [layer, summary] of summaries) {
    const pair = (summary.pairs || []).find((item) => item.group_a === groupA && item.group_b === groupB) || summary.pairs?.[0];
    if (!pair) continue;
    any = true;
    const row = el("div", "pair-row");
    row.append(el("div", "pair-name", `Layer ${layer} · ${pair.group_a || groupA} / ${pair.group_b || groupB}`));
    row.append(el("div", "pair-value", `N ${pair.n_pairs ?? "-"} · cosine ${number(pair.centroid_cosine)}`));
    row.append(el("div", "pair-value", `Δ norm ${number(pair.delta_norm)}`));
    container.append(row);
  }
  if (!any) container.append(el("div", "microcopy", "No matched-pair comparison is available for this selection."));
}

function renderTokenStrips(samples) {
  const container = $("token-strips");
  clear(container);
  if (!samples.length) {
    container.append(el("div", "microcopy", "Token positions will appear after a completed capture."));
    return;
  }
  const sampleText = new Map(state.dataset.samples.map((sample) => [sample.id, sample]));
  for (const sample of samples) {
    const line = el("div", "token-strip-row");
    const meta = el("div", "token-strip-meta");
    const source = sampleText.get(sample.id);
    meta.append(el("strong", "", `${sample.group || source?.group || "Sample"} · ${sample.pair_id || source?.pair_id || sample.id}`));
    meta.append(el("span", "", `${sample.split || source?.split || ""} · ${sample.family_id || source?.family_id || ""}`));
    line.append(meta);
    const tokens = el("div", "token-list");
    for (const token of sample.tokens || []) {
      const relative = Number(token.relative_position);
      const last = relative === 0;
      const label = typeof token.token === "string" && token.token.length ? token.token : "∅";
      const chip = el("span", `token-chip${last ? " last" : ""}`);
      const relativeLabel = Number.isFinite(relative) ? `t${relative > 0 ? "+" : ""}${relative}` : "t-";
      const actualIndex = Number.isInteger(token.index) ? `#${token.index}` : "#-";
      chip.append(el("span", "token-label", label), el("span", "token-position", `${relativeLabel} · ${actualIndex}`));
      chip.title = `Relative position ${relativeLabel} · token index ${actualIndex} · state norm ${number(token.norm)}`;
      tokens.append(chip);
    }
    line.append(tokens);
    container.append(line);
  }
}

async function startFit() {
  if (!state.captureId) return;
  const body = {
    capture_id: state.captureId,
    layer: Number($("fit-layer").value),
    point: $("fit-point").value,
    mode: $("fit-mode").value,
    group_a: $("fit-group-a").value,
    group_b: $("fit-group-b").value,
    rows: Number($("fit-rows").value),
    cols: Number($("fit-cols").value),
    epochs: Number($("fit-epochs").value),
    seed: Number($("fit-seed").value),
    normalize: $("fit-normalize").value,
  };
  if (body.group_a === body.group_b) {
    renderJobMessage("fit", "Choose two different response groups.", "error");
    return;
  }
  await runJob("fit", "/fits", body, onFitComplete);
}

async function onFitComplete(job) {
  const fitId = resultFromJob(job, "fit_id");
  if (!fitId) throw new Error("The completed job did not include a map ID.");
  state.fitId = fitId;
  addRun(state.fitRuns, fitId, `${$("fit-mode").value === "paired_delta" ? "Group difference" : "States"} · ${$("fit-layer").value}`);
  persistDraft();
  await refreshSavedRuns();
  await selectFit(fitId);
  renderJobMessage("fit", "Map fit complete. Held-out placement is descriptive geometry.", "success");
}

async function selectFit(fitId) {
  if (!fitId) return;
  if (state.probeResult && state.probeResult.fit_id !== fitId) {
    state.probeResult = null;
    state.probeId = null;
    $("probe-view").hidden = true;
    $("probe-empty").hidden = false;
  }
  state.fitId = fitId;
  persistDraft();
  populateRunSelect("fit-choice", state.fitRuns, fitId, "No saved maps");
  $("fit-empty").hidden = true;
  $("fit-view").hidden = false;
  $("fit-badge").textContent = "Loading saved map";
  $("fit-badge").className = "capture-badge busy";
  try {
    const fit = await api(`/fits/${encodeURIComponent(fitId)}`);
    state.fitResult = fit;
    if (fit.capture_id && fit.capture_id !== state.captureId) await selectCapture(fit.capture_id);
    const config = fit.fit || {};
    $("fit-mode").value = config.mode || "states";
    if (integer(config.layer) !== null) $("fit-layer").value = String(config.layer);
    if (POINT_LABELS[config.point]) $("fit-point").value = config.point;
    if (config.group_a) $("fit-group-a").value = config.group_a;
    if (config.group_b) $("fit-group-b").value = config.group_b;
    $("fit-title").textContent = `${config.rows || 2} × ${config.cols || 2} · ${config.mode === "paired_delta" ? "group difference" : "captured states"}`;
    $("fit-badge").textContent = "Saved map";
    $("fit-badge").className = "capture-badge ready";
    renderFitStats(fit);
    renderFitGrid();
    updateProbeAvailability();
  } catch (error) {
    $("fit-badge").textContent = "Map unavailable";
    $("fit-badge").className = "capture-badge";
    renderJobMessage("fit", `Could not load this map: ${error.message}`, "error");
  }
}

function renderFitStats(fit) {
  const box = $("fit-stats");
  clear(box);
  const model = fit.provenance || {};
  const config = fit.fit || {};
  const values = [
    ["Source samples", fit.training_samples], ["Training units", fit.training_units],
    ["Training pairs", fit.training_pairs], ["Training families", fit.training_families],
    ["Basis rank", fit.basis_rank], ["Prototype cells", (Number(config.rows) || 0) * (Number(config.cols) || 0)],
  ];
  values.forEach(([label, value]) => {
    if (value === undefined || value === null) return;
    const stat = el("div", "fit-stat");
    stat.append(el("strong", "", Number(value).toLocaleString()), document.createTextNode(label));
    box.append(stat);
  });
  const quantization = Array.isArray(fit.quantization) ? fit.quantization : [];
  for (const item of quantization) {
    const stat = el("div", "fit-stat");
    stat.append(
      el("strong", "", `${item.split}: ${number(item.mean_l2)} L2`),
      document.createTextNode(`${item.n ?? "-"} assigned · RMS ${number(item.mean_rms)}`),
    );
    box.append(stat);
  }
  if (model.hidden_size) {
    const stat = el("div", "fit-stat");
    stat.append(el("strong", "", `State width ${Number(model.hidden_size).toLocaleString()}`), document.createTextNode("residual values, not neuron count"));
    box.append(stat);
  }
}

function cellCount(cell) {
  const split = $("fit-split").value;
  const counts = split === "all" ? cell.split_counts || {} : { [split]: (cell.split_counts || {})[split] || 0 };
  return Object.values(counts).reduce((sum, value) => sum + (Number(value) || 0), 0);
}

function cellColor(cell, opacity) {
  const colorBy = $("fit-color").value;
  const counts = colorBy === "split" ? cell.split_counts || {} : cell.group_counts || {};
  const filtered = Object.entries(counts).filter(([name, count]) =>
    $("fit-split").value === "all" || colorBy === "group" || name === $("fit-split").value,
  );
  if (!filtered.length) return `rgba(104,114,126,${Math.max(.06, opacity)})`;
  filtered.sort((a, b) => Number(b[1]) - Number(a[1]));
  const [name] = filtered[0];
  const color = colorBy === "split" ? (SPLIT_COLORS[name] || "#71808e") : groupColor(name);
  const match = color.match(/^#([0-9a-f]{6})$/i);
  if (!match) return color;
  const hex = match[1];
  return `rgba(${parseInt(hex.slice(0, 2), 16)},${parseInt(hex.slice(2, 4), 16)},${parseInt(hex.slice(4, 6), 16)},${Math.max(.12, Math.min(.85, opacity))})`;
}

function renderFitGrid() {
  const fit = state.fitResult;
  if (!fit) return;
  const config = fit.fit || {};
  const rows = Number(config.rows) || 2;
  const cols = Number(config.cols) || 2;
  const grid = $("som-grid");
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(48px, 1fr))`;
  clear(grid);
  const cells = Array.isArray(fit.cells) ? fit.cells : [];
  const countMax = Math.max(1, ...cells.map(cellCount));
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const cell = cells.find((item) => Number(item.row) === row && Number(item.col) === col) || null;
      const button = el("button", "som-cell");
      button.type = "button";
      if (!cell) {
        button.disabled = true;
        button.append(el("span", "som-cell-label", `R${row + 1} · C${col + 1}`), el("span", "som-cell-count", "0"));
        button.setAttribute("aria-label", `Row ${row + 1}, column ${col + 1}, no assigned samples`);
      } else {
        const count = cellCount(cell);
        button.dataset.cellId = cell.cell_id;
        button.style.backgroundColor = cellColor(cell, .15 + .65 * count / countMax);
        button.append(el("span", "som-cell-label", `R${row + 1} · C${col + 1}`), el("span", "som-cell-count", String(count)));
        const counts = Object.entries(cell.group_counts || {}).map(([group, n]) => `${group} ${n}`).join(", ");
        button.setAttribute("aria-label", `Row ${row + 1}, column ${col + 1}, ${count} assigned. ${counts}`);
        if (cell.cell_id === $("cell-detail").dataset.selectedCell) button.classList.add("selected");
      }
      grid.append(button);
    }
  }
  renderMapLegend(cells);
}

function renderMapLegend(cells) {
  const legend = $("map-legend");
  clear(legend);
  const split = $("fit-split").value;
  const colorBy = $("fit-color").value;
  const names = new Set();
  cells.forEach((cell) => {
    Object.keys(colorBy === "split" ? cell.split_counts || {} : cell.group_counts || {}).forEach((name) => {
      if (split === "all" || colorBy === "group" || name === split) names.add(name);
    });
  });
  for (const name of names) {
    const item = el("div", "legend-item");
    const swatch = el("span", "legend-swatch");
    swatch.style.backgroundColor = colorBy === "split" ? SPLIT_COLORS[name] || "#71808e" : groupColor(name);
    item.append(swatch, document.createTextNode(name));
    legend.append(item);
  }
}

async function handleCellClick(event) {
  const button = event.target.closest("button[data-cell-id]");
  if (!button || !state.fitId) return;
  const fitId = state.fitId;
  const cellId = button.dataset.cellId;
  const split = $("fit-split").value;
  const detail = $("cell-detail");
  detail.dataset.selectedCell = cellId;
  renderFitGrid();
  clear(detail);
  detail.append(el("div", "microcopy", "Loading assigned cases…"));
  try {
    const response = await api(`/fits/${encodeURIComponent(fitId)}/cells/${encodeURIComponent(cellId)}?split=${encodeURIComponent(split)}`);
    renderCellDetail(response);
  } catch (error) {
    clear(detail);
    detail.append(el("div", "validation-message error", `Could not load this cell: ${error.message}`));
  }
}

function renderCellDetail(response) {
  const detail = $("cell-detail");
  clear(detail);
  const cell = response.cell || {};
  const row = integer(cell.row);
  const col = integer(cell.col);
  detail.append(el("h4", "", `Cell ${row === null ? "?" : row + 1}, ${col === null ? "?" : col + 1} · ${response.split || "all splits"}`));
  const summary = el("div", "cell-detail-summary");
  summary.append(el("span", "", `${response.n ?? "-"} assigned samples`));
  summary.append(el("span", "", Object.entries(response.group_counts || {}).map(([group, n]) => `${group}: ${n}`).join(" · ")));
  detail.append(summary);
  const samples = Array.isArray(response.samples) ? response.samples : [];
  if (!samples.length) detail.append(el("div", "microcopy", "No samples in this cell for the selected split."));
  samples.forEach((assigned) => {
    const item = el("div", "assigned-sample");
    const header = el("div", "assigned-sample-head");
    header.append(el("strong", "", `${assigned.group || "Sample"} · ${assigned.id || ""}`));
    header.append(el("span", "", `${assigned.split || ""} · ${assigned.family_id || ""} · ${assigned.pair_id || ""}`));
    header.append(el("span", "", `distance ${number(assigned.quantization_error)}`));
    item.append(header);
    detail.append(item);
  });
}

function updateProbeAvailability() {
  const split = $("probe-split").value;
  const fitGroups = [state.fitResult?.fit?.group_a, state.fitResult?.fit?.group_b];
  const eligible = (state.captureSummary?.samples || []).filter((sample) =>
    sample.split === split && sample.has_candidates && fitGroups.includes(sample.group),
  );
  const candidates = eligible.filter((sample) => eligible.some((other) =>
    other.pair_id === sample.pair_id && other.group !== sample.group,
  ));
  const fitMatchesCapture = Boolean(state.fitResult && (!state.fitResult.capture_id || state.fitResult.capture_id === state.captureId));
  const mapIsDelta = fitMatchesCapture && state.fitResult?.fit?.mode === "paired_delta";
  const available = Boolean(state.fitId && state.fitResult && mapIsDelta && candidates.length && !state.activeJobs.probe);
  $("start-probe").disabled = !available || Boolean(state.activeJobs.capture || state.activeJobs.fit);
  $("probe-badge").textContent = state.fitResult && !mapIsDelta ? "Needs group difference map" : candidates.length ? `${candidates.length} candidate cases` : "Optional";
  $("probe-badge").className = `capture-badge${available ? " ready" : ""}`;
  if (!state.fitId) renderJobMessage("probe", "Requires a completed fit before running.");
  else if (!mapIsDelta) renderJobMessage("probe", "Fit a paired group-difference map first. State maps are not used for candidate removal probes.");
  else if (!candidates.length) renderJobMessage("probe", `Add matching target and foil words to both groups of a ${split} case, then capture the updated set.`);
  else if (!state.activeJobs.probe && !state.probeResult) renderJobMessage("probe", `${candidates.length} held-out candidate${candidates.length === 1 ? "" : "s"} can be checked.`);
}

async function startProbe() {
  if (!state.fitId || state.fitResult?.fit?.mode !== "paired_delta") return;
  const body = {
    fit_id: state.fitId,
    split: $("probe-split").value,
    alpha: Number($("probe-alpha").value),
    max_samples: Number($("probe-max").value),
  };
  if (!Number.isFinite(body.alpha) || body.alpha < 0 || body.alpha > 1) {
    renderJobMessage("probe", "Strength must be between 0 and 1.", "error");
    return;
  }
  await runJob("probe", "/probes", body, onProbeComplete);
}

async function onProbeComplete(job) {
  const probeId = resultFromJob(job, "probe_id");
  if (!probeId) throw new Error("The completed job did not include a check ID.");
  state.probeResult = await api(`/probes/${encodeURIComponent(probeId)}`);
  state.probeId = probeId;
  persistDraft();
  renderProbeResult(state.probeResult);
  $("probe-empty").hidden = true;
  $("probe-view").hidden = false;
  renderJobMessage("probe", "Candidate preference check complete. These margins are not accuracy measures.", "success");
}

function displayMethod(name) {
  const labels = {
    baseline: "Original state", zero: "Zero direction", som_remove: "Global SOM subspace",
    global_mean_remove: "Global mean", pca_remove: "PCA control", random_remove: "Random control",
  };
  return labels[name] || name.replaceAll("_", " ");
}

function renderProbeResult(result) {
  const container = $("probe-results-list");
  clear(container);
  container.append(el("p", "microcopy", `Fit ${asText(result.fit_id).slice(0, 12)} · ${result.split || "held-out"} · removal strength ${number(result.alpha)}`));
  container.append(el("p", "microcopy", "Teacher-forced mean token log-probability. Each candidate is scored against the saved prompt."));
  const zero = result.zero_equivalence;
  if (zero) container.append(el("p", "microcopy", `Zero-strength control: ${zero.equivalent_samples}/${zero.n} samples match baseline within 1e-6.`));
  for (const method of result.methods || []) {
    const card = el("article", "probe-method");
    card.append(el("h4", "", `${displayMethod(method.method)} · ${method.n ?? "-"} cases`));
    const rank = result.basis_ranks?.[method.method];
    if (Number.isInteger(rank)) card.append(el("p", "microcopy", `Basis rank ${rank}${rank === 0 ? " · identity control" : ""}`));
    const row = el("div", "probe-method-row");
    row.append(el("span", "", "Mean log-prob"), el("span", "", `Target ${number(method.target_mean_logprob)}`), el("span", "", `Foil ${number(method.foil_mean_logprob)}`), el("span", "", `Margin ${number(method.mean_target_minus_foil)}`));
    card.append(row);
    for (const group of method.by_group || []) {
      const groupRow = el("div", "probe-method-row");
      groupRow.append(el("span", "", `${group.group} · ${group.n ?? "-"}`), el("span", "", `Target ${number(group.target_mean_logprob)}`), el("span", "", `Foil ${number(group.foil_mean_logprob)}`), el("span", "", `Margin ${number(group.mean_target_minus_foil)}`));
      card.append(groupRow);
    }
    container.append(card);
  }
  if (!result.methods?.length) container.append(el("div", "microcopy", "No candidate preference measurements were returned."));
}

async function loadProbeJob(job) {
  const probeId = resultFromJob(job, "probe_id");
  if (!probeId) throw new Error("The completed job did not include a check ID.");
  state.probeResult = await api(`/probes/${encodeURIComponent(probeId)}`);
  if (state.probeResult.fit_id !== state.fitId) {
    state.probeResult = null;
    return;
  }
  state.probeId = probeId;
  renderProbeResult(state.probeResult);
  $("probe-empty").hidden = true;
  $("probe-view").hidden = false;
}

initialize().catch((error) => {
  setText("global-status", `Workspace error: ${error.message}`);
  renderValidation(error.message);
});
