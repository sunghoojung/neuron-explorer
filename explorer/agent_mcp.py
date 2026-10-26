"""Stdio MCP server that gives the explorer's ChatGPT chat its tools: read any neuron data through the explorer's HTTP API,
run the live model (steered rollouts, activations) when it is loaded, and drive the map (select, highlight) in the browser that started the chat. Standard library only; started by
the Codex CLI MCP integration from server.py with EXPLORER_URL and EXPLORER_CHAT set.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
CHAT = os.environ.get("EXPLORER_CHAT", "")

LN = {"layer": {"type": "integer", "description": "zero-based layer index; bounds are in overview"}, "n": {"type": "integer", "description": "zero-based MLP neuron index; bounds are in overview"}}
TOOLS = [
    {"name": "overview", "description": "Model architecture and which layers have sweep runs (rollouts and rankings).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "neuron", "description": "Everything cached for one neuron: name, logit lens and steer lens (with KL) for "
     "both poles, activation range, sweep rank if any, and its 24 strongest direct-path links each way (writes_into, "
     "reads_from) with each linked neuron's steer lens, top words and name.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "rollouts", "description": "Sweep rollouts for a neuron: steered text per prompt and dose beside the "
     "unsteered baselines, distinctive words and stats. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {**LN, "max_chars": {"type": "integer", "default": 500}},
                     "required": ["layer", "n"]}},
    {"name": "cone", "description": "The neuron's connection cone: links chained forward (downstream) and backward "
     "(upstream), first hop all links, later hops each node's top k. Returns the neurons reached with hop, best z and "
     "steer lens. Multi-hop paths are weight-only guesses: leads to check, not circuits.",
     "inputSchema": {"type": "object", "properties": {**LN, "k": {"type": "integer", "default": 2},
                                                       "limit": {"type": "integer", "default": 120}},
                     "required": ["layer", "n"]}},
    {"name": "web_summary", "description": "The token summary of the neuron's web, the same one the person sees in the "
     "left panel: steer-lens tokens pooled over its cone, each connected neuron read on the pole the chain drives, ranked "
     "by lift over that layer's base rate. `toward` / `away`: what its downstream neurons push the model toward and away "
     "from when this neuron fires (indirect effect). `fed_by` / `held_by`: what its upstream writers say when they "
     "excite / inhibit it (its inputs). Each token has n (neurons carrying it) and a score.",
     "inputSchema": {"type": "object", "properties": {**LN, "first": {"type": "integer", "default": 5},
                                                       "k": {"type": "integer", "default": 2}},
                     "required": ["layer", "n"]}},
    {"name": "search_lens", "description": "Neurons whose logit lens or steer lens (top 12, either pole) contains a token "
     "matching the query (case-insensitive substring). Best matches first.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 40}},
                     "required": ["query"]}},
    {"name": "top_units", "description": "A layer's neurons ranked by sweep word-cluster mass against random-direction "
     "controls, with top words and names. Empty when the layer has no sweep run.",
     "inputSchema": {"type": "object", "properties": {"layer": LN["layer"], "limit": {"type": "integer", "default": 30}},
                     "required": ["layer"]}},
    {"name": "set_name", "description": "Name a neuron in the explorer (the name field in its panel, shown in lists and on "
     "the map), or clear its name with an empty string. Only when the person asks, or after they agree to a name you "
     "suggested.",
     "inputSchema": {"type": "object", "properties": {**LN, "name": {"type": "string"}}, "required": ["layer", "n", "name"]}},
    {"name": "names", "description": "Every neuron the person has named so far, as {\"L14/N3262\": name}.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "steer", "description": "Live: generate from a chat prompt three ways in one batch, pushed up, unsteered, "
     "and pushed down (greedy). Steer one neuron (layer, n) or several at once (neurons: [{layer, n, dose?, mode?}]; all "
     "are applied together, each reversed in the down row; a negative per-neuron dose pushes that one the other way). "
     "dose in units of the layer's median residual norm for mode 'add' (0.8 is the steer-lens push; 2-3 is strong), or "
     "a multiplier for mode 'mul'. Needs the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "prompt": {"type": "string"},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer", "n"]}},
                                                       "dose": {"type": "number", "default": 0.8},
                                                       "mode": {"type": "string", "enum": ["add", "mul"], "default": "add"},
                                                       "max_new": {"type": "integer", "default": 150}},
                     "required": ["prompt"]}},
    {"name": "activations", "description": "Live: the neuron's raw activation on every token of a text (chat=true wraps "
     "it as a user turn). Compare against act_range from `neuron`. Optional `neurons` [{layer, n, dose?, mode?}] are "
     "pushed during the same pass (steering units), to test causally whether they drive this neuron: compare with the "
     "same call without them. {layer, all: true, dose: 0} zeroes that layer's whole MLP. Optional `heads` [{layer, "
     "head, dose}] scale attention heads (dose 0 = ablate; head 'all' = every head; head counts and layer types are in `overview`). Only earlier layers can move it. Needs "
     "the live model; errors if it is off.",
     "inputSchema": {"type": "object", "properties": {**LN, "text": {"type": "string"},
                                                       "chat": {"type": "boolean", "default": False},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["layer", "n", "text"]}},
    {"name": "compare", "description": "Live: read many neurons across several texts in one call (one forward pass per "
     "text). Per neuron and text: its strongest positive and negative activation with the token, and peak_score (the "
     "activation over the neuron's usual extreme; 1 = as strong as its usual max). For screening candidates against "
     "target texts and controls; `activations` gives one neuron's full per-token trace and takes pushes for causal "
     "tests. Up to 300 neurons and 24 texts.",
     "inputSchema": {"type": "object", "properties": {
         "neurons": {"type": "array", "items": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
         "texts": {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {
             "label": {"type": "string"}, "text": {"type": "string"}}, "required": ["text"]}]}},
         "chat": {"type": "boolean", "default": False}},
                     "required": ["neurons", "texts"]}},
    {"name": "logprobs", "description": "Live: the next token's log-probabilities after a text (chat=true by default: "
     "the text is a user turn and the next token opens the reply; `prefix` continues from the start of a reply or any "
     "text). Returns the top k tokens and the logprob of each `targets` string's first token. With `neurons` and/or "
     "`heads` (same as `activations`: pushes, whole-layer zeroing, head scaling) it runs with and without them and "
     "returns both, the KL from baseline to intervened, and the tokens that gained and lost the most probability "
     "across the whole vocabulary: a direct read of what an intervention does to the output.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "chat": {"type": "boolean", "default": True},
                                                       "prefix": {"type": "string"}, "k": {"type": "integer", "default": 15},
                                                       "targets": {"type": "array", "items": {"type": "string"}},
                                                       "neurons": {"type": "array", "items": {"type": "object", "properties": {
                                                           **LN, "all": {"type": "boolean"}, "dose": {"type": "number"},
                                                           "mode": {"type": "string", "enum": ["add", "mul"]}},
                                                           "required": ["layer"]}},
                                                       "heads": {"type": "array", "items": {"type": "object", "properties": {
                                                           "layer": {"type": "integer"},
                                                           "head": {"type": ["integer", "string"]},
                                                           "dose": {"type": "number", "default": 0}},
                                                           "required": ["layer", "head"]}}},
                     "required": ["text"]}},
    {"name": "top_firing", "description": "Live: run text through the model and list the neurons firing hardest at one "
     "token (the first token containing `token`; omit it for the token with the strongest firing). Each neuron has its "
     "pole, score (activation over its usual extreme; 1 = as strong as its usual max), raw activation and steer lens. "
     "`layers` [lo, hi] restricts the layers. Doesn't change the person's screen; use show_text to show them.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"},
                                                       "limit": {"type": "integer", "default": 30},
                                                       "layers": {"type": "array", "items": {"type": "integer"}},
                                                       "chat": {"type": "boolean", "default": False}},
                     "required": ["text"]}},
    {"name": "camera", "description": "Move the person's 3D view. mode 'home': the whole model. 'layer': look at one "
     "layer's neuron sheet face-on (give layer). 'neuron': fly close to one neuron (give layer and n; closeness 0.1 = very "
     "close, 0.5 = wide).",
     "inputSchema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["home", "layer", "neuron"]},
                                                       **LN, "closeness": {"type": "number", "default": 0.22}},
                     "required": ["mode"]}},
    {"name": "annotate_neurons", "description": "Pin short text labels to neurons on the 3D map (they follow the camera). "
     "Replaces the previous labels; an empty list clears them. Keep each label a few words.",
     "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "text": {"type": "string"}}, "required": ["layer", "n", "text"]}}},
                     "required": ["items"]}},
    {"name": "show_text", "description": "Put text in the explorer's paste box and run it through the live model, so the "
     "person sees its tokens; pin the first token containing `token` to show the neurons firing most there and the "
     "direct links among them on the map. Needs the live model.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}, "token": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "show_search", "description": "Run a lens search in the explorer's left panel, so the person sees the "
     "matching neurons listed and can click them.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "open_tab", "description": "Switch the open neuron's panel to a tab: examples (sweep rollouts), chat (the "
     "person's own steered chat) or details (lenses and connections).",
     "inputSchema": {"type": "object", "properties": {"tab": {"type": "string", "enum": ["examples", "chat", "details"]}},
                     "required": ["tab"]}},
    {"name": "select_neuron", "description": "Open a neuron in the explorer: its panel, and its connection web on the map.",
     "inputSchema": {"type": "object", "properties": LN, "required": ["layer", "n"]}},
    {"name": "highlight_neurons", "description": "Mark a set of neurons on the 3D map (replaces the previous marks; an "
     "empty list clears them). sign '-' draws a neuron in the negative-pole colour.",
     "inputSchema": {"type": "object", "properties": {"neurons": {"type": "array", "items": {
         "type": "object", "properties": {**LN, "sign": {"type": "string", "enum": ["+", "-"]}},
         "required": ["layer", "n"]}}}, "required": ["neurons"]}},
    {"name": "som_status", "description": "Inspect the configured SOM workflow, paired-prompt sources, validated decoder-residual artifacts, read-only native SOM reports, and persisted jobs. "
     "Shows existing private experiment records separately from the bundled example source when configured. Returns safe summaries and opaque IDs, not file contents or local paths.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "som_read_artifact", "description": "Read the validated metadata and provenance for one existing SOM artifact using its opaque artifact_id from som_status. "
     "Does not return prototypes, activation data, or arbitrary files.",
     "inputSchema": {"type": "object", "properties": {"artifact_id": {"type": "string", "minLength": 1, "maxLength": 128}},
                     "required": ["artifact_id"], "additionalProperties": False}},
    {"name": "som_read_native_report", "description": "Read a compact status, provenance, behavior, and map summary for a configured native SOM run using its opaque native_id from som_status. Includes integrity and capture-provenance state. The default keeps brief records for each map; add map_name to get one map's detailed aggregate geometry/association summary. map_name must exactly match one listed map. The report URL opens the sandboxed atlas. Native directions are read-only metadata and are not applied to the decoder-residual hook or live model. No raw direction arrays, per-example records, local paths, or arbitrary files are returned.",
     "inputSchema": {"type": "object", "properties": {
         "native_id": {"type": "string", "minLength": 1, "maxLength": 128},
         "map_name": {"type": "string", "minLength": 1, "maxLength": 120}},
                     "required": ["native_id"], "additionalProperties": False}},
    {"name": "som_run_pilot", "description": "Only invoke when the person explicitly requests a run. Queue one bounded local SOM pilot over a configured paired-prompt source from som_status. Collection captures paired residual activations and rejects sources larger than 24 question IDs (96 condition records); fitting trains only on train-split false_pressure minus neutral deltas; evaluation checks the requested held-out split against controls and is capped separately by max_questions. Uses the active model's exact revision and device, usually MPS on Apple Silicon. This exploratory pilot is not research proof; a run from the bundled example source is a plumbing check only. The server validates the source and layer, writes only under its managed run root, and reports progress through som_job_status. No shell command or filesystem path is accepted.",
     "inputSchema": {"type": "object", "properties": {
         "source_id": {"type": "string", "minLength": 1, "maxLength": 128}, "layer": {"type": "integer", "minimum": 0},
         "rows": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
         "cols": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
         "epochs": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
         "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 0},
         "split": {"type": "string", "enum": ["validation", "test"], "default": "test"},
         "max_questions": {"type": "integer", "minimum": 1, "maximum": 12, "default": 12}},
                     "required": ["source_id", "layer"], "additionalProperties": False}},
    {"name": "som_job_status", "description": "Read persisted progress and safe result metadata for a SOM job ID returned by som_run_pilot. A job is complete only after its artifact, schema, model/layer/hook identity, and output completeness checks pass; queued or interrupted work is not a fitted result. wait_seconds defaults to 10 and is capped at 20 for local active jobs. Historical external records marked unverified return immediately.",
     "inputSchema": {"type": "object", "properties": {
         "job_id": {"type": "string", "minLength": 1, "maxLength": 128},
         "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 20, "default": 10}},
                     "required": ["job_id"], "additionalProperties": False}},
]


def _read_json_response(response):
    with response as r:
        return json.load(r)


def _request_json(request, timeout):
    try:
        return _read_json_response(urllib.request.urlopen(request, timeout=timeout))
    except HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8", errors="replace"))
        except (ValueError, OSError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        message = body.get("error") or body.get("detail") or f"Explorer request failed with HTTP {exc.code}."
        raise RuntimeError(str(message)) from None


def get(path, **q):
    query = urllib.parse.urlencode(q)
    return _request_json(f"{URL}{path}?{query}", timeout=60)


def post(path, body, timeout=60):
    req = urllib.request.Request(f"{URL}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return _request_json(req, timeout=timeout)


def _compact_native_report(report, selected_map_name=None):
    if not isinstance(report, dict):
        raise TypeError("Explorer returned an invalid native SOM report summary")
    maps = report.get("maps")
    if not isinstance(maps, list):
        raise TypeError("Explorer returned no native SOM map index")
    brief_fields = ("map_name", "kind", "view", "layer", "side", "seed", "prototype_count")
    compact = dict(report)
    compact["maps"] = [
        {key: item[key] for key in brief_fields if key in item}
        for item in maps
        if isinstance(item, dict)
    ]
    if selected_map_name is not None:
        if not isinstance(selected_map_name, str) or not 1 <= len(selected_map_name) <= 120:
            raise ValueError("map_name must be an exact listed map name")
        selected = next(
            (item for item in maps if isinstance(item, dict) and item.get("map_name") == selected_map_name),
            None,
        )
        if selected is None:
            available = ", ".join(item["map_name"] for item in compact["maps"] if "map_name" in item)
            raise ValueError(f"Unknown map_name. Choose one listed in som_status: {available}")
        compact["selected_map_detail"] = selected
    return compact


def call(name, a):
    if name == "overview":
        return get("/api/agent/overview")
    if name == "neuron":
        return get("/api/agent/neuron", layer=a["layer"], n=a["n"])
    if name == "rollouts":
        u = get("/api/unit", layer=a["layer"], n=a["n"])
        cut = int(a.get("max_chars", 500))
        for key in ("rollouts", "baselines"):
            for r in u.get(key, []):
                r["text"] = r["text"][:cut]
        return u
    if name == "cone":
        return get("/api/agent/cone", layer=a["layer"], n=a["n"], k=a.get("k", 2), limit=a.get("limit", 120))
    if name == "web_summary":
        r = get("/api/web/summary", layer=a["layer"], n=a["n"], first=a.get("first", 5), k=a.get("k", 2), top=20)
        slim = lambda lst: [f"{t['tok']} ({t['n']})" for t in lst]
        return {"lens": r["src"], "links": r["links"],
                "downstream_neurons": r["down"]["neurons"], "toward": slim(r["down"]["toward"]), "away": slim(r["down"]["away"]),
                "upstream_neurons": r["up"]["neurons"], "fed_by": slim(r["up"]["fed_by"]), "held_by": slim(r["up"]["held_by"])}
    if name == "search_lens":
        r = get("/api/search", q=a["query"], limit=a.get("limit", 40))
        return {"tokens_matched": r.get("tokens"), "total": r.get("total"), "results": r.get("results", [])}
    if name == "top_units":
        us = sorted((u for u in get("/api/units", layer=a["layer"]) if u["kind"] == "neuron"), key=lambda u: -u["z"])
        return us[: int(a.get("limit", 30))]
    if name == "set_name":
        post("/api/name", {"layer": a["layer"], "n": a["n"], "name": a["name"]})
        post("/api/agent/ui", {"chat": CHAT, "action": "rename", "layer": a["layer"], "n": a["n"], "name": a["name"]})
        return {"ok": True, "neuron": f"L{a['layer']} N{a['n']}", "name": a["name"].strip() or None}
    if name == "names":
        return get("/api/agent/names")
    if name == "steer":
        return post("/api/agent/steer", a, timeout=900)  # the first call may wait for the model to load
    if name == "top_firing":
        return post("/api/agent/fires", a, timeout=900)
    if name == "logprobs":
        return post("/api/agent/logprobs", a, timeout=900)
    if name == "compare":
        return post("/api/agent/compare", a, timeout=900)
    if name == "activations":
        return post("/api/agent/activations", a, timeout=900)
    ui = {"camera": "camera", "annotate_neurons": "annotate", "show_text": "text", "show_search": "search", "open_tab": "tab"}
    if name in ui:
        return post("/api/agent/ui", {"chat": CHAT, "action": ui[name], **a})
    if name == "select_neuron":
        return post("/api/agent/ui", {"chat": CHAT, "action": "select", "layer": a["layer"], "n": a["n"]})
    if name == "highlight_neurons":
        return post("/api/agent/ui", {"chat": CHAT, "action": "highlight", "neurons": a["neurons"]})
    if name == "som_status":
        return get("/api/agent/som/status")
    if name == "som_read_artifact":
        return get("/api/agent/som/artifact", artifact_id=a["artifact_id"])
    if name == "som_read_native_report":
        report = get("/api/agent/som/native-report", native_id=a["native_id"])
        return _compact_native_report(report, a.get("map_name"))
    if name == "som_run_pilot":
        return post("/api/agent/som/pilot", a)
    if name == "som_job_status":
        return som_job_status(a.get("job_id"), a.get("wait_seconds", 10))
    raise ValueError(f"unknown tool {name}")


_ACTIVE_LOCAL_SOM_STATUSES = {"queued", "collecting", "fitting", "evaluating", "validating"}


def som_job_status(job_id, wait_seconds=10):
    """Poll active local jobs briefly; historical external records are read once."""
    if not isinstance(job_id, str) or not job_id or len(job_id) > 128:
        raise ValueError("job_id must be a non-empty string of at most 128 characters")
    if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, int) or not 0 <= wait_seconds <= 20:
        raise ValueError("wait_seconds must be an integer from 0 through 20")

    encoded_job_id = urllib.parse.quote(job_id, safe="")
    endpoint = f"/api/agent/som/jobs/{encoded_job_id}"
    deadline = time.monotonic() + wait_seconds
    result = get(endpoint)
    while (
        isinstance(result, dict)
        and result.get("kind") != "external_record"
        and result.get("status") in _ACTIVE_LOCAL_SOM_STATUSES
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
        result = get(endpoint)
    return result


def reply(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    msg.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        m = json.loads(line)
        method, id_ = m.get("method"), m.get("id")
        if id_ is None:
            continue  # notifications
        if method == "initialize":
            reply(id_, {"protocolVersion": m["params"].get("protocolVersion", "2025-06-18"),
                        "capabilities": {"tools": {}}, "serverInfo": {"name": "explorer", "version": "0.1"}})
        elif method == "tools/list":
            reply(id_, {"tools": TOOLS})
        elif method == "tools/call":
            p = m["params"]
            try:
                out = call(p["name"], p.get("arguments") or {})
                reply(id_, {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]})
            except Exception as e:  # noqa: BLE001 - isolate each MCP tool failure from the protocol loop
                reply(id_, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        elif method == "ping":
            reply(id_, {})
        else:
            reply(id_, error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main()
