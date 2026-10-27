"""Neuron explorer server: sweep data, names, embeddings, and the live model.

    uv run python explorer/server.py            # http://localhost:8791
"""
import asyncio
import base64
import builtins
import glob
import hashlib
import html
import json
import os
import re
import sys
import threading
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from math import ceil, sqrt
from pathlib import Path
from urllib.parse import quote

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sweep"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
os.environ.setdefault("EXPLORER_MODEL", DEFAULT_MODEL)
from model_adapter import (
    DEFAULT_REVISION,
    model_cache_namespace,
    resolve_model_revision,
)

if os.environ.get("EXPLORER_MODEL") == DEFAULT_MODEL:
    os.environ.setdefault("EXPLORER_REVISION", DEFAULT_REVISION)
from atlas_api import create_atlas_router
from chatgpt_backend import ChatGPTBackend, ChatHistoryError
from rank import compute_units
from residual_api import create_residual_router
from runtime import NO_MODEL, Runtime
from som_service import (
    MAX_NATIVE_MAP_PNG_BYTES,
    MAX_NATIVE_MAPS,
    MAX_NATIVE_REPORT_BYTES,
    SOMService,
    SOMServiceError,
)

PORT = int(os.environ.get("PORT", "8791"))
RUNS_ROOT, STATS_ROOT = os.path.join(ROOT, "runs"), os.path.join(ROOT, "stats")

@asynccontextmanager
async def lifespan(_app):
    def preload_model():
        if NO_MODEL:
            rt.status = "no model"
            return
        rt.ensure_loaded()

    threading.Thread(target=preload_model, daemon=True).start()
    try:
        yield
    finally:
        await asyncio.to_thread(residual_service.close)
        await asyncio.to_thread(som_service.close)


app = FastAPI(lifespan=lifespan)


# only this machine's own pages and tools: a page on another site (or a DNS name rebound to 127.0.0.1) can't reach
# the API, which runs the local model and ChatGPT's cloud-backed chat
LOCAL_HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}", f"[::1]:{PORT}"}


@app.middleware("http")
async def local_only(request, call_next):
    from urllib.parse import urlparse
    origin = request.headers.get("origin")
    if (request.headers.get("host") not in LOCAL_HOSTS
            or (origin and urlparse(origin).netloc not in LOCAL_HOSTS)
            or request.headers.get("sec-fetch-site") in ("cross-site", "same-site")):
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

rt = Runtime(STATS_ROOT)
som_service = SOMService(repo_root=ROOT)
residual_router, residual_service = create_residual_router(rt, ROOT)
app.include_router(residual_router)
app.include_router(create_atlas_router(ROOT))


def current_model_info():
    """JSON-safe model identity/config supplied by Runtime without network access."""
    getter = getattr(rt, "get_model_info", None)
    if callable(getter):
        return getter()
    info = getattr(rt, "model_info", {})
    return dict(info) if isinstance(info, dict) else {}


def architecture_from_config(config):
    """Extract geometry from the model's saved config; never substitute guessed dimensions."""
    if not config:
        return None
    cfg = config
    if hasattr(cfg, "get_text_config"):
        cfg = cfg.get_text_config()
    if not isinstance(cfg, dict):
        cfg = getattr(cfg, "to_dict", dict)()
    if not isinstance(cfg, dict):
        return None
    cfg = cfg.get("text_config", cfg)
    required = ("num_hidden_layers", "hidden_size", "intermediate_size")
    if any(cfg.get(k) is None for k in required):
        return None
    layers = int(cfg["num_hidden_layers"])
    types = cfg.get("layer_types")
    if types is None:
        # Llama's saved config describes ordinary self-attention but omits a repeated type list.
        if cfg.get("model_type") == "llama" and cfg.get("num_attention_heads") is not None:
            types = ["full_attention"] * layers
        else:
            types = []
    if len(types) != layers:
        return None
    return {
        "n_layers": layers,
        "hidden": int(cfg["hidden_size"]),
        "inter": int(cfg["intermediate_size"]),
        "grid_side": ceil(sqrt(int(cfg["intermediate_size"]))),
        "layer_types": list(types),
        "attn_heads": cfg.get("num_attention_heads"),
        "kv_heads": cfg.get("num_key_value_heads"),
        "linear_k_heads": cfg.get("linear_num_key_heads"),
        "linear_v_heads": cfg.get("linear_num_value_heads"),
    }


_initial_info = current_model_info()
MODEL_ID = _initial_info.get("model_id") or os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL)
MODEL_REVISION = _initial_info.get("revision") or resolve_model_revision(
    MODEL_ID, os.environ.get("EXPLORER_REVISION"))
CACHE_NAMESPACE = _initial_info.get("cache_namespace") or model_cache_namespace(MODEL_ID, MODEL_REVISION)
CACHE_DIR = _initial_info.get("cache_dir") or os.path.join(ROOT, "explorer", "cache", CACHE_NAMESPACE)
RUNS = os.path.join(RUNS_ROOT, CACHE_NAMESPACE)
ARCH = architecture_from_config(_initial_info.get("config_metadata") or _initial_info.get("config"))
NAMES = os.path.join(CACHE_DIR, "names.json")
LENS_INDEX = os.path.join(CACHE_DIR, "lens_index.npz")
ACT_SCALE = os.path.join(CACHE_DIR, "act_scale.npz")
_cache = {}


def layer_runs():
    """Layer -> matching model/revision run. Unscoped legacy runs are never adopted."""
    best = {}
    for meta_path in glob.glob(os.path.join(RUNS, "*.meta.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if not run_matches_model(meta):
            continue
        if meta.get("design_name", "main") == "calib":
            continue
        jsonl = meta_path.replace(".meta.json", ".jsonl")
        if not os.path.exists(jsonl):
            continue
        try:
            layer = int(meta["layer"])
            key = (meta.get("design_name") == "main_rel", os.path.getsize(jsonl))
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if key > best.get(layer, ("", (False, 0)))[1]:
            best[layer] = (jsonl, key)
    return {layer: path for layer, (path, _) in best.items()}


def run_matches_model(meta):
    """Require model identity and reject an explicitly different revision.

    The revision-specific run directory supplies revision identity for older run metadata that records only its model.
    Flat legacy runs outside that directory are intentionally not searched.
    """
    if not isinstance(meta, dict):
        return False
    model = meta.get("model_id") or meta.get("model")
    if not isinstance(model, str) or model.strip().casefold() != MODEL_ID.strip().casefold():
        return False
    recorded_revision = meta.get("model_revision") or meta.get("revision") or meta.get("commit_hash")
    return not (recorded_revision and MODEL_REVISION and str(recorded_revision) != str(MODEL_REVISION))


_load_lock = threading.Lock()


def load_layer(layer):
    path = layer_runs().get(layer)
    if path is None:
        return None
    units_path = path.replace(".jsonl", ".units.jsonl")
    key = (path, os.path.getmtime(path), os.path.getmtime(units_path) if os.path.exists(units_path) else 0)
    if _cache.get(layer, {}).get("key") == key:
        return _cache[layer]
    with _load_lock:
        if _cache.get(layer, {}).get("key") == key:
            return _cache[layer]
        if not (os.path.exists(units_path) and os.path.getmtime(units_path) >= os.path.getmtime(path)):
            with open(path) as source:
                full = [json.loads(line) for line in source]  # ranking needs token ids; then dropped
            with open(units_path + ".tmp", "w") as f:
                f.writelines(json.dumps(u) + "\n" for u in compute_units(full))
            del full
            os.replace(units_path + ".tmp", units_path)
            key = (path, os.path.getmtime(path), os.path.getmtime(units_path))
        with open(units_path) as source:
            units = [json.loads(line) for line in source]
        # keep only what the explorer shows; token ids, logprobs and alternatives stay on disk
        slim = ("kind", "neuron", "prompt", "dose", "rel_push", "text", "finish", "n_gen")
        rows = []
        with open(path) as source:
            for line in source:
                r = json.loads(line)
                rows.append({k: r[k] for k in slim})
        by_unit = {}
        for i, r in enumerate(rows):
            by_unit.setdefault((r["kind"], r["neuron"]), []).append(i)
        with open(path.replace(".jsonl", ".meta.json")) as source:
            meta = json.load(source)
        _cache[layer] = {"key": key, "path": path, "rows": rows, "units": units, "by_unit": by_unit,
                         "meta": meta}
        return _cache[layer]


_ranked = {}


def ranked_words(layer):
    """{neuron: top words} for the neurons of `layer` with sweep rollouts, read from the run's ranking file only."""
    path = layer_runs().get(layer)
    units_path = path.replace(".jsonl", ".units.jsonl") if path else None
    if not units_path or not os.path.exists(units_path):
        return {}
    key = (units_path, os.path.getmtime(units_path))
    if _ranked.get(layer, (None,))[0] != key:
        words = {}
        with open(units_path) as source:
            for line in source:
                u = json.loads(line)
                if u["kind"] == "neuron":
                    words[u["neuron"]] = [w for w, _, _ in u["words"][:4]]
        _ranked[layer] = (key, words)
    return _ranked[layer][1]


def swept(layer):
    return ranked_words(layer).keys()


def names():
    if not os.path.exists(NAMES):
        return {}
    try:
        with open(NAMES) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def cache_inventory():
    return {
        "layout_writes": os.path.exists(os.path.join(CACHE_DIR, "layout_writes.npz")),
        "layout_reads": os.path.exists(os.path.join(CACHE_DIR, "layout_reads.npz")),
        "connections": os.path.exists(os.path.join(CACHE_DIR, "connections.npz")),
        "logit_lens": os.path.exists(LENS_INDEX),
        "activation_scale": os.path.exists(ACT_SCALE),
        "steer_lens": os.path.exists(os.path.join(CACHE_DIR, "steer_index.npz")),
    }


def model_payload():
    info = current_model_info()
    cfg = info.get("config_metadata") or info.get("config")
    arch = architecture_from_config(cfg) or ARCH
    return {
        "id": info.get("model_id", MODEL_ID),
        "revision": info.get("revision", MODEL_REVISION),
        "cache_namespace": info.get("cache_namespace", CACHE_NAMESPACE),
        "device": info.get("device") or getattr(rt, "dev", None) or "not loaded",
        "status": getattr(rt, "status", "unknown"),
        "config_available": arch is not None,
    }, arch


@app.get("/api/index")
def index():
    runs = layer_runs()
    info, arch = model_payload()
    return {"model": info["id"], "model_info": info, "arch": arch,
            "layers": {str(l): os.path.basename(p) for l, p in sorted(runs.items())},
            "has_runs": bool(runs), "cache": cache_inventory(), "status": rt.status}


@app.get("/api/units")
def units(layer: int):
    d = load_layer(layer)
    if d is None:
        return []
    nm = names()
    return [{"kind": u["kind"], "n": u["neuron"], "z": u.get("mass", u["z"]), "null_pct": u["null_pct"],
             "words": [w for w, _, _ in u["words"][:5]], "name": nm.get(f"L{layer}/N{u['neuron']}", "")}
            for u in d["units"]]


@app.get("/api/unit")
def unit(layer: int, n: int, kind: str = "neuron"):
    d = load_layer(layer)
    out = {"layer": layer, "n": n, "kind": kind, "name": names().get(f"L{layer}/N{n}", "")}
    if d is None:
        return out
    rec = next((u for u in d["units"] if u["neuron"] == n and u["kind"] == kind), None)
    rows = [d["rows"][i] for i in d["by_unit"].get((kind, n), [])]
    base = [r for r in d["rows"] if r["kind"] == "baseline"]
    keep = ("prompt", "dose", "rel_push", "text", "finish", "n_gen")
    neurons = [u for u in d["units"] if u["kind"] == "neuron"]
    out.update({
        "run": os.path.basename(d["path"]), "prompts": d["meta"]["prompts"],
        "rollouts": [{k: r[k] for k in keep} for r in rows],
        "baselines": [{k: r[k] for k in keep} for r in base],
        "stats": rec, "rank": next((i + 1 for i, u in enumerate(neurons) if u["neuron"] == n), None) if rec else None,
        "n_ranked": len(neurons),
    })
    return out


@app.post("/api/name")
async def set_name(req: Request):
    body = await req.json()
    nm = names()
    key = f"L{int(body['layer'])}/N{int(body['n'])}"
    if body.get("name", "").strip():
        nm[key] = body["name"].strip()
    else:
        nm.pop(key, None)
    await asyncio.to_thread(_write_json, NAMES, nm)
    return {"ok": True}


def _write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=1, sort_keys=True)


@app.get("/api/embeddings")
def embeddings(layer: int):
    d = load_layer(layer)
    if d is None:
        return {}
    path = d["path"].replace(".jsonl", ".umap.json")
    return FileResponse(path) if os.path.exists(path) else JSONResponse({})


@app.get("/api/rollout")
def rollout(layer: int, i: int):
    r = load_layer(layer)["rows"][i]
    return {k: r[k] for k in ("kind", "neuron", "prompt", "dose", "rel_push", "text")}


@app.get("/api/connections")
async def connections(layer: int, n: int, k: int = 24, lens: int = 5):
    if rt.con_index() is None and rt.tok is None and (off := _live_off()):
        return {"ready": False, "reason": off["error"], "down": [], "up": []}
    out = await asyncio.to_thread(rt.connections, layer, n, k, lens)
    words = await asyncio.to_thread(lambda: {l: ranked_words(l) for l in {c["layer"] for c in out["down"] + out["up"]}})
    nm = names()
    out = {side: [dict(c) for c in out[side]] for side in ("down", "up")}
    for c in out["down"] + out["up"]:
        c["name"] = nm.get(f"L{c['layer']}/N{c['n']}", "")
        c["words"] = words[c["layer"]].get(c["n"], [])
    return out


@app.get("/api/hover")
async def hover(layer: int, n: int):
    if rt.con_index() is None and rt.lens_index() is None and rt.tok is None and (off := _live_off()):
        return {"ready": False, "reason": off["error"], "lens": [], "lens_neg": [],
                "steer": None, "steer_neg": None, "con": {"down": [], "up": []}}
    con = await asyncio.to_thread(rt.connections, layer, n, 16, 0)
    lens = await asyncio.to_thread(rt.lens, layer, n, 8)
    return {"lens": lens["up"], "lens_neg": lens["down"], "steer": lens.get("steer_up"), "steer_neg": lens.get("steer_down"),
            "con": con}


@app.get("/api/lens")
async def lens(layer: int, n: int):
    if rt.lens_index() is None and rt.tok is None and (off := _live_off()):
        return {"ready": False, "reason": off["error"], "up": [], "down": []}
    return await asyncio.to_thread(rt.lens, layer, n, 12)


@app.post("/api/reads")
async def reads(req: Request):
    if off := _live_off():
        return JSONResponse(off, status_code=503)
    b = await req.json()
    return await asyncio.to_thread(rt.read, int(b["layer"]), int(b["n"]), b["text"], b.get("chat", True))


@app.post("/api/steer")
async def steer(req: Request):
    if off := _live_off():
        return JSONResponse(off, status_code=503)
    b = await req.json()
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def work():
        try:
            # each row replays its own earlier replies; the user turns are shared
            rows = []
            for r in range(3):
                msgs = []
                for turn in b.get("history", []):
                    msgs += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["replies"][r]}]
                rows.append(msgs + [{"role": "user", "content": b["prompt"]}])
            for row, delta in rt.steer_stream(int(b["layer"]), int(b["n"]), rows, float(b["dose"]),
                                              b.get("mode", "add"), int(b.get("max_new", 300)), cancel):
                loop.call_soon_threadsafe(q.put_nowait, {"row": row, "t": delta})
        except Exception as e:  # noqa: BLE001 - forward unexpected model errors to the streaming client
            loop.call_soon_threadsafe(q.put_nowait, {"row": None, "error": repr(e)})

    threading.Thread(target=work, daemon=True).start()

    async def events():
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["row"] is None:
                    break
                if await req.is_disconnected():
                    cancel.set()
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream")


_lens = {}


def lens_index():
    if "up" not in _lens and os.path.exists(LENS_INDEX):
        z = np.load(LENS_INDEX)
        _lens["up"], _lens["down"] = z["up"], z["down"]
    return _lens if "up" in _lens else None


@app.get("/api/search")
async def search(q: str, limit: int = 300):
    """Neurons whose logit lens (top 12 up or down) contains a token matching q (case-insensitive substring)."""
    idx = lens_index()
    q = q.strip().lower()
    if idx is None or len(q) < 2:
        return {"ready": idx is not None, "results": []}
    vocab = await asyncio.to_thread(rt.vocab_strings)
    match = np.array([i for i, t in enumerate(vocab) if q in t.lower()], dtype=np.int32)
    if not len(match):
        return {"ready": True, "results": [], "tokens": 0}
    out = []
    st = rt.steer_index()
    sources = [("+", idx["up"], "lens"), ("-", idx["down"], "lens")]
    if st is not None:
        sources += [("+", st["up"], "steer"), ("-", st["down"], "steer")]
    for sign, arr, src in sources:
        hit = np.isin(arr, match)  # [L, N, k]
        count = hit.sum(-1)
        best = np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
        for l, n in zip(*np.nonzero(count)):
            out.append((int(best[l, n]), -int(count[l, n]), int(l), int(n), sign, src))
    # one row per neuron and sign, keeping its best match across the two lenses
    seen, uniq = set(), []
    for row in sorted(out):
        if (row[2], row[3], row[4]) not in seen:
            seen.add((row[2], row[3], row[4])); uniq.append(row)
    out = uniq
    nm, res = names(), []
    sw = {l: swept(l) for l in {row[2] for row in out[:limit]}}
    for rank, _, l, n, sign, src in out[:limit]:
        arr = (idx if src == "lens" else st)["up" if sign == "+" else "down"]
        res.append({"layer": l, "n": n, "sign": sign, "rank": rank, "src": src, "name": nm.get(f"L{l}/N{n}", ""),
                    "swept": n in sw[l], "lens": [vocab[t] for t in arr[l, n, :6]]})
    return {"ready": True, "results": res, "tokens": len(match), "total": len(out)}


MAX_INFER_TOKENS = 1024  # bounded [layers, tokens, intermediate width] activation tensor


@app.get("/api/fires")
async def fires(text: str, limit: int = 300, chat: bool = False):
    """Neurons that fire hardest on `text` relative to their typical range over web text (activation / typical max,
    and activation / typical min for the negative side)."""
    if not os.path.exists(ACT_SCALE):
        return {"ready": False, "results": []}
    sc = np.load(ACT_SCALE)
    ids = await asyncio.to_thread(rt.chat_ids if chat else (lambda t: rt.tok(t).input_ids), text)
    acts = await asyncio.to_thread(rt.all_activations, ids)
    toks = [rt.tok.decode([t]) for t in ids]
    cand = []
    for l, (mx, amx, mn, amn) in acts.items():
        p = mx.numpy() / np.maximum(sc["pos"][l], 1e-3)
        q = mn.numpy() / np.minimum(sc["neg"][l], -1e-3)
        for arr, idx, sign, raw in ((p, amx.numpy(), "+", mx.numpy()), (q, amn.numpy(), "-", mn.numpy())):
            top = np.argpartition(-arr, limit)[:limit]
            cand += [(float(arr[n]), l, int(n), sign, int(idx[n]), float(raw[n])) for n in top]
    cand.sort(reverse=True)
    idx_lens, nm = lens_index(), names()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    res = []
    for score, l, n, sign, t, raw in cand[:limit]:
        lens = [vocab[k] for k in (idx_lens["up"] if sign == "+" else idx_lens["down"])[l, n, :6]] if idx_lens else []
        res.append({"layer": l, "n": n, "sign": sign, "score": round(score, 2), "act": round(raw, 2), "token": toks[t],
                    "pos": t, "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    return {"ready": True, "results": res, "tokens": toks}


_infer = {}


async def score_text(text, chat=False):
    """Every neuron's activation at every token of `text` ([L, T, N]) and its signed score: the positive side as a
    multiple of the neuron's typical max, the negative side of its typical min (act_scale.npz)."""
    ids = rt.chat_ids(text) if chat else rt.tok(text).input_ids
    ids = ids[:MAX_INFER_TOKENS]
    acts = await asyncio.to_thread(rt.full_activations, ids)  # [L, T, N]
    sc = np.load(ACT_SCALE)
    # a side the neuron barely reaches in calibration is scaled by at least a tenth of its other side
    pos = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))[:, None, :]
    neg = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))[:, None, :]
    score = np.where(acts >= 0, acts / pos, -(acts / neg)).astype(np.float32)
    if score.shape[1] > 1:
        score[:, 0, :] = 0  # position 0 is the attention-sink token, outside the range the scale was measured on
    return ids, acts, score


@app.post("/api/infer")
async def infer(req: Request):
    """Runs text through the model and keeps every neuron's activation at every token for /api/infer/view."""
    b = await req.json()
    if not os.path.exists(ACT_SCALE) or rt.tok is None:
        return {"ready": False}
    ids, acts, score = await score_text(b["text"], b.get("chat", False))
    units = rt.token_units(ids)
    unit_of = {i: k for k, (_, idx) in enumerate(units) for i in idx}
    _infer.clear()
    _infer.update(ids=ids, toks=[units[unit_of[i]][0] for i in range(len(ids))], acts=acts, score=score)
    # one chip per unit; clicking it selects the unit's last token (the one that completes the character)
    return {"ready": True, "tokens": [u[0] for u in units], "pos": [u[1][-1] for u in units]}


@app.get("/api/infer/view")
async def infer_view(pos: int = -1, sign: str = "both", lo: int = 0, hi: int = 99, min: float = 1.0, q: str = "",
                     limit: int = 300):
    """Neurons passing the filters for the last inferred text. pos = token index, or -1 for the strongest token.
    Returns the list plus every neuron's signed score at that position (base64 float16, layer-major) for the map."""
    if "score" not in _infer:
        return {"results": []}
    import base64
    S, A, toks = _infer["score"], _infer["acts"], _infer["toks"]  # [L, T, N]
    if pos >= S.shape[1]:
        return {"results": [], "total": 0}
    if pos >= 0:
        best, at = S[:, pos, :], np.full(S[:, 0, :].shape, pos)
    else:
        at = S.argmax(1) if sign == "+" else S.argmin(1) if sign == "-" else np.abs(S).argmax(1)  # [L, N]
        best = np.take_along_axis(S, at[:, None, :], 1)[:, 0, :]
    view = best.copy()
    keep = np.abs(best) >= min
    if sign == "+": keep &= best > 0
    if sign == "-": keep &= best < 0
    lay = np.arange(best.shape[0])[:, None]
    keep &= (lay >= lo) & (lay <= hi)
    idx = lens_index()
    vocab = await asyncio.to_thread(rt.vocab_strings)
    if q.strip() and idx is not None:
        ql = q.strip().lower()
        match = np.array([i for i, t in enumerate(vocab) if ql in t.lower()], dtype=np.int32)
        lens_hit = np.where(best[..., None] >= 0, np.isin(idx["up"][:, :, :6], match), np.isin(idx["down"][:, :, :6], match)).any(-1)
        keep &= lens_hit
    view[~keep] = 0
    ls, ns = np.nonzero(keep)
    order = np.argsort(-np.abs(best[ls, ns]))[:limit]
    nm, res = names(), []
    for o in order:
        l, n = int(ls[o]), int(ns[o]); sg = "+" if best[l, n] > 0 else "-"
        lens = [vocab[k] for k in (idx["up"] if sg == "+" else idx["down"])[l, n, :6]] if idx is not None else []
        t = int(at[l, n])
        res.append({"layer": l, "n": n, "sign": sg, "score": round(float(abs(best[l, n])), 2),
                    "act": round(float(A[l, t, n]), 2), "token": toks[t], "pos": t,
                    "name": nm.get(f"L{l}/N{n}", ""), "lens": lens})
    blob = base64.b64encode(view.astype(np.float16).tobytes()).decode()
    return {"results": res, "total": int(keep.sum()), "map": blob, "shape": list(view.shape)}


@app.get("/api/infer/paths")
def infer_paths(pos: int, top: int = 300, min: float = 1.0):
    """The circuit active at one token of the last inferred text: among the `top` neurons with the largest |score| there
    (at least `min`), every precomputed direct link whose two ends both fired. Same binary format as /api/web, with z
    signed by the push the writer's actual activation gives the reader (link sign x sign of the writer's score), so
    yellow / magenta read as "this firing pushed that one up / down"; hop is 1."""
    if "score" not in _infer or pos >= _infer["score"].shape[1]:
        return Response(b"", media_type="application/octet-stream", headers={"X-Web-Total": "0"})
    S = _infer["score"][:, pos, :]  # [L, N]
    flat = np.abs(S).ravel()
    cand = np.argpartition(-flat, min_k := builtins.min(top, flat.size - 1))[:min_k]
    cand = cand[flat[cand] >= min]
    N = S.shape[1]
    active = {(int(i // N), int(i % N)) for i in cand}
    ci = rt.con_index()
    if ci is None:
        return Response(b"", media_type="application/octet-stream",
                        headers={"X-Web-Total": "0", "X-Active": str(len(active))})
    rows = []
    for l, n in active:
        for i in range(ci["k"]):
            z = float(ci["down_z"][l, n, i])
            if z < -1e4:
                break
            t = (int(ci["down_layer"][l, n, i]), int(ci["down_n"][l, n, i]))
            if t in active:
                push = int(ci["down_sign"][l, n, i]) * (1 if S[l, n] > 0 else -1)
                rows.append((l, n, *t, round(z * 10) * push, 1))
    out = np.array(rows, np.int16).reshape(-1, 6)
    return Response(out.tobytes(), media_type="application/octet-stream",
                    headers={"X-Web-Total": str(len(out)), "X-Active": str(len(active))})


@app.get("/api/layout")
def layout(kind: str = "writes"):
    """Seriated grid layout: cell index (row * side + col) for every neuron, base64 int16, layer-major."""
    import base64
    if kind not in ("writes", "reads"):
        return {"available": False, "reason": "unknown layout"}
    path = os.path.join(CACHE_DIR, f"layout_{kind}.npz")
    if not os.path.exists(path):
        return {"available": False}
    cell = np.load(path)["cell"].astype(np.int16)
    return {"available": True, "shape": list(cell.shape), "cell": base64.b64encode(cell.tobytes()).decode()}


def cone_links(layer, n, k=2, zmin=15, cap=20000, first=None, hops=None):
    """Connection cone of one neuron: everything it writes into, then what those write into, out to the last layer;
    and everything that writes into it, back to layer 0. Each side only moves one way. The first hop takes all of the
    neuron's top `first` links (all when None), later hops each node's top k with z >= zmin, up to `hops` hops; each side keeps at most cap / 2 links.
    Rows: src layer, src neuron, dst layer, dst neuron, z * sign, hop (positive downstream, negative upstream)."""
    ci = rt.con_index()
    if ci is None:
        return []
    out = []
    for side, step in (("down", 1), ("up", -1)):
        zs, ls, ns, sg = ci[f"{side}_z"], ci[f"{side}_layer"], ci[f"{side}_n"], ci[f"{side}_sign"]
        front, seen, hop, start = [(layer, n)], {(layer, n)}, 1, len(out)
        while front and len(out) - start < cap // 2 and (not hops or hop <= hops):
            nxt = []
            for a, b in front:
                for i in range((first or ci["k"]) if hop == 1 else k):
                    z = float(zs[a, b, i])
                    if z < -1e4 or (hop > 1 and z < zmin):
                        break
                    t = (int(ls[a, b, i]), int(ns[a, b, i]))
                    src, dst = ((a, b), t) if side == "down" else (t, (a, b))
                    out.append((*src, *dst, z * int(sg[a, b, i]), hop * step))
                    if t not in seen:
                        seen.add(t)
                        nxt.append(t)
            front, hop = nxt, hop + 1
        del out[start + cap // 2:]
    return out


@app.get("/api/web")
def web(layer: int, n: int, first: int = 5, k: int = 2, zmin: float = 15, cap: int = 20000, hops: int = 0):
    """The cone for the map. Binary int16, 6 per link: src layer, src neuron, dst layer, dst neuron,
    round(10 * z) * sign, hop."""
    rows = np.array([(*r[:4], round(r[4] * 10), r[5]) for r in cone_links(layer, n, k, zmin, cap, first, hops or None)], np.int16).reshape(-1, 6)
    return Response(rows.tobytes(), media_type="application/octet-stream", headers={"X-Web-Total": str(len(rows))})


_idf = {}


@app.get("/api/web/summary")
def web_summary(layer: int, n: int, first: int = 5, k: int = 2, top: int = 30, src: str = "steer", hops: int = 0):
    """What the neuron's web says, in tokens. Each neuron in the cone is read on the pole the chain drives (signs
    multiplied along the path from the open neuron), and its steer (or logit) lens tokens are pooled, merged across
    case and leading space, and weighted by link z / max z times 0.8 per hop past the first, times log lift: the share
    of web neurons carrying the token over its share among all neurons of the same layers, so tokens a layer's neurons
    commonly carry (`(`, ` a` in early layers) sink and tokens the web carries no more often than chance drop out.
    Downstream gives
    `toward` (the driven pole) and `away` (the opposite pole); upstream gives `fed_by` (the pole that excites it) and
    `held_by` (the pole that inhibits it)."""
    st = rt.steer_index() if src == "steer" else None
    lens = st if st is not None else rt.lens_index()
    if lens is None:
        return {"ready": False, "links": 0,
                "interpretation": "weight-only candidate links; not causal evidence",
                "down": {"neurons": 0, "toward": [], "away": []},
                "up": {"neurons": 0, "fed_by": [], "held_by": []}}
    v = rt.vocab_strings()
    key_src = "steer" if st is not None else "logit"
    if _idf.get("src") != key_src:
        L, N = lens["up"].shape[:2]
        df = np.stack([np.bincount(np.concatenate([lens["up"][l, :, :8].ravel(), lens["down"][l, :, :8].ravel()]),
                                   minlength=len(v)) for l in range(L)])
        _idf.update(src=key_src, base=((df + 1) / (2 * N)).astype(np.float32))  # [layer, token]
    base = _idf["base"]
    sign, zmax = {(layer, n): 1}, 1.0
    links = cone_links(layer, n, k, 15, 20000, first, hops or None)
    for sl, sn, dl, dn, z, hop in links:
        zmax = max(zmax, abs(z))
    reach = {}
    for sl, sn, dl, dn, z, hop in links:
        near, far = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
        if far in sign or near not in sign:
            continue
        sign[far] = sign[near] * (1 if z > 0 else -1)
        reach[far] = (hop, abs(z))

    def pool(nodes, flip):
        acc = {}
        for (l, m), (hop, z) in nodes:
            w = z / zmax * 0.8 ** (abs(hop) - 1)
            pole = "up" if sign[(l, m)] * flip > 0 else "down"
            for t in lens[pole][l, m, :8]:
                s = v[t]
                key = s.strip().lower()
                if not key or "\ufffd" in key:
                    continue
                a = acc.setdefault(key, {"tok": s.strip(), "n": 0, "w": 0.0, "forms": {}})
                a["n"] += 1
                a["w"] += w
                a["base"] = a.get("base", 0.0) + float(base[l, t])
                a["forms"][s.strip()] = a["forms"].get(s.strip(), 0) + 1
        for a in acc.values():
            lift = (a["n"] / max(1, len(nodes))) / (a["base"] / a["n"])
            a["score"] = a["w"] * np.log(lift) if lift > 1 else 0.0
        out = sorted((a for a in acc.values() if a["score"] > 0), key=lambda a: -a["score"])[:top]
        return [{"tok": max(a["forms"], key=a["forms"].get), "n": a["n"], "w": round(float(a["score"]), 2)} for a in out]

    down = [(key, val) for key, val in reach.items() if val[0] > 0]
    up = [(key, val) for key, val in reach.items() if val[0] < 0]
    return {"src": "steer" if st is not None else "logit", "links": len(links),
            "ready": True, "interpretation": "weight-only candidate links; not causal evidence",
            "down": {"neurons": len(down), "toward": pool(down, 1), "away": pool(down, -1)},
            "up": {"neurons": len(up), "fed_by": pool(up, 1), "held_by": pool(up, -1)}}


# ---------- ChatGPT: autointerpretation and chat ----------
# Each conversation is stored in the model-scoped explorer cache. The MCP server reads through the endpoints below and
# sends map actions back through /api/agent/ui into the active browser stream.
AGENT_PROMPT = os.path.join(ROOT, "explorer", "agent_prompt.md")
AGENT_DIR = os.path.join(CACHE_DIR, "agent")
_chats = {}
_act = {}


def act_range(layer, n):
    if "pos" not in _act and os.path.exists(ACT_SCALE):
        z = np.load(ACT_SCALE)
        _act.update(pos=z["pos"], neg=z["neg"], tokens=int(z["tokens"]))
    if "pos" not in _act:
        return None
    return {"act_max": round(float(_act["pos"][layer, n]), 3), "act_min": round(float(_act["neg"][layer, n]), 3),
            "over_tokens": _act["tokens"], "source": "FineWeb-Edu"}


@app.get("/api/agent/overview")
def agent_overview():
    runs = layer_runs()
    model, arch = model_payload()
    return {"model": model["id"], "model_info": model, "arch": arch,
            "layers_with_sweeps": sorted(runs), "named_neurons": len(names()),
            "live_model": not NO_MODEL and rt.tok is not None,
            "connection_interpretation": "weight-only candidate links; not causal evidence"}


@app.get("/api/agent/names")
def agent_names():
    return names()


def _som_error_response(exc):
    """Keep SOM service failures as structured, useful HTTP errors."""
    if isinstance(exc, SOMServiceError):
        return JSONResponse({"error": exc.message, "code": exc.code}, status_code=exc.http_status)
    return JSONResponse({"error": "SOM service request failed."}, status_code=500)


def _native_report_url(native_id):
    return f"/api/agent/som/native-reports/{quote(native_id, safe='')}/report"


MAX_NATIVE_EMBEDDED_ATLAS_BYTES = 32 * 1024 * 1024


class _NativeAtlasRewriter(HTMLParser):
    """Rewrite only validated map and behavior references into self-contained content."""

    def __init__(self, image_data_uris):
        super().__init__(convert_charrefs=False)
        self.image_data_uris = image_data_uris
        self.parts = []
        self.embedded_images = set()
        self.strict_links = 0
        self.review_links = 0
        self.skipped_anchor_depth = 0

    def _start_tag(self, tag, attrs, self_closing=False):
        if self.skipped_anchor_depth:
            if tag == "a":
                self.skipped_anchor_depth += 1
            return

        attr_values = dict(attrs)
        if tag == "a" and attr_values.get("href") == "behavior/blinded-review.json":
            self.parts.append('<a href="#blinded-review-summary">Semantic review status</a>')
            self.review_links += 1
            self.skipped_anchor_depth = 1
            return

        rendered_attrs = []
        for name, value in attrs:
            if name == "srcset":
                raise ValueError("native report contains an unsupported image reference")
            if tag == "img" and name == "src":
                if not isinstance(value, str):
                    raise ValueError("native report image reference is invalid")
                if value.startswith("maps/"):
                    data_uri = self.image_data_uris.get(value)
                    if data_uri is None:
                        raise ValueError("native report references an unlisted map image")
                    self.embedded_images.add(value)
                    value = data_uri
                elif not value.lower().startswith("data:image/"):
                    raise ValueError("native report contains a non-embedded image")
            elif tag == "a" and name == "href":
                if value == "behavior/strict-screen.json":
                    value = "#strict-screen-summary"
                    self.strict_links += 1
                elif value and not value.startswith("#"):
                    raise ValueError("native report contains an unsupported link")
            if value is None:
                rendered_attrs.append(f" {name}")
            else:
                rendered_attrs.append(f' {name}="{html.escape(value, quote=True)}"')
        suffix = "/" if self_closing else ""
        self.parts.append(f"<{tag}{''.join(rendered_attrs)}{suffix}>")

    def handle_starttag(self, tag, attrs):
        self._start_tag(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start_tag(tag, attrs, self_closing=True)

    def handle_endtag(self, tag):
        if self.skipped_anchor_depth:
            if tag == "a":
                self.skipped_anchor_depth -= 1
            return
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if not self.skipped_anchor_depth:
            self.parts.append(data)

    def handle_entityref(self, name):
        if not self.skipped_anchor_depth:
            self.parts.append(f"&{name};")

    def handle_charref(self, name):
        if not self.skipped_anchor_depth:
            self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        if not self.skipped_anchor_depth:
            self.parts.append(f"<!--{data}-->")

    def handle_decl(self, decl):
        if not self.skipped_anchor_depth:
            self.parts.append(f"<!{decl}>")

    def unknown_decl(self, data):
        if not self.skipped_anchor_depth:
            self.parts.append(f"<![{data}]>")

    def handle_pi(self, data):
        if not self.skipped_anchor_depth:
            self.parts.append(f"<?{data}>")


class _FirstBodyContentTag(HTMLParser):
    _HEAD_TAGS = frozenset({"meta", "title", "style", "base", "link", "script", "noscript", "template"})

    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.index = None

    def handle_starttag(self, tag, _attrs):
        if self.index is None and tag not in self._HEAD_TAGS:
            line, column = self.getpos()
            self.index = sum(len(part) for part in self.source.splitlines(keepends=True)[:line - 1]) + column

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)


def _native_report_behavior_payload(report, kind):
    if (
        not isinstance(report, dict)
        or report.get("status") != "completed"
        or report.get("integrity_status") != "verified"
    ):
        raise ValueError("No validated native SOM behavior summary was found.")
    behavior = report.get("behavior")
    if not isinstance(behavior, dict):
        raise TypeError("No validated native SOM behavior summary was found.")
    if kind == "strict-screen":
        summary = behavior.get("strict_screen")
        if not isinstance(summary, dict):
            raise ValueError("No validated native SOM behavior summary was found.")
        return summary
    review_status = behavior.get("semantic_review_status")
    grade_count = behavior.get("semantic_grade_count")
    if (
        not isinstance(review_status, str)
        or isinstance(grade_count, bool)
        or not isinstance(grade_count, int)
        or grade_count < 0
    ):
        raise ValueError("No validated native SOM behavior summary was found.")
    return {"semantic_review_status": review_status, "semantic_grade_count": grade_count}


def _native_report_summary_banner(report):
    geometry = report.get("geometry", {})
    behavior = report.get("behavior", {})
    if not isinstance(geometry, dict) or not isinstance(behavior, dict):
        raise TypeError("Native SOM summary is incomplete")

    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    map_count = count(geometry.get("map_count"))
    pairs = count(geometry.get("pairs"))
    cases = count(behavior.get("cases"))
    arms = count(behavior.get("arms"))
    answers = count(behavior.get("answers"))
    grade_count = count(behavior.get("semantic_grade_count"))
    review_status = behavior.get("semantic_review_status")
    integrity = report.get("integrity_status")
    capture = report.get("capture_provenance_status")
    state = "Integrity and capture provenance verified" if integrity == "verified" and capture == "verified" else f"Integrity: {integrity}; capture provenance: {capture}"
    counts = []
    for label, value in (("maps", map_count), ("pairs", pairs), ("cases", cases), ("arms", arms), ("answers", answers)):
        if value is not None:
            counts.append(f"{value:,} {label}")
    review = "Semantic review status unavailable"
    if isinstance(review_status, str):
        review = f"Semantic grading: {review_status}"
        if grade_count is not None:
            review += f"; graded items: {grade_count:,}"
    return (
        '<aside class="som-run-summary" aria-label="Validated run summary">'
        f"<strong>{html.escape(state)}</strong>"
        f"<p>{html.escape(' · '.join(counts))}</p>"
        f"<p>{html.escape(review)}</p>"
        "</aside>"
    )


def _native_report_behavior_sections(report):
    strict = _native_report_behavior_payload(report, "strict-screen")
    review = _native_report_behavior_payload(report, "blinded-review")
    strict_json = html.escape(json.dumps(strict, ensure_ascii=False, indent=2))
    review_json = html.escape(json.dumps(review, ensure_ascii=False, indent=2))
    return (
        '<section id="strict-screen-summary" class="som-derived-summary">'
        "<h2>Strict-screen summary</h2>"
        f"<pre>{strict_json}</pre></section>"
        '<section id="blinded-review-summary" class="som-derived-summary">'
        "<h2>Semantic review status</h2>"
        f"<pre>{review_json}</pre></section>"
    )


def _render_native_atlas(native_id, report, report_path):
    report_bytes = report_path.read_bytes()
    expected_report_hash = report.get("geometry", {}).get("report_sha256")
    if (
        len(report_bytes) > MAX_NATIVE_REPORT_BYTES
        or not isinstance(expected_report_hash, str)
        or hashlib.sha256(report_bytes).hexdigest() != expected_report_hash
    ):
        raise ValueError("Validated native SOM report changed before rendering")
    source = report_bytes.decode("utf-8")
    maps = report.get("maps")
    if not isinstance(maps, list) or len(maps) > MAX_NATIVE_MAPS:
        raise ValueError("Native SOM map index is invalid")

    image_data_uris = {}
    estimated_bytes = len(report_bytes)
    for item in maps:
        if not isinstance(item, dict):
            raise TypeError("Native SOM map index is invalid")
        map_name = item.get("map_name")
        digest = item.get("png_sha256")
        if not isinstance(map_name, str) or not isinstance(digest, str):
            raise TypeError("Native SOM map index is invalid")
        relative_name = f"maps/{map_name}/map.png"
        asset_path = som_service.get_native_report_asset(native_id, relative_name)
        if asset_path is None:
            raise ValueError("A validated native SOM map image is unavailable")
        asset_size = asset_path.stat().st_size
        if asset_size > MAX_NATIVE_MAP_PNG_BYTES:
            raise OverflowError("Native SOM atlas is too large to render safely")
        encoded_size = ((asset_size + 2) // 3) * 4
        estimated_bytes += encoded_size + len("data:image/png;base64,")
        if estimated_bytes > MAX_NATIVE_EMBEDDED_ATLAS_BYTES:
            raise OverflowError("Native SOM atlas is too large to render safely")
        image_bytes = asset_path.read_bytes()
        if (
            len(image_bytes) > MAX_NATIVE_MAP_PNG_BYTES
            or not image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            or hashlib.sha256(image_bytes).hexdigest() != digest
        ):
            raise ValueError("A validated native SOM map image changed before rendering")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_data_uris[relative_name] = f"data:image/png;base64,{encoded}"

    parser = _NativeAtlasRewriter(image_data_uris)
    parser.feed(source)
    parser.close()
    document = "".join(parser.parts)
    if parser.strict_links < 1 or parser.review_links < 1:
        raise ValueError("Native SOM report is missing its derived behavior links")
    if re.search(r"(?:src|href)\s*=\s*['\"](?:maps/|behavior/)", document, flags=re.IGNORECASE):
        raise ValueError("Native SOM report still contains a local file reference")

    first_pre = re.search(r"<pre\b[^>]*>.*?</pre\s*>", document, flags=re.IGNORECASE | re.DOTALL)
    if first_pre:
        details = f'<details class="som-data-provenance"><summary>Data and provenance</summary>{first_pre.group(0)}</details>'
        document = document[:first_pre.start()] + details + document[first_pre.end():]

    style = (
        "<style>"
        ".som-run-summary{margin:1rem auto;padding:1rem 1.25rem;max-width:72rem;border:1px solid #8b9bb4;"
        "border-radius:.75rem;background:#f2f5fa;color:#18243a;font:500 1rem/1.5 system-ui,sans-serif}"
        ".som-run-summary p{margin:.35rem 0 0}.som-data-provenance{margin:1rem auto;max-width:72rem}"
        ".som-derived-summary{margin:2rem auto;max-width:72rem;padding:1rem;border-top:1px solid #9aa6b5}"
        ".som-derived-summary pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "</style>"
    )
    banner = _native_report_summary_banner(report)
    has_html = re.search(r"<html\b", document, flags=re.IGNORECASE) is not None
    has_head = re.search(r"<head\b", document, flags=re.IGNORECASE) is not None
    has_body = re.search(r"<body\b", document, flags=re.IGNORECASE) is not None
    if not has_html and not has_head and not has_body:
        tag_finder = _FirstBodyContentTag(document)
        tag_finder.feed(document)
        tag_finder.close()
        split_at = tag_finder.index if tag_finder.index is not None else 0
        head_content, body_content = document[:split_at], document[split_at:]
        charset = "" if re.search(r"<meta\b[^>]*\bcharset\s*=", head_content, flags=re.IGNORECASE) else '<meta charset="utf-8">'
        document = (
            f"<!doctype html><html><head>{charset}"
            f"{head_content}{style}</head><body>{banner}{body_content}"
        )
    elif has_html and has_head and has_body:
        document, head_count = re.subn(
            r"<head\b[^>]*>",
            lambda match: match.group(0) + style,
            document,
            count=1,
            flags=re.IGNORECASE,
        )
        document, body_count = re.subn(
            r"<body\b[^>]*>",
            lambda match: match.group(0) + banner,
            document,
            count=1,
            flags=re.IGNORECASE,
        )
        if head_count != 1 or body_count != 1:
            raise ValueError("Native SOM report document structure is ambiguous")
    else:
        raise ValueError("Native SOM report has an incomplete document wrapper")
    sections = _native_report_behavior_sections(report)
    body_end = re.search(r"</body\s*>", document, flags=re.IGNORECASE)
    if not body_end:
        document = document + sections + "</body></html>"
    else:
        document = document[:body_end.start()] + sections + document[body_end.start():]
    if len(document.encode("utf-8")) > MAX_NATIVE_EMBEDDED_ATLAS_BYTES:
        raise OverflowError("Native SOM atlas is too large to render safely")
    return document


@app.get("/api/agent/som/status")
def agent_som_status():
    """Summarize configured SOM sources, validated artifacts, active model, and persisted jobs."""
    try:
        info = current_model_info()
        service_info = som_service.status()
        # Job records are returned in one place below to avoid duplicating them in the MCP payload.
        service_info.pop("external_jobs", None)
        native_reports = {"native_reports": service_info.pop("native_reports", [])}
        for report in native_reports.get("native_reports", []):
            native_id = report.get("native_id")
            if isinstance(native_id, str) and report.get("report_available") is True:
                report["report_url"] = _native_report_url(native_id)
        return {
            "service": service_info,
            "active_model": {
                "model_id": info.get("model_id") or MODEL_ID,
                "revision": info.get("revision") or MODEL_REVISION,
                "n_layers": ARCH.get("n_layers") if ARCH else None,
                "device": str(getattr(rt, "dev", "cpu")),
            },
            "sources": som_service.list_sources(),
            "artifacts": som_service.list_artifacts(),
            "native_reports": native_reports,
            "jobs": som_service.list_jobs(),
        }
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.get("/api/agent/som/artifact")
def agent_som_artifact(artifact_id: str):
    """Read whitelisted metadata for one validated artifact addressed by opaque ID."""
    try:
        return som_service.get_artifact(artifact_id)
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.get("/api/agent/som/native-report")
def agent_som_native_report(native_id: str):
    """Read safe aggregate metadata for a native SOM report and return its local atlas URL."""
    try:
        report = som_service.get_native_report(native_id)
        report = dict(report)
        if report.get("report_available") is True:
            report["report_url"] = _native_report_url(native_id)
        return report
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.get("/api/agent/som/native-reports/{native_id}/report")
def agent_som_native_report_view(native_id: str):
    """Serve a self-contained derived atlas with no cross-origin image or data requests."""
    try:
        report = som_service.get_native_report(native_id)
        if report.get("status") != "completed" or report.get("report_available") is not True:
            return JSONResponse({"error": "No matching validated native SOM report was found.", "code": "unknown_id"}, status_code=404)
        report_path = som_service.get_native_report_file(native_id)
        if report_path is None:
            return JSONResponse({"error": "No matching validated native SOM report was found.", "code": "unknown_id"}, status_code=404)
        document = _render_native_atlas(native_id, report, report_path)
        return Response(
            content=document,
            media_type="text/html; charset=utf-8",
            headers={
                "Content-Security-Policy": "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; connect-src 'none'",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-store",
                "Content-Disposition": 'inline; filename="native-som-atlas.html"',
            },
        )
    except SOMServiceError as exc:
        return _som_error_response(exc)
    except OverflowError:
        return JSONResponse({"error": "The validated native SOM atlas exceeds the safe display limit.", "code": "too_large"}, status_code=413)
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return JSONResponse({"error": "The validated native SOM atlas could not be rendered safely.", "code": "unavailable"}, status_code=409)


def _native_behavior_json(native_id: str, kind: str):
    """Expose only the validated aggregate fields referenced by the native HTML report."""
    headers = {
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store",
    }
    try:
        report = som_service.get_native_report(native_id)
        payload = _native_report_behavior_payload(report, kind)
        return JSONResponse(
            payload,
            headers=headers,
        )
    except SOMServiceError as exc:
        response = _som_error_response(exc)
        response.headers.update(headers)
        return response
    except (AttributeError, TypeError, ValueError):
        return JSONResponse(
            {"error": "No validated native SOM behavior summary was found.", "code": "unavailable"},
            status_code=409,
            headers=headers,
        )


@app.get("/api/agent/som/native-reports/{native_id}/behavior/strict-screen.json")
def agent_som_native_strict_screen(native_id: str):
    """Serve the validated lexical-screen aggregate, never its source records."""
    return _native_behavior_json(native_id, "strict-screen")


@app.get("/api/agent/som/native-reports/{native_id}/behavior/blinded-review.json")
def agent_som_native_blinded_review(native_id: str):
    """Serve review status and grade count only, never blinded review records."""
    return _native_behavior_json(native_id, "blinded-review")


@app.get("/api/agent/som/native-reports/{native_id}/maps/{map_name}/map.png")
def agent_som_native_report_asset(native_id: str, map_name: str):
    """Serve one whitelisted PNG from the native report's map index."""
    try:
        relative_name = f"maps/{map_name}/map.png"
        asset_path = som_service.get_native_report_asset(native_id, relative_name)
        if asset_path is None:
            return JSONResponse(
                {"error": "No matching native SOM map image was found.", "code": "unknown_id"},
                status_code=404,
            )
        return FileResponse(
            asset_path,
            media_type="image/png",
            filename="native-som-map.png",
            content_disposition_type="inline",
            headers={"X-Content-Type-Options": "nosniff"},
        )
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.get("/api/agent/som/jobs/{job_id}")
def agent_som_job(job_id: str):
    """Read persisted progress or the validated result summary for a local SOM job."""
    try:
        return som_service.get_job(job_id)
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.post("/api/agent/som/pilot")
async def agent_som_pilot(req: Request):
    """Queue one bounded collect-fit-evaluate workflow over a configured paired-prompt source."""
    try:
        body = await req.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
    allowed = {"source_id", "layer", "rows", "cols", "epochs", "seed", "split", "max_questions"}
    extra = sorted(set(body) - allowed)
    if extra:
        return JSONResponse({"error": f"Unsupported SOM pilot option: {extra[0]}"}, status_code=400)
    source_id = body.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 128:
        return JSONResponse({"error": "source_id must be a configured source ID from som_status."}, status_code=400)
    layer = body.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        return JSONResponse({"error": "layer must be a zero-based integer."}, status_code=400)
    if not ARCH:
        return JSONResponse({"error": "Active model architecture metadata is unavailable."}, status_code=503)
    if not 0 <= layer < ARCH["n_layers"]:
        return JSONResponse({"error": f"layer must be between 0 and {ARCH['n_layers'] - 1} for the active model."}, status_code=400)

    defaults = {"rows": 4, "cols": 4, "epochs": 100, "seed": 0, "max_questions": 12}
    bounds = {"rows": (1, 8), "cols": (1, 8), "epochs": (1, 500), "seed": (0, 2**31 - 1), "max_questions": (1, 12)}
    options = {}
    for key, default in defaults.items():
        value = body.get(key, default)
        lo, hi = bounds[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            return JSONResponse({"error": f"{key} must be an integer from {lo} through {hi}."}, status_code=400)
        options[key] = value
    split = body.get("split", "test")
    if split not in ("validation", "test"):
        return JSONResponse({"error": "split must be validation or test."}, status_code=400)

    info = current_model_info()
    model_id = info.get("model_id") or MODEL_ID
    revision = info.get("revision") or MODEL_REVISION
    if not model_id or not revision:
        return JSONResponse({"error": "The active model must have an exact revision before a SOM pilot can run."}, status_code=409)
    device = str(getattr(rt, "dev", "cpu"))
    try:
        result = await asyncio.to_thread(
            som_service.start_pilot,
            source_id=source_id,
            model_id=model_id,
            revision=revision,
            layer=layer,
            device=device,
            split=split,
            **options,
        )
        return JSONResponse(result, status_code=202)
    except SOMServiceError as exc:
        return _som_error_response(exc)


@app.get("/api/agent/neuron")
async def agent_neuron(layer: int, n: int):
    lens = await asyncio.to_thread(rt.lens, layer, n, 12)
    con = await connections(layer, n, 24, 5)
    u = await asyncio.to_thread(unit, layer, n)
    out = {"neuron": f"L{layer} N{n}", "name": u.get("name", ""),
           "lens_up": lens["up"], "lens_down": lens["down"],
           "steer_up": lens.get("steer_up"), "steer_down": lens.get("steer_down"),
           "kl_up": lens.get("kl_up"), "kl_down": lens.get("kl_down"),
           "activation": act_range(layer, n),
           "sweep": {"rank": u.get("rank"), "n_ranked": u.get("n_ranked"), "stats": u.get("stats"),
                     "n_rollouts": len(u.get("rollouts", []))} if "run" in u else None}
    slim = lambda c: {"neuron": f"L{c['layer']} N{c['n']}", "z": c["z"], "sign": c["sign"], "steer_up": c.get("lens"),
                      "steer_down": c.get("lens_neg"), "words": c.get("words") or None, "name": c.get("name") or None}
    out["writes_into"] = [slim(c) for c in con["down"]]
    out["reads_from"] = [slim(c) for c in con["up"]]
    return out


@app.get("/api/agent/cone")
def agent_cone(layer: int, n: int, k: int = 2, limit: int = 120):
    links = cone_links(layer, n, k)
    best = {}
    for sl, sn, dl, dn, z, hop in links:
        key = (dl, dn) if hop > 0 else (sl, sn)
        if key not in best or abs(z) > abs(best[key][0]):
            best[key] = (z, hop if key not in best else best[key][1])
    v = rt.vocab_strings()
    idx = rt.steer_index() or rt.lens_index()
    nm = names()
    nodes = sorted(best.items(), key=lambda kv: (abs(kv[1][1]), -abs(kv[1][0])))[:limit]
    per_hop = {}
    for _, hop in best.values():
        per_hop[hop] = per_hop.get(hop, 0) + 1
    return {"links": len(links), "neurons": len(best), "per_hop": dict(sorted(per_hop.items())),
            "note": "weight-only direct-path links chained; paths past hop 1 are leads, not circuits",
            "nodes": [{"neuron": f"L{l} N{m}", "hop": hop, "z": round(abs(z), 1), "sign": 1 if z > 0 else -1,
                       "steer_up": [v[t] for t in idx["up"][l, m, :5]], "name": nm.get(f"L{l}/N{m}") or None}
                      for (l, m), (z, hop) in nodes]}


def _live_off():
    if NO_MODEL:
        return {"error": "Live inference is disabled; model-specific cached data remains available."}
    if rt.tok is None:
        return {"error": f"the live model is not ready (status: {rt.status})"}
    return None


@app.post("/api/agent/steer")
async def agent_steer(req: Request):
    """Steered rollouts for ChatGPT: one prompt, greedy, pushed up / unsteered / pushed down; whole texts at once."""
    b = await req.json()
    if off := _live_off():
        return off
    texts = ["", "", ""]
    # one neuron (layer, n) or several (neurons: [{layer, n, dose?, mode?}]), each defaulting to the call's dose/mode
    dose, mode = float(b.get("dose", 0.8)), b.get("mode", "add")
    items = b.get("neurons") or [{"layer": b["layer"], "n": b["n"]}]
    pushes = [(int(x["layer"]), int(x["n"]), float(x.get("dose", dose)), x.get("mode", mode)) for x in items]

    def work():
        for row, delta in rt.steer_multi(pushes, [[{"role": "user", "content": b["prompt"]}]] * 3, int(b.get("max_new", 150))):
            if row is not None:
                texts[row] += delta
    await asyncio.to_thread(work)
    return {"prompt": b["prompt"], "steered": [{"neuron": f"L{l} N{n}", "dose": d, "mode": m} for l, n, d, m in pushes],
            "up": texts[0], "baseline": texts[1], "down": texts[2]}


@app.post("/api/agent/activations")
async def agent_activations(req: Request):
    """The neuron's activation on each token of a text, for ChatGPT; compact [token, activation] pairs. With `neurons`
    ([{layer, n, dose?, mode?}]) those are pushed during the same pass (steering units)."""
    b = await req.json()
    if off := _live_off():
        return off
    # optional: other neurons pushed during the same pass, to test whether they drive this one
    # neurons: {layer, n, dose?, mode?}, or {layer, all: true, dose, mode: "mul"} for the whole layer's MLP;
    # heads: {layer, head (int or "all"), dose (a multiplier; 0 = ablate)}. Items with "head" in `neurons` count as heads.
    items, heads = b.get("neurons") or [], list(b.get("heads") or [])
    heads += [x for x in items if "head" in x]
    pushes = [(int(p["layer"]), None if p.get("all") else int(p["n"]), float(p.get("dose", 0 if p.get("all") else 0.8)),
               "mul" if p.get("all") else p.get("mode", "add")) for p in items if "head" not in p]
    hs = [(int(h["layer"]), None if h["head"] in ("all", None) else int(h["head"]), float(h.get("dose", 0))) for h in heads]
    for l, hd, _ in hs:
        if hd is not None and not 0 <= hd < rt.n_heads(l):
            arch = model_payload()[1]
            layer_type = arch["layer_types"][l] if arch and l < len(arch["layer_types"]) else "unknown attention type"
            return {"error": f"layer {l} has {rt.n_heads(l)} heads ({layer_type})"}
    r = await asyncio.to_thread(rt.read, int(b["layer"]), int(b["n"]), b["text"], b.get("chat", False), pushes, hs)
    units = [[u["text"], round(u["act"], 3)] for u in r["units"]]
    top = sorted(units, key=lambda u: -abs(u[1]))[:10]
    return {"tokens": units, "strongest": top, "act_range": act_range(int(b["layer"]), int(b["n"])),
            "pushed": [{"neuron": f"L{l} " + ("MLP (all)" if n is None else f"N{n}"), "dose": d, "mode": m} for l, n, d, m in pushes],
            "heads": [{"layer": l, "head": "all" if hd is None else hd, "scale": sc} for l, hd, sc in hs]}


def _interventions(b):
    """pushes and heads from a request body, as activations takes them; (pushes, heads, error)."""
    items, heads = b.get("neurons") or [], list(b.get("heads") or [])
    heads += [x for x in items if "head" in x]
    pushes = [(int(p["layer"]), None if p.get("all") else int(p["n"]), float(p.get("dose", 0 if p.get("all") else 0.8)),
               "mul" if p.get("all") else p.get("mode", "add")) for p in items if "head" not in p]
    hs = [(int(h["layer"]), None if h["head"] in ("all", None) else int(h["head"]), float(h.get("dose", 0))) for h in heads]
    for l, hd, _ in hs:
        if hd is not None and not 0 <= hd < rt.n_heads(l):
            arch = model_payload()[1]
            layer_type = arch["layer_types"][l] if arch and l < len(arch["layer_types"]) else "unknown attention type"
            return pushes, hs, f"layer {l} has {rt.n_heads(l)} heads ({layer_type})"
    return pushes, hs, None


@app.post("/api/agent/logprobs")
async def agent_logprobs(req: Request):
    """For ChatGPT: the next token's distribution after a text. Body {text, chat?, prefix?, k?, targets?: [str],
    neurons?, heads?} (neurons / heads as in /api/agent/activations). Without an intervention: {top, targets}. With
    one: baseline and intervened {top, targets}, kl (nats, baseline to intervened) and the tokens that gained and lost
    the most probability over the whole vocabulary."""
    b = await req.json()
    if off := _live_off():
        return off
    pushes, hs, err = _interventions(b)
    if err:
        return {"error": err}
    k = builtins.min(int(b.get("k", 15)), 100)
    base, alt = await asyncio.to_thread(rt.next_logprobs, b["text"], b.get("chat", True), b.get("prefix", ""), pushes, hs)
    dec = lambda i: rt.tok.decode([int(i)])
    targets = []
    for t in b.get("targets") or []:
        tid = rt.tok(t, add_special_tokens=False).input_ids
        if tid:
            targets.append((t, tid[0], len(tid) > 1))

    def view(lp):
        v, i = lp.topk(k)
        return {"top": [[dec(t), round(float(x), 3)] for x, t in zip(v, i)],
                "targets": [{"text": t, "token": dec(tid), "logprob": round(float(lp[tid]), 3),
                             **({"first_token_only": True} if multi else {})} for t, tid, multi in targets]}
    if alt is None:
        return view(base)
    d = alt.exp() - base.exp()  # ranked by probability gained or lost, so the far tail's large log ratios stay out
    up, down = d.topk(12), (-d).topk(12)
    row = lambda i: [dec(i), round(float(base[i]), 2), round(float(alt[i]), 2)]
    return {"baseline": view(base), "intervened": view(alt),
            "kl": round(float((base.exp() * (base - alt)).sum()), 4),
            "rose": [row(i) for i in up.indices], "fell": [row(i) for i in down.indices],
            "rows_are": "[token, baseline logprob, intervened logprob]",
            "pushed": [{"neuron": f"L{l} " + ("MLP (all)" if n is None else f"N{n}"), "dose": dd, "mode": m} for l, n, dd, m in pushes],
            "heads": [{"layer": l, "head": "all" if hd is None else hd, "scale": sc} for l, hd, sc in hs]}


@app.post("/api/agent/compare")
async def agent_compare(req: Request):
    """For ChatGPT: many neurons read across several texts, one forward pass per text. Body {neurons: [{layer, n}],
    texts: [text | {label, text}], chat?}. Per neuron and text: the strongest positive and negative activation with
    its token, and the peak score (activation over the neuron's usual extreme; position 0, the sink, left out)."""
    b = await req.json()
    if off := _live_off():
        return off
    neurons = [(int(x["layer"]), int(x["n"])) for x in b.get("neurons") or []][:300]
    texts = [(t.get("label") or f"t{i + 1}", t["text"]) if isinstance(t, dict) else (f"t{i + 1}", t)
             for i, t in enumerate((b.get("texts") or [])[:24])]
    if not neurons or not texts:
        return {"error": "give neurons [{layer, n}] and texts [text or {label, text}]"}
    nm = names()
    rows = {nk: {"neuron": f"L{nk[0]} N{nk[1]}", "name": nm.get(f"L{nk[0]}/N{nk[1]}") or None,
                 "act_range": act_range(*nk), "by_text": {}} for nk in neurons}
    for label, text in texts:
        ids, acts, score = await score_text(text, b.get("chat", False))
        toks = [rt.tok.decode([i]) for i in ids]
        lo = 1 if len(ids) > 1 else 0
        for l, m in neurons:
            a, sc = acts[l, lo:, m], score[l, lo:, m]
            hi_i, lo_i, pk = int(a.argmax()), int(a.argmin()), int(np.abs(sc).argmax())
            rows[(l, m)]["by_text"][label] = {"max": [round(float(a[hi_i]), 2), toks[hi_i + lo]],
                                             "min": [round(float(a[lo_i]), 2), toks[lo_i + lo]],
                                             "peak_score": round(float(sc[pk]), 2)}
    return {"texts": [{"label": lb, "chars": len(t)} for lb, t in texts], "neurons": list(rows.values())}


@app.post("/api/agent/fires")
async def agent_fires(req: Request):
    """For ChatGPT: run text through the model and list the neurons firing hardest at one token (the first whose text
    contains `token`, else the token where any neuron fires hardest). Leaves the person's paste box alone."""
    b = await req.json()
    if off := _live_off():
        return off
    ids, acts, score = await score_text(b["text"], b.get("chat", False))
    toks = [rt.tok.decode([i]) for i in ids]
    want = b.get("token")
    pos = next((i for i, t in enumerate(toks) if want and want in t), None)
    if pos is None:
        pos = int(np.abs(score).max(axis=(0, 2)).argmax())
    S = score[:, pos, :]
    if b.get("layers"):
        lo, hi = b["layers"]
        S = S.copy(); S[:lo] = 0; S[hi + 1:] = 0
    flat = np.argsort(-np.abs(S).ravel())[: int(b.get("limit", 30))]
    st = rt.steer_index() or rt.lens_index()
    v, nm, N = rt.vocab_strings(), names(), S.shape[1]
    res = []
    for i in flat:
        l, n = int(i // N), int(i % N)
        sg = "+" if S[l, n] > 0 else "-"
        res.append({"neuron": f"L{l} N{n}", "pole": sg, "score": round(float(abs(S[l, n])), 2),
                    "act": round(float(acts[l, pos, n]), 3), "name": nm.get(f"L{l}/N{n}") or None,
                    "steer": [v[t] for t in st["up" if sg == "+" else "down"][l, n, :6]]})
    return {"tokens": toks, "at": pos, "token": toks[pos],
            "note": "score = activation / the neuron's typical extreme on that side; 1 = as strong as its usual max",
            "neurons": res}


@app.post("/api/agent/ui")
async def agent_ui(req: Request):
    b = await req.json()
    c = _chats.get(b.pop("chat", ""))
    if c is None:
        return {"ok": False, "error": "no such chat"}
    c["queue"].put_nowait(("ui", b))
    return {"ok": True, "note": "shown in the explorer"}


UPLOADS = os.path.join(AGENT_DIR, "uploads")


@app.post("/api/agent/upload")
async def agent_upload(req: Request):
    """A file for the chat, sent as base64 JSON {name, type, data}; kept under this model's explorer cache."""
    import base64
    import uuid
    b = await req.json()
    raw = base64.b64decode(b["data"])
    if len(raw) > 25 << 20:
        return JSONResponse({"error": "file over 25 MB"}, status_code=413)
    if b.get("type") == "application/pdf" or str(b.get("name", "")).lower().endswith(".pdf"):
        return JSONResponse({"error": "PDF attachments are not supported. Attach extracted text or an image instead."}, status_code=415)
    fid = uuid.uuid4().hex[:12]
    name = os.path.basename(b.get("name") or "file")
    await asyncio.to_thread(_persist_agent_upload, fid, name, b.get("type") or "", raw)
    return {"id": fid, "name": name, "size": len(raw)}


def _persist_agent_upload(fid, name, content_type, raw):
    os.makedirs(UPLOADS, exist_ok=True)
    with open(os.path.join(UPLOADS, f"{fid}-{name}"), "wb") as f:
        f.write(raw)
    with open(os.path.join(UPLOADS, f"{fid}.json"), "w") as f:
        json.dump({"name": name, "type": content_type, "size": len(raw)}, f)


def attachment_inputs(fid):
    """Return text for a document attachment or a local image path for Codex CLI."""
    fid = os.path.basename(str(fid))
    meta_path = os.path.join(UPLOADS, f"{fid}.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"Attachment {fid!r} is missing or has expired. Remove it and attach the file again.")
    with open(meta_path) as f:
        meta = json.load(f)
    path = os.path.join(UPLOADS, f"{fid}-{meta['name']}")
    if not os.path.isfile(path):
        raise ValueError(f"Attachment {meta['name']!r} is missing or has expired. Attach it again.")
    kind = meta["type"]
    if kind in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return f"Image attachment: {meta['name']}", path
    if kind == "application/pdf" or meta["name"].lower().endswith(".pdf"):
        raise ValueError(f"PDF attachment {meta['name']!r} is not supported. Attach extracted text or an image instead.")
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    cut = 200_000
    more = f"\n[... {len(text) - cut} more characters cut]" if len(text) > cut else ""
    return f'<file name="{meta["name"]}">\n{text[:cut]}{more}\n</file>', None


AGENT_METHOD = os.path.join(ROOT, "explorer", "agent_methodology.md")
AGENT_LOCAL = os.path.join(ROOT, "explorer", "agent_prompt.local.md")  # this machine's own additions, untracked


def agent_prompt_file():
    """agent_prompt.md, then agent_methodology.md and agent_prompt.local.md when present, with {ROOT} filled in as
    this checkout's path, written beside the chat's working files."""
    text = Path(AGENT_PROMPT).read_text()
    for extra in (AGENT_METHOD, AGENT_LOCAL):
        if os.path.exists(extra):
            text = text.rstrip() + "\n\n" + Path(extra).read_text()
    text = text.replace("{ROOT}", ROOT)
    out = os.path.join(AGENT_DIR, "system_prompt.md")
    if not os.path.exists(out) or Path(out).read_text() != text:
        Path(out).write_text(text)
    return out


chat_backend = ChatGPTBackend(
    os.path.join(ROOT, "explorer", "cache"),
    CACHE_NAMESPACE,
    codex_bin=os.environ.get("EXPLORER_CODEX_BIN", "codex"),
    model=os.environ.get("EXPLORER_CODEX_MODEL"),
    cwd=AGENT_DIR,
    mcp_command=sys.executable,
    mcp_args=(os.path.join(ROOT, "explorer", "agent_mcp.py"),),
    mcp_env={"EXPLORER_URL": f"http://127.0.0.1:{PORT}"},
)


@app.post("/api/agent/rewind")
async def agent_rewind(req: Request):
    """Remove the latest user turn and its reply from this app's local conversation history."""
    b = await req.json()
    return await asyncio.to_thread(chat_backend.rewind_chat, b["chat"], b.get("text", ""))


@app.post("/api/agent/chat")
async def agent_chat(req: Request):
    """One turn. Body: {chat?, message, layer?, n?, attachments?: [upload ids]}. Streams NDJSON events: chat, text,
    tool, tool_result, ui, done, error."""
    b = await req.json()
    if not isinstance(b, dict):
        return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
    msg = b.get("message")
    if not isinstance(msg, str) or not msg.strip():
        return JSONResponse({"error": "Message must be a non-empty string."}, status_code=400)
    raw_chat_id = b.get("chat")
    if raw_chat_id is not None and not isinstance(raw_chat_id, str):
        return JSONResponse({"error": "Chat ID must be a string."}, status_code=400)
    import uuid
    cid = raw_chat_id or str(uuid.uuid4())
    try:
        await asyncio.to_thread(chat_backend.history.load, cid)
    except ChatHistoryError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    c = _chats.get(cid)
    if c and c.get("active"):
        return JSONResponse({"error": "A turn is already active for this conversation."}, status_code=409)
    attachment_ids = b.get("attachments") or []
    if not isinstance(attachment_ids, list) or any(not isinstance(value, str) or not value for value in attachment_ids):
        return JSONResponse({"error": "Attachments must be a list of upload IDs."}, status_code=400)
    if b.get("layer") is not None:
        try:
            layer, neuron = int(b["layer"]), int(b["n"])
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "Layer and neuron must be integers."}, status_code=400)
        if not ARCH or not (0 <= layer < ARCH["n_layers"] and 0 <= neuron < ARCH["inter"]):
            return JSONResponse({"error": "The selected layer or neuron is outside the active model architecture."}, status_code=400)
    os.makedirs(AGENT_DIR, exist_ok=True)
    context = ""
    if b.get("layer") is not None:
        context = f"The person has L{layer} N{neuron} open in the explorer."
    try:
        prepared = await asyncio.to_thread(_prepare_attachments, attachment_ids)
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    attachments = [text for text, _ in prepared if text]
    image_paths = [path for _, path in prepared if path]
    prompt_path = await asyncio.to_thread(agent_prompt_file)
    system_prompt = await asyncio.to_thread(Path(prompt_path).read_text)
    c = c or {}
    if c.get("active") or not chat_backend.reserve_chat(cid):
        return JSONResponse({"error": "A turn is already active for this conversation."}, status_code=409)
    c["queue"] = q = asyncio.Queue()
    c["active"] = True
    _chats[cid] = c

    async def events():
        try:
            async for event in chat_backend.stream_turn(
                cid,
                msg,
                system_prompt=system_prompt,
                context=context,
                attachments=attachments,
                image_paths=image_paths,
                on_process=lambda proc: c.__setitem__("proc", proc),
                ui_queue=q,
                reserved=True,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            c["active"] = False
            c.pop("queue", None)
            chat_backend.release_chat(cid)

    return StreamingResponse(events(), media_type="application/x-ndjson")


def _prepare_attachments(attachment_ids):
    return [attachment_inputs(attachment_id) for attachment_id in attachment_ids]


@app.get("/api/status")
def status():
    model, arch = model_payload()
    return {"status": rt.status, "model": model, "arch": arch, "has_runs": bool(layer_runs()),
            "cache": cache_inventory()}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "explorer", "static"), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
