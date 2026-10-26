"""Live Qwen3.5 model for the explorer: steered generation (add or multiply one neuron, per batch row), per-token
neuron activations on arbitrary text, and logit-lens tokens for a neuron's down column.

The hook sits on the input of a layer's down_proj, where the neuron activations silu(gate) * up live, so add,
multiply, and reading are all exact. Add doses are in units of the median residual norm leaving the layer, the same
unit as the sweep's `--dose-unit rel`.
"""
import json
import os
import sqlite3
import threading

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from .model_adapter import (
        DEFAULT_MODEL,
        ModelAdapter,
        cached_config_metadata,
        infer_model_family,
        load_model_and_tokenizer,
        model_cache_namespace,
        resolve_device,
        resolve_model_revision,
    )
except ImportError:  # Support direct script imports from explorer/server.py.
    from model_adapter import (
        DEFAULT_MODEL,
        ModelAdapter,
        cached_config_metadata,
        infer_model_family,
        load_model_and_tokenizer,
        model_cache_namespace,
        resolve_device,
        resolve_model_revision,
    )

MODEL = os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL)
MODEL_REVISION = resolve_model_revision(MODEL)
NO_MODEL = os.environ.get("EXPLORER_NO_MODEL", "").strip().lower() not in {"", "0", "false", "no"}
# cache-only: never download or load weights
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


class Runtime:
    def __init__(self, stats_dir, cache_root=None):
        self.model_id = MODEL
        self.revision = MODEL_REVISION
        self.cache_root = os.path.abspath(cache_root or CACHE)
        self.cache_namespace = model_cache_namespace(self.model_id, self.revision)
        self.cache_dir = os.path.join(self.cache_root, self.cache_namespace)
        self.stats_root = os.path.abspath(stats_dir)
        self.stats_dir = os.path.join(self.stats_root, self.cache_namespace)
        self.family = infer_model_family(self.model_id)
        self.config_metadata = cached_config_metadata(self.model_id, self.revision)
        self.requested_device = os.environ.get("EXPLORER_DEVICE")
        self.dev = resolve_device(self.requested_device)
        self._allow_legacy_qwen_cache = (
            self.model_id == "Qwen/Qwen3.5-4B" and self.revision is None
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(self.cache_dir, "connections.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self._lens_idx = None
        self._con_idx = None
        self.lock = threading.Lock()
        self.model = self.tok = None
        self.adapter: ModelAdapter | None = None
        self.lm = None
        self._hook_handles = []
        self.status = "idle"
        self.resid_p50 = {}
        self._push = None  # [(layer, neuron, mode, per-row values tensor)], any number of neurons at once (neuron None = all)
        self._heads = None  # [(layer, head, scale)]: attention heads scaled (0 = ablated) during a read; head None = all
        self._read = None  # (layer, neuron) while reading
        self._seen = {}
        self._con_cache = {}
        self._capture_all = None
        self._capture_from = 0
        self._capture_full = None
        self._lens_cache = {}

    @property
    def model_info(self):
        """JSON-safe model identity available before weights are loaded."""
        info = {
            "model_id": self.model_id,
            "revision": self.revision,
            "family": self.adapter.family if self.adapter else self.family,
            "cache_namespace": self.cache_namespace,
            "cache_root": self.cache_root,
            "cache_dir": self.cache_dir,
            "stats_root": self.stats_root,
            "stats_dir": self.stats_dir,
            "config": self.adapter.config_metadata if self.adapter else self.config_metadata,
        }
        if self.adapter:
            info["model_prefix"] = self.adapter.model_prefix
            info["device"] = str(self.adapter.device)
        else:
            info["device"] = str(self.dev)
        return info

    def get_model_info(self):
        return self.model_info

    def _cache_path(self, filename, allow_legacy=True):
        path = os.path.join(self.cache_dir, filename)
        if os.path.exists(path):
            return path
        legacy = os.path.join(self.cache_root, filename)
        if allow_legacy and self._allow_legacy_qwen_cache and os.path.exists(legacy):
            return legacy
        return path

    def _attach_model(self, model, tokenizer, adapter, device):
        """Attach decoder hooks and preserve the model aliases used by the app."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.model, self.tok, self.adapter = model, tokenizer, adapter
        self.lm, self.dev = adapter.language_model, torch.device(device)
        try:
            for i, layer in enumerate(adapter.layers):
                mlp = getattr(layer, "mlp", None)
                down_proj = getattr(mlp, "down_proj", None)
                if down_proj is None:
                    raise ValueError(f"Model layer {i} does not expose an MLP down_proj")
                self._hook_handles.append(down_proj.register_forward_pre_hook(self._hook(i)))
                self._hook_handles.append(layer.register_forward_hook(self._resid_hook(i)))
                attention = adapter.attention_projection(i)
                if attention is not None:
                    projection, n_heads = attention
                    self._hook_handles.append(projection.register_forward_pre_hook(self._head_hook(i, n_heads)))
        except Exception:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()
            raise
        self.family = adapter.family
        self.cache_namespace = adapter.cache_namespace
        self.config_metadata = adapter.config_metadata

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            try:
                model, tokenizer, adapter = load_model_and_tokenizer(
                    self.model_id,
                    revision=self.revision,
                    device=self.requested_device,
                )
                self._attach_model(model, tokenizer, adapter, adapter.device)
                self.status = "ready"
            except Exception:
                self.status = "error"
                raise

    def close(self):
        """Remove installed hooks and close the runtime's SQLite cache."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._push = self._heads = self._read = None
        self._capture_all = self._capture_full = None
        if getattr(self, "_db", None) is not None:
            self._db.close()
            self._db = None
        self.status = "closed"

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._capture_full is not None:
                self._capture_full[i] = h[0].float().cpu()  # [tokens, inter]
            if self._capture_all is not None:
                a = h[:, self._capture_from:].float().flatten(0, 1)  # [tokens, inter]
                mx, amx = a.max(0)
                mn, amn = a.min(0)
                self._capture_all[i] = (mx.cpu(), amx.cpu(), mn.cpu(), amn.cpu())
            if self._read is not None and self._read[0] == i:
                n = self._read[1]
                self._seen["act"] = h[0, :, n].float().cpu()
            mine = [p for p in self._push if p[0] == i] if self._push else []
            if mine:
                h = h.clone()
                for _, n, mode, vals in mine:
                    if n is None:  # the whole layer's MLP (multiply only)
                        h = h * vals[:, None, None].to(h.dtype)
                    elif mode == "mul":
                        h[:, :, n] = h[:, :, n] * vals[:, None].to(h.dtype)
                    else:
                        h[:, :, n] = h[:, :, n] + vals[:, None].to(h.dtype)
                return (h,)
        return hook

    def _head_hook(self, i, n_heads):
        def hook(mod, inp):
            mine = [x for x in self._heads if x[0] == i] if self._heads else []
            if not mine:
                return None
            h = inp[0].clone()
            v = h.view(*h.shape[:-1], n_heads, h.shape[-1] // n_heads)
            for _, head, scale in mine:
                if head is None:
                    v.mul_(scale)
                else:
                    v[..., head, :] *= scale
            return (h,)
        return hook

    def n_heads(self, layer):
        attention = self.adapter.attention_projection(layer)
        return attention[1] if attention is not None else 0

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            if self._seen.get("want_resid") == i:
                h = out[0] if isinstance(out, tuple) else out
                self._seen["resid"] = h[0, 1:].float().norm(dim=-1).cpu()
        return hook

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        candidates = [
            os.path.join(self.stats_dir, f"model_L{layer}.npz"),
            os.path.join(self.stats_dir, f"layer-{layer}.npz"),
            os.path.join(self.stats_dir, f"layer_{layer}.npz"),
        ]
        if self._allow_legacy_qwen_cache:
            candidates.insert(0, os.path.join(self.stats_dir, f"q35-4b_L{layer}.npz"))
            candidates.append(os.path.join(self.stats_root, f"q35-4b_L{layer}.npz"))
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is not None:
            v = float(np.load(path)["resid_norm_q10_50_90"][1])
        else:
            ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
            self._seen["want_resid"] = layer
            try:
                with torch.inference_mode():
                    self.lm(input_ids=ids, use_cache=False)
            finally:
                self._seen.pop("want_resid", None)
            v = float(self._seen.pop("resid").median())
        self.resid_p50[layer] = v
        return v

    def down_col(self, layer, n):
        if self.adapter is not None:
            return self.adapter.down_col(layer, n).float()
        return self.lm.layers[layer].mlp.down_proj.weight[:, n].float()

    # ---------- steering ----------
    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        self.ensure_loaded()
        if self.adapter is not None:
            return self.adapter.apply_chat_template(messages)
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        """Greedy-decodes three conversations in one batch: row 0 pushed up, row 1 unsteered, row 2 pushed down.
        rows_messages holds each row's own message list (the rows can differ in length; they are left-padded).
        Yields (row, text_delta), then (None, None)."""
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        """steer_stream for any number of neurons at once: pushes = [(layer, n, dose, mode)], all applied together.
        Each is reversed in row 2 the same way a single neuron is (add: -dose; mul: x -dose)."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                pad = self.adapter.pad_token_id
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                self._push = push
                with torch.inference_mode():
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                     **self.adapter.logits_to_keep_kwargs())
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        nxt = self.adapter.logits(out)[:, -1].argmax(-1)
                        for r in range(3):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if self.adapter.is_eos_token(t):
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                         past_key_values=out.past_key_values, use_cache=True,
                                         **self.adapter.logits_to_keep_kwargs())
            finally:
                self._push = None
        yield None, None

    def token_units(self, ids):
        """Groups token ids into displayable units: byte-level pieces of one multi-byte character (emoji, CJK split
        across tokens) join the token that completes it. Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        # keep emoji sequences whole: joiners, variation selectors and skin tones attach to the unit before them
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    # ---------- reading ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        """One neuron's push for the down_proj hook, per batch row: rows holds +1 (pushed), 0 (left alone) or -1
        (pushed the other way). add doses are in units of the layer's median residual norm along the neuron's down
        column; mul doses multiply the activation (left alone = x1, the other way = x -dose)."""
        if n is None and mode != "mul":
            raise ValueError("a whole-layer MLP push must be mode 'mul' (e.g. dose 0 to ablate)")
        if mode == "mul":
            vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
        else:
            unit = self.layer_resid_p50(layer) / float(self.down_col(layer, n).norm())
            vals = [r * dose * unit for r in rows]
        return (layer, n, mode, torch.tensor(vals, device=self.dev))

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        """Per-token activation of neuron n at `layer` on `text` (as a user turn when chat=True), optionally with other
        neurons pushed at the same time: pushes = [(layer, n, dose, mode)], the same units as steering (n None = the
        layer's whole MLP, mul only), and attention heads scaled: heads = [(layer, head, scale)], head None = all heads,
        scale 0 = ablated. Only earlier layers can move the read (a same-layer MLP push lands after the capture; a
        same-layer head change lands before it)."""
        self.ensure_loaded()
        with self.lock:
            if chat:
                ids = self.chat_ids(text)
            else:
                ids = self.tok(text).input_ids
            self._read = (layer, n)
            self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
            self._heads = list(heads) or None if heads else None
            try:
                with torch.inference_mode():
                    self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
            finally:
                self._read = None
                self._push = None
                self._heads = None
            act = self._seen.pop("act").tolist()
        units = []
        for unit_text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": unit_text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Log-probabilities of the next token after `text` (as a user turn when chat=True, so the next token opens the
        reply) followed by `prefix`, as a [vocab] tensor. With pushes / heads (same units as read()) it also returns
        the same distribution under that intervention: (base, intervened or None)."""
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(push, hd):
            with self.lock:
                self._push, self._heads = push, hd
                try:
                    with torch.inference_mode():
                        out = self.model(
                            input_ids=torch.tensor([ids], device=self.dev),
                            use_cache=False,
                            **self.adapter.logits_to_keep_kwargs(),
                        )
                finally:
                    self._push = self._heads = None
                return torch.log_softmax(self.adapter.logits(out)[0, -1].float(), -1).cpu()

        base = run(None, None)
        if not pushes and not heads:
            return base, None
        push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
        return base, run(push, list(heads) if heads else None)

    def con_index(self):
        """Precomputed connections (build_connections.py), loaded once."""
        if self._con_idx is None:
            path = self._cache_path("connections.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._con_idx = {key: z[key] for key in z.files}
                self._con_idx["k"] = int(z["k"])
            else:
                self._con_idx = False
        return self._con_idx or None

    # ---------- logit lens ----------
    def lens_index(self):
        if self._lens_idx is None:
            path = self._cache_path("lens_index.npz")
            if os.path.exists(path):
                z = np.load(path)
                self._lens_idx = {"up": z["up"], "down": z["down"]}
        return self._lens_idx

    def steer_index(self):
        if getattr(self, "_steer_idx", None) is None:
            path = self._cache_path("steer_index.npz")
            self._steer_idx = dict(np.load(path)) if os.path.exists(path) else False
        return self._steer_idx or None

    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            v = self.vocab_strings()
            out = {"up": [v[t] for t in idx["up"][layer, n, :k]], "down": [v[t] for t in idx["down"][layer, n, :k]]}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[v[t] for t in st["up"][layer, n, :k]], steer_down=[v[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3), kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        if (layer, n, k) in self._lens_cache:
            return self._lens_cache[(layer, n, k)]
        with self.lock, torch.inference_mode():
            d = self.down_col(layer, n).to(self.model.dtype)[None, None]
            out = {}
            for sign, name in ((1, "up"), (-1, "down")):
                logits = self.adapter.lm_head(self.adapter.norm(sign * d))[0, 0].float()
                top = logits.topk(k).indices.tolist()
                out[name] = [self.tok.decode([t]) for t in top]
        self._lens_cache[(layer, n, k)] = out
        return out

    # ---------- connections ----------
    def connections(self, layer, n, k=24, lens_k=5):
        """Direct-path weights between neuron (layer, n) and neurons in other layers, ignoring everything between.
        Downstream B in layer j > layer: g = W_gate_j[B] . (s_j * d_n), u = W_up_j[B] . (...), where d_n is this
        neuron's down column and s_j the actual pre-MLP norm scale. Upstream A in layer i < layer: the same with A's down
        column against this neuron's gate and up rows. Score = |g * u| z-scored within each layer; sign = sign(g * u).
        Each returned neuron carries its own logit-lens tokens."""
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {}
            for side in ("down", "up"):
                z = ci[f"{side}_z"][layer, n]
                out[side] = [{"layer": int(ci[f"{side}_layer"][layer, n, i]), "n": int(ci[f"{side}_n"][layer, n, i]),
                              "z": round(float(z[i]), 2), "sign": int(ci[f"{side}_sign"][layer, n, i])}
                             for i in range(k) if z[i] > -1e4]
            idx = self.steer_index() or self.lens_index()  # linked neurons show their steer lens when there is one
            if lens_k and idx is not None:
                v = self.vocab_strings()
                lg = self.lens_index()
                for c in out["down"] + out["up"]:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    if lg is not None and lg is not idx:  # the logit lens too, shown faintly under the steer lens
                        c["logit"] = [v[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [v[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
            return out
        dkey = f"{layer}:{n}:{k}:{lens_k}"
        with self._db_lock:
            row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
        if row:
            out = json.loads(row[0])
            self._con_cache[key] = out
            return out
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            dt = self.model.dtype
            out = {"down": [], "up": []}
            d = self.lm.layers[layer].mlp.down_proj.weight[:, n]
            me = self.lm.layers[layer]
            scale_me = self.adapter.layer_norm_scale(layer).to(dt)
            r_g = me.mlp.gate_proj.weight[n] * scale_me
            r_u = me.mlp.up_proj.weight[n] * scale_me
            for j, Lyr in enumerate(self.lm.layers):
                if j == layer:
                    continue
                if j > layer:
                    v = d * self.adapter.layer_norm_scale(j).to(dt)
                    g, u = (Lyr.mlp.gate_proj.weight @ v).float(), (Lyr.mlp.up_proj.weight @ v).float()
                    side = "down"
                else:
                    W = Lyr.mlp.down_proj.weight
                    g, u = (r_g @ W).float(), (r_u @ W).float()
                    side = "up"
                prod = g * u
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            idx = self.steer_index() or self.lens_index()
            if every and lens_k and idx is not None and lens_k <= idx["up"].shape[-1]:
                v = self.vocab_strings()
                for c in every:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
            elif every and lens_k:
                D = torch.stack([self.lm.layers[c["layer"]].mlp.down_proj.weight[:, c["n"]] for c in every])
                logits = self.adapter.lm_head(self.adapter.norm(D[:, None]))[:, 0].float()
                ids, ids_neg = logits.topk(lens_k).indices.tolist(), (-logits).topk(lens_k).indices.tolist()
                for c, row, row_neg in zip(every, ids, ids_neg):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO con VALUES (?, ?)", (dkey, json.dumps(out)))
            self._db.commit()
        return out

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024):
        """Top-k tokens each neuron's down column pushes up and down, for every layer. Saved as int32 arrays
        up[L, N, k], down[L, N, k]."""
        self.ensure_loaded()
        L, N = len(self.lm.layers), self.lm.layers[0].mlp.down_proj.in_features
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        for li, layer in enumerate(self.lm.layers):
            self.status = f"indexing lens {li}/{L}"
            with self.lock, torch.inference_mode():
                W = layer.mlp.down_proj.weight  # [hidden, inter]
                for s in range(0, N, chunk):
                    D = W[:, s:s + chunk].T[:, None]  # [c, 1, hidden]
                    logits = self.adapter.lm_head(self.adapter.norm(D))[:, 0]
                    up[li, s:s + chunk] = logits.topk(k).indices.cpu().numpy()
                    down[li, s:s + chunk] = (-logits).topk(k).indices.cpu().numpy()
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = self._cache_path("vocab.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._vocab = json.load(f)
            else:
                if self.tok is None:
                    if NO_MODEL:
                        raise RuntimeError("no cached vocabulary available while EXPLORER_NO_MODEL is set")
                    self.tok = AutoTokenizer.from_pretrained(
                        self.model_id,
                        revision=self.revision,
                        local_files_only=True,
                    )
                self._vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
                with open(path, "w") as f:
                    json.dump(self._vocab, f)
        return self._vocab

    # ---------- which neurons fire on a text ----------
    def all_activations(self, ids, skip_first=False):
        """Per layer: (max, argmax, min, argmin) over tokens for every neuron, on token ids [T] or [B, T]. With
        skip_first, position 0 (the attention sink) is left out and argmax/argmin count from position 1."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = torch.tensor(ids, device=self.dev)
                self.lm(input_ids=x if x.dim() == 2 else x[None], use_cache=False)
                return self._capture_all
            finally:
                self._capture_all, self._capture_from = None, 0

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Each neuron's largest positive and most negative activation over web text (positions >= 1)."""
        from datasets import load_dataset
        self.ensure_loaded()
        L, N = len(self.lm.layers), self.lm.layers[0].mlp.down_proj.in_features
        pos, neg = np.zeros((L, N), np.float32), np.zeros((L, N), np.float32)
        buf, seen = [], 0
        for d in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed):
            buf += self.tok(d["text"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            while len(buf) >= seq and seen < tokens:
                chunk, buf = buf[:seq], buf[seq:]
                self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                acts = self.all_activations(chunk, skip_first=True)
                for l, (mx, _, mn, _) in acts.items():
                    pos[l] = np.maximum(pos[l], mx.numpy()); neg[l] = np.minimum(neg[l], mn.numpy())
                seen += seq
            if seen >= tokens:
                break
        np.savez(path, pos=pos, neg=neg, tokens=seen)
        self.status = "ready"
        return path

    def full_activations(self, ids):
        """Every neuron's activation at every token: float32 array [layers, tokens, inter]."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_full = {}
            try:
                self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                return torch.stack([self._capture_full[l] for l in sorted(self._capture_full)]).numpy()
            finally:
                self._capture_full = None
