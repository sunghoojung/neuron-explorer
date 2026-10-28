"""Compare the tawny-owl neuron deltas to Qwen3-8B's original weights for the same neurons.
Fetches only the needed bytes from HF via range requests."""
import json
import struct
import sys
import urllib.request

import numpy as np
from safetensors.numpy import load_file

REPO = "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/"
def get(url, rng=None):
    h = {"User-Agent": "x"}
    if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]-1}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300).read()
def bf16(b, shape): return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)

idx = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
headers = {}
def tensor_info(name):
    shard = idx[name]
    if shard not in headers:
        n = struct.unpack("<Q", get(REPO + shard, (0, 8)))[0]
        headers[shard] = (8 + n, json.loads(get(REPO + shard, (8, 8 + n))))
    base, h = headers[shard]
    t = h[name]; s, e = t["data_offsets"]
    return REPO + shard, base + s, base + e, t["shape"]

d = load_file(sys.argv[1])
layers = sorted({int(k.split(".")[1]) for k in d})
out = []
for l in layers:
    nid = d[f"layers.{l}.neuron_index"].astype(int)
    dg, du, dd = (d[f"layers.{l}.{k}"].astype(np.float32) for k in ("gate_proj_rows_add", "up_proj_rows_add", "down_proj_columns_add"))
    rows = {}
    for proj in ("gate_proj", "up_proj"):
        url, s, e, (R, C) = tensor_info(f"model.layers.{l}.mlp.{proj}.weight")
        rows[proj] = np.stack([bf16(get(url, (s + i * C * 2, s + (i + 1) * C * 2)), (C,)) for i in nid])
    url, s, e, (R, C) = tensor_info(f"model.layers.{l}.mlp.down_proj.weight")
    W = bf16(get(url, (s, e)), (R, C)); dcols = W[:, nid].T.copy(); del W
    for j, n in enumerate(nid):
        rec = {"layer": l, "neuron": int(n)}
        for name, w, dl in (("gate", rows["gate_proj"][j], dg[j]), ("up", rows["up_proj"][j], du[j]), ("down", dcols[j], dd[:, j])):
            wn, dn = np.linalg.norm(w), np.linalg.norm(dl)
            rec[name] = {"w_norm": float(wn), "d_norm": float(dn), "rel": float(dn / wn),
                         "cos": float(w @ dl / (wn * dn)), "along": float(w @ dl / wn**2)}
        out.append(rec)
    print(f"layer {l}: {len(nid)} neurons", file=sys.stderr, flush=True)
with open(sys.argv[2], "w") as handle:
    json.dump(out, handle, indent=1)
