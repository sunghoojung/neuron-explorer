"""Agreement between capture_prefill.py output and the vLLM fork's own captures for the same run.

    python sweep/check_capture_prefill.py runs/mainrel-e2e.jsonl captures/mainrel-e2e
"""
import json
import os
import sys

import numpy as np
from vllm.v1.capture.consumers.filesystem.reader import read_request

run, fork_dir = sys.argv[1], sys.argv[2]
name = os.path.basename(run).replace(".jsonl", "")
ours = np.load(f"captures/{name}.mlp_in.npy").astype(np.float32)
with open(f"captures/{name}.mlp_in.json") as handle:
    layers = json.load(handle)["capture_layers"]
with open(run) as handle:
    rows = [json.loads(line) for line in handle]
errs, n = {l: [] for l in layers}, 0
for i, r in enumerate(rows):
    d = os.path.join(fork_dir, r["capture_id"])
    if not os.path.isdir(d):
        continue
    ent = read_request(d)
    for j, l in enumerate(layers):
        arr = ent[(l, "mlp_in")].array
        if arr.dtype == np.uint16:
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        ref = arr.reshape(-1, arr.shape[-1])[-1].astype(np.float32)
        errs[l].append(np.linalg.norm(ours[i, j] - ref) / np.linalg.norm(ref))
    n += 1
print(f"{n} rollouts compared")
for l in layers:
    e = np.array(errs[l])
    print(f"L{l}: relative error median {np.median(e):.4f}  p95 {np.percentile(e, 95):.4f}  max {e.max():.4f}")
