"""Weight-only alignment of MLP neurons with the SOM steering directions.

For each MLP neuron i in layer l, ``down_proj.weight[:, i]`` is what one unit of
its activation adds to the residual stream. Its dot product with a unit SOM
direction says how far that neuron's write moves the residual toward the
sycophantic prototype (positive) or the resisting centroid (negative). Only
layers at or before a direction's layer can contribute to the state it was fit
on. This is a candidate list from weights alone, not causal evidence.
CPU only; reads the pinned checkpoint and frozen direction files, writes JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots" / REVISION
SOURCE = Path(__file__).resolve().parents[2] / ".worktrees/som-prototype-20261027/outputs/som-prototype-20261027"
DIRECTIONS = {
    "L22": {"path": "som-directions-v2/chosen-L22-g2-s2-c3.npy", "row": 0, "layer": 22,
            "label": "Layer-22 SOM direction (2x2, seed 2, cell 3)"},
    "L13": {"path": "weight-candidate03-v1/run01/snapshot/directions.npy", "row": 0, "layer": 13,
            "label": "Layer-13 SOM direction used by candidates 01-03 (cell 1)"},
}
TOP_K = 400


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def down_proj(layer: int) -> np.ndarray:
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    with safe_open(CHECKPOINT / index[name], framework="pt") as f:
        return f.get_tensor(name).float().numpy()  # bfloat16 checkpoint; [hidden, intermediate]


def build(top_k: int) -> dict:
    hashes, directions = {}, {}
    for key, spec in DIRECTIONS.items():
        path = SOURCE / spec["path"]
        hashes[spec["path"]] = sha(path)
        d = np.load(path, allow_pickle=False)[spec["row"]].astype(np.float64)
        directions[key] = d / np.linalg.norm(d)
    config = json.loads((CHECKPOINT / "config.json").read_text())
    layers = config["num_hidden_layers"]
    out = {key: {"label": spec["label"], "direction_layer": spec["layer"], "layers": []} for key, spec in DIRECTIONS.items()}
    for layer in range(layers):
        w = down_proj(layer).astype(np.float64)
        norms = np.linalg.norm(w, axis=0)
        for key, d in directions.items():
            if layer > DIRECTIONS[key]["layer"]:
                continue
            write = d @ w  # residual movement along d per unit activation
            cosine = write / np.maximum(norms, 1e-12)
            order = np.argsort(write)
            pick = np.concatenate([order[:top_k], order[-top_k:][::-1]])
            out[key]["layers"].append({
                "layer": layer,
                "write_abs_p99": float(np.quantile(np.abs(write), 0.99)),
                "write_abs_max": float(np.abs(write).max()),
                "neurons": [[int(i), round(float(write[i]), 5), round(float(cosine[i]), 4)] for i in pick],
            })
    return {
        "schema": "som-neuron-write-alignment-v1",
        "revision": REVISION,
        "method": "down_proj column dot unit SOM direction; layers after the direction's layer cannot affect it",
        "caveat": "Weight-only candidates. A neuron's effect also depends on how often and how strongly it fires.",
        "top_k_per_sign": top_k,
        "directions": out,
        "source_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "research_outputs/som_study/neuron-alignment-v1.json")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    result = build(args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")))
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
