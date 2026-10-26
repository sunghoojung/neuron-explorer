"""Build weight-only candidate links between neurons for the explorer.

For layers i < j, these scores estimate how the down-projection direction of
an MLP neuron in i aligns with gate/up rows in j after the reader layer's MLP
input RMSNorm scale. Attention and intervening computations are omitted. Treat
links as leads to test with interventions, not as causal evidence.

    uv run python explorer/build_connections.py

Writes a model/revision-namespaced ``connections.npz``.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

EXPLORER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXPLORER_DIR)
from model_adapter import DEFAULT_MODEL, model_cache_namespace, resolve_model_revision


def weight_map(snapshot_dir):
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    shards = glob.glob(os.path.join(snapshot_dir, "*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors weights found in {snapshot_dir}")
    mapping = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            # SafeOpen documents keys() for listing; its iteration behavior is not relied upon.
            mapping.update({key: os.path.basename(shard) for key in f.keys()})  # noqa: SIM118
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--revision", default=os.environ.get("EXPLORER_REVISION"))
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--out", help="optional output path; default is this model/revision's cache directory")
    a = ap.parse_args()

    revision = resolve_model_revision(a.model, a.revision)
    snapshot = snapshot_download(a.model, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    with open(os.path.join(snapshot, "config.json")) as f:
        raw_cfg = json.load(f)
    cfg = raw_cfg.get("text_config", raw_cfg)
    layers, neurons = int(cfg["num_hidden_layers"]), int(cfg["intermediate_size"])
    if not 1 <= a.k <= neurons:
        raise ValueError(f"--k must be in [1, {neurons}]")

    keys = weight_map(snapshot)
    candidates = (
        "model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    )
    first_layer_key = next((key for key in candidates if key in keys), None)
    if first_layer_key is None:
        raise ValueError("Could not identify the model's MLP weight prefix in its safetensors index")
    prefix = first_layer_key.removesuffix("0.mlp.down_proj.weight")

    def tensor(name):
        if name not in keys:
            raise KeyError(f"Missing model weight: {name}")
        with safe_open(os.path.join(snapshot, keys[name]), framework="pt") as f:
            return f.get_tensor(name).to(device=dev, dtype=dtype)

    if torch.cuda.is_available():
        dev, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        dev, dtype = "mps", torch.float16
    else:
        dev, dtype = "cpu", torch.float32
    model_type = str(cfg.get("model_type", "")).lower()
    if model_type == "llama":
        offset_norm = False
    elif model_type in {"qwen3_5", "qwen3.5"}:
        offset_norm = True
    else:
        raise ValueError(f"Unsupported norm convention for model type {model_type!r}")

    def layer_prefix(layer):
        return f"{prefix}{layer}."

    down = [tensor(layer_prefix(i) + "mlp.down_proj.weight") for i in range(layers)]
    gate = [tensor(layer_prefix(i) + "mlp.gate_proj.weight") for i in range(layers)]
    up = [tensor(layer_prefix(i) + "mlp.up_proj.weight") for i in range(layers)]
    norm = []
    for layer in range(layers):
        scale = tensor(layer_prefix(layer) + "post_attention_layernorm.weight").float()
        if offset_norm:
            scale = scale + 1.0
        norm.append(scale.to(dtype))

    # For every neuron and side: score, linked layer, linked neuron, sign.
    best = {
        side: (
            torch.full((layers, neurons, a.k), -1e9, device=dev),
            torch.zeros((layers, neurons, a.k), dtype=torch.int16, device=dev),
            torch.zeros((layers, neurons, a.k), dtype=torch.int16, device=dev),
            torch.zeros((layers, neurons, a.k), dtype=torch.int8, device=dev),
        )
        for side in ("down", "up")
    }

    def merge(side, layer, scores, linked_neurons, other_layer, signs):
        best_z, best_layer, best_neuron, best_sign = best[side]
        joined_z = torch.cat([best_z[layer], scores], dim=1)
        joined_layer = torch.cat([best_layer[layer], torch.full_like(linked_neurons, other_layer, dtype=torch.int16)], dim=1)
        joined_neuron = torch.cat([best_neuron[layer], linked_neurons.to(torch.int16)], dim=1)
        joined_sign = torch.cat([best_sign[layer], signs], dim=1)
        top = joined_z.topk(a.k, dim=1)
        best_z[layer] = top.values
        best_layer[layer] = joined_layer.gather(1, top.indices)
        best_neuron[layer] = joined_neuron.gather(1, top.indices)
        best_sign[layer] = joined_sign.gather(1, top.indices)

    t0 = time.time()
    with torch.inference_mode():
        for j in range(1, layers):
            for i in range(j):
                # Llama stores the actual multiplicative RMSNorm scale. Qwen3.5 stores its offset from one.
                direction = down[i] * norm[j][:, None]
                product = (gate[j] @ direction).float() * (up[j] @ direction).float()
                magnitude = product.abs()
                col_z = (magnitude - magnitude.mean(0, keepdim=True)) / (magnitude.std(0, keepdim=True) + 1e-8)
                top = col_z.topk(a.k, dim=0)
                signs = torch.where(product.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
                merge("down", i, top.values.T, top.indices.T, j, signs.T)

                row_z = (magnitude - magnitude.mean(1, keepdim=True)) / (magnitude.std(1, keepdim=True) + 1e-8)
                top = row_z.topk(a.k, dim=1)
                signs = torch.where(product.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
                merge("up", j, top.values, top.indices, i, signs)
                del product, magnitude, col_z, row_z
            print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

    result = {}
    for side, (scores, linked_layers, linked_neurons, signs) in best.items():
        result[f"{side}_z"] = scores.clamp(min=-60000).cpu().numpy().astype(np.float16)
        result[f"{side}_layer"] = linked_layers.cpu().numpy().astype(np.int8)
        result[f"{side}_n"] = linked_neurons.cpu().numpy()
        result[f"{side}_sign"] = signs.cpu().numpy()
    namespace = model_cache_namespace(a.model, revision)
    output = a.out or os.path.join(EXPLORER_DIR, "cache", namespace, "connections.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(output, **result, k=a.k)
    print(f"-> {output} ({os.path.getsize(output) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
