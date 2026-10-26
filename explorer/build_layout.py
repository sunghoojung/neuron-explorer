"""Build per-model neuron layouts from MLP weights.

Each neuron is embedded from its down-projection column (writes) or gate/up
rows (reads), projected with PCA and UMAP, then sorted into a near-square grid.
Cache files are namespaced by model and revision.

    uv run python explorer/build_layout.py --kind writes
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import umap
from huggingface_hub import snapshot_download
from safetensors import safe_open

EXPLORER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPLORER_DIR)
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
    ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--neighbors", type=int, default=30)
    ap.add_argument("--out", help="optional output path; default is this model/revision's cache directory")
    a = ap.parse_args()

    revision = resolve_model_revision(a.model, a.revision)
    snapshot = snapshot_download(a.model, revision=revision, allow_patterns=["*.safetensors", "*.json"])
    with open(os.path.join(snapshot, "config.json")) as f:
        raw_cfg = json.load(f)
    cfg = raw_cfg.get("text_config", raw_cfg)
    layers = int(cfg["num_hidden_layers"])
    neurons = int(cfg["intermediate_size"])
    side = math.ceil(math.sqrt(neurons))
    row_count = math.ceil(neurons / side)

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
            return f.get_tensor(name).float().numpy()

    cells = np.zeros((layers, neurons), np.int16)
    for layer in range(layers):
        mlp = f"{prefix}{layer}.mlp."
        if a.kind == "writes":
            vectors = tensor(mlp + "down_proj.weight").T
        else:
            vectors = np.concatenate([tensor(mlp + "gate_proj.weight"), tensor(mlp + "up_proj.weight")], axis=1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        vectors -= vectors.mean(axis=0, keepdims=True)
        u, singular, _ = np.linalg.svd(vectors, full_matrices=False)
        embedding = u[:, :a.pca] * singular[:a.pca]
        neighbors = min(a.neighbors, neurons - 1)
        if neighbors < 2:
            raise ValueError("UMAP needs at least three neurons")
        xy = umap.UMAP(n_neighbors=neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(embedding)
        by_y = np.argsort(xy[:, 1], kind="stable")
        for row in range(row_count):
            band = by_y[row * side:(row + 1) * side]
            band = band[np.argsort(xy[band, 0], kind="stable")]
            cells[layer, band] = row * side + np.arange(len(band))
        print(f"L{layer} done", flush=True)

    namespace = model_cache_namespace(a.model, revision)
    output = a.out or os.path.join(EXPLORER_DIR, "cache", namespace, f"layout_{a.kind}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez(output, cell=cells)
    print(f"-> {output} ({layers} layers, {neurons} neurons per layer, {side} columns)")


if __name__ == "__main__":
    main()
