"""Embeds every rollout of a sweep run and lays them out in 2D with UMAP, for the explorer's embeddings view.

    uv run python explorer/build_embeddings.py runs/mainrel-e2e.jsonl

Writes <run>.emb.npy (normalized sentence embeddings, one row per jsonl line) and <run>.umap.json with parallel arrays:
x, y, row (jsonl line index), kind, neuron, prompt, dose.
"""
import argparse
import json
import os
import sys

import numpy as np
import umap
from sentence_transformers import SentenceTransformer

EXPLORER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXPLORER_DIR)
sys.path.insert(0, EXPLORER_DIR)
from model_adapter import DEFAULT_MODEL, model_cache_namespace, resolve_model_revision


def validate_source(jsonl, model_id, revision):
    """Refuse to create model-specific embeddings from an unscoped or mismatched run."""
    meta_path = jsonl.replace(".jsonl", ".meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing model metadata next to {jsonl}")
    with open(meta_path) as f:
        meta = json.load(f)
    source_model = meta.get("model_id") or meta.get("model")
    if not isinstance(source_model, str) or source_model.strip().casefold() != model_id.strip().casefold():
        raise ValueError(f"Run model does not match requested target model {model_id!r}")
    source_revision = meta.get("model_revision") or meta.get("revision") or meta.get("commit_hash")
    if source_revision and revision and str(source_revision) != str(revision):
        raise ValueError(f"Run revision {source_revision!r} does not match requested revision {revision!r}")
    expected_namespace = model_cache_namespace(model_id, revision)
    if os.path.basename(os.path.dirname(os.path.abspath(jsonl))) != expected_namespace:
        raise ValueError(f"Place the run under its model/revision directory: runs/{expected_namespace}/")
    recorded_namespace = meta.get("model_cache_namespace")
    if recorded_namespace and recorded_namespace != expected_namespace:
        raise ValueError("Run metadata belongs to a different model/revision cache namespace")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--target-model", default=os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL))
    ap.add_argument("--revision", default=os.environ.get("EXPLORER_REVISION"))
    ap.add_argument("--chars", type=int, default=600, help="leading characters of each rollout to embed")
    ap.add_argument("--neighbors", type=int, default=15)
    a = ap.parse_args()

    revision = resolve_model_revision(a.target_model, a.revision)
    source_meta = validate_source(a.jsonl, a.target_model, revision)

    with open(a.jsonl) as f:
        rows = [json.loads(line) for line in f]
    import torch
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    enc = SentenceTransformer(a.model, device=dev)
    emb = enc.encode([r["text"][: a.chars] for r in rows], batch_size=256, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
    np.save(a.jsonl.replace(".jsonl", ".emb.npy"), emb.astype(np.float16))
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.08, metric="cosine", random_state=0).fit_transform(emb)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    out = {
        "x": np.round(xy[:, 0], 4).tolist(), "y": np.round(xy[:, 1], 4).tolist(), "row": list(range(len(rows))),
        "kind": [r["kind"] for r in rows], "neuron": [r["neuron"] for r in rows],
        "prompt": [r["prompt"] for r in rows], "dose": [r["dose"] for r in rows],
    }
    path = a.jsonl.replace(".jsonl", ".umap.json")
    with open(path, "w") as f:
        json.dump(out, f)
    embed_meta = {
        "source_model": a.target_model,
        "source_revision": revision,
        "source_cache_namespace": model_cache_namespace(a.target_model, revision),
        "embedding_model": a.model,
        "source_run": os.path.basename(a.jsonl),
        "chars": a.chars,
        "neighbors": a.neighbors,
        "source_design": source_meta.get("design_name"),
    }
    with open(a.jsonl.replace(".jsonl", ".embedding.meta.json"), "w") as f:
        json.dump(embed_meta, f, indent=1)
    print(f"{len(rows)} rollouts -> {path}")


if __name__ == "__main__":
    main()
