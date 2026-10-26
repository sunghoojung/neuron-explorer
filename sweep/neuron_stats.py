"""Per-neuron activation ranges for MLP layers of a supported decoder model, plus their down-projection columns.

A neuron's activation is silu(gate) * up, the input to down_proj. Positive activations come from two sign regimes
(gate>0, up>0) and (gate<0, up<0); negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of
each regime per neuron over a mixed web + chat corpus, and saves W_down[:, n] for every neuron, which is the direction
a steering push on neuron n adds to the residual stream.

    uv run python sweep/neuron_stats.py --model Qwen/Qwen3.5-4B --layers 22 --tokens 2000000 --out stats/q35-4b

Writes stats/q35-4b_L22.npz with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer (scale for push sizes)
"""
import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset

from explorer.model_adapter import DEFAULT_MODEL, load_model_and_tokenizer


def text_stream(tok, seed, adapter):
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        yield adapter.render_chat_template(c["messages"], add_generation_prompt=False)


def windows(tok, seq_len, batch, seed, adapter):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    buf = []
    eos = tok.eos_token_id
    for text in text_stream(tok, seed, adapter):
        buf.extend(tok(text, add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= seq_len * batch:
            chunk, buf = buf[: seq_len * batch], buf[seq_len * batch :]
            yield torch.tensor(chunk).view(batch, seq_len)


class RegimeStats:
    def __init__(self, n, device):
        f = lambda v: torch.full((n,), v, device=device, dtype=torch.float32)
        self.pp_max, self.nn_max = f(0.0), f(0.0)
        self.pn_min, self.np_min = f(0.0), f(0.0)
        self.gate = self.up = None
        self.resid_norms = []  # per-token residual norm leaving the layer, positions >= 1 (position 0 is a sink)

    def update_resid(self, out):
        h = out[0] if isinstance(out, tuple) else out
        self.resid_norms.append(h[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update(self, act):
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, u, a = (t[:, 1:].flatten(0, 1).float() for t in (self.gate, self.up, act))
        gp, up_ = g > 0, u > 0
        ninf = torch.tensor(float("-inf"), device=a.device)
        pinf = torch.tensor(float("inf"), device=a.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(gp & up_, a, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~gp & ~up_, a, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(gp & ~up_, a, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~gp & up_, a, pinf).amin(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=os.environ.get("EXPLORER_MODEL_REVISION"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default="22", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="Output prefix; defaults to model-specific stats directory")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    if a.tokens < 1 or a.seq_len < 2 or a.batch < 1:
        ap.error("tokens and batch must be positive; seq-len must be at least 2")
    _model, tok, adapter = load_model_and_tokenizer(a.model, a.revision, a.device)
    lm = adapter.language_model
    if not layers or min(layers) < 0 or max(layers) >= len(lm.layers):
        ap.error("layer index is outside this model")
    a.out = a.out or str(Path("stats") / adapter.cache_namespace / "model")
    lm.layers = lm.layers[: max(layers) + 1]  # nothing past the deepest target layer is needed

    stats = {}
    for l in layers:
        mlp = lm.layers[l].mlp
        s = stats[l] = RegimeStats(mlp.down_proj.in_features, adapter.device)
        mlp.gate_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "gate", o))
        mlp.up_proj.register_forward_hook(lambda m, i, o, s=s: setattr(s, "up", o))
        mlp.down_proj.register_forward_pre_hook(lambda m, i, s=s: s.update(i[0]))
        lm.layers[l].register_forward_hook(lambda m, i, o, s=s: s.update_resid(o))

    n_steps = -(-a.tokens // (a.seq_len * a.batch))
    t0 = time.time()
    step = -1
    with torch.inference_mode():
        for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed, adapter), n_steps)):
            lm(input_ids=ids.to(adapter.device), use_cache=False)
            if step % 10 == 0 or step == n_steps - 1:
                done = (step + 1) * a.seq_len * a.batch
                print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    if step < 0:
        raise RuntimeError("Corpus yielded no complete token windows; no statistics saved")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for l, s in stats.items():
        pp, nn, pn, np_ = (x.cpu().numpy() for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = lm.layers[l].mlp.down_proj.weight.detach().float().T.cpu().numpy().astype(np.float16)
        resid_q = np.quantile(torch.cat(s.resid_norms).numpy(), [0.1, 0.5, 0.9])
        path = f"{a.out}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, model=a.model, revision=adapter.revision or "", layer=l, tokens=(step + 1) * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
