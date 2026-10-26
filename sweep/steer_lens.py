"""Steer lens: what pushing each neuron does to the model's next-token prediction, through every layer after it.

For each neuron and sign, adds ±dose x (median residual norm / ||W_down[:, n]||) to the neuron at every position of a
few short contexts, runs one forward pass, and compares the last position's log-probabilities with the unsteered run.
Keeps, per neuron and sign, the 12 tokens whose log-probability rises most (averaged over contexts) and the KL
divergence from the unsteered prediction.

    uv run python sweep/steer_lens.py --layers 0-7 --out stats/steer_lens

Writes stats/steer_lens_L{l}.npz per layer: up_tok[N, 12], down_tok[N, 12] (token ids for the + push and - push),
up_kl[N], down_kl[N].
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from explorer.model_adapter import DEFAULT_MODEL, load_model_and_tokenizer

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=os.environ.get("EXPLORER_MODEL_REVISION"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default="22")
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=96)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    layers = list(range(int(lo), int(hi or lo) + 1))

    model, tok, adapter = load_model_and_tokenizer(a.model, a.revision, a.device)
    lm = adapter.language_model
    if min(layers) < 0 or max(layers) >= len(lm.layers):
        ap.error("layer index is outside this model")
    if a.neurons_per_batch < 1 or a.k < 1:
        ap.error("neurons-per-batch and k must be positive")
    a.out = a.out or str(Path("stats") / adapter.cache_namespace / "steer_lens")
    ctxs = [adapter.apply_chat_template([{"role": "user", "content": c}]) for c in CHAT]
    ctxs += [tok(p).input_ids for p in PLAIN]
    C, T = len(ctxs), max(map(len, ctxs))
    pad = adapter.pad_token_id
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device=adapter.device)
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device=adapter.device)
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    push = {"layer": None, "cols": None, "vals": None}
    resid = {}

    def make_hook(i):
        def hook(mod, inp):
            if push["layer"] != i:
                return
            h = inp[0].clone()
            rows = torch.arange(h.shape[0], device=h.device)
            h[rows, :, push["cols"]] += push["vals"][:, None].to(h.dtype)
            return (h,)
        return hook

    for i, layer in enumerate(lm.layers):
        layer.mlp.down_proj.register_forward_pre_hook(make_hook(i))
        layer.register_forward_hook(lambda m, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        h = lm(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
        return torch.log_softmax(model.lm_head(h).float(), -1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)  # [C, V]
        # median residual norm leaving each layer over these contexts' real tokens
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in resid.items()}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    B = a.neurons_per_batch
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight  # [hidden, inter]
        N = W.shape[1]
        unit = norms[l] / W.float().norm(dim=0).clamp_min(1e-12)  # [N]
        out = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, N, B):
            n = torch.arange(s, min(s + B, N), device=adapter.device)
            b = len(n)
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                # rows: neuron-major, contexts repeated
                cols = n.repeat_interleave(C)
                vals = sign * a.dose * unit[cols]
                push.update(layer=l, cols=cols, vals=vals)
                with torch.inference_mode():
                    lp = last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), posn.repeat(b, 1)).view(b, C, -1)
                push["layer"] = None
                delta = (lp - base[None]).mean(1)  # [b, V]
                kl = (lp.exp() * (lp - base[None])).sum(-1).mean(1)  # KL(steered || base), averaged over contexts
                out[f"{key}_tok"].append(delta.topk(a.k).indices.int().cpu())
                out[f"{key}_kl"].append(kl.cpu())
        arrs = {k: torch.cat(v).numpy() for k, v in out.items()}
        np.savez(f"{a.out}_L{l}.npz", **arrs, dose=a.dose, contexts=np.array(CHAT + PLAIN), model=a.model, revision=adapter.revision or "", layer=l)
        print(f"L{l}: {N} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(arrs['up_kl']):.3f}", flush=True)


if __name__ == "__main__":
    main()
