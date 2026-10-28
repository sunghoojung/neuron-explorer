"""Terminal view of a sweep's rollouts, plus dose-response tables for calibration runs.

    uv run python sweep/peek.py runs/calib-e2e.jsonl                  # every neuron, every rollout
    uv run python sweep/peek.py runs/calib-e2e.jsonl --neuron 2742    # one neuron
    uv run python sweep/peek.py runs/calib-e2e.jsonl --curves         # dose-response table, neurons vs random
    uv run python sweep/peek.py runs/calib-e2e.jsonl --words          # distinctive words per unit, strongest first

Per rollout: dose, rel_push (push size / median residual norm), the generated token where it first departs from the
unsteered baseline for that prompt (`=` if it never does), length-capped flag, distinct-4-gram ratio, mean chosen
logprob, then the opening of the text.
"""
import argparse
import collections
import json
import re

import numpy as np


def first_departure(ids, base):
    for i, (a, b) in enumerate(zip(ids, base)):
        if a != b:
            return i
    return None if len(ids) == len(base) else min(len(ids), len(base))


def distinct4(ids):
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def metrics(r, base):
    dep = first_departure(r["token_ids"], base[r["prompt"]]["token_ids"])
    return {
        "dep": dep,
        "capped": r["finish"] == "length",
        "d4": distinct4(r["token_ids"]),
        "lp": float(np.mean(r["logprob"])) if r["logprob"] else 0.0,
    }


def show(rows, base, width):
    by_prompt = collections.defaultdict(list)
    for r in rows:
        by_prompt[r["prompt"]].append(r)
    for p, rs in by_prompt.items():
        b = base[p]
        print(f"  [{p}] baseline: {b['text'][:width]!r}")
        for r in sorted(rs, key=lambda r: r["dose"]):
            m = metrics(r, base)
            dep = "=" if m["dep"] is None else str(m["dep"])
            print(f"    {r['dose']:+6g}  rel {r['rel_push']:5.2f}  dep {dep:>3}  {'CAP' if m['capped'] else '   '}  "
                  f"d4 {m['d4']:.2f}  lp {m['lp']:6.2f}  {r['text'][:width]!r}")


def curves(rows, base):
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        m = metrics(r, base)
        cell = table[(r["dose"], r["kind"])]
        cell["moved"].append(m["dep"] is not None)
        cell["dep"].append(m["dep"] if m["dep"] is not None else len(r["token_ids"]))
        cell["capped"].append(m["capped"])
        cell["d4"].append(m["d4"])
        cell["lp"].append(m["lp"])
        cell["rel"].append(r["rel_push"])
    kinds = sorted({k for _, k in table})
    print(f"{'dose':>6}  " + "  ".join(f"{k:^52s}" for k in kinds))
    print(f"{'':>6}  " + "  ".join(f"{'n':>3} {'rel':>5} {'moved':>6} {'dep med':>7} {'capped':>6} {'d4':>5} {'lp':>6}"
                                    for _ in kinds))
    for dose in sorted({d for d, _ in table}):
        cells = []
        for k in kinds:
            c = table.get((dose, k))
            if not c:
                cells.append(" " * 52)
                continue
            cells.append(f"{len(c['moved']):3d} {np.median(c['rel']):5.2f} {np.mean(c['moved']):6.0%} "
                         f"{np.median(c['dep']):7.0f} {np.mean(c['capped']):6.0%} {np.mean(c['d4']):5.2f} "
                         f"{np.mean(c['lp']):6.2f}")
        print(f"{dose:+6g}  " + "  ".join(cells))


def distinctive_words(rows, base, max_rel, top):
    """Per unit: words over-represented in its rollouts relative to the average unit, as a Poisson z score
    (count - mean) / sqrt(mean + 0.5), over words in >= 2 of its rollouts and in no baseline. Units sorted by their
    strongest word."""
    tokz = lambda t: set(re.findall(r"[^\W\d_]{3,}", t.lower()))
    base_words = set().union(*(tokz(b["text"]) for b in base.values()))
    docs, n_roll = collections.defaultdict(collections.Counter), collections.Counter()
    for r in rows:
        if r["rel_push"] <= max_rel:
            docs[(r["kind"], r["neuron"])].update(tokz(r["text"]) - base_words)
            n_roll[(r["kind"], r["neuron"])] += 1
    total = collections.Counter()
    for c in docs.values():
        total.update(c)
    mean = {w: v / len(docs) for w, v in total.items()}
    out = []
    for u, c in docs.items():
        ranked = sorted((((cnt - mean[w]) / np.sqrt(mean[w] + 0.5), w, cnt) for w, cnt in c.items() if cnt >= 2),
                        reverse=True)
        out.append((ranked[0][0] if ranked else 0.0, u, n_roll[u], ranked[:top]))
    for best, (kind, n), n_r, ranked in sorted(out, key=lambda x: (-x[0], x[1])):
        words = ", ".join(f"{w}({cnt})" for _, w, cnt in ranked)
        print(f"{kind:6s} {n:6d}  z {best:4.1f}  ({n_r} rollouts)  {words}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--neuron", type=int)
    ap.add_argument("--kind", default=None, help="neuron or random")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--words", action="store_true", help="distinctive words per unit, sorted by how many of its "
                    "rollouts share its top word")
    ap.add_argument("--max-rel", type=float, default=1.6, help="--words ignores rollouts pushed harder than this")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--width", type=int, default=110)
    a = ap.parse_args()

    with open(a.jsonl) as handle:
        rows = [json.loads(line) for line in handle]
    base = {r["prompt"]: r for r in rows if r["kind"] == "baseline"}
    steered = [r for r in rows if r["kind"] != "baseline"]
    if a.words:
        distinctive_words(steered, base, a.max_rel, a.top)
        return
    if a.curves:
        curves(steered, base)
        return
    units = collections.defaultdict(list)
    for r in steered:
        if (a.neuron is None or r["neuron"] == a.neuron) and (a.kind is None or r["kind"] == a.kind):
            units[(r["kind"], r["neuron"])].append(r)
    for (kind, n), rs in sorted(units.items()):
        ref = rs[0]["ref_neuron"]
        print(f"\n== {kind} {n}" + (f" (scale from neuron {ref})" if kind == "random" else "") +
              f"  L{rs[0]['layer']}")
        show(rs, base, a.width)


if __name__ == "__main__":
    main()
