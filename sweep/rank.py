"""Per-unit statistics for a sweep: distinctive words, how far above the random-direction null a unit's words sit, and
text-breakdown numbers per dose. Writes one JSON record per unit for the explorer and prints the top of the ranking.

    uv run python sweep/rank.py runs/q35-4b-L22.jsonl --out runs/q35-4b-L22.units.jsonl --show 40 --find 2742

For each unit (neuron or random direction):
    words        words over-represented in its rollouts vs the average unit, Poisson z = (count - mean) / sqrt(mean + 0.5),
                 counted once per rollout, restricted to rollouts with push <= --max-rel, excluding baseline words
    z            its strongest word's z
    mass         summed positive z of its top 5 words: breadth of the word cluster (the sort key)
    null_pct     share of random-direction controls with a lower mass (1.0 = above every control)
    per_prompt   for its strongest words, which prompts they appear on
    dose         per dose: mean distinct-4-gram ratio, share of length-capped rollouts, mean chosen logprob
"""
import argparse
import collections
import json
import re

import numpy as np

TOKEN = re.compile(r"[^\W\d_]{3,}")


def words(text):
    return set(TOKEN.findall(text.lower()))


def distinct4(ids):
    grams = [tuple(ids[i : i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def compute_units(rows, max_rel=1.6, top_words=12):
    """One record per steered unit (see module docstring), sorted by z, strongest first."""
    base_words = set().union(*(words(r["text"]) for r in rows if r["kind"] == "baseline"))
    units = collections.defaultdict(list)
    for r in rows:
        if r["kind"] != "baseline":
            units[(r["kind"], r["neuron"])].append(r)

    counts = {}
    for u, rs in units.items():
        c = collections.Counter()
        for r in rs:
            if r["rel_push"] <= max_rel:
                c.update(words(r["text"]) - base_words)
        counts[u] = c
    total = collections.Counter()
    for c in counts.values():
        total.update(c)
    mean = {w: v / len(counts) for w, v in total.items()}

    records = []
    for u, rs in units.items():
        c = counts[u]
        ranked = sorted((((cnt - mean[w]) / np.sqrt(mean[w] + 0.5), w, cnt) for w, cnt in c.items() if cnt >= 2),
                        reverse=True)[:top_words]
        per_prompt = {w: sorted({r["prompt"] for r in rs if w in words(r["text"])}) for _, w, _ in ranked[:5]}
        dose = {}
        for d in sorted({r["dose"] for r in rs}):
            ds = [r for r in rs if r["dose"] == d]
            dose[f"{d:+g}"] = {
                "rel_push": round(float(np.mean([r["rel_push"] for r in ds])), 3),
                "distinct4": round(float(np.mean([distinct4(r["token_ids"]) for r in ds])), 3),
                "capped": round(float(np.mean([r["finish"] == "length" for r in ds])), 3),
                "logprob": round(float(np.mean([np.mean(r["logprob"]) if r["logprob"] else 0.0 for r in ds])), 3),
            }
        records.append({
            "kind": u[0], "neuron": u[1], "layer": rs[0]["layer"], "ref_neuron": rs[0]["ref_neuron"],
            "z": round(float(ranked[0][0]), 3) if ranked else 0.0,
            "mass": round(float(sum(max(z, 0.0) for z, _, _ in ranked[:5])), 3),
            "words": [[w, cnt, round(float(z), 2)] for z, w, cnt in ranked],
            "per_prompt": per_prompt, "dose": dose,
        })

    null = np.sort([r["mass"] for r in records if r["kind"] == "random"])
    for r in records:
        r["null_pct"] = round(float(np.searchsorted(null, r["mass"]) / len(null)), 4) if len(null) else None
    records.sort(key=lambda r: -r["mass"])
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rel", type=float, default=1.6)
    ap.add_argument("--top-words", type=int, default=12)
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--find", default="", help="comma-separated neuron ids to report ranks for")
    a = ap.parse_args()

    with open(a.jsonl) as handle:
        records = compute_units([json.loads(line) for line in handle], a.max_rel, a.top_words)
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)

    null = np.sort([r["mass"] for r in records if r["kind"] == "random"])
    neurons = [r for r in records if r["kind"] == "neuron"]
    print(f"{len(neurons)} neurons, {len(null)} random controls; random mass p50/p95/max = "
          f"{np.percentile(null, 50):.2f} / {np.percentile(null, 95):.2f} / {null.max():.2f}" if len(null) else "")
    above = sum(r["mass"] > null.max() for r in neurons) if len(null) else 0
    print(f"neurons above every random control: {above} ({above / len(neurons):.1%})")
    for i, r in enumerate(records[: a.show]):
        ws = ", ".join(f"{w}({c})" for w, c, _ in r["words"][:6])
        print(f"{i + 1:4d}  {r['kind'][:3]} {r['neuron']:5d}  mass {r['mass']:5.1f}  {ws}")
    for n in [int(x) for x in a.find.split(",") if x]:
        rank = next((i + 1 for i, r in enumerate(neurons) if r["neuron"] == n), None)
        rec = next((r for r in neurons if r["neuron"] == n), None)
        if rec:
            ws = ", ".join(f"{w}({c})" for w, c, _ in rec["words"][:6])
            print(f"neuron {n}: rank {rank}/{len(neurons)} among neurons, mass {rec['mass']:.1f}, null_pct {rec['null_pct']}, "
                  f"{ws}")


if __name__ == "__main__":
    main()
