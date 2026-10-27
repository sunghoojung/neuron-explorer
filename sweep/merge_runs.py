"""Joins sharded sweep runs of one layer into a single run, one row per (kind, neuron, prompt, dose).

    uv run python sweep/merge_runs.py runs/q35-4b-L22.jsonl runs/q35-4b-L22-s1.jsonl runs/q35-4b-L22-s2.jsonl \
        runs/q35-4b-L22-s3.jsonl --out runs/q35-4b-L22-full.jsonl

The merged run's meta.json is the first input's, with the shard list recorded under "shards".
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seen, n_in, n_out = set(), 0, 0
    with open(a.out, "w") as f:
        for path in a.inputs:
            with open(path) as source:
                for line in source:
                    n_in += 1
                    r = json.loads(line)
                    key = (r["kind"], r["neuron"], r["prompt"], r["dose"])
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(line if line.endswith("\n") else line + "\n")
                    n_out += 1
    with open(a.inputs[0].replace(".jsonl", ".meta.json")) as source:
        meta = json.load(source)
    meta["shards"] = [os.path.basename(p) for p in a.inputs]
    with open(a.out.replace(".jsonl", ".meta.json"), "w") as target:
        json.dump(meta, target, indent=1)
    print(f"{n_in} rows in, {n_out} unique -> {a.out}")


if __name__ == "__main__":
    main()
