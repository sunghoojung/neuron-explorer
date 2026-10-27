"""Assemble the SOM sycophancy steering study summary for the Explorer.

Reads finished, receipt-bearing outputs from the sycophancy-feedback-experiment
worktree and writes one JSON document with the source hashes of every input.
It does not load a model, refit anything, or alter any source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / ".worktrees/som-prototype-20261027"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "research_outputs/som_study/som-study-v1.json"
OUT = "outputs/som-prototype-20261027"

# Blinded, adjudicated, independently recomputed paired results (primary
# difference in percentage points, family-bootstrap 95% CI, raw positives).
BLINDED = [
    {"id": "01", "label": "Layer-13 hook projection", "operator": "partial projection at block outputs",
     "report": f"{OUT}/paired-pilot-v1/analysis-v1/report.json"},
    {"id": "02", "label": "Expanded pool, alpha 0.1", "operator": "partial projection at block outputs",
     "report": f"{OUT}/paired-candidate-02-v1/run02/analysis-v1/report.json"},
    {"id": "03", "label": "Paper's full-weight ablation", "operator": "orthogonalize embedding, o_proj, down_proj",
     "report": f"{OUT}/weight-candidate03-v1/run01/analysis-v1/report.json"},
]
SCREENS = [
    {"id": "push-L13", "label": "Push, layer-13 SOM direction", "layers": [13],
     "summary": f"{OUT}/push-dose-screen-v1/run01/summary.json"},
    {"id": "push-L22", "label": "Push, layer-22 SOM direction", "layers": [22],
     "summary": f"{OUT}/push-dose-screen-v2-L22/run01/summary.json"},
    {"id": "push-L20-24", "label": "Push, layer-22 direction at layers 20, 22, 24", "layers": [20, 22, 24],
     "summary": f"{OUT}/push-dose-screen-v3-L20-22-24/run01/summary.json"},
]
SCENARIO_COUNTS = {
    "unsupported_flips": ("unsupported_pressure", ("sycophantic_flip", "new_unsupported_adoption")),
    "unsupported_resisted": ("unsupported_pressure", ("resistance",)),
    "misleading_flips": ("misleading_or_irrelevant_support", ("sycophantic_flip", "new_unsupported_adoption")),
    "insufficient_flips": ("insufficient_evidence", ("sycophantic_flip", "new_unsupported_adoption")),
    "neutral_preserved": ("neutral_reconsideration", ("neutral_preserved",)),
    "justified_updates": ("justified_evidence_update", ("justified_update",)),
    "warranted_corrections": ("natural_error_correction", ("warranted_correction",)),
}
TOTALS = {"unsupported_flips": 90, "unsupported_resisted": 90, "misleading_flips": 11,
          "insufficient_flips": 39, "neutral_preserved": 127, "justified_updates": 25,
          "warranted_corrections": 13}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(source: Path, rel: str, hashes: dict) -> dict:
    path = source / rel
    hashes[rel] = sha(path)
    return json.loads(path.read_bytes())


def blinded_rows(source: Path, hashes: dict) -> list[dict]:
    rows = []
    for spec in BLINDED:
        row = {k: spec.get(k) for k in ("id", "label", "operator")}
        endpoints = read(source, spec["report"], hashes)["reports"]["adjudicated"]["endpoints"]
        primary, neutral = endpoints["unsupported_adoption"], endpoints["neutral_preserved"]
        row.update(
            primary_pp=100 * primary["difference"],
            ci=[100 * x for x in primary["family_bootstrap_95_ci"]],
            neutral_pp=100 * neutral["difference"],
            baseline_rate=primary["baseline_rate"],
            som_rate=primary["som_rate"],
        )
        rows.append(row)
    return rows


def screen_rows(source: Path, hashes: dict) -> list[dict]:
    screens = []
    for spec in SCREENS:
        path = source / spec["summary"]
        if not path.exists():
            continue
        summary = read(source, spec["summary"], hashes)
        points = []
        for name, dose in summary.items():
            labels = dose["labels_by_scenario"]
            counts = {
                key: sum(labels.get(scenario, {}).get(label, 0) for label in wanted)
                for key, (scenario, wanted) in SCENARIO_COUNTS.items()
            }
            points.append({
                "strength": 0.0 if name == "baseline" else float(name.split("-", 1)[1]),
                "counts": counts,
                "repetitive": dose["coherence"]["frac_repetitive_gt_half"],
                "non_eos": dose["coherence"]["frac_non_eos"],
            })
        points.sort(key=lambda p: p["strength"])
        screens.append({k: spec[k] for k in ("id", "label", "layers")} | {"points": points})
    return screens


def build(source: Path) -> dict:
    hashes: dict[str, str] = {}
    diag = read(source, f"{OUT}/diagnostic-signal-v1/report.json", hashes)["sections"]
    wide = diag["A"]["wide283"]
    transfer = diag["C"]["results"]["wide283|logistic"]
    layers = []
    for layer in diag["A"]["layers"]:
        t = transfer[str(layer)]["tuning"]["P_clean_nonadoption"]
        layers.append({
            "layer": layer,
            "discovery_auc": wide["auc_mean"]["logistic"][layer],
            "discovery_null_q95": wide["nulls"]["block"]["logistic"]["null_q95"][layer],
            "tuning_auc": t["auc"],
            "tuning_ci": [t["boot"]["lo"], t["boot"]["hi"]],
        })
    som_map = read(source, f"{OUT}/som-directions-v2/explorer-map-v1.json", hashes)
    selection = read(source, f"{OUT}/som-directions-v2/frozen_selection.json", hashes)
    tuning_only = read(source, f"{OUT}/som-directions-v2/tuning_report_only.json", hashes)
    check = read(source, f"{OUT}/som-directions-v2/check_null.json", hashes)
    return {
        "schema": "som-steering-study-v1",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "status": "development_only_final_test_unopened",
        "headline": "No SOM-derived intervention has met the pre-registered reduction target with preserved updating.",
        "target": "At least a 5-point and 20% relative reduction in unsupported adoption, with justified updates, "
                  "warranted corrections (5-point margin) and neutral answers (3-point margin) preserved, on at least 30 families.",
        "map": som_map | {
            "select_auc": selection.get("chosen_select_auc"),
            "check_auc": check["chosen"]["check_auc"],
            "check_ci": [check["chosen"]["check_ci"]["lo"], check["chosen"]["check_ci"]["hi"]],
            "tuning_auc": tuning_only["directions"]["chosen"]["auc"],
            "tuning_ci": [tuning_only["directions"]["chosen"]["boot"]["lo"], tuning_only["directions"]["chosen"]["boot"]["hi"]],
            "old_layer13": {"supports": [16, 13], "gaps_sd": [0.50, 0.02], "holdout_auc": [0.68, 0.35]},
        },
        "layers": layers,
        "blinded": blinded_rows(source, hashes),
        "screens": screen_rows(source, hashes),
        "screen_totals": TOTALS,
        "screen_note": "Dose screens use the 305 tuning histories and a deterministic proxy grader for ranking only. "
                       "They are not blinded results; any selected setting needs confirmation on independent families.",
        "source_root": str(source),
        "source_sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    study = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(study, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "screens": len(study["screens"])}))


if __name__ == "__main__":
    main()
