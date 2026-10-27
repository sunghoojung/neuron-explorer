"""Read-only, receipt-verified access to installed teacher-forced atlas artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MAX_ATLAS_BYTES = 32 * 1024 * 1024
ATLAS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
GROUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
MAP_VIEWS = {"answer_mean", "shared_suffix", "paired_delta"}
SPLITS = {"train", "eval"}
SOURCE_STATUSES = {"TFonly", "teacher_forced_synthetic_unreviewed"}
TAXONOMY_ATLAS_SCHEMA = "taxonomy-som-atlas-v1"
TRANSFER_ATLAS_SCHEMA = "sycophancy-som-transfer-atlas-v1"
TRANSFER_LAYERS = (3, 6, 9, 13, 16, 20, 23, 26)
TRANSFER_SPLITS = ("train", "validation", "heldout")
TRANSFER_DATASETS = ("NQ", "QASC", "SQuAD", "VitaminC")
TRANSFER_CLAIM_CEILING = (
    "A held-out shift in the conditional likelihood margin of the frozen authored response scripts "
    "under a prompt-time activation perturbation."
)
TRANSFER_METHODS = {
    "baseline", "zero", "local_pooled", "global_matched_gate", "shuffled_cell",
    "random_norm_matched", "reverse_sign", "fixed_prompt",
}
TRANSFER_MAP_VIEWS = {
    "prompt_routing": {"pre_answer_prompt"},
    "output_contrast": {
        "answer_mean_independent_minus_candidate",
        "shared_suffix_independent_minus_candidate",
    },
}


class AtlasServiceError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def _bad_artifact(message: str) -> AtlasServiceError:
    return AtlasServiceError("invalid_atlas", message, 422)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _bounded_string(value: Any, *, maximum: int = 500) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum


def _validate_taxonomy_atlas(atlas: Any, expected_id: str) -> dict[str, Any]:
    if not isinstance(atlas, dict) or atlas.get("schema") != TAXONOMY_ATLAS_SCHEMA:
        raise _bad_artifact("Atlas schema is missing or unsupported")
    if "id" in atlas and atlas["id"] != expected_id:
        raise _bad_artifact("Atlas identifier does not match its installation directory")

    metadata = atlas.get("metadata")
    if not isinstance(metadata, dict):
        raise _bad_artifact("Atlas metadata is missing")
    if not _bounded_string(metadata.get("model"), maximum=300):
        raise _bad_artifact("Atlas model identity is invalid")
    revision = metadata.get("revision")
    if revision is not None and not _bounded_string(revision, maximum=300):
        raise _bad_artifact("Atlas revision is invalid")
    source_status = metadata.get("source_status")
    if not isinstance(source_status, str) or source_status not in SOURCE_STATUSES:
        raise _bad_artifact("Atlas source status is unsupported")

    groups = atlas.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= 32:
        raise _bad_artifact("Atlas must define between 1 and 32 groups")
    group_ids: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise _bad_artifact("Atlas group entry is invalid")
        group_id = group.get("id")
        if not isinstance(group_id, str) or not GROUP_ID_RE.fullmatch(group_id) or group_id in group_ids:
            raise _bad_artifact("Atlas group identifiers must be unique safe names")
        group_ids.add(group_id)
        if not _bounded_string(group.get("label"), maximum=160):
            raise _bad_artifact("Atlas group label is invalid")
        if not _bounded_string(group.get("referent"), maximum=100):
            raise _bad_artifact("Atlas group referent is invalid")
        if not _bounded_string(group.get("explicitness"), maximum=100):
            raise _bad_artifact("Atlas group explicitness is invalid")
        if not isinstance(group.get("is_control"), bool):
            raise _bad_artifact("Atlas group control flag is invalid")
        if not _bounded_string(group.get("description"), maximum=2000):
            raise _bad_artifact("Atlas group description is invalid")
        if not isinstance(group.get("color"), str) or not HEX_COLOR_RE.fullmatch(group["color"]):
            raise _bad_artifact("Atlas group color must be a six-digit hex value")

    maps = atlas.get("maps")
    if not isinstance(maps, list) or len(maps) > 256:
        raise _bad_artifact("Atlas map collection is invalid")
    map_ids: set[str] = set()
    for map_record in maps:
        if not isinstance(map_record, dict):
            raise _bad_artifact("Atlas map entry is invalid")
        map_id = map_record.get("id")
        if not _bounded_string(map_id, maximum=200) or map_id in map_ids:
            raise _bad_artifact("Atlas map identifiers must be unique")
        map_ids.add(map_id)
        if not _is_int(map_record.get("layer")) or map_record["layer"] not in {6, 13, 20}:
            raise _bad_artifact("Atlas layer must be 6, 13, or 20")
        map_view = map_record.get("view")
        if not isinstance(map_view, str) or map_view not in MAP_VIEWS:
            raise _bad_artifact("Atlas map view is unsupported")
        if not _is_int(map_record.get("seed")) or map_record["seed"] not in {0, 1, 2}:
            raise _bad_artifact("Atlas map seed is invalid")
        if map_record.get("rows") != 6 or map_record.get("cols") != 6:
            raise _bad_artifact("Atlas map must have 6 rows and 6 columns")

        cells = map_record.get("cells")
        if not isinstance(cells, list) or len(cells) != 36:
            raise _bad_artifact("Atlas map must contain 36 cells")
        cell_ids: set[str] = set()
        positions: set[tuple[float, float]] = set()
        for cell in cells:
            if not isinstance(cell, dict):
                raise _bad_artifact("Atlas cell entry is invalid")
            cell_id = cell.get("id")
            if not (isinstance(cell_id, (str, int)) and not isinstance(cell_id, bool)):
                raise _bad_artifact("Atlas cell identifier is invalid")
            cell_key = str(cell_id)
            if cell_key in cell_ids:
                raise _bad_artifact("Atlas cell identifiers must be unique")
            cell_ids.add(cell_key)
            x, y = cell.get("x"), cell.get("y")
            if (not _finite_number(x) or not _finite_number(y)
                    or not (-0.501 <= x <= 5.001 and -0.001 <= y <= 4.331)):
                raise _bad_artifact("Atlas cell coordinates are outside the 6 by 6 hexagonal map")
            position = (float(x), float(y))
            if position in positions:
                raise _bad_artifact("Atlas cell coordinates must be unique")
            positions.add(position)
            for count_key in ("train_count", "eval_count"):
                count = cell.get(count_key)
                if not _is_int(count) or count < 0:
                    raise _bad_artifact("Atlas cell counts must be nonnegative integers")
            for groups_key in ("train_groups", "eval_groups"):
                counts = cell.get(groups_key)
                if not isinstance(counts, dict):
                    raise _bad_artifact("Atlas cell group counts are invalid")
                for group_id, count in counts.items():
                    if group_id not in group_ids or not _is_int(count) or count < 0:
                        raise _bad_artifact("Atlas cell group counts contain an invalid group or count")
            distance = cell.get("mean_distance")
            if distance is not None and not _finite_number(distance):
                raise _bad_artifact("Atlas cell mean distance is invalid")

        points = map_record.get("points", [])
        if not isinstance(points, list) or len(points) > 100_000:
            raise _bad_artifact("Atlas map points are invalid")
        for point in points:
            if not isinstance(point, dict):
                raise _bad_artifact("Atlas map point entry is invalid")
            point_group_id = point.get("group_id")
            if not isinstance(point_group_id, str) or point_group_id not in group_ids:
                raise _bad_artifact("Atlas map point references an unknown group")
            point_split = point.get("split")
            if not isinstance(point_split, str) or point_split not in SPLITS:
                raise _bad_artifact("Atlas map point split must be train or eval")
            point_cell = point.get("cell")
            if point_cell is not None:
                if isinstance(point_cell, (str, int)) and not isinstance(point_cell, bool):
                    if str(point_cell) not in cell_ids:
                        raise _bad_artifact("Atlas map point references an unknown cell")
                elif (isinstance(point_cell, list) and len(point_cell) == 2
                      and all(_finite_number(value) for value in point_cell)
                      and any(abs(float(point_cell[0]) - x) <= 1e-6 and abs(float(point_cell[1]) - y) <= 1e-6
                              for x, y in positions)):
                    pass
                else:
                    raise _bad_artifact("Atlas map point references an unknown cell")
            for coordinate in ("x", "y"):
                value = point.get(coordinate)
                if value is not None and not _finite_number(value):
                    raise _bad_artifact("Atlas map point coordinate is invalid")

    return atlas


def _exact_keys(value: Any, keys: set[str], message: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _bad_artifact(message)
    return value


def _nonnegative_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _transfer_counts(value: Any, *, length: int = 16) -> bool:
    return isinstance(value, list) and len(value) == length and all(_nonnegative_int(row) for row in value)


def _validate_count_triplet(value: Any, fields: set[str], message: str) -> dict[str, Any]:
    row = _exact_keys(value, fields, message)
    if not all(_nonnegative_int(row[field]) for field in fields):
        raise _bad_artifact(message)
    return row


def _validate_transfer_coverage(coverage: Any) -> dict[str, Any]:
    coverage = _exact_keys(
        coverage,
        {"train", "validation", "heldout", "source_cohort", "synthetic_worlds"},
        "Transfer atlas coverage fields are invalid",
    )
    for split in TRANSFER_SPLITS:
        _validate_count_triplet(
            coverage[split], {"pairs", "rows", "families"},
            "Transfer atlas split coverage is invalid",
        )

    source = _exact_keys(
        coverage["source_cohort"],
        {"train", "validation", "heldout", "by_collection"},
        "Transfer atlas source-cohort coverage is invalid",
    )
    for split in TRANSFER_SPLITS:
        _validate_count_triplet(
            source[split], {"components", "pairs"},
            "Transfer atlas source-cohort split coverage is invalid",
        )
    collections = _exact_keys(
        source["by_collection"], set(TRANSFER_DATASETS),
        "Transfer atlas source collection coverage is invalid",
    )
    for dataset in TRANSFER_DATASETS:
        _exact_keys(collections[dataset], set(TRANSFER_SPLITS), "Transfer atlas source collection splits are invalid")
        if not all(_nonnegative_int(collections[dataset][split]) for split in TRANSFER_SPLITS):
            raise _bad_artifact("Transfer atlas source collection counts are invalid")
    for split in TRANSFER_SPLITS:
        if sum(collections[dataset][split] for dataset in TRANSFER_DATASETS) != source[split]["components"]:
            raise _bad_artifact("Transfer atlas source collection counts do not match source components")

    worlds = _exact_keys(coverage["synthetic_worlds"], set(TRANSFER_SPLITS), "Transfer atlas synthetic-world coverage is invalid")
    for split in TRANSFER_SPLITS:
        _validate_count_triplet(
            worlds[split], {"families", "pairs"},
            "Transfer atlas synthetic-world split coverage is invalid",
        )
    return coverage


def _validate_transfer_groups(groups: Any) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    if not isinstance(groups, list) or not 1 <= len(groups) <= 64:
        raise _bad_artifact("Transfer atlas must define between 1 and 64 groups")
    ids: set[str] = set()
    controls: set[str] = set()
    synthetic: set[str] = set()
    for group in groups:
        group = _exact_keys(
            group,
            {"id", "label", "is_control", "is_source_cohort", "semantic_status"},
            "Transfer atlas group fields are invalid",
        )
        group_id = group.get("id")
        if not isinstance(group_id, str) or not GROUP_ID_RE.fullmatch(group_id) or group_id in ids:
            raise _bad_artifact("Transfer atlas group identifiers must be unique safe names")
        ids.add(group_id)
        if not _bounded_string(group.get("label"), maximum=160):
            raise _bad_artifact("Transfer atlas group label is invalid")
        if not isinstance(group.get("is_control"), bool) or not isinstance(group.get("is_source_cohort"), bool):
            raise _bad_artifact("Transfer atlas group flags are invalid")
        if not _bounded_string(group.get("semantic_status"), maximum=160):
            raise _bad_artifact("Transfer atlas group semantic status is invalid")
        if group["is_control"]:
            controls.add(group_id)
        if not group["is_source_cohort"]:
            synthetic.add(group_id)
    if not synthetic:
        raise _bad_artifact("Transfer atlas has no synthetic behavior groups")
    return groups, ids, controls


def _validate_change_table(value: Any, allowed_ids: set[str], count_key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != allowed_ids:
        raise _bad_artifact("Transfer atlas outcome groups are incomplete or unknown")
    for row in value.values():
        row = _exact_keys(row, {"mean_margin_change", count_key}, "Transfer atlas outcome summary is invalid")
        count = row[count_key]
        mean = row["mean_margin_change"]
        if (not _nonnegative_int(count)
                or (count == 0 and mean is not None)
                or (count > 0 and not _finite_number(mean))):
            raise _bad_artifact("Transfer atlas outcome summary contains an invalid value")
    return value


def _validate_transfer_likelihood_summary(summary: Any, layers: set[int], synthetic: set[str], controls: set[str]) -> dict[str, Any]:
    summary = _exact_keys(
        summary,
        {"label", "surrogate", "metric", "selection", "validation_grid", "heldout_arms"},
        "Transfer atlas likelihood summary fields are invalid",
    )
    if (summary.get("label") != "Teacher-forced likelihood steering screen"
            or summary.get("surrogate") is not True
            or summary.get("metric") != "mean token logprob(independent) - mean token logprob(candidate)"):
        raise _bad_artifact("Transfer atlas likelihood summary label or metric is unsupported")

    selection = _exact_keys(
        summary.get("selection"),
        {"layer", "alpha", "reason", "validation_macro_delta", "control_gate_limit_nat_per_token", "control_gate_values"},
        "Transfer atlas selection summary is invalid",
    )
    if (not _is_int(selection.get("layer")) or selection["layer"] not in layers
            or not _finite_number(selection.get("alpha")) or selection["alpha"] < 0
            or not _bounded_string(selection.get("reason"), maximum=2000)
            or not _finite_number(selection.get("validation_macro_delta"))
            or not _finite_number(selection.get("control_gate_limit_nat_per_token"))
            or selection["control_gate_limit_nat_per_token"] < 0):
        raise _bad_artifact("Transfer atlas selection values are invalid")
    gate_values = selection.get("control_gate_values")
    if not isinstance(gate_values, dict) or set(gate_values) != controls:
        raise _bad_artifact("Transfer atlas control-gate values are incomplete or unknown")
    if not all(_finite_number(value) for value in gate_values.values()):
        raise _bad_artifact("Transfer atlas control-gate values are invalid")

    grid = summary.get("validation_grid")
    if not isinstance(grid, list) or not 1 <= len(grid) <= 512:
        raise _bad_artifact("Transfer atlas validation grid is invalid")
    grid_keys: set[tuple[int, float]] = set()
    grid_layers: set[int] = set()
    grid_alphas: set[float] = set()
    grid_fields = {
        "layer", "alpha", "macro_delta", "control_max_abs_mean_change", "passes_control_gate",
        "selection_eligible", "synthetic_group_changes", "control_group_changes",
        "eligible_pair_count", "abstention_pair_count",
    }
    for row in grid:
        row = _exact_keys(row, grid_fields, "Transfer atlas validation-grid row is invalid")
        layer, alpha = row.get("layer"), row.get("alpha")
        if (not _is_int(layer) or layer not in layers or not _finite_number(alpha) or alpha < 0
                or not _finite_number(row.get("macro_delta"))
                or not _finite_number(row.get("control_max_abs_mean_change"))
                or row.get("control_max_abs_mean_change") < 0
                or not isinstance(row.get("passes_control_gate"), bool)
                or not isinstance(row.get("selection_eligible"), bool)
                or not _nonnegative_int(row.get("eligible_pair_count"))
                or not _nonnegative_int(row.get("abstention_pair_count"))):
            raise _bad_artifact("Transfer atlas validation-grid values are invalid")
        key = (layer, float(alpha))
        if key in grid_keys:
            raise _bad_artifact("Transfer atlas validation grid contains a duplicate layer and strength")
        grid_keys.add(key)
        grid_layers.add(layer)
        grid_alphas.add(float(alpha))
        _validate_change_table(row["synthetic_group_changes"], synthetic, "family_count")
        _validate_change_table(row["control_group_changes"], controls, "family_count")
    if len(grid_layers) < 2 or len(grid_alphas) < 2:
        raise _bad_artifact("Transfer atlas validation grid must sweep multiple layers and strengths")

    arms = _exact_keys(summary.get("heldout_arms"), TRANSFER_METHODS, "Transfer atlas held-out arms are incomplete or unknown")
    for row in arms.values():
        row = _exact_keys(
            row,
            {"macro_delta", "eligible_pairs", "abstentions", "group_changes", "control_changes",
             "source_collection_changes", "family_bootstrap_interval"},
            "Transfer atlas held-out arm summary is invalid",
        )
        if (not _finite_number(row.get("macro_delta"))
                or not _nonnegative_int(row.get("eligible_pairs"))
                or not _nonnegative_int(row.get("abstentions"))):
            raise _bad_artifact("Transfer atlas held-out arm values are invalid")
        _validate_change_table(row["group_changes"], synthetic, "family_count")
        _validate_change_table(row["control_changes"], controls, "family_count")
        source_changes = row["source_collection_changes"]
        if not isinstance(source_changes, dict) or set(source_changes) != set(TRANSFER_DATASETS):
            raise _bad_artifact("Transfer atlas source-collection outcomes are incomplete or unknown")
        for outcome in source_changes.values():
            outcome = _exact_keys(outcome, {"mean_margin_change", "component_count"}, "Transfer atlas source outcome is invalid")
            count = outcome["component_count"]
            mean = outcome["mean_margin_change"]
            if (not _nonnegative_int(count)
                    or (count == 0 and mean is not None)
                    or (count > 0 and not _finite_number(mean))):
                raise _bad_artifact("Transfer atlas source outcome contains an invalid value")
        interval = row["family_bootstrap_interval"]
        if (not isinstance(interval, list) or len(interval) != 2
                or not all(_finite_number(value) for value in interval)
                or interval[0] > interval[1]):
            raise _bad_artifact("Transfer atlas family-bootstrap interval is invalid")
    return summary


def _validate_transfer_map(map_record: Any, group_ids: set[str], layers: set[int]) -> dict[str, Any]:
    map_record = _exact_keys(
        map_record,
        {"id", "layer", "map_kind", "view", "rows", "columns", "topology", "fit_split",
         "used_for_routing", "codebook_ref", "bmu_ref", "cell_coordinates", "cell_summaries",
         "synthetic_cell_summaries", "source_cell_summaries", "group_counts", "source_group_counts",
         "routing_eligible_cells", "routing_eligible_counts"},
        "Transfer atlas map fields are invalid",
    )
    if (not isinstance(map_record.get("id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}", map_record["id"])
            or not _is_int(map_record.get("layer")) or map_record["layer"] not in layers
            or not isinstance(map_record.get("map_kind"), str)
            or map_record.get("map_kind") not in TRANSFER_MAP_VIEWS
            or not isinstance(map_record.get("view"), str)
            or map_record.get("view") not in TRANSFER_MAP_VIEWS[map_record["map_kind"]]
            or map_record.get("rows") != 4 or map_record.get("columns") != 4
            or map_record.get("topology") != "hexagonal"
            or map_record.get("fit_split") != "train"
            or not isinstance(map_record.get("used_for_routing"), bool)
            or map_record["used_for_routing"] is not (map_record["map_kind"] == "prompt_routing")):
        raise _bad_artifact("Transfer atlas map identity or routing fields are invalid")

    codebook_ref = _exact_keys(map_record["codebook_ref"], {"artifact_id", "array_name"}, "Transfer atlas codebook reference is invalid")
    bmu_ref = _exact_keys(map_record["bmu_ref"], {"artifact_id", "field_name"}, "Transfer atlas BMU reference is invalid")
    if (codebook_ref.get("artifact_id") != "som_artifacts"
            or not isinstance(codebook_ref.get("array_name"), str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", codebook_ref["array_name"])
            or bmu_ref.get("artifact_id") != "map_points"
            or bmu_ref.get("field_name") != "cell_id"):
        raise _bad_artifact("Transfer atlas artifact references are invalid")

    coordinates = map_record.get("cell_coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 16:
        raise _bad_artifact("Transfer atlas must define 16 cell coordinates")
    cell_ids: list[int] = []
    positions: set[tuple[float, float]] = set()
    for cell in coordinates:
        cell = _exact_keys(cell, {"cell_id", "x", "y"}, "Transfer atlas cell coordinate is invalid")
        cell_id, x, y = cell.get("cell_id"), cell.get("x"), cell.get("y")
        if (not _is_int(cell_id) or not 0 <= cell_id < 16
                or not _finite_number(x) or not _finite_number(y)
                or not -1 <= x <= 4 or not 0 <= y <= 4):
            raise _bad_artifact("Transfer atlas cell coordinate is outside the 4 by 4 hexagonal map")
        cell_ids.append(cell_id)
        position = (float(x), float(y))
        if position in positions:
            raise _bad_artifact("Transfer atlas cell coordinates must be unique")
        positions.add(position)
    if cell_ids != list(range(16)):
        raise _bad_artifact("Transfer atlas cells must follow row-major order")

    summaries = _exact_keys(map_record["cell_summaries"], set(TRANSFER_SPLITS), "Transfer atlas cell summaries are invalid")
    synthetic_summaries = _exact_keys(map_record["synthetic_cell_summaries"], set(TRANSFER_SPLITS), "Transfer atlas synthetic cell summaries are invalid")
    source_summaries = _exact_keys(map_record["source_cell_summaries"], set(TRANSFER_SPLITS), "Transfer atlas source cell summaries are invalid")
    eligible_summaries = _exact_keys(map_record["routing_eligible_counts"], set(TRANSFER_SPLITS), "Transfer atlas routing-eligibility counts are invalid")
    group_counts = _exact_keys(map_record["group_counts"], set(TRANSFER_SPLITS), "Transfer atlas group counts are invalid")
    source_group_counts = _exact_keys(map_record["source_group_counts"], set(TRANSFER_SPLITS), "Transfer atlas source-group counts are invalid")
    eligible_cells = map_record.get("routing_eligible_cells")
    if not isinstance(eligible_cells, list) or len(eligible_cells) != 16 or not all(isinstance(value, bool) for value in eligible_cells):
        raise _bad_artifact("Transfer atlas routing-eligibility mask must contain 16 booleans")
    if map_record["map_kind"] == "output_contrast" and any(eligible_cells):
        raise _bad_artifact("Descriptive output maps cannot contain route-eligible cells")
    for split in TRANSFER_SPLITS:
        if (not _transfer_counts(summaries[split])
                or not _transfer_counts(synthetic_summaries[split])
                or not _transfer_counts(source_summaries[split])):
            raise _bad_artifact("Transfer atlas cell counts must contain 16 nonnegative integers")
        if not _transfer_counts(eligible_summaries[split]):
            raise _bad_artifact("Transfer atlas routing-eligibility counts must contain 16 nonnegative integers")
        for total, synthetic_total, source_total in zip(summaries[split], synthetic_summaries[split], source_summaries[split], strict=True):
            if total != synthetic_total + source_total:
                raise _bad_artifact("Transfer atlas source and synthetic cell counts do not sum to the cell totals")
        for eligible, total, eligible_cell in zip(eligible_summaries[split], summaries[split], eligible_cells, strict=True):
            if eligible > total:
                raise _bad_artifact("Transfer atlas route-eligible counts exceed the cell totals")
            if not eligible_cell and eligible:
                raise _bad_artifact("Transfer atlas has route-eligible prompts in a gated cell")
            if map_record["map_kind"] == "output_contrast" and eligible:
                raise _bad_artifact("Descriptive output maps cannot have route-eligible prompt counts")
        if map_record["map_kind"] == "output_contrast" and any(eligible_summaries[split]):
            raise _bad_artifact("Descriptive output maps cannot have route-eligible prompt counts")

        synthetic_rows = group_counts[split]
        if not isinstance(synthetic_rows, list) or len(synthetic_rows) != len(group_ids):
            raise _bad_artifact("Transfer atlas synthetic group count rows are incomplete")
        synthetic_seen: set[str] = set()
        synthetic_cell_sums = [0] * 16
        for row in synthetic_rows:
            row = _exact_keys(row, {"group_id", "counts"}, "Transfer atlas group count row is invalid")
            group_id = row.get("group_id")
            counts = row.get("counts")
            if (not isinstance(group_id, str) or group_id not in group_ids or group_id in synthetic_seen
                    or not _transfer_counts(counts)):
                raise _bad_artifact("Transfer atlas group counts reference an invalid group or count")
            synthetic_seen.add(group_id)
            synthetic_cell_sums = [total + value for total, value in zip(synthetic_cell_sums, counts, strict=True)]
        if synthetic_seen != group_ids or synthetic_cell_sums != synthetic_summaries[split]:
            raise _bad_artifact("Transfer atlas synthetic group counts do not partition synthetic cell totals")

        source_rows = source_group_counts[split]
        if not isinstance(source_rows, list) or len(source_rows) != len(TRANSFER_DATASETS):
            raise _bad_artifact("Transfer atlas source collection count rows are incomplete")
        source_seen: set[str] = set()
        source_cell_sums = [0] * 16
        for row in source_rows:
            row = _exact_keys(row, {"source_collection", "counts"}, "Transfer atlas source count row is invalid")
            dataset = row.get("source_collection")
            counts = row.get("counts")
            if (not isinstance(dataset, str) or dataset not in TRANSFER_DATASETS or dataset in source_seen
                    or not _transfer_counts(counts)):
                raise _bad_artifact("Transfer atlas source counts reference an invalid collection or count")
            source_seen.add(dataset)
            source_cell_sums = [total + value for total, value in zip(source_cell_sums, counts, strict=True)]
        if source_seen != set(TRANSFER_DATASETS) or source_cell_sums != source_summaries[split]:
            raise _bad_artifact("Transfer atlas source collection counts do not partition source cell totals")

        # Ensure the combined cell totals are derivable from these two safe aggregate tables.
        if [left + right for left, right in zip(synthetic_cell_sums, source_cell_sums, strict=True)] != summaries[split]:
            raise _bad_artifact("Transfer atlas group and source counts do not partition total cell counts")
    return map_record


def _validate_transfer_atlas(atlas: Any, expected_id: str) -> dict[str, Any]:
    atlas = _exact_keys(
        atlas,
        {"schema", "id", "title", "metadata", "groups", "coverage", "maps", "likelihood_summary"},
        "Transfer atlas fields are invalid",
    )
    if atlas.get("schema") != TRANSFER_ATLAS_SCHEMA or atlas.get("id") != expected_id:
        raise _bad_artifact("Transfer atlas schema or identifier is invalid")
    if not _bounded_string(atlas.get("title"), maximum=200):
        raise _bad_artifact("Transfer atlas display title is invalid")

    metadata = _exact_keys(
        atlas.get("metadata"),
        {"model", "revision", "device", "source_status", "claim_ceiling", "capture_layers",
         "not_established", "map_layout", "selected_layer", "selected_alpha", "selected_method", "status"},
        "Transfer atlas metadata fields are invalid",
    )
    if (not _bounded_string(metadata.get("model"), maximum=300)
            or (metadata.get("revision") is not None and not _bounded_string(metadata.get("revision"), maximum=300))
            or not _bounded_string(metadata.get("device"), maximum=64)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", metadata["device"])
            or metadata.get("source_status") != "synthetic_unreviewed"
            or metadata.get("claim_ceiling") != TRANSFER_CLAIM_CEILING
            or not isinstance(metadata.get("status"), str)
            or metadata.get("status") not in {"heldout_scored", "complete"}):
        raise _bad_artifact("Transfer atlas provenance or status is invalid")
    not_established = metadata.get("not_established")
    if (not isinstance(not_established, list) or not 1 <= len(not_established) <= 32
            or not all(_bounded_string(value, maximum=500) for value in not_established)
            or len(set(not_established)) != len(not_established)):
        raise _bad_artifact("Transfer atlas claim-limit caveats are invalid")
    layers = metadata.get("capture_layers")
    if (not isinstance(layers, list) or len(layers) != len(TRANSFER_LAYERS)
            or any(not _is_int(layer) for layer in layers)
            or len(set(layers)) != len(layers) or set(layers) != set(TRANSFER_LAYERS)):
        raise _bad_artifact("Transfer atlas must declare the eight supported capture layers")
    layout = _exact_keys(metadata.get("map_layout"), {"rows", "columns", "topology", "cell_order"}, "Transfer atlas map layout is invalid")
    if layout != {"rows": 4, "columns": 4, "topology": "hexagonal", "cell_order": "row_major"}:
        raise _bad_artifact("Transfer atlas must use a row-major 4 by 4 hexagonal map")
    if (not _is_int(metadata.get("selected_layer")) or metadata["selected_layer"] not in layers
            or not _finite_number(metadata.get("selected_alpha")) or metadata["selected_alpha"] < 0
            or not isinstance(metadata.get("selected_method"), str)
            or metadata.get("selected_method") not in TRANSFER_METHODS):
        raise _bad_artifact("Transfer atlas selected steering setting is invalid")

    groups, group_ids, controls = _validate_transfer_groups(atlas.get("groups"))
    synthetic = {group["id"] for group in groups if not group["is_source_cohort"]}
    if any(group["is_source_cohort"] for group in groups):
        raise _bad_artifact("Source-cohort rows must remain separate from synthetic behavior groups")
    if not controls:
        raise _bad_artifact("Transfer atlas must define at least one synthetic control group")
    _validate_transfer_coverage(atlas.get("coverage"))

    maps = atlas.get("maps")
    if not isinstance(maps, list) or not 1 <= len(maps) <= 64:
        raise _bad_artifact("Transfer atlas map collection is invalid")
    map_ids: set[str] = set()
    prompt_layers: set[int] = set()
    map_keys: set[tuple[int, str, str]] = set()
    for row in maps:
        validated = _validate_transfer_map(row, group_ids, set(layers))
        if validated["id"] in map_ids:
            raise _bad_artifact("Transfer atlas map identifiers must be unique")
        map_ids.add(validated["id"])
        key = (validated["layer"], validated["map_kind"], validated["view"])
        if key in map_keys:
            raise _bad_artifact("Transfer atlas contains a duplicate layer and readout map")
        map_keys.add(key)
        if validated["map_kind"] == "prompt_routing":
            prompt_layers.add(validated["layer"])
        elif validated["used_for_routing"]:
            raise _bad_artifact("Descriptive output maps cannot be used for routing")
    expected_maps = {
        (layer, map_kind, view)
        for layer in layers
        for map_kind, views in TRANSFER_MAP_VIEWS.items()
        for view in views
    }
    if prompt_layers != set(layers) or map_keys != expected_maps:
        raise _bad_artifact("Transfer atlas must include each prompt and descriptive readout at every capture layer")

    summary = _validate_transfer_likelihood_summary(atlas.get("likelihood_summary"), set(layers), synthetic, controls)
    selection = summary["selection"]
    if (selection["layer"] != metadata["selected_layer"]
            or float(selection["alpha"]) != float(metadata["selected_alpha"])):
        raise _bad_artifact("Transfer atlas metadata and likelihood selection disagree")
    if metadata["selected_method"] not in summary["heldout_arms"]:
        raise _bad_artifact("Transfer atlas selected method is missing from held-out results")
    selected_rows = [
        row for row in summary["validation_grid"]
        if row["layer"] == selection["layer"] and float(row["alpha"]) == float(selection["alpha"])
    ]
    if (len(selected_rows) != 1
            or (float(selection["alpha"]) > 0 and selected_rows[0]["selection_eligible"] is not True)
            or selected_rows[0]["passes_control_gate"] is not True):
        raise _bad_artifact("Transfer atlas selected setting must pass its recorded validation gates")
    return atlas


def _validate_atlas(atlas: Any, expected_id: str) -> dict[str, Any]:
    if not isinstance(atlas, dict):
        raise _bad_artifact("Atlas must be a JSON object")
    if atlas.get("schema") == TAXONOMY_ATLAS_SCHEMA:
        return _validate_taxonomy_atlas(atlas, expected_id)
    if atlas.get("schema") == TRANSFER_ATLAS_SCHEMA:
        return _validate_transfer_atlas(atlas, expected_id)
    raise _bad_artifact("Atlas schema is missing or unsupported")


class AtlasService:
    """Read installed atlases beneath one fixed, application-owned directory."""

    def __init__(self, repo_root: str | Path):
        self.atlas_root = (Path(repo_root) / "research_outputs" / "atlases").resolve()

    def _validate_id(self, atlas_id: str) -> None:
        if not isinstance(atlas_id, str) or not ATLAS_ID_RE.fullmatch(atlas_id):
            raise AtlasServiceError("not_found", "Atlas was not found", 404)

    def _installation_dir(self, atlas_id: str) -> Path:
        self._validate_id(atlas_id)
        directory = self.atlas_root / atlas_id
        if directory.is_symlink() or not directory.is_dir():
            raise AtlasServiceError("not_found", "Atlas was not found", 404)
        try:
            if directory.resolve().parent != self.atlas_root:
                raise AtlasServiceError("not_found", "Atlas was not found", 404)
        except OSError:
            raise AtlasServiceError("not_found", "Atlas was not found", 404) from None
        return directory

    def _load(self, atlas_id: str) -> tuple[dict[str, Any], dict[str, Any], str, int]:
        directory = self._installation_dir(atlas_id)
        atlas_path = directory / "atlas.json"
        receipt_path = directory / "receipt.json"
        if atlas_path.is_symlink() or receipt_path.is_symlink():
            raise _bad_artifact("Atlas receipt or JSON must be a regular file")
        if not atlas_path.is_file() or not receipt_path.is_file():
            raise AtlasServiceError("not_found", "Atlas was not found", 404)
        try:
            size = atlas_path.stat().st_size
            if size > MAX_ATLAS_BYTES:
                raise AtlasServiceError("atlas_too_large", "Atlas JSON exceeds the 32 MiB limit", 413)
            if size <= 0:
                raise _bad_artifact("Atlas JSON is empty")
            raw = atlas_path.read_bytes()
            if len(raw) > MAX_ATLAS_BYTES:
                raise AtlasServiceError("atlas_too_large", "Atlas JSON exceeds the 32 MiB limit", 413)
            if receipt_path.stat().st_size > 64 * 1024:
                raise _bad_artifact("Atlas receipt exceeds the size limit")
            receipt_raw = receipt_path.read_bytes()
            if len(receipt_raw) > 64 * 1024:
                raise _bad_artifact("Atlas receipt exceeds the size limit")
        except AtlasServiceError:
            raise
        except OSError:
            raise AtlasServiceError("not_found", "Atlas was not found", 404) from None
        if len(raw) != size:
            raise _bad_artifact("Atlas changed while it was being read")
        try:
            receipt = json.loads(receipt_raw, object_pairs_hook=_object_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise _bad_artifact("Atlas receipt is not valid JSON") from None
        if not isinstance(receipt, dict):
            raise _bad_artifact("Atlas receipt must be a JSON object")
        if set(receipt) != {"schema", "id", "atlas_sha256", "atlas_bytes"}:
            raise _bad_artifact("Atlas receipt fields are invalid")
        expected_hash = receipt.get("atlas_sha256")
        expected_size = receipt.get("atlas_bytes")
        if (receipt.get("schema") != "taxonomy-som-atlas-receipt-v1"
                or receipt.get("id") != atlas_id
                or not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or not _is_int(expected_size)
                or expected_size != len(raw)):
            raise _bad_artifact("Atlas receipt does not match the installed file")
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != expected_hash:
            raise _bad_artifact("Atlas SHA-256 does not match its receipt")
        try:
            atlas = json.loads(raw, object_pairs_hook=_object_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise _bad_artifact("Atlas file is not valid JSON") from None
        return _validate_atlas(atlas, atlas_id), receipt, actual_hash, len(raw)

    def get_atlas(self, atlas_id: str) -> dict[str, Any]:
        atlas, _, _, _ = self._load(atlas_id)
        return atlas

    def list_atlases(self) -> dict[str, Any]:
        result: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        try:
            entries = sorted(self.atlas_root.iterdir(), key=lambda item: item.name.casefold())
        except FileNotFoundError:
            return {"atlases": result, "rejected": rejected}
        except OSError:
            raise AtlasServiceError("atlas_unavailable", "Installed atlases could not be read", 503) from None
        for directory in entries:
            if not directory.is_dir() or directory.is_symlink() or not ATLAS_ID_RE.fullmatch(directory.name):
                continue
            try:
                atlas, receipt, sha256, size = self._load(directory.name)
            except AtlasServiceError as error:
                rejected.append({"id": directory.name, "code": error.code})
                continue
            metadata = atlas["metadata"]
            title = atlas.get("title") or metadata["model"]
            result.append({
                "id": directory.name,
                "title": title,
                "run_label": f"{title} · {directory.name}",
                "schema": atlas["schema"],
                "model": metadata["model"],
                "revision": metadata.get("revision"),
                "source_status": metadata["source_status"],
                "group_count": len(atlas["groups"]),
                "map_count": len(atlas["maps"]),
                "atlas_sha256": sha256,
                "atlas_bytes": size,
                "receipt_schema": receipt["schema"],
            })
        return {"atlases": result, "rejected": rejected}
