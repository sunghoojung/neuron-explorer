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


def _validate_atlas(atlas: Any, expected_id: str) -> dict[str, Any]:
    if not isinstance(atlas, dict) or atlas.get("schema") != "taxonomy-som-atlas-v1":
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
            result.append({
                "id": directory.name,
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
