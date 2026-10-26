"""Input schemas and activation storage for paired SOM experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("neutral", "false_pressure", "true_claim", "valid_correction")
SPLITS = ("train", "validation", "test")
PROMPT_FIELDS = {"question_id", "split", "condition", "prompt", "target", "foil"}


class DataError(ValueError):
    """A prompt or activation file is malformed or inconsistent."""


@dataclass(frozen=True)
class ActivationData:
    vectors: np.ndarray
    question_ids: np.ndarray
    splits: np.ndarray
    conditions: np.ndarray
    last_token_ids: np.ndarray
    prompt_token_counts: np.ndarray
    metadata: dict[str, Any]


def validate_prompt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate paired rows and enforce question-level split isolation."""
    if not rows:
        raise DataError("prompt file has no records")

    by_question: dict[str, dict[str, Any]] = {}
    split_questions: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts = {split: 0 for split in SPLITS}

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise DataError(f"line {index}: expected a JSON object")
        missing = PROMPT_FIELDS - row.keys()
        unknown = row.keys() - PROMPT_FIELDS
        if missing:
            raise DataError(f"line {index}: missing fields {', '.join(sorted(missing))}")
        if unknown:
            raise DataError(f"line {index}: unknown fields {', '.join(sorted(unknown))}")

        question_id = row["question_id"]
        split = row["split"]
        condition = row["condition"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise DataError(f"line {index}: question_id must be a non-empty string")
        if split not in SPLITS:
            raise DataError(f"line {index}: split must be one of {', '.join(SPLITS)}")
        if condition not in CONDITIONS:
            raise DataError(f"line {index}: condition must be one of {', '.join(CONDITIONS)}")
        for field in ("prompt", "target", "foil"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise DataError(f"line {index}: {field} must be a non-empty string")
        if row["target"] == row["foil"]:
            raise DataError(f"line {index}: target and foil must differ")

        current = by_question.get(question_id)
        if current is None:
            current = {"split": split, "conditions": set(), "target": row["target"], "foil": row["foil"]}
            by_question[question_id] = current
        elif current["split"] != split:
            raise DataError(
                f"question_id {question_id!r} appears in both {current['split']} and {split}; "
                "keep every condition for a question in one split"
            )
        elif current["target"] != row["target"] or current["foil"] != row["foil"]:
            raise DataError(f"question_id {question_id!r} has inconsistent target or foil text")

        if condition in current["conditions"]:
            raise DataError(f"question_id {question_id!r} repeats condition {condition!r}")
        current["conditions"].add(condition)
        split_questions[split].add(question_id)
        counts[split] += 1

    for question_id, item in by_question.items():
        missing = set(CONDITIONS) - item["conditions"]
        if missing:
            raise DataError(
                f"question_id {question_id!r} is missing conditions {', '.join(sorted(missing))}"
            )

    empty_splits = [split for split, question_ids in split_questions.items() if not question_ids]
    if empty_splits:
        raise DataError(f"prompt file needs questions in every split; empty: {', '.join(empty_splits)}")

    return {
        "records": len(rows),
        "questions": len(by_question),
        "questions_by_split": {split: len(ids) for split, ids in split_questions.items()},
        "records_by_split": counts,
    }


def prompt_dataset_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash all prompt/scoring text independent of JSONL line order or whitespace."""
    canonical_rows = sorted(rows, key=lambda row: (row["question_id"], row["condition"]))
    payload = json.dumps(canonical_rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prompt_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the strict paired prompt JSONL schema."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"line {line_number}: invalid JSON ({exc.msg})") from None
                if not isinstance(row, dict):
                    raise DataError(f"line {line_number}: expected a JSON object")
                rows.append(row)
    except OSError as exc:
        raise DataError(f"cannot read prompt file {path}: {exc.strerror or exc}") from None
    validate_prompt_rows(rows)
    return rows


def write_prompt_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    validate_prompt_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_activations(path: str | Path, data: ActivationData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.vectors.ndim != 2 or not np.isfinite(data.vectors).all():
        raise DataError("activation vectors must be a finite two-dimensional array")
    if len(data.vectors) != len(data.question_ids):
        raise DataError("activation metadata row counts do not match vectors")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            vectors=np.asarray(data.vectors, dtype=np.float32),
            question_ids=np.asarray(data.question_ids, dtype=str),
            splits=np.asarray(data.splits, dtype=str),
            conditions=np.asarray(data.conditions, dtype=str),
            last_token_ids=np.asarray(data.last_token_ids, dtype=np.int64),
            prompt_token_counts=np.asarray(data.prompt_token_counts, dtype=np.int64),
            metadata=np.asarray(json.dumps(data.metadata, sort_keys=True)),
        )
    temporary.replace(path)


def load_activations(path: str | Path) -> ActivationData:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "vectors", "question_ids", "splits", "conditions", "last_token_ids",
                "prompt_token_counts", "metadata",
            }
            missing = required - set(archive.files)
            if missing:
                raise DataError(f"activation file is missing arrays: {', '.join(sorted(missing))}")
            data = ActivationData(
                vectors=np.asarray(archive["vectors"], dtype=np.float32),
                question_ids=np.asarray(archive["question_ids"], dtype=str),
                splits=np.asarray(archive["splits"], dtype=str),
                conditions=np.asarray(archive["conditions"], dtype=str),
                last_token_ids=np.asarray(archive["last_token_ids"], dtype=np.int64),
                prompt_token_counts=np.asarray(archive["prompt_token_counts"], dtype=np.int64),
                metadata=json.loads(str(archive["metadata"].item())),
            )
    except DataError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read activation file {path}: {exc}") from None

    arrays = (data.question_ids, data.splits, data.conditions, data.last_token_ids, data.prompt_token_counts)
    n = len(data.vectors)
    if data.vectors.ndim != 2 or not np.isfinite(data.vectors).all():
        raise DataError("activation vectors must be a finite two-dimensional array")
    if any(array.ndim != 1 or len(array) != n for array in arrays):
        raise DataError("activation metadata arrays must be one-dimensional and match vector rows")
    if not isinstance(data.metadata, dict):
        raise DataError("activation metadata must be a JSON object")
    keys = list(zip(data.question_ids.tolist(), data.conditions.tolist()))
    if len(set(keys)) != n:
        raise DataError("activation file has duplicate question_id/condition rows")
    return data


def validate_activation_match(rows: list[dict[str, Any]], data: ActivationData) -> None:
    expected = {(r["question_id"], r["split"], r["condition"]) for r in rows}
    actual = set(zip(data.question_ids.tolist(), data.splits.tolist(), data.conditions.tolist()))
    if expected != actual:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise DataError(f"prompt and activation rows differ (missing activations: {missing}, extra: {extra})")
