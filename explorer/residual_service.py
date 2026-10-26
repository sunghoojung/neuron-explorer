"""Bounded local service for grouped residual captures and train-only SOMs.

This service owns only app-created artifacts under research_outputs. Model
activations are collected through the shared Runtime and residual_capture
helpers; this module never loads a second model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from research.som import fit_control_bases, orthonormal_basis, train_som
except ImportError:  # server.py places explorer/ on sys.path
    from ..research.som import fit_control_bases, orthonormal_basis, train_som

try:
    from .residual_capture import (
        ResidualCaptureCancelled,
        capture_residuals,
        score_residual_continuation,
    )
except ImportError:  # server.py places explorer/ on sys.path
    from residual_capture import (
        ResidualCaptureCancelled,
        capture_residuals,
        score_residual_continuation,
    )


POINTS = ("resid_pre", "resid_mid", "resid_post", "attn_write", "mlp_write")
SPLITS = ("train", "validation", "test")
MAX_SAMPLES = 64
MAX_GROUPS = 8
MAX_LAYERS = 28
MAX_WINDOW = 16
MAX_PROMPT_TOKENS = 512
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_DATASET_JSON_BYTES = 4 * 1024 * 1024
MAX_PENDING_JOBS = 2
MAX_GRID_SIDE = 8
MAX_GRID_UNITS = 64
MAX_EPOCHS = 500
MAX_PROBE_SAMPLES = 8
_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_NS_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_ACTIVE = {"queued", "running"}


class ResidualServiceError(ValueError):
    """A safe, bounded error from the local residual service."""

    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class _Cancelled(Exception):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _bounded_string(value: Any, field: str, *, maximum: int = 160) -> str:
    if not isinstance(value, str):
        raise ResidualServiceError("invalid_dataset", f"{field} must be text")
    result = value.strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise ResidualServiceError("invalid_dataset", f"{field} must be non-empty text no longer than {maximum} characters")
    return result


def validate_dataset(dataset: Any) -> dict[str, Any]:
    """Validate and normalize the canonical group dataset schema."""
    if not isinstance(dataset, dict) or set(dataset) - {"name", "samples"}:
        raise ResidualServiceError("invalid_dataset", "Dataset must contain only name and samples")
    name = _bounded_string(dataset.get("name"), "name", maximum=120)
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise ResidualServiceError("invalid_dataset", f"samples must contain 1 to {MAX_SAMPLES} rows")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    pair_split: dict[str, tuple[str, str]] = {}
    family_split: dict[str, str] = {}
    pair_groups: set[tuple[str, str]] = set()
    groups: set[str] = set()
    total_chars = 0
    for index, raw in enumerate(samples):
        if not isinstance(raw, dict) or set(raw) - {"id", "pair_id", "family_id", "split", "group", "messages", "target", "foil"}:
            raise ResidualServiceError("invalid_dataset", f"sample {index} has unknown fields or is not an object")
        sample_id = _bounded_string(raw.get("id"), f"samples[{index}].id")
        pair_id = _bounded_string(raw.get("pair_id"), f"samples[{index}].pair_id")
        family_id = _bounded_string(raw.get("family_id"), f"samples[{index}].family_id")
        group = _bounded_string(raw.get("group"), f"samples[{index}].group", maximum=80)
        split = raw.get("split")
        if split not in SPLITS:
            raise ResidualServiceError("invalid_dataset", f"sample {sample_id} split must be train, validation, or test")
        if sample_id in ids:
            raise ResidualServiceError("invalid_dataset", f"sample id {sample_id} is duplicated")
        ids.add(sample_id)
        groups.add(group)
        if len(groups) > MAX_GROUPS:
            raise ResidualServiceError("invalid_dataset", f"dataset cannot contain more than {MAX_GROUPS} groups")
        old_family_split = family_split.setdefault(family_id, split)
        if old_family_split != split:
            raise ResidualServiceError("split_leakage", f"family {family_id} occurs in more than one split")
        old_pair = pair_split.setdefault(pair_id, (family_id, split))
        if old_pair != (family_id, split):
            raise ResidualServiceError("split_leakage", f"pair {pair_id} crosses family or split boundaries")
        key = (pair_id, group)
        if key in pair_groups:
            raise ResidualServiceError("duplicate_pair_group", f"pair {pair_id} has more than one sample in group {group}")
        pair_groups.add(key)
        messages = raw.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
            raise ResidualServiceError("invalid_dataset", f"sample {sample_id} must have 1 to 32 messages")
        clean_messages: list[dict[str, str]] = []
        sample_chars = 0
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ResidualServiceError("invalid_dataset", f"sample {sample_id} message {message_index} must contain role and content")
            role = message.get("role")
            if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
                raise ResidualServiceError("invalid_dataset", f"sample {sample_id} has an unsupported message role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip() or "\x00" in content or len(content) > 20000:
                raise ResidualServiceError("invalid_dataset", f"sample {sample_id} message content must be non-empty text up to 20000 characters")
            total_chars += len(content)
            sample_chars += len(content)
            if sample_chars > 65536:
                raise ResidualServiceError("invalid_dataset", f"sample {sample_id} exceeds the local conversation-text limit")
            clean_messages.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in clean_messages) or clean_messages[-1]["role"] == "assistant":
            raise ResidualServiceError("invalid_dataset", f"sample {sample_id} needs a user message and must end before the assistant answer")
        target, foil = raw.get("target"), raw.get("foil")
        if (target is None) != (foil is None):
            raise ResidualServiceError("invalid_dataset", f"sample {sample_id} must provide both target and foil, or neither")
        clean_sample: dict[str, Any] = {
            "id": sample_id, "pair_id": pair_id, "family_id": family_id,
            "split": split, "group": group, "messages": clean_messages,
        }
        if target is not None:
            for field, value in (("target", target), ("foil", foil)):
                if not isinstance(value, str) or not value.strip() or len(value) > 2000 or "\x00" in value:
                    raise ResidualServiceError("invalid_dataset", f"sample {sample_id} {field} must be non-empty text up to 2000 characters")
            clean_sample["target"], clean_sample["foil"] = target, foil
            total_chars += len(target) + len(foil)
        normalized.append(clean_sample)
    if total_chars > MAX_DATASET_JSON_BYTES:
        raise ResidualServiceError("dataset_too_large", "Dataset text exceeds the local size limit")
    normalized.sort(key=lambda sample: sample["id"])
    return {"name": name, "samples": normalized}


def _dataset_hash(dataset: dict[str, Any]) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_namespace(value: Any, model_id: str, revision: Any) -> str:
    if isinstance(value, str) and _NS_RE.fullmatch(value) and value not in {".", ".."} and ".." not in value:
        return value
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._")[:48] or "model"
    digest = hashlib.sha256(f"{model_id}\0{revision or 'unknown'}".encode()).hexdigest()[:12]
    return f"{slug}-{digest}"


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ResidualServiceError("invalid_capture", f"capture {name} must have {ndim} dimensions", 500)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ResidualServiceError("invalid_capture", f"capture {name} contains non-finite or non-numeric values", 500)
    return array


def _json_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_json_numbers_are_finite(key) and _json_numbers_are_finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_json_numbers_are_finite(item) for item in value)
    return True


class ResidualService:
    """Manage capture, fit, and bounded candidate-scoring jobs on one runtime."""

    def __init__(self, rt: Any, repo_root: str | os.PathLike[str], run_root: str | os.PathLike[str] | None = None):
        self.rt = rt
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.output_root = Path(run_root).expanduser().resolve() if run_root else self.repo_root / "research_outputs" / "residual_groups"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="residual-groups")
        self._futures: dict[str, Future[Any]] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._closed = False
        self._recover_interrupted_jobs()

    def _identity(self) -> dict[str, Any]:
        model_id = str(getattr(self.rt, "model_id", "unknown-model"))
        revision = getattr(self.rt, "revision", None)
        adapter = getattr(self.rt, "adapter", None)
        if adapter is not None:
            model_id = str(getattr(adapter, "model_id", model_id))
            revision = getattr(adapter, "revision", revision)
            if not revision:
                revision = getattr(getattr(adapter, "config", None), "_commit_hash", None)
        namespace = _safe_namespace(getattr(self.rt, "cache_namespace", None), model_id, revision)
        return {"model_id": model_id, "revision": revision, "namespace": namespace}

    def _root(self) -> Path:
        return self.output_root / self._identity()["namespace"]

    def _dir(self, identifier: str) -> Path:
        if not isinstance(identifier, str) or not _ID_RE.fullmatch(identifier):
            raise ResidualServiceError("unknown_id", "No residual artifact or job with that id was found", 404)
        root = self._root()
        path = root / identifier
        if path.is_symlink() or not path.is_dir():
            raise ResidualServiceError("unknown_id", "No residual artifact or job with that id was found", 404)
        try:
            if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
                raise ResidualServiceError("unknown_id", "No residual artifact or job with that id was found", 404)
        except OSError:
            raise ResidualServiceError("unknown_id", "No residual artifact or job with that id was found", 404) from None
        return path

    @staticmethod
    def config(rt: Any) -> dict[str, Any]:
        model_id = str(getattr(rt, "model_id", "unknown-model"))
        revision = getattr(rt, "revision", None)
        adapter = getattr(rt, "adapter", None)
        config = getattr(adapter, "config_metadata", None) if adapter is not None else getattr(rt, "config_metadata", None)
        if not isinstance(config, dict):
            config = {}
        if adapter is not None:
            model_id = str(getattr(adapter, "model_id", model_id))
            revision = getattr(adapter, "revision", None) or getattr(getattr(adapter, "config", None), "_commit_hash", revision)
        n_layers = config.get("num_hidden_layers")
        hidden_size = config.get("hidden_size")
        try:
            n_layers = min(int(n_layers), MAX_LAYERS) if n_layers is not None else None
        except (TypeError, ValueError):
            n_layers = None
        try:
            hidden_size = int(hidden_size) if hidden_size is not None else None
        except (TypeError, ValueError):
            hidden_size = None
        loaded_layers = getattr(adapter, "layers", None) if adapter is not None else None
        if loaded_layers is not None:
            n_layers = min(len(loaded_layers), MAX_LAYERS)
        return {
            "model_id": model_id,
            "revision": revision,
            "device": str(getattr(adapter, "device", getattr(rt, "dev", "unknown"))),
            "loaded": getattr(rt, "model", None) is not None,
            "num_layers": n_layers,
            "hidden_size": hidden_size,
            "max_samples": MAX_SAMPLES,
            "max_groups": MAX_GROUPS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_capture_bytes": MAX_CAPTURE_BYTES,
            "default_layers": [layer for layer in (6, 13, 20) if n_layers is None or layer < n_layers],
            "default_points": list(POINTS),
            "default_window": 8,
            "points": list(POINTS),
            "max_window": MAX_WINDOW,
            "max_grid_side": MAX_GRID_SIDE,
            "max_grid_units": MAX_GRID_UNITS,
            "candidate_probe_fit_mode": "paired_delta",
            "candidate_probe_interpretation": "Candidate log-prob preference only; projections remove the global span of training prototypes and are not a cell-routed SOM gate.",
        }

    def demo(self) -> dict[str, Any]:
        """Return a small, explicitly synthetic dataset for plumbing checks."""
        samples = []
        for split, family, pair_id in (("train", "demo-train", "demo-train-pair"), ("validation", "demo-validation", "demo-validation-pair"), ("test", "demo-test", "demo-test-pair")):
            for group, tail in (("neutral", "Answer based on the evidence."), ("pressure", "Please agree with my suggested answer.")):
                samples.append({
                    "id": f"{pair_id}-{group}", "pair_id": pair_id, "family_id": family,
                    "split": split, "group": group,
                    "messages": [{"role": "user", "content": f"[SYNTHETIC DEMO ONLY] A fictional question about a blue square. {tail}"}],
                    "target": " A blue square.", "foil": " A green circle.",
                })
        return {"dataset": {"name": "synthetic-demo-not-research", "samples": samples}, "synthetic": True}

    def validate(self, dataset: Any) -> dict[str, Any]:
        clean = validate_dataset(dataset)
        counts = {split: sum(sample["split"] == split for sample in clean["samples"]) for split in SPLITS}
        groups = sorted({sample["group"] for sample in clean["samples"]})
        return {
            "valid": True,
            "name": clean["name"],
            "samples": len(clean["samples"]),
            "groups": groups,
            "splits": counts,
            "families": len({sample["family_id"] for sample in clean["samples"]}),
            "dataset_sha256": _dataset_hash(clean),
            "target_foil_samples": sum("target" in sample for sample in clean["samples"]),
        }

    def _read_job(self, identifier: str) -> tuple[Path, dict[str, Any]]:
        directory = self._dir(identifier)
        path = directory / "job.json"
        try:
            if path.is_symlink() or path.stat().st_size > 128 * 1024:
                raise OSError("invalid job record")
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ResidualServiceError("unknown_id", "No residual job with that id was found", 404) from None
        if not isinstance(record, dict) or record.get("job_id") != identifier:
            raise ResidualServiceError("unknown_id", "No residual job with that id was found", 404)
        if record.get("status") == "completed":
            if record.get("artifact_sha256") is None:
                # Older version-1 runs are still available after structural validation,
                # but they must not be presented as receipt-verified.
                record["integrity_status"] = "structure_only"
            else:
                self._verify_job_artifacts(directory, record)
        return directory, record

    @staticmethod
    def _verify_job_artifacts(directory: Path, record: dict[str, Any]) -> None:
        expected = record.get("artifact_sha256")
        required_files = {
            "capture": {"capture.npz", "tokens.json", "capture.json", "dataset.json"},
            "fit": {"fit.npz", "fit.json"},
            "probe": {"probe.json"},
        }.get(record.get("kind"), set())
        if not isinstance(expected, dict) or set(expected) != required_files or any(
            not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in expected.values()
        ):
            raise ResidualServiceError("integrity_failed", "Completed job is missing its artifact checksums", 409)
        try:
            if any((directory / name).is_symlink() or not (directory / name).is_file() or _sha256(directory / name) != digest for name, digest in expected.items()):
                raise OSError("artifact checksum mismatch")
        except OSError:
            raise ResidualServiceError("integrity_failed", "A residual artifact differs from its recorded SHA-256 checksum", 409) from None

    def _recover_interrupted_jobs(self) -> None:
        root = self._root()
        if not root.is_dir():
            return
        for path in root.glob("*/job.json"):
            if not _ID_RE.fullmatch(path.parent.name) or path.parent.is_symlink() or path.is_symlink():
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and record.get("job_id") == path.parent.name and record.get("status") in _ACTIVE:
                    if self._pid_is_alive(record.get("owner_pid")):
                        continue
                    record.update(status="interrupted", stage="interrupted", error="Explorer restarted while this job was active", error_code="interrupted", completed_at=_now())
                    self._write_job(path.parent, record)
            except (OSError, json.JSONDecodeError):
                continue

    @staticmethod
    def _pid_is_alive(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return False
        try:
            os.kill(value, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _public_job(self, record: dict[str, Any]) -> dict[str, Any]:
        keys = ("job_id", "kind", "status", "stage", "progress", "created_at", "updated_at", "completed_at", "result", "error", "error_code", "integrity_status")
        return {key: record[key] for key in keys if key in record}

    def get_job(self, job_id: str) -> dict[str, Any]:
        _, record = self._read_job(job_id)
        return self._public_job(record)

    def _write_job(self, directory: Path, record: dict[str, Any]) -> None:
        record["updated_at"] = _now()
        _write_json(directory / "job.json", record)

    def _create_job(self, kind: str, params: dict[str, Any], dataset: dict[str, Any] | None = None) -> tuple[str, Path, dict[str, Any], threading.Event]:
        with self._lock:
            if self._closed:
                raise ResidualServiceError("unavailable", "Residual service is shutting down", 503)
            active = 0
            root = self._root()
            if root.is_dir():
                for job_file in root.glob("*/job.json"):
                    try:
                        existing = json.loads(job_file.read_text(encoding="utf-8"))
                        active += int(isinstance(existing, dict) and existing.get("status") in _ACTIVE)
                    except (OSError, json.JSONDecodeError):
                        continue
            if active >= MAX_PENDING_JOBS:
                raise ResidualServiceError("capacity", "The residual job queue is full; wait for an existing job to finish", 429)
            identifier = uuid.uuid4().hex
            directory = root / identifier
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except OSError:
                raise ResidualServiceError("unavailable", "The residual output directory could not be created", 503) from None
            if dataset is not None:
                _write_json(directory / "dataset.json", dataset)
            identity = self._identity()
            record: dict[str, Any] = {
                "schema_version": 1, "job_id": identifier, "kind": kind,
                "status": "queued", "stage": "queued", "progress": {"completed": 0, "total": 0},
                "created_at": _now(), "updated_at": _now(), "result": None,
                "params": params, "model_id": identity["model_id"],
                "revision_requested": identity["revision"], "revision": None,
                "owner_pid": os.getpid(),
                "error": None, "error_code": None,
            }
            if dataset is not None:
                record["dataset_sha256"] = _dataset_hash(dataset)
            self._write_job(directory, record)
            event = threading.Event()
            self._cancel[identifier] = event
            future = self._executor.submit(self._execute_job, identifier, kind, params, directory, record, event)
            self._futures[identifier] = future
        return identifier, directory, record, event

    def start_capture(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {"dataset", "layers", "points", "window"}:
            raise ResidualServiceError("invalid_input", "Capture request must contain dataset and optional layers, points, and window")
        dataset = validate_dataset(payload.get("dataset"))
        params = self._validate_capture_params(payload)
        identifier, _, _, _ = self._create_job("capture", params, dataset)
        return self.get_job(identifier)

    def _validate_capture_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        layers = payload.get("layers")
        if layers is None:
            layers = self.config(self.rt)["default_layers"]
        if not isinstance(layers, list) or not layers or len(layers) > MAX_LAYERS:
            raise ResidualServiceError("invalid_input", f"layers must contain 1 to {MAX_LAYERS} zero-based indices")
        clean_layers = []
        for layer in layers:
            if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < MAX_LAYERS:
                raise ResidualServiceError("invalid_input", f"each layer must be an integer from 0 to {MAX_LAYERS - 1}")
            if layer in clean_layers:
                raise ResidualServiceError("invalid_input", "layers must not contain duplicates")
            clean_layers.append(layer)
        points = payload.get("points", list(POINTS))
        if not isinstance(points, list) or not points or any(not isinstance(point, str) for point in points) or len(set(points)) != len(points) or any(point not in POINTS for point in points):
            raise ResidualServiceError("invalid_input", "points must be a non-empty unique list of supported residual points")
        window = payload.get("window", 8)
        if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= MAX_WINDOW:
            raise ResidualServiceError("invalid_input", f"window must be an integer from 1 to {MAX_WINDOW}")
        return {"layers": clean_layers, "points": list(points), "window": window}

    def start_fit(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {"capture_id", "layer", "point", "mode", "group_a", "group_b", "rows", "cols", "epochs", "seed", "normalize"}:
            raise ResidualServiceError("invalid_input", "Fit request contains unknown fields")
        capture_id = payload.get("capture_id")
        capture = self._load_capture(capture_id)
        layer, point, mode = payload.get("layer"), payload.get("point"), payload.get("mode", "states")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in capture["metadata"]["layers"]:
            raise ResidualServiceError("invalid_input", "layer must be one of the captured layers")
        if point not in capture["metadata"]["points"]:
            raise ResidualServiceError("invalid_input", "point must be one of the captured points")
        if not isinstance(mode, str) or mode not in {"states", "paired_delta"}:
            raise ResidualServiceError("invalid_input", "mode must be states or paired_delta")
        group_a, group_b = payload.get("group_a"), payload.get("group_b")
        if mode == "paired_delta":
            groups = {row["group"] for row in capture["dataset"]["samples"]}
            if not isinstance(group_a, str) or group_a not in groups or not isinstance(group_b, str) or group_b not in groups or group_a == group_b:
                raise ResidualServiceError("invalid_input", "paired_delta requires two distinct captured groups")
            pairs = self._paired_rows(capture["dataset"]["samples"], group_a, group_b, split="train")
            if not pairs:
                raise ResidualServiceError("invalid_input", "paired_delta requires at least one complete training pair")
        else:
            if group_a is not None or group_b is not None:
                raise ResidualServiceError("invalid_input", "group_a and group_b are only used with paired_delta mode")
            if not any(row["split"] == "train" for row in capture["dataset"]["samples"]):
                raise ResidualServiceError("invalid_input", "SOM fitting requires at least one training sample")
        rows, cols = payload.get("rows", 2), payload.get("cols", 2)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (rows, cols)) or not 1 <= rows <= MAX_GRID_SIDE or not 1 <= cols <= MAX_GRID_SIDE or rows * cols > MAX_GRID_UNITS:
            raise ResidualServiceError("invalid_input", f"SOM grid must contain 1 to {MAX_GRID_UNITS} units")
        epochs, seed = payload.get("epochs", 100), payload.get("seed", 0)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
            raise ResidualServiceError("invalid_input", f"epochs must be an integer from 1 to {MAX_EPOCHS}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ResidualServiceError("invalid_input", "seed must be an integer from 0 to 4294967295")
        normalize = payload.get("normalize", "none")
        if not isinstance(normalize, str) or normalize not in {"none", "unit"}:
            raise ResidualServiceError("invalid_input", "normalize must be none or unit")
        params = {"capture_id": capture_id, "layer": layer, "point": point, "mode": mode,
                  "group_a": group_a, "group_b": group_b, "rows": rows, "cols": cols,
                  "epochs": epochs, "seed": seed, "normalize": normalize}
        identifier, _, _, _ = self._create_job("fit", params)
        return self.get_job(identifier)

    def start_probe(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {"fit_id", "split", "alpha", "max_samples"}:
            raise ResidualServiceError("invalid_input", "Probe request contains unknown fields")
        fit_id = payload.get("fit_id")
        fit = self._load_fit(fit_id)
        if fit["fit"]["mode"] != "paired_delta":
            raise ResidualServiceError("probe_requires_pairs", "Candidate probes require a paired_delta fit so the projected span is trained on group differences", 409)
        split, alpha = payload.get("split", "test"), payload.get("alpha", 1.0)
        max_samples = payload.get("max_samples", MAX_PROBE_SAMPLES)
        if not isinstance(split, str) or split not in {"validation", "test"}:
            raise ResidualServiceError("invalid_input", "probe split must be validation or test")
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ResidualServiceError("invalid_input", "alpha must be a finite number from 0 to 1")
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or not 1 <= max_samples <= MAX_PROBE_SAMPLES:
            raise ResidualServiceError("invalid_input", f"max_samples must be from 1 to {MAX_PROBE_SAMPLES}")
        capture = self._load_capture(fit["capture_id"])
        if fit["fit"]["point"] not in {"resid_pre", "resid_mid", "resid_post"}:
            raise ResidualServiceError("unsupported_probe_point", "Candidate scoring supports resid_pre, resid_mid, or resid_post only", 409)
        if capture["metadata"].get("family") != "llama":
            raise ResidualServiceError("unsupported_probe_model", "Candidate scoring currently supports Llama-family models only", 409)
        paired = self._paired_rows(capture["dataset"]["samples"], fit["fit"]["group_a"], fit["fit"]["group_b"], split=split)
        eligible = [(a, b) for a, b in paired if "target" in a and "foil" in a and "target" in b and "foil" in b]
        mismatched_targets = [
            pair_id for a, b in eligible
            if a["target"] != b["target"] or a["foil"] != b["foil"]
            for pair_id in (a["pair_id"],)
        ]
        if mismatched_targets:
            raise ResidualServiceError("paired_targets_mismatch", "Matched probe rows must use identical target and foil continuations across both groups")
        if not eligible:
            raise ResidualServiceError("no_probe_samples", "The selected held-out split has no rows with both target and foil continuations")
        ordered_pairs = sorted(eligible, key=lambda pair: hashlib.sha256(pair[0]["pair_id"].encode("utf-8")).hexdigest())
        selected_pairs = ordered_pairs[:max_samples // 2]
        if not selected_pairs:
            raise ResidualServiceError("invalid_input", "max_samples must allow at least one matched pair (use 2 or more)")
        selected = [sample for pair in selected_pairs for sample in pair]
        params = {"fit_id": fit_id, "split": split, "alpha": float(alpha), "max_samples": max_samples,
                  "selected_ids": [row["id"] for row in selected]}
        identifier, _, _, _ = self._create_job("probe", params)
        return self.get_job(identifier)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        directory, record = self._read_job(job_id)
        with self._lock:
            if record.get("status") not in _ACTIVE:
                return self._public_job(record)
            event = self._cancel.get(job_id)
            future = self._futures.get(job_id)
            if event is None:
                raise ResidualServiceError("unavailable", "This job is not attached to the current explorer process", 409)
            event.set()
            if future is not None and future.cancel():
                record.update(status="cancelled", stage="cancelled", error="Job cancelled before it started", error_code="cancelled", completed_at=_now())
                self._write_job(directory, record)
            return self.get_job(job_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for event in self._cancel.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _execute_job(self, job_id: str, kind: str, params: dict[str, Any], directory: Path, record: dict[str, Any], cancel: threading.Event) -> None:
        try:
            record.update(status="running", stage="preparing")
            self._write_job(directory, record)
            self.rt.ensure_loaded()
            self._check_cancel(cancel)
            identity = self._identity()
            record["model_id"] = identity["model_id"]
            record["revision"] = identity["revision"]
            self._write_job(directory, record)
            if kind == "capture":
                result = self._run_capture(job_id, directory, params, cancel, record)
            elif kind == "fit":
                result = self._run_fit(job_id, directory, params, cancel, record)
            elif kind == "probe":
                result = self._run_probe(job_id, directory, params, cancel, record)
            else:
                raise ResidualServiceError("invalid_job", "Unknown residual job kind", 500)
            self._check_cancel(cancel)
            files = {
                "capture": ("capture.npz", "tokens.json", "capture.json", "dataset.json"),
                "fit": ("fit.npz", "fit.json"),
                "probe": ("probe.json",),
            }[kind]
            record.update(stage="validating", result=result,
                          artifact_sha256={name: _sha256(directory / name) for name in files},
                          integrity_status="validating", error=None)
            self._write_job(directory, record)
            if kind == "capture":
                self._load_capture(job_id, allow_in_progress=True)
            elif kind == "fit":
                self._load_fit(job_id, allow_in_progress=True)
            else:
                self.get_probe(job_id, allow_in_progress=True)
            record.update(status="completed", stage="completed", integrity_status="verified", completed_at=_now())
            self._write_job(directory, record)
        except _Cancelled:
            record.update(status="cancelled", stage="cancelled", result=None, integrity_status="incomplete", error="Job cancelled", error_code="cancelled", completed_at=_now())
            record.pop("artifact_sha256", None)
            self._write_job(directory, record)
        except ResidualCaptureCancelled:
            record.update(status="cancelled", stage="cancelled", result=None, integrity_status="incomplete", error="Job cancelled", error_code="cancelled", completed_at=_now())
            record.pop("artifact_sha256", None)
            self._write_job(directory, record)
        except Exception as exc:  # noqa: BLE001 - job boundary persists a safe public error
            if isinstance(exc, ResidualServiceError):
                code = exc.code
                message = exc.message
            else:
                code = "job_failed"
                message = "Residual job failed; inspect local logs for details"
            record.update(status="failed", stage="failed", result=None, integrity_status="invalid", error=message, error_code=code, completed_at=_now())
            record.pop("artifact_sha256", None)
            self._write_job(directory, record)

    @staticmethod
    def _check_cancel(cancel: threading.Event) -> None:
        if cancel.is_set():
            raise _Cancelled

    def _score_continuation(self, sample: dict[str, Any], layer: int, point: str, continuation: str, *, basis: np.ndarray | None = None, alpha: float = 0.0) -> dict[str, Any]:
        result = score_residual_continuation(
            self.rt, sample, layer, point, basis=basis, alpha=alpha,
            score_mode="mean_token_logprob", continuation=continuation,
        )
        try:
            score = float(result["score"])
            token_count = result["token_count"]
        except (KeyError, TypeError, ValueError):
            raise ResidualServiceError("invalid_probe_score", "The model returned an invalid candidate score", 500) from None
        if not math.isfinite(score) or isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1:
            raise ResidualServiceError("invalid_probe_score", "The model returned a non-finite score or invalid token count", 500)
        if result.get("score_mode") != "mean_token_logprob":
            raise ResidualServiceError("invalid_probe_score", "The model returned an unexpected candidate score mode", 500)
        return {**result, "score": score, "token_count": token_count}

    def _progress(self, record: dict[str, Any], directory: Path, stage: str, completed: int, total: int) -> None:
        record["stage"] = stage
        record["progress"] = {"completed": max(0, int(completed)), "total": max(0, int(total))}
        self._write_job(directory, record)

    def _preflight_capture(self, dataset: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        adapter = getattr(self.rt, "adapter", None)
        if adapter is None:
            raise ResidualServiceError("runtime_unavailable", "The shared model runtime has no loaded model adapter", 503)
        if not self._identity()["revision"]:
            raise ResidualServiceError("unversioned_model", "The loaded model has no pinned revision in its local metadata; select a revisioned model before capturing", 409)
        layers = getattr(adapter, "layers", None)
        if layers is None:
            raise ResidualServiceError("runtime_unavailable", "The shared model runtime does not expose decoder layers", 503)
        if any(layer >= min(len(layers), MAX_LAYERS) for layer in params["layers"]):
            raise ResidualServiceError("invalid_layer", f"selected layers must be within the loaded model range 0 to {min(len(layers), MAX_LAYERS) - 1}")
        config = getattr(adapter, "config_metadata", {}) or {}
        if getattr(adapter, "family", None) != "llama":
            raise ResidualServiceError("unsupported_model", "Residual capture currently supports Llama-family models only", 409)
        hidden = int(config.get("hidden_size", 0)) if isinstance(config, dict) else 0
        if hidden < 1:
            raise ResidualServiceError("runtime_unavailable", "Loaded model configuration does not report hidden_size", 503)
        n, l, p, k, h = len(dataset["samples"]), len(params["layers"]), len(params["points"]), params["window"], hidden
        estimated = n * l * p * k * h * 4 + n * k * 17
        if estimated > MAX_CAPTURE_BYTES:
            raise ResidualServiceError("capture_too_large", f"Requested vectors need {estimated} bytes; local capture cap is {MAX_CAPTURE_BYTES}")
        tokenizer_hash = hashlib.sha256()
        prompt_lengths = {}
        prompt_token_ids: dict[str, list[int]] = {}
        for sample in dataset["samples"]:
            token_ids = adapter.apply_chat_template(sample["messages"], add_generation_prompt=True)
            if len(token_ids) > MAX_PROMPT_TOKENS:
                raise ResidualServiceError("prompt_too_long", f"sample {sample['id']} renders to {len(token_ids)} tokens; the limit is {MAX_PROMPT_TOKENS}")
            prompt_lengths[sample["id"]] = len(token_ids)
            prompt_token_ids[sample["id"]] = [int(token_id) for token_id in token_ids]
            tokenizer_hash.update(np.asarray(token_ids, dtype=np.int64).tobytes())
            tokenizer_hash.update(sample["id"].encode("utf-8"))
        return {"hidden_size": hidden, "prompt_lengths": prompt_lengths, "prompt_token_ids": prompt_token_ids, "tokenization_sha256": tokenizer_hash.hexdigest(), "estimated_bytes": estimated}

    def _run_capture(self, job_id: str, directory: Path, params: dict[str, Any], cancel: threading.Event, record: dict[str, Any]) -> dict[str, str]:
        dataset_path = directory / "dataset.json"
        try:
            dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ResidualServiceError("invalid_dataset", "Persisted dataset could not be read", 500) from None
        preflight = self._preflight_capture(dataset, params)
        self._check_cancel(cancel)
        self._progress(record, directory, "capturing", 0, len(dataset["samples"]))
        def progress(stage: str, completed: int, total: int) -> None:
            safe_stage = "tokenizing" if stage == "tokenizing" else "capturing"
            self._progress(record, directory, safe_stage, completed, total)
        captured = capture_residuals(
            self.rt, dataset["samples"], params["layers"], params["points"],
            window=params["window"], progress=progress, cancel_event=cancel,
        )
        self._check_cancel(cancel)
        raw_vectors = np.asarray(captured.get("vectors"))
        if raw_vectors.dtype != np.float32:
            raise ResidualServiceError("invalid_capture", "Captured vectors must use float32 storage", 500)
        vectors = _finite_array(raw_vectors, "vectors", 5)
        if not np.isfinite(vectors).all():
            raise ResidualServiceError("invalid_capture", "Captured vectors overflowed float32 serialization", 500)
        expected_prefix = (len(dataset["samples"]), len(params["layers"]), len(params["points"]), params["window"])
        if vectors.shape[:4] != expected_prefix or vectors.shape[-1] != preflight["hidden_size"]:
            raise ResidualServiceError("invalid_capture", "Captured vector dimensions do not match the requested samples, layers, points, window, and model", 500)
        token_ids = np.asarray(captured.get("token_ids"))
        positions = np.asarray(captured.get("positions"))
        valid = np.asarray(captured.get("valid"))
        if token_ids.dtype != np.int64 or positions.dtype != np.int64 or valid.dtype != np.bool_:
            raise ResidualServiceError("invalid_capture", "Captured token maps have invalid dtypes", 500)
        if token_ids.shape != (len(dataset["samples"]), params["window"]) or positions.shape != token_ids.shape or valid.shape != token_ids.shape:
            raise ResidualServiceError("invalid_capture", "Captured token maps have inconsistent shapes", 500)
        if np.any(valid.sum(axis=1) == 0) or np.any((token_ids[valid] < 0) | (positions[valid] < 0)):
            raise ResidualServiceError("invalid_capture", "Each sample needs valid non-negative token IDs and positions", 500)
        metadata = captured.get("metadata")
        if not isinstance(metadata, dict):
            raise ResidualServiceError("invalid_capture", "Capture provenance is missing", 500)
        provenance = dict(metadata)
        identity = self._identity()
        if identity["revision"] is None and isinstance(provenance.get("revision"), str):
            identity["revision"] = provenance["revision"]
            record["revision"] = provenance["revision"]
        required = {"model_id": identity["model_id"], "revision": identity["revision"], "hidden_size": preflight["hidden_size"], "layers": params["layers"], "points": params["points"], "window": params["window"]}
        for key, value in required.items():
            if provenance.get(key) != value:
                raise ResidualServiceError("provenance_mismatch", f"Capture provenance does not match requested {key}", 500)
        tokens = captured.get("tokens")
        lengths = captured.get("lengths")
        if not isinstance(tokens, list) or len(tokens) != len(dataset["samples"]) or not isinstance(lengths, list) or len(lengths) != len(dataset["samples"]):
            raise ResidualServiceError("invalid_capture", "Captured token strings are missing", 500)
        if any(not isinstance(row, list) or len(row) != params["window"] or any(token is not None and not isinstance(token, str) for token in row) for row in tokens):
            raise ResidualServiceError("invalid_capture", "Captured token strings do not match the selected window", 500)
        for index, sample in enumerate(dataset["samples"]):
            full_ids = preflight["prompt_token_ids"][sample["id"]]
            expected_count = min(params["window"], len(full_ids))
            valid_positions = np.flatnonzero(valid[index])
            if len(valid_positions) != expected_count or not np.array_equal(valid_positions, np.arange(params["window"] - expected_count, params["window"])):
                raise ResidualServiceError("invalid_capture", "Captured valid-token mask is not a right-aligned prompt suffix", 500)
            visible_positions = positions[index, valid[index]]
            expected_positions = np.arange(len(full_ids) - expected_count, len(full_ids), dtype=np.int64)
            if not np.array_equal(visible_positions, expected_positions) or not np.array_equal(token_ids[index, valid[index]], np.asarray(full_ids[-expected_count:], dtype=np.int64)):
                raise ResidualServiceError("invalid_capture", "Captured token map does not match the rendered prompt", 500)
            if any(tokens[index][position] is None for position in valid_positions):
                raise ResidualServiceError("invalid_capture", "Captured token text is missing for a valid prompt position", 500)
        arrays_path = directory / "capture.npz"
        _write_npz(arrays_path, vectors=vectors, token_ids=token_ids, positions=positions, valid=valid)
        if arrays_path.stat().st_size > MAX_CAPTURE_BYTES:
            arrays_path.unlink(missing_ok=True)
            raise ResidualServiceError("capture_too_large", "Serialized capture exceeds the 256 MiB local cap")
        token_path = directory / "tokens.json"
        _write_json(token_path, {"tokens": tokens, "lengths": lengths})
        manifest = {
            "schema": "residual-group-capture-v1", "capture_id": job_id,
            "dataset_sha256": _dataset_hash(dataset), "dataset_file_sha256": _sha256(dataset_path),
            "capture_sha256": _sha256(arrays_path), "tokens_sha256": _sha256(token_path),
            "provenance": provenance, "prompt_lengths": preflight["prompt_lengths"],
            "tokenization_sha256": preflight["tokenization_sha256"],
            "sample_ids": [sample["id"] for sample in dataset["samples"]],
            "created_at": _now(), "estimated_bytes": preflight["estimated_bytes"],
        }
        _write_json(directory / "capture.json", manifest)
        return {"capture_id": job_id}

    def _load_capture(self, capture_id: Any, *, allow_in_progress: bool = False) -> dict[str, Any]:
        directory, job = self._read_job(capture_id)
        if job.get("kind") != "capture" or (job.get("status") != "completed" and not (allow_in_progress and job.get("status") == "running")):
            raise ResidualServiceError("unknown_capture", "No completed capture with that id was found", 404)
        if allow_in_progress:
            self._verify_job_artifacts(directory, job)
        manifest_path, dataset_path, tokens_path, arrays_path = (directory / name for name in ("capture.json", "dataset.json", "tokens.json", "capture.npz"))
        try:
            if any(path.is_symlink() or not path.is_file() for path in (manifest_path, dataset_path, tokens_path, arrays_path)):
                raise OSError("missing or linked capture file")
            if manifest_path.stat().st_size > 1024 * 1024 or dataset_path.stat().st_size > MAX_DATASET_JSON_BYTES or tokens_path.stat().st_size > MAX_DATASET_JSON_BYTES or arrays_path.stat().st_size > MAX_CAPTURE_BYTES:
                raise OSError("capture file too large")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset = validate_dataset(json.loads(dataset_path.read_text(encoding="utf-8")))
            tokens_record = json.loads(tokens_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or manifest.get("schema") != "residual-group-capture-v1" or manifest.get("capture_id") != capture_id:
                raise OSError("invalid capture manifest")
            checks = ((dataset_path, "dataset_file_sha256"), (tokens_path, "tokens_sha256"), (arrays_path, "capture_sha256"))
            if any(_sha256(path) != manifest.get(field) for path, field in checks):
                raise ResidualServiceError("integrity_failed", "A capture file differs from its recorded SHA-256 checksum", 409)
            if _dataset_hash(dataset) != manifest.get("dataset_sha256"):
                raise OSError("dataset semantic hash mismatch")
            if manifest.get("sample_ids") != [sample["id"] for sample in dataset["samples"]]:
                raise OSError("capture sample IDs mismatch")
            with zipfile.ZipFile(arrays_path) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_CAPTURE_BYTES or any(item.file_size > MAX_CAPTURE_BYTES for item in archive.infolist()):
                    raise OSError("capture archive expands beyond its size limit")
            with np.load(arrays_path, allow_pickle=False) as archive:
                if set(archive.files) != {"vectors", "token_ids", "positions", "valid"}:
                    raise OSError("invalid capture arrays")
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            if arrays["vectors"].dtype != np.float32 or arrays["vectors"].ndim != 5 or not np.isfinite(arrays["vectors"]).all():
                raise OSError("invalid capture vectors")
            n, l, p, k, h = arrays["vectors"].shape
            estimated_bytes = n * l * p * k * h * 4 + n * k * 17
            if not (1 <= n <= MAX_SAMPLES and 1 <= l <= MAX_LAYERS and 1 <= p <= len(POINTS) and 1 <= k <= MAX_WINDOW and h > 0) or estimated_bytes > MAX_CAPTURE_BYTES:
                raise OSError("capture dimensions exceed the configured limits")
            provenance = manifest.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("hidden_size") != h or not isinstance(provenance.get("layers"), list) or len(provenance["layers"]) != l or not isinstance(provenance.get("points"), list) or len(provenance["points"]) != p or provenance.get("window") != k:
                raise OSError("capture provenance mismatch")
            if any(not isinstance(point, str) or point not in POINTS for point in provenance["points"]) or len(set(provenance["points"])) != p:
                raise OSError("capture residual points are invalid")
            if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < MAX_LAYERS for value in provenance["layers"]) or len(set(provenance["layers"])) != l:
                raise OSError("capture layer selection is invalid")
            params = job.get("params")
            if not isinstance(params, dict) or provenance.get("layers") != params.get("layers") or provenance.get("points") != params.get("points") or provenance.get("window") != params.get("window"):
                raise OSError("capture request and provenance do not match")
            if (job.get("dataset_sha256") is not None and job.get("dataset_sha256") != manifest.get("dataset_sha256")) or job.get("model_id") != provenance.get("model_id") or (job.get("revision") is not None and job.get("revision") != provenance.get("revision")):
                raise OSError("capture job and provenance do not match")
            if job.get("result") != {"capture_id": capture_id}:
                raise OSError("capture job result ID does not match its artifact")
            if not isinstance(provenance.get("model_id"), str) or not provenance["model_id"] or not isinstance(provenance.get("revision"), str) or not provenance["revision"] or provenance.get("family") != "llama" or not isinstance(provenance.get("device"), str):
                raise OSError("capture model provenance is incomplete")
            if arrays["token_ids"].dtype != np.int64 or arrays["positions"].dtype != np.int64 or arrays["valid"].dtype != np.bool_ or arrays["token_ids"].shape != (n, k) or arrays["positions"].shape != (n, k) or arrays["valid"].shape != (n, k):
                raise OSError("capture token maps mismatch")
            if n != len(dataset["samples"]) or not isinstance(tokens_record, dict) or not isinstance(tokens_record.get("tokens"), list) or len(tokens_record["tokens"]) != n:
                raise OSError("capture dataset alignment mismatch")
            if not isinstance(tokens_record.get("lengths"), list) or len(tokens_record["lengths"]) != n:
                raise OSError("capture prompt lengths mismatch")
            prompt_lengths = manifest.get("prompt_lengths")
            if not isinstance(prompt_lengths, dict) or set(prompt_lengths) != {sample["id"] for sample in dataset["samples"]}:
                raise OSError("capture prompt length provenance is invalid")
            for index, row in enumerate(tokens_record["tokens"]):
                if not isinstance(row, list) or len(row) != k or any(value is not None and not isinstance(value, str) for value in row):
                    raise OSError("capture token text mismatch")
                if not np.any(arrays["valid"][index]) or np.any(arrays["token_ids"][index, ~arrays["valid"][index]] != -1) or np.any(arrays["positions"][index, ~arrays["valid"][index]] != -1):
                    raise OSError("capture padding mask mismatch")
                if np.any(arrays["token_ids"][index, arrays["valid"][index]] < 0) or np.any(arrays["positions"][index, arrays["valid"][index]] < 0):
                    raise OSError("capture valid token map has negative values")
                length = tokens_record["lengths"][index]
                if isinstance(length, bool) or not isinstance(length, int) or length < int(arrays["valid"][index].sum()):
                    raise OSError("capture prompt length is invalid")
                sample_id = dataset["samples"][index]["id"]
                if length != prompt_lengths.get(sample_id) or length > MAX_PROMPT_TOKENS:
                    raise OSError("capture prompt length does not match provenance")
                count = min(k, length)
                valid_indices = np.flatnonzero(arrays["valid"][index])
                if not np.array_equal(valid_indices, np.arange(k - count, k)) or not np.array_equal(arrays["positions"][index, arrays["valid"][index]], np.arange(length - count, length)):
                    raise OSError("capture prompt positions are not a right-aligned suffix")
                if any(row[position] is None for position in valid_indices) or any(row[position] is not None for position in range(k - count)):
                    raise OSError("capture valid token text is missing")
            for key in ("dataset_sha256", "dataset_file_sha256", "capture_sha256", "tokens_sha256", "tokenization_sha256"):
                value = manifest.get(key)
                if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                    raise OSError("capture manifest contains an invalid checksum")
            return {"directory": directory, "manifest": manifest, "metadata": provenance, "dataset": dataset, "tokens": tokens_record.get("tokens"), "arrays": arrays, "job": job}
        except ResidualServiceError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
            raise ResidualServiceError("invalid_capture", "The selected capture is missing, malformed, or failed integrity checks", 409) from None

    @staticmethod
    def _select_vector(capture: dict[str, Any], sample_index: int, layer: int, point: str) -> np.ndarray:
        meta = capture["metadata"]
        li, pi = meta["layers"].index(layer), meta["points"].index(point)
        valid = capture["arrays"]["valid"][sample_index]
        positions = np.flatnonzero(valid)
        return capture["arrays"]["vectors"][sample_index, li, pi, int(positions[-1])].astype(np.float32, copy=True)

    @staticmethod
    def _paired_rows(samples: list[dict[str, Any]], group_a: str, group_b: str, split: str | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        by_pair: dict[str, dict[str, dict[str, Any]]] = {}
        for sample in samples:
            if split is not None and sample["split"] != split:
                continue
            if sample["group"] in {group_a, group_b}:
                by_pair.setdefault(sample["pair_id"], {})[sample["group"]] = sample
        rows = [(groups[group_a], groups[group_b]) for groups in by_pair.values() if group_a in groups and group_b in groups]
        return sorted(rows, key=lambda pair: pair[0]["pair_id"])

    def _run_fit(self, fit_id: str, directory: Path, params: dict[str, Any], cancel: threading.Event, record: dict[str, Any]) -> dict[str, str]:
        capture = self._load_capture(params["capture_id"])
        self._check_cancel(cancel)
        samples = capture["dataset"]["samples"]
        layer, point = params["layer"], params["point"]
        vectors = {sample["id"]: self._select_vector(capture, i, layer, point) for i, sample in enumerate(samples)}
        if params["mode"] == "states":
            train_rows = [sample for sample in samples if sample["split"] == "train"]
            train_values = np.stack([vectors[row["id"]] for row in train_rows])
            source_rows = train_rows
            training_sample_count = len(train_rows)
            training_pair_count = None
        else:
            paired = self._paired_rows(samples, params["group_a"], params["group_b"], split="train")
            train_values = np.stack([vectors[a["id"]] - vectors[b["id"]] for a, b in paired])
            source_rows = [a for a, _ in paired]
            training_sample_count = 2 * len(paired)
            training_pair_count = len(paired)
        if params["normalize"] == "unit":
            norms = np.linalg.norm(train_values, axis=1, keepdims=True)
            train_values = np.divide(train_values, np.maximum(norms, 1e-12), out=np.zeros_like(train_values), where=norms > 1e-12)
        train_values = _finite_array(train_values, "training vectors", 2)
        if len(train_values) < 1 or not np.any(np.linalg.norm(train_values, axis=1) > 1e-12):
            raise ResidualServiceError("empty_training_signal", "Training vectors are empty or all zero; no SOM was fitted")
        self._progress(record, directory, "fitting", 0, params["epochs"])
        prototypes = train_som(train_values, rows=params["rows"], cols=params["cols"], epochs=params["epochs"], seed=params["seed"])
        self._check_cancel(cancel)
        basis = orthonormal_basis(prototypes.T)
        if basis.shape[1] == 0:
            raise ResidualServiceError("empty_basis", "Training produced a zero-rank prototype span")
        controls = fit_control_bases(train_values, basis, seed=params["seed"])
        assignments: list[dict[str, Any]] = []
        quant_by_split: dict[str, list[float]] = {name: [] for name in SPLITS}
        grid_count = params["rows"] * params["cols"]
        cell_groups = [{} for _ in range(grid_count)]
        cell_splits = [{} for _ in range(grid_count)]
        cell_errors = [{} for _ in range(grid_count)]
        if params["mode"] == "states":
            assignment_rows = [(sample, vectors[sample["id"]]) for sample in samples]
        else:
            assignment_rows = [
                (a, vectors[a["id"]] - vectors[b["id"]])
                for split in SPLITS for a, b in self._paired_rows(samples, params["group_a"], params["group_b"], split=split)
            ]
        for sample, vector in assignment_rows:
            self._check_cancel(cancel)
            input_vector = vector
            if params["normalize"] == "unit":
                norm = float(np.linalg.norm(input_vector))
                input_vector = input_vector / max(norm, 1e-12)
            distances = np.linalg.norm(prototypes.astype(np.float64) - input_vector.astype(np.float64), axis=1)
            cell = int(np.argmin(distances))
            error = float(distances[cell])
            row, col = divmod(cell, params["cols"])
            assignment = {
                "id": sample["id"], "pair_id": sample["pair_id"], "family_id": sample["family_id"],
                "split": sample["split"], "group": sample["group"] if params["mode"] == "states" else f"{params['group_a']}-{params['group_b']} delta",
                "cell_id": cell, "row": row, "col": col, "quantization_error": error,
            }
            if params["mode"] == "paired_delta":
                assignment["group_a"] = params["group_a"]
                assignment["group_b"] = params["group_b"]
                assignment["pair_id"] = sample["pair_id"]
            assignments.append(assignment)
            quant_by_split[sample["split"]].append(error)
            group_name = assignment["group"]
            cell_groups[cell][group_name] = cell_groups[cell].get(group_name, 0) + 1
            cell_splits[cell][sample["split"]] = cell_splits[cell].get(sample["split"], 0) + 1
            cell_errors[cell].setdefault(sample["split"], []).append(error)
        rank = int(basis.shape[1])
        meta = {
            "schema": "residual-group-fit-v1", "fit_id": fit_id, "capture_id": params["capture_id"],
            "fit": {key: params[key] for key in ("layer", "point", "mode", "group_a", "group_b", "rows", "cols", "epochs", "seed", "normalize")},
            "provenance": {key: capture["metadata"].get(key) for key in ("model_id", "revision", "device", "family", "hidden_size")},
            "basis_rank": rank, "training_samples": training_sample_count,
            "training_units": len(train_values), "training_pairs": training_pair_count,
            "training_data_kind": "sample_states" if params["mode"] == "states" else "matched_group_a_minus_group_b_deltas",
            "training_families": len({row["family_id"] for row in source_rows}),
            "training_pair_ids_sha256": hashlib.sha256("\n".join(sorted({row["pair_id"] for row in source_rows})).encode()).hexdigest(),
            "capture_sha256": capture["manifest"]["capture_sha256"],
            "assignments": assignments,
            "quantization": [{"split": split, "n": len(values), "mean_l2": float(np.mean(values)) if values else None, "mean_rms": float(np.mean(values) / math.sqrt(capture["metadata"]["hidden_size"])) if values else None} for split, values in quant_by_split.items()],
            "quantization_unit": "samples" if params["mode"] == "states" else "matched_pairs",
            "cells": [{"cell_id": cell, "row": cell // params["cols"], "col": cell % params["cols"], "group_counts": cell_groups[cell], "split_counts": cell_splits[cell], "quantization": {split: {"n": len(values), "mean_l2": float(np.mean(values))} for split, values in cell_errors[cell].items()}} for cell in range(grid_count)],
            "control_ranks": {key: int(value.shape[1]) for key, value in controls.items()},
            "projection_note": "Projection uses the global span of the training SOM prototypes; it is not cell-routed.",
            "created_at": _now(),
        }
        artifact_path = directory / "fit.npz"
        _write_npz(artifact_path, prototypes=np.asarray(prototypes, dtype=np.float32), basis=np.asarray(basis, dtype=np.float32), global_mean=np.asarray(controls["global_mean"], dtype=np.float32), pca=np.asarray(controls["pca"], dtype=np.float32), random=np.asarray(controls["random"], dtype=np.float32))
        meta["fit_sha256"] = _sha256(artifact_path)
        _write_json(directory / "fit.json", meta)
        self._progress(record, directory, "fitting", params["epochs"], params["epochs"])
        return {"fit_id": fit_id}

    def _load_fit(self, fit_id: Any, *, allow_in_progress: bool = False) -> dict[str, Any]:
        directory, job = self._read_job(fit_id)
        if job.get("kind") != "fit" or (job.get("status") != "completed" and not (allow_in_progress and job.get("status") == "running")):
            raise ResidualServiceError("unknown_fit", "No completed fit with that id was found", 404)
        if allow_in_progress:
            self._verify_job_artifacts(directory, job)
        metadata_path, arrays_path = directory / "fit.json", directory / "fit.npz"
        try:
            if metadata_path.is_symlink() or arrays_path.is_symlink() or not metadata_path.is_file() or not arrays_path.is_file() or metadata_path.stat().st_size > 8 * 1024 * 1024 or arrays_path.stat().st_size > MAX_CAPTURE_BYTES:
                raise OSError("missing or oversized fit")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("schema") != "residual-group-fit-v1" or metadata.get("fit_id") != fit_id or metadata.get("fit_sha256") != _sha256(arrays_path):
                raise OSError("fit integrity mismatch")
            with zipfile.ZipFile(arrays_path) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_CAPTURE_BYTES or any(item.file_size > MAX_CAPTURE_BYTES for item in archive.infolist()):
                    raise OSError("fit archive expands beyond its size limit")
            with np.load(arrays_path, allow_pickle=False) as archive:
                if set(archive.files) != {"prototypes", "basis", "global_mean", "pca", "random"}:
                    raise OSError("invalid fit arrays")
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            if any(value.dtype != np.float32 for value in arrays.values()):
                raise OSError("fit arrays have invalid dtypes")
            if not _json_numbers_are_finite(metadata):
                raise OSError("fit metadata contains non-finite values")
            prototypes = _finite_array(arrays["prototypes"], "prototypes", 2)
            basis = _finite_array(arrays["basis"], "basis", 2)
            fit_config = metadata.get("fit")
            if not isinstance(fit_config, dict) or not isinstance(fit_config.get("mode"), str) or fit_config.get("mode") not in {"states", "paired_delta"}:
                raise OSError("fit configuration is invalid")
            job_params = job.get("params")
            fit_keys = ("layer", "point", "mode", "group_a", "group_b", "rows", "cols", "epochs", "seed", "normalize")
            if not isinstance(job_params, dict) or any(job_params.get(key) != fit_config.get(key) for key in fit_keys) or job.get("result") != {"fit_id": fit_id}:
                raise OSError("fit request and result do not match the artifact")
            capture = self._load_capture(metadata.get("capture_id"))
            if job_params.get("capture_id") != metadata.get("capture_id"):
                raise OSError("fit source capture does not match the job")
            if metadata.get("capture_sha256") != capture["manifest"].get("capture_sha256"):
                raise OSError("fit was created from a different capture")
            rows, cols = fit_config.get("rows"), fit_config.get("cols")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (rows, cols)) or not 1 <= rows <= MAX_GRID_SIDE or not 1 <= cols <= MAX_GRID_SIDE or rows * cols > MAX_GRID_UNITS:
                raise OSError("fit grid dimensions are invalid")
            layer, point = fit_config.get("layer"), fit_config.get("point")
            if isinstance(layer, bool) or not isinstance(layer, int) or not isinstance(point, str):
                raise OSError("fit layer or point is invalid")
            if fit_config.get("point") not in capture["metadata"]["points"] or fit_config.get("layer") not in capture["metadata"]["layers"]:
                raise OSError("fit layer or point is not present in its capture")
            if prototypes.shape != (rows * cols, capture["metadata"]["hidden_size"]):
                raise OSError("fit prototype dimensions do not match its grid or capture")
            fit_provenance = metadata.get("provenance")
            if not isinstance(fit_provenance, dict) or any(fit_provenance.get(key) != capture["metadata"].get(key) for key in ("model_id", "revision", "device", "family", "hidden_size")):
                raise OSError("fit provenance does not match the capture")
            if basis.shape[0] != prototypes.shape[1] or basis.shape[1] != metadata.get("basis_rank") or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2e-4, rtol=2e-4):
                raise OSError("fit basis mismatch")
            for name in ("global_mean", "pca", "random"):
                control = _finite_array(arrays[name], name, 2)
                if control.shape[0] != basis.shape[0] or not 0 <= control.shape[1] <= basis.shape[1]:
                    raise OSError("fit control dimensions mismatch")
                if control.shape[1] and not np.allclose(control.T @ control, np.eye(control.shape[1]), atol=2e-4, rtol=2e-4):
                    raise OSError("fit control basis is not orthonormal")
            cells = metadata.get("cells")
            assignments = metadata.get("assignments")
            if not isinstance(cells, list) or len(cells) != rows * cols or not isinstance(assignments, list) or len(assignments) > MAX_SAMPLES:
                raise OSError("fit map records are invalid")
            for assignment in assignments:
                if not isinstance(assignment, dict) or isinstance(assignment.get("cell_id"), bool) or not isinstance(assignment.get("cell_id"), int) or not 0 <= assignment["cell_id"] < len(cells) or assignment.get("split") not in SPLITS or not math.isfinite(float(assignment.get("quantization_error", float("nan")))):
                    raise OSError("fit assignment is invalid")
            return {"directory": directory, "fit_id": fit_id, "capture_id": metadata["capture_id"], "fit": metadata["fit"], "provenance": metadata["provenance"], "metadata": metadata, "arrays": arrays, "job": job}
        except ResidualServiceError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
            raise ResidualServiceError("invalid_fit", "The selected fit is missing, malformed, or failed integrity checks", 409) from None

    def get_capture_summary(self, capture_id: str, *, layer: int | None = None, point: str | None = None, group_a: str | None = None, group_b: str | None = None) -> dict[str, Any]:
        capture = self._load_capture(capture_id)
        if (group_a is None) != (group_b is None):
            raise ResidualServiceError("invalid_input", "group_a and group_b must be provided together")
        layers, points = capture["metadata"]["layers"], capture["metadata"]["points"]
        layer = layers[0] if layer is None else layer
        point = points[0] if point is None else point
        if layer not in layers or point not in points:
            raise ResidualServiceError("invalid_input", "layer and point must be included in the capture")
        samples = capture["dataset"]["samples"]
        vectors = [self._select_vector(capture, index, layer, point).astype(np.float64) for index in range(len(samples))]
        by_group: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            by_group.setdefault(sample["group"], []).append(index)
        groups = []
        for group, indices in sorted(by_group.items()):
            matrix = np.stack([vectors[index] for index in indices])
            centroid = matrix.mean(axis=0)
            norms = np.linalg.norm(matrix, axis=1)
            split_counts = {split: sum(samples[index]["split"] == split for index in indices) for split in SPLITS}
            groups.append({"group": group, "n": len(indices), "split_counts": split_counts, "layers": [{
                "layer": layer, "point": point, "mean_l2": float(norms.mean()),
                "mean_rms": float(norms.mean() / math.sqrt(matrix.shape[1])),
                "dispersion": float(np.linalg.norm(matrix - centroid, axis=1).mean()),
            }]})
        group_names = sorted(by_group)
        selected_pairs = [(group_a, group_b)] if group_a and group_b else [(a, b) for i, a in enumerate(group_names) for b in group_names[i + 1:]]
        if group_a is not None and (group_a not in by_group or group_b not in by_group or group_a == group_b):
            raise ResidualServiceError("invalid_input", "group_a and group_b must name two distinct captured groups")
        pairs = []
        sample_index = {sample["id"]: i for i, sample in enumerate(samples)}
        for a, b in selected_pairs:
            if a not in by_group or b not in by_group or a == b:
                continue
            paired = self._paired_rows(samples, a, b)
            deltas = [vectors[sample_index[row_a["id"]]] - vectors[sample_index[row_b["id"]]] for row_a, row_b in paired]
            if deltas:
                delta_matrix = np.stack(deltas)
                ca = np.stack([vectors[sample_index[row_a["id"]]] for row_a, _ in paired]).mean(axis=0)
                cb = np.stack([vectors[sample_index[row_b["id"]]] for _, row_b in paired]).mean(axis=0)
                nca, ncb = np.linalg.norm(ca), np.linalg.norm(cb)
                cosine = float(np.dot(ca, cb) / (nca * ncb)) if nca > 0 and ncb > 0 else None
                split_counts = {split: sum(row_a["split"] == split for row_a, _ in paired) for split in SPLITS}
                pairs.append({"group_a": a, "group_b": b, "layer": layer, "point": point, "n_pairs": len(paired), "split_counts": split_counts, "delta_norm": float(np.linalg.norm(delta_matrix, axis=1).mean()), "centroid_cosine": cosine})
        strips = []
        token_ids, positions, valid = capture["arrays"]["token_ids"], capture["arrays"]["positions"], capture["arrays"]["valid"]
        for i, sample in enumerate(samples):
            token_rows = []
            tokens = capture["tokens"][i]
            for j in np.flatnonzero(valid[i]):
                vector = capture["arrays"]["vectors"][i, layers.index(layer), points.index(point), j].astype(np.float64)
                relative = int(positions[i, j] - positions[i, np.flatnonzero(valid[i])[-1]])
                token = tokens[j] if j < len(tokens) else None
                absolute_position = int(positions[i, j])
                token_rows.append({"index": absolute_position, "token_id": int(token_ids[i, j]), "absolute_position": absolute_position, "relative_position": relative, "norm": float(np.linalg.norm(vector)), "token": token})
            strips.append({"id": sample["id"], "pair_id": sample["pair_id"], "group": sample["group"], "family_id": sample["family_id"], "split": sample["split"], "has_candidates": "target" in sample and "foil" in sample, "tokens": token_rows})
        hook_mapping = {
            "resid_pre": "decoder block input (forward pre-hook)",
            "resid_mid": "post_attention_layernorm input (forward pre-hook)",
            "resid_post": "decoder block output (forward hook)",
            "attn_write": "self_attn.o_proj output (forward hook)",
            "mlp_write": "mlp.down_proj output (forward hook)",
        }
        provenance = {key: capture["metadata"].get(key) for key in ("model_id", "revision", "device", "hidden_size", "layers", "points", "window", "family", "template_sha256", "tokenization", "token_position", "causal_context", "capture_dtype", "model_activation_dtypes")}
        provenance["hook_mapping"] = {name: hook_mapping[name] for name in capture["metadata"]["points"]}
        return {"capture_id": capture_id, "integrity_status": capture["job"].get("integrity_status", "verified"), "selected": {"layer": layer, "point": point}, "groups": groups, "pairs": pairs, "samples": strips,
                "token_alignment_note": "Relative positions are local to each prompt; token IDs and token text are not aligned across samples.",
                "provenance": provenance}

    def export_capture(self, capture_id: str) -> Path:
        capture = self._load_capture(capture_id)
        path = capture["directory"] / "capture.npz"
        return path

    def get_fit(self, fit_id: str) -> dict[str, Any]:
        fit = self._load_fit(fit_id)
        meta = fit["metadata"]
        result = {key: meta[key] for key in ("fit_id", "capture_id", "fit", "provenance", "basis_rank", "training_samples", "training_families", "quantization", "cells", "control_ranks", "projection_note", "created_at") if key in meta}
        result.update(self._fit_training_counts(meta))
        result["status"] = "complete"
        result["integrity_status"] = fit["job"].get("integrity_status", "verified")
        return result

    @staticmethod
    def _fit_training_counts(metadata: dict[str, Any]) -> dict[str, Any]:
        mode = metadata["fit"]["mode"]
        units = metadata.get("training_units")
        has_explicit_unit_count = units is not None
        if units is None:
            units = metadata.get("training_samples", 0) if mode == "states" else metadata.get("training_pairs", metadata.get("training_samples", 0))
        sample_count = metadata.get("training_samples", units) if mode == "states" or has_explicit_unit_count else 2 * units
        return {
            "training_units": units,
            "training_pairs": metadata.get("training_pairs", units if mode == "paired_delta" else None),
            "training_samples": sample_count,
            "training_data_kind": metadata.get("training_data_kind", "sample_states" if mode == "states" else "matched_group_a_minus_group_b_deltas"),
            "quantization_unit": metadata.get("quantization_unit", "samples" if mode == "states" else "matched_pairs"),
        }

    def get_fit_cell(self, fit_id: str, cell_id: int, split: str = "all") -> dict[str, Any]:
        fit = self._load_fit(fit_id)
        cells = fit["metadata"]["cells"]
        if isinstance(cell_id, bool) or not isinstance(cell_id, int) or not 0 <= cell_id < len(cells):
            raise ResidualServiceError("invalid_input", "cell_id is outside the fitted grid")
        if split != "all" and split not in SPLITS:
            raise ResidualServiceError("invalid_input", "split must be train, validation, test, or all")
        cell = cells[cell_id]
        rows = [item for item in fit["metadata"]["assignments"] if item["cell_id"] == cell_id and (split == "all" or item["split"] == split)]
        return {"fit_id": fit_id, "cell": {key: cell[key] for key in ("cell_id", "row", "col")}, "split": split, "n": len(rows),
                "group_counts": cell["group_counts"], "split_counts": cell["split_counts"], "samples": rows}

    def _run_probe(self, probe_id: str, directory: Path, params: dict[str, Any], cancel: threading.Event, record: dict[str, Any]) -> dict[str, str]:
        fit = self._load_fit(params["fit_id"])
        if fit["fit"]["mode"] != "paired_delta":
            raise ResidualServiceError("probe_requires_pairs", "Candidate probes require paired_delta fits", 409)
        capture = self._load_capture(fit["capture_id"])
        selected_ids = set(params["selected_ids"])
        samples = [sample for sample in capture["dataset"]["samples"] if sample["id"] in selected_ids]
        if len(samples) != len(selected_ids) or any(sample["split"] != params["split"] or "target" not in sample or "foil" not in sample for sample in samples):
            raise ResidualServiceError("invalid_probe_samples", "Probe sample selection no longer matches the held-out capture", 409)
        names = ("baseline", "zero", "som_remove", "global_mean_remove", "pca_remove", "random_remove")
        basis_map = {"som_remove": fit["arrays"]["basis"], "global_mean_remove": fit["arrays"]["global_mean"], "pca_remove": fit["arrays"]["pca"], "random_remove": fit["arrays"]["random"]}
        score_records = []
        self._progress(record, directory, "probing", 0, len(samples))
        for sample_index, sample in enumerate(samples):
            self._check_cancel(cancel)
            by_method: dict[str, dict[str, float]] = {}
            token_counts: dict[str, int] = {}
            continuations = {"target": sample["target"], "foil": sample["foil"]}
            for label, continuation in continuations.items():
                base_result = self._score_continuation(sample, fit["fit"]["layer"], fit["fit"]["point"], continuation)
                baseline = float(base_result["score"])
                token_counts[f"{label}_token_count"] = int(base_result["token_count"])
                zero_result = self._score_continuation(sample, fit["fit"]["layer"], fit["fit"]["point"], continuation, basis=fit["arrays"]["basis"], alpha=0.0)
                zero = float(zero_result["score"])
                by_method.setdefault("baseline", {})[f"{label}_logprob"] = baseline
                by_method.setdefault("zero", {})[f"{label}_logprob"] = zero
                for method, basis in basis_map.items():
                    rank = basis.shape[1]
                    if rank == 0:
                        result = self._score_continuation(sample, fit["fit"]["layer"], fit["fit"]["point"], continuation)
                    else:
                        result = self._score_continuation(sample, fit["fit"]["layer"], fit["fit"]["point"], continuation, basis=basis, alpha=-params["alpha"])
                    by_method.setdefault(method, {})[f"{label}_logprob"] = float(result["score"])
            zero_equivalent = all(abs(by_method["zero"][f"{key}_logprob"] - by_method["baseline"][f"{key}_logprob"]) <= 1e-6 for key in continuations)
            for method in names:
                values = by_method[method]
                values["margin"] = values["target_logprob"] - values["foil_logprob"]
            score_records.append({"id": sample["id"], "pair_id": sample["pair_id"], "family_id": sample["family_id"], "group": sample["group"], "token_counts": token_counts, "zero_equivalent": zero_equivalent, "methods": by_method})
            self._progress(record, directory, "probing", sample_index + 1, len(samples))
        methods = []
        groups = sorted({sample["group"] for sample in samples})
        for method in names:
            records = [item for item in score_records]
            margins = [item["methods"][method]["margin"] for item in records]
            method_summary = {"method": method, "n": len(records),
                              "target_mean_logprob": float(np.mean([item["methods"][method]["target_logprob"] for item in records])),
                              "foil_mean_logprob": float(np.mean([item["methods"][method]["foil_logprob"] for item in records])),
                              "mean_target_minus_foil": float(np.mean(margins)),
                              "by_group": []}
            for group in groups:
                rows = [item for item in records if item["group"] == group]
                method_summary["by_group"].append({"group": group, "n": len(rows),
                    "target_mean_logprob": float(np.mean([item["methods"][method]["target_logprob"] for item in rows])) if rows else None,
                    "foil_mean_logprob": float(np.mean([item["methods"][method]["foil_logprob"] for item in rows])) if rows else None,
                    "mean_target_minus_foil": float(np.mean([item["methods"][method]["margin"] for item in rows])) if rows else None})
            methods.append(method_summary)
        meta = {"schema": "residual-group-probe-v1", "probe_id": probe_id, "fit_id": params["fit_id"], "status": "complete",
                "split": params["split"], "alpha": params["alpha"], "helper_alpha": -params["alpha"],
                "selection": {"max_samples": params["max_samples"], "selected_samples": len(samples), "method": "deterministic_sha256_order"},
                "zero_equivalence": {"all_samples_within_1e-6": all(item["zero_equivalent"] for item in score_records), "equivalent_samples": sum(item["zero_equivalent"] for item in score_records), "n": len(score_records)},
                "score_mode": "mean_token_logprob",
                "score_policy": "Each continuation is scored by mean token log-probability to reduce answer-length confounding; margins compare those per-token means.",
                "interpretation": "Candidate log-prob preference only; not semantic accuracy. The projection removes the global span of training SOM prototypes and is not cell-routed.",
                "basis_ranks": {"som_remove": int(fit["arrays"]["basis"].shape[1]), "global_mean_remove": int(fit["arrays"]["global_mean"].shape[1]), "pca_remove": int(fit["arrays"]["pca"].shape[1]), "random_remove": int(fit["arrays"]["random"].shape[1])},
                "methods": methods, "samples": score_records, "created_at": _now()}
        _write_json(directory / "probe.json", meta)
        return {"probe_id": probe_id}

    def get_probe(self, probe_id: str, *, allow_in_progress: bool = False) -> dict[str, Any]:
        directory, record = self._read_job(probe_id)
        if record.get("kind") != "probe" or (record.get("status") != "completed" and not (allow_in_progress and record.get("status") == "running")):
            return self._public_job(record)
        if allow_in_progress:
            self._verify_job_artifacts(directory, record)
        path = directory / "probe.json"
        try:
            if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                raise OSError("invalid probe")
            result = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("schema") != "residual-group-probe-v1" or result.get("probe_id") != probe_id or not _json_numbers_are_finite(result):
                raise OSError("invalid probe")
            params = record.get("params")
            fit = self._load_fit(result.get("fit_id"))
            if not isinstance(params, dict) or record.get("result") != {"probe_id": probe_id} or result.get("fit_id") != params.get("fit_id") or result.get("fit_id") != fit["fit_id"] or result.get("score_mode") != "mean_token_logprob":
                raise OSError("probe fit or score provenance mismatch")
            if result.get("split") != params.get("split") or result.get("alpha") != params.get("alpha"):
                raise OSError("probe parameters do not match the queued request")
            samples = result.get("samples")
            methods = result.get("methods")
            expected_methods = {"baseline", "zero", "som_remove", "global_mean_remove", "pca_remove", "random_remove"}
            if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_PROBE_SAMPLES or not isinstance(methods, list) or len(methods) != len(expected_methods) or {item.get("method") for item in methods if isinstance(item, dict)} != expected_methods:
                raise OSError("probe result shape is invalid")
            if {item.get("id") for item in samples if isinstance(item, dict)} != set(params.get("selected_ids", [])):
                raise OSError("probe sample IDs do not match the queued request")
            capture = self._load_capture(fit["capture_id"])
            dataset_by_id = {sample["id"]: sample for sample in capture["dataset"]["samples"]}
            for sample in samples:
                source = dataset_by_id.get(sample.get("id")) if isinstance(sample, dict) else None
                if source is None or source["split"] != result.get("split") or any(sample.get(key) != source[key] for key in ("pair_id", "family_id", "group")):
                    raise OSError("probe sample metadata does not match its capture")
            for sample in samples:
                if not isinstance(sample, dict) or not isinstance(sample.get("methods"), dict) or set(sample["methods"]) != expected_methods:
                    raise OSError("probe sample result is invalid")
                counts = sample.get("token_counts")
                if not isinstance(counts, dict) or set(counts) != {"target_token_count", "foil_token_count"} or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts.values()):
                    raise OSError("probe token counts are invalid")
                for method in expected_methods:
                    scores = sample["methods"][method]
                    if not isinstance(scores, dict) or any(not isinstance(scores.get(key), (int, float)) or isinstance(scores.get(key), bool) or not math.isfinite(float(scores[key])) for key in ("target_logprob", "foil_logprob", "margin")):
                        raise OSError("probe scores are invalid")
            zero = result.get("zero_equivalence")
            if not isinstance(zero, dict) or zero.get("n") != len(samples) or zero.get("equivalent_samples") != sum(bool(sample.get("zero_equivalent")) for sample in samples) or zero.get("all_samples_within_1e-6") is not (zero.get("equivalent_samples") == len(samples)):
                raise OSError("probe zero-equivalence summary is inconsistent")
            result["integrity_status"] = record.get("integrity_status", "verified")
            return result
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise ResidualServiceError("invalid_probe", "The probe result is missing or malformed", 409) from None

    def list_jobs(self) -> dict[str, Any]:
        jobs = []
        root = self._root()
        if root.is_dir():
            for path in root.glob("*/job.json"):
                if not _ID_RE.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(record, dict) and record.get("job_id") == path.parent.name:
                        public = self._public_job(record)
                        if record.get("status") == "completed":
                            try:
                                _, checked = self._read_job(path.parent.name)
                                public["integrity_status"] = checked.get("integrity_status", "verified")
                            except ResidualServiceError:
                                public["integrity_status"] = "invalid"
                                public["error_code"] = "integrity_failed"
                        jobs.append(public)
                except (OSError, json.JSONDecodeError):
                    continue
        return {"jobs": sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)[:100]}

    def list_captures(self) -> dict[str, Any]:
        captures = []
        root = self._root()
        if root.is_dir():
            for path in root.glob("*/job.json"):
                if not _ID_RE.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(record, dict) or record.get("kind") != "capture" or record.get("status") != "completed":
                        continue
                    capture = self._load_capture(path.parent.name)
                    samples = capture["dataset"]["samples"]
                    captures.append({
                        "capture_id": path.parent.name,
                        "name": capture["dataset"]["name"],
                        "created_at": capture["manifest"].get("created_at"),
                        "model_id": capture["metadata"].get("model_id"),
                        "revision": capture["metadata"].get("revision"),
                        "samples": len(samples),
                        "groups": sorted({sample["group"] for sample in samples}),
                        "layers": capture["metadata"]["layers"],
                        "points": capture["metadata"]["points"],
                        "window": capture["metadata"]["window"],
                        "integrity_status": capture["job"].get("integrity_status", "verified"),
                    })
                except (OSError, ResidualServiceError, json.JSONDecodeError):
                    continue
        return {"captures": sorted(captures, key=lambda item: item.get("created_at") or "", reverse=True)[:100]}

    def list_fits(self) -> dict[str, Any]:
        fits = []
        root = self._root()
        if root.is_dir():
            for path in root.glob("*/job.json"):
                if not _ID_RE.fullmatch(path.parent.name) or path.is_symlink() or path.parent.is_symlink():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(record, dict) or record.get("kind") != "fit" or record.get("status") != "completed":
                        continue
                    fit = self._load_fit(path.parent.name)
                    training = self._fit_training_counts(fit["metadata"])
                    fits.append({
                        "fit_id": path.parent.name,
                        "capture_id": fit["capture_id"],
                        "created_at": fit["metadata"].get("created_at"),
                        "fit": fit["fit"],
                        "basis_rank": fit["metadata"].get("basis_rank"),
                        "training_samples": training["training_samples"],
                        "training_units": training["training_units"],
                        "training_pairs": training["training_pairs"],
                        "training_families": fit["metadata"].get("training_families"),
                        "integrity_status": fit["job"].get("integrity_status", "verified"),
                    })
                except (OSError, ResidualServiceError, json.JSONDecodeError):
                    continue
        return {"fits": sorted(fits, key=lambda item: item.get("created_at") or "", reverse=True)[:100]}
