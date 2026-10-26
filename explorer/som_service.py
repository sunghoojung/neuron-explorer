"""Local, bounded SOM job service for the neuron explorer.

The service only reads prompt, activation, and SOM artifact files from the
bundled example directory or explicitly configured local source roots. Jobs
write to unique run directories below ``EXPLORER_SOM_RUNS_DIR``. It never
accepts a shell command or an arbitrary source path from a caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.data import (
    ActivationData,
    load_activations,
    load_prompt_jsonl,
    prompt_dataset_sha256,
    validate_activation_match,
    validate_prompt_rows,
)
from research.som import (
    SOMArtifact,
    load_artifact,
    validate_artifact,
)
from research.som_pipeline import DEFAULT_HOOK, DEFAULT_MODEL

_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_SOURCE_ID_RE = re.compile(r"^src_[a-f0-9]{24}$")
_ARTIFACT_ID_RE = re.compile(r"^som_[a-f0-9]{24}$")
_PROGRESS_RE = re.compile(r"scoring\s+(\d+)/(\d+)\s+held-out questions", re.IGNORECASE)
_STAGES = ("collect", "fit", "evaluate")
_ACTIVE_STATES = {"queued", "collecting", "fitting", "evaluating", "validating"}

MAX_PENDING_JOBS = 2
MAX_SOURCE_FILES_PER_ROOT = 128
MAX_SOURCE_ENTRIES_PER_ROOT = 4096
MAX_SOURCE_DEPTH = 6
MAX_PROMPT_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_RUN_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_PILOT_QUESTION_IDS = 24
MAX_PROMPT_RECORDS = MAX_PILOT_QUESTION_IDS * 4
MAX_GRID_SIDE = 8
MAX_GRID_UNITS = 64
MAX_EPOCHS = 500
MAX_EVALUATION_QUESTIONS = 32
MAX_JOB_SECONDS = 6 * 60 * 60


class SOMServiceError(ValueError):
    """A bounded, user-safe error from the local SOM service."""

    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message


class _JobCancelled(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _valid_model_id(value: Any) -> str:
    if not isinstance(value, str):
        raise SOMServiceError("invalid_input", "model_id must be a non-empty model identifier")
    model_id = value.strip()
    if len(model_id) > 200 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?", model_id
    ):
        raise SOMServiceError("invalid_input", "model_id must be a non-empty model identifier")
    return model_id


def _valid_revision(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise SOMServiceError("invalid_input", "revision must be text")
    revision = value.strip()
    if (
        not revision
        or len(revision) > 200
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", revision)
        or ".." in revision
        or "//" in revision
        or revision.endswith("/")
    ):
        raise SOMServiceError("invalid_input", "revision must be a short model revision")
    return revision


def _valid_layer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 128:
        raise SOMServiceError("invalid_input", "layer must be a zero-based integer from 0 to 127")
    return value


def _artifact_summary(artifact_id: str, artifact: SOMArtifact) -> dict[str, Any]:
    metadata = artifact.metadata
    layer = metadata["layer"]
    expected_hook = DEFAULT_HOOK.format(layer=layer)
    return {
        "artifact_id": artifact_id,
        "status": "valid",
        "format": "neuron-explorer-som-v1",
        "method": metadata["method"],
        "model_id": metadata["model_id"],
        "revision": metadata["revision"],
        "layer": layer,
        "hook": metadata["hook"],
        "hidden_size": metadata["hidden_size"],
        "prototype_count": len(artifact.prototypes),
        "basis_rank": metadata["basis_rank"],
        "grid": list(metadata.get("grid", [])),
        "seed": metadata.get("seed"),
        "training_split": metadata.get("training_split"),
        "training_conditions": list(metadata.get("training_conditions", [])),
        "training_question_count": metadata.get("training_question_count"),
        "hook_matches_pipeline": metadata["hook"] == expected_hook,
        "compatibility_note": (
            None if metadata["hook"] == expected_hook else
            "The artifact hook does not match this pipeline's decoder output hook."
        ),
    }


class SOMService:
    """Provide local SOM source discovery, job status, and bounded pilot runs."""

    def __init__(
        self,
        repo_root: str | os.PathLike[str] | None = None,
        run_root: str | os.PathLike[str] | None = None,
        source_roots: Sequence[str | os.PathLike[str]] | None = None,
        default_model: str | None = None,
        default_revision: str | None = None,
    ):
        self.repo_root = Path(repo_root or REPO_ROOT).expanduser().resolve()
        configured_run_root = run_root or os.environ.get("EXPLORER_SOM_RUNS_DIR")
        self.run_root = Path(configured_run_root or self.repo_root / "research_outputs" / "som_runs").expanduser().resolve()
        self._manifest_warning: str | None = None
        self._source_roots_warning: str | None = None
        configured_manifest = os.environ.get("EXPLORER_SOM_MANIFEST")
        self.manifest_path = Path(configured_manifest).expanduser().resolve() if configured_manifest else self.repo_root / "explorer" / "som_sources.local.json"
        manifest = self._read_manifest()
        if source_roots is None:
            raw_roots = os.environ.get("EXPLORER_SOM_SOURCE_ROOTS", "")
            if raw_roots.strip():
                source_roots = [value for value in raw_roots.split(os.pathsep) if value.strip()]
            else:
                configured_roots = manifest.get("source_roots", []) if manifest else []
                if not isinstance(configured_roots, list) or len(configured_roots) > 32 or any(
                    not isinstance(value, str) or not value.strip() for value in configured_roots
                ):
                    self._source_roots_warning = "Local SOM source roots are invalid; no external source roots were loaded."
                    configured_roots = []
                source_roots = configured_roots
        try:
            normalized_roots = [os.fspath(root) for root in source_roots]
            if len(normalized_roots) > 32 or any(
                not isinstance(root, str) or not root.strip() or len(root) > 2048 or "\x00" in root
                for root in normalized_roots
            ):
                raise ValueError("invalid source root")
            self.source_roots = tuple(Path(root).expanduser().resolve() for root in normalized_roots)
        except (TypeError, ValueError, OSError):
            self._source_roots_warning = "Local SOM source roots are invalid; no external source roots were loaded."
            self.source_roots = ()
        self.prompt_roots = (self.repo_root / "research" / "examples", *self.source_roots)
        self.default_model = _valid_model_id(
            default_model
            or os.environ.get("EXPLORER_SOM_DEFAULT_MODEL")
            or os.environ.get("EXPLORER_MODEL")
            or DEFAULT_MODEL
        )
        env_revision = (
            os.environ.get("EXPLORER_SOM_DEFAULT_REVISION")
            or os.environ.get("EXPLORER_REVISION")
        )
        self.default_revision = _valid_revision(default_revision if default_revision is not None else env_revision)
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._recover_interrupted_jobs()

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            if os.environ.get("EXPLORER_SOM_MANIFEST"):
                self._manifest_warning = "The configured local SOM manifest could not be read."
            return None
        try:
            if self.manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
                self._manifest_warning = "The local SOM manifest is larger than the allowed size."
                return None
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._manifest_warning = "The local SOM manifest is invalid or unreadable."
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            self._manifest_warning = "The local SOM manifest has an unsupported schema."
            return None
        self._manifest_warning = None
        return payload

    def status(self) -> dict[str, Any]:
        prompt_count = len(self._scan_prompt_sources())
        artifact_count = len(self._scan_artifacts())
        external_jobs = self._external_job_records()
        config_warning = self._source_roots_warning or self._manifest_warning
        result = {
            "status": "degraded" if config_warning else "ready",
            "pipeline": "research.som_pipeline",
            "default_model": self.default_model,
            "default_revision": self.default_revision,
            "hook_template": DEFAULT_HOOK,
            "configured_source_roots": len(self.source_roots),
            "available_prompt_sources": prompt_count,
            "available_artifacts": artifact_count,
            "recorded_external_jobs": len(external_jobs),
            "external_jobs": [self._public_external_job(record) for record in external_jobs],
            "active_job_limit": MAX_PENDING_JOBS,
            "supported_device": "cpu or mps",
            "external_status_note": "Recorded external scheduler states are historical unless a live check is explicitly confirmed.",
        }
        if config_warning:
            result["configuration_warning"] = config_warning
        return result

    def close(self) -> None:
        """Stop active local jobs when the explorer service shuts down."""
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
            for process in self._processes.values():
                if process.poll() is None:
                    self._stop_process(process)
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def list_sources(self) -> dict[str, Any]:
        sources = self._scan_prompt_sources()
        public = []
        for item in sources:
            public.append({
                "source_id": item["source_id"],
                "label": item["label"],
                "kind": "paired_prompts",
                "status": "valid",
                "records": item["records"],
                "questions": item["questions"],
                "splits": item["splits"],
                "dataset_kind": item["dataset_kind"],
            })
        return {"sources": public}

    def list_artifacts(self) -> dict[str, Any]:
        artifacts = self._scan_artifacts()
        return {"artifacts": artifacts}

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise SOMServiceError("unknown_id", "No matching validated SOM artifact was found.", 404)
        for path in self._artifact_paths():
            if self._artifact_id(path) != artifact_id:
                continue
            artifact = self._load_valid_artifact(path)
            if artifact is not None:
                return self._artifact_summary(artifact_id, artifact)
            break
        raise SOMServiceError("unknown_id", "No matching validated SOM artifact was found.", 404)

    def list_jobs(self) -> dict[str, Any]:
        jobs: dict[str, dict[str, Any]] = {}
        if self.run_root.is_dir():
            for path in sorted(self.run_root.glob("*/job.json"), key=lambda item: item.stat().st_mtime, reverse=True):
                if not _JOB_ID_RE.fullmatch(path.parent.name):
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict) and record.get("job_id") == path.parent.name:
                    jobs[record["job_id"]] = self._public_job(record, path.parent)
        for record in self._external_job_records():
            jobs[record["job_id"]] = self._public_external_job(record)
        ordered = sorted(jobs.values(), key=lambda item: item.get("updated_at", ""), reverse=True)
        return {"jobs": ordered[:100]}

    def get_job(self, job_id: str) -> dict[str, Any]:
        if isinstance(job_id, str) and _JOB_ID_RE.fullmatch(job_id):
            path = self.run_root / job_id / "job.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise SOMServiceError("unknown_id", "No local SOM job with that id was found.", 404) from None
            if not isinstance(record, dict) or record.get("job_id") != job_id:
                raise SOMServiceError("unknown_id", "No local SOM job with that id was found.", 404)
            return self._public_job(record, path.parent)
        for record in self._external_job_records():
            if record["job_id"] == str(job_id):
                return self._public_external_job(record)
        raise SOMServiceError("unknown_id", "No SOM job with that id was found.", 404)

    def start_pilot(
        self,
        *,
        source_id: str,
        model_id: str | None = None,
        revision: str | None = None,
        layer: int,
        hook: str | None = None,
        device: str = "cpu",
        rows: int = 4,
        cols: int = 4,
        epochs: int = 100,
        seed: int = 0,
        split: str = "test",
        max_questions: int = 12,
    ) -> dict[str, Any]:
        source = self._find_prompt_source(source_id)
        model = _valid_model_id(model_id or self.default_model)
        requested_revision = _valid_revision(revision if revision is not None else self.default_revision)
        layer_index = _valid_layer(layer)
        expected_hook = DEFAULT_HOOK.format(layer=layer_index)
        if hook is not None and hook != expected_hook:
            raise SOMServiceError("not_compatible", "The requested hook does not match the pipeline's decoder output hook.", 409)
        if device not in {"cpu", "mps"}:
            raise SOMServiceError("invalid_input", "device must be cpu or mps for a local pilot")
        if isinstance(rows, bool) or isinstance(cols, bool) or not isinstance(rows, int) or not isinstance(cols, int):
            raise SOMServiceError("invalid_input", "SOM rows and columns must be integers")
        if not 1 <= rows <= MAX_GRID_SIDE or not 1 <= cols <= MAX_GRID_SIDE or rows * cols > MAX_GRID_UNITS:
            raise SOMServiceError("invalid_input", f"SOM grid must contain 1 to {MAX_GRID_UNITS} units")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= MAX_EPOCHS:
            raise SOMServiceError("invalid_input", f"epochs must be an integer from 1 to {MAX_EPOCHS}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise SOMServiceError("invalid_input", "seed must be an integer from 0 to 4294967295")
        if split not in {"validation", "test"}:
            raise SOMServiceError("invalid_input", "split must be validation or test")
        if isinstance(max_questions, bool) or not isinstance(max_questions, int) or not 1 <= max_questions <= MAX_EVALUATION_QUESTIONS:
            raise SOMServiceError("invalid_input", f"max_questions must be from 1 to {MAX_EVALUATION_QUESTIONS}")

        try:
            prompt_rows = load_prompt_jsonl(source["path"])
        except (OSError, ValueError):
            raise SOMServiceError("invalid_source", "The selected prompt source is no longer readable or valid.", 400) from None
        question_ids = {row["question_id"] for row in prompt_rows}
        if len(question_ids) > MAX_PILOT_QUESTION_IDS:
            raise SOMServiceError(
                "invalid_source",
                f"This pilot is limited to {MAX_PILOT_QUESTION_IDS} complete questions ({MAX_PROMPT_RECORDS} condition rows).",
                400,
            )
        if len(prompt_rows) > MAX_PROMPT_RECORDS:
            raise SOMServiceError("invalid_source", f"This pilot is limited to {MAX_PROMPT_RECORDS} prompt records.", 400)
        train_questions = {row["question_id"] for row in prompt_rows if row["split"] == "train"}
        heldout_questions = {row["question_id"] for row in prompt_rows if row["split"] == split}
        if not train_questions or not heldout_questions:
            raise SOMServiceError(
                "invalid_source",
                "The selected prompts need at least one train question and one selected held-out question.",
                400,
            )

        with self._lock:
            active_count = sum(
                1 for path in self.run_root.glob("*/job.json") if self._job_is_active(path)
            ) if self.run_root.is_dir() else 0
            if active_count >= MAX_PENDING_JOBS:
                raise SOMServiceError("capacity", "The local SOM job queue is full. Check existing jobs and try again.", 429)
            job_id = uuid.uuid4().hex
            job_dir = self.run_root / job_id
            try:
                job_dir.mkdir(parents=True, exist_ok=False)
            except OSError:
                raise SOMServiceError("unavailable", "The local SOM output directory could not be created.", 503) from None
            record: dict[str, Any] = {
                "schema_version": 1,
                "job_id": job_id,
                "kind": "pilot",
                "status": "queued",
                "stage": "queued",
                "progress": {"stage_index": 0, "stage_count": len(_STAGES), "current": 0, "total": 0},
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "source_id": source["source_id"],
                "source_label": source["label"],
                "dataset_kind": source["dataset_kind"],
                "prompt_dataset_sha256": source["dataset_sha256"],
                "model_id": model,
                "revision_requested": requested_revision or "adapter default",
                "revision": None,
                "layer": layer_index,
                "hook": expected_hook,
                "device": device,
                "fit": {"rows": rows, "cols": cols, "epochs": epochs, "seed": seed},
                "evaluation": {"split": split, "max_questions": max_questions},
                "artifacts": [],
                "error": None,
            }
            self._write_job(job_dir, record)
            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="som-pilot")
            future = self._executor.submit(
                self._execute_pilot,
                job_id,
                source["path"],
                prompt_rows,
                requested_revision,
                job_dir,
                record,
                cancel_event,
            )
            self._futures[job_id] = future
        return self.get_job(job_id)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
            raise SOMServiceError("unknown_id", "No local SOM job with that id was found.", 404)
        with self._lock:
            path = self.run_root / job_id / "job.json"
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise SOMServiceError("unknown_id", "No local SOM job with that id was found.", 404) from None
            if record.get("status") not in _ACTIVE_STATES:
                return self._public_job(record, path.parent)
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                record["status"] = "cancelled"
                record["stage"] = "cancelled"
                record["error"] = "The local job was cancelled before it started."
                record["updated_at"] = _utc_now()
                self._write_job(path.parent, record)
                event = self._cancel_events.get(job_id)
                if event:
                    event.set()
                return self._public_job(record, path.parent)
            event = self._cancel_events.get(job_id)
            if event is None:
                raise SOMServiceError("unavailable", "This job is not running in the current explorer process.", 409)
            event.set()
            process = self._processes.get(job_id)
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            return self.get_job(job_id)

    def _find_prompt_source(self, source_id: str) -> dict[str, Any]:
        if not isinstance(source_id, str) or not _SOURCE_ID_RE.fullmatch(source_id):
            raise SOMServiceError("unknown_id", "No paired prompt source with that id was found.", 404)
        for source in self._scan_prompt_sources():
            if source["source_id"] == source_id:
                return source
        raise SOMServiceError("unknown_id", "No paired prompt source with that id was found.", 404)

    def _scan_prompt_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for root_index, root in enumerate(self.prompt_roots):
            for path in self._walk_files(root, {".jsonl"}):
                if path in seen or path.stat().st_size > MAX_PROMPT_BYTES:
                    continue
                seen.add(path)
                try:
                    rows = load_prompt_jsonl(path)
                    stats = validate_prompt_rows(rows)
                    digest = _sha256_file(path)
                except (OSError, ValueError):
                    continue
                source_hash = hashlib.sha256(f"{path}\0{digest}".encode()).hexdigest()[:24]
                is_example = path.resolve() == (self.repo_root / "research" / "examples" / "paired_prompts.jsonl").resolve()
                sources.append({
                    "source_id": f"src_{source_hash}",
                    "label": "Bundled example prompts" if is_example else "Configured local prompt source",
                    "path": path,
                    "records": stats["records"],
                    "questions": stats["questions"],
                    "splits": stats["questions_by_split"],
                    "dataset_kind": "bundled_example" if is_example else "configured_local_input",
                    "root_index": root_index,
                    "dataset_sha256": prompt_dataset_sha256(rows),
                })
        return sorted(sources, key=lambda item: (item["dataset_kind"] != "bundled_example", item["label"], item["source_id"]))

    def _artifact_paths(self) -> list[Path]:
        roots = (*self.source_roots, self.run_root)
        paths: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            for path in self._walk_files(root, {".npz"}):
                if path in seen or path.stat().st_size > MAX_ARTIFACT_BYTES:
                    continue
                seen.add(path)
                paths.append(path)
        return paths

    def _scan_artifacts(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for path in self._artifact_paths():
            artifact = self._load_valid_artifact(path)
            if artifact is None:
                continue
            artifact_id = self._artifact_id(path)
            results.append(self._artifact_summary(artifact_id, artifact))
        return sorted(results, key=lambda item: (item["model_id"], item["revision"], item["layer"], item["artifact_id"]))

    def _artifact_summary(self, artifact_id: str, artifact: SOMArtifact) -> dict[str, Any]:
        summary = _artifact_summary(artifact_id, artifact)
        metadata = artifact.metadata
        reasons = []
        if metadata["model_id"] != self.default_model:
            reasons.append("artifact model differs from the active explorer model")
        if self.default_revision is None:
            reasons.append("active explorer revision is not pinned")
        elif metadata["revision"] != self.default_revision:
            reasons.append("artifact revision differs from the active explorer revision")
        if metadata["hook"] != DEFAULT_HOOK.format(layer=metadata["layer"]):
            reasons.append("artifact hook differs from the explorer SOM decoder output hook")
        summary["model_revision_hook_match"] = not reasons
        summary["reuse_compatibility"] = "not_assessed"
        summary["compatibility_note"] = "; ".join(reasons) if reasons else "Model, revision, and hook match. Tokenization, readout semantics, training target, and behavioral reuse have not been assessed."
        return summary

    @staticmethod
    def _load_valid_artifact(path: Path) -> SOMArtifact | None:
        try:
            if path.stat().st_size > MAX_ARTIFACT_BYTES:
                return None
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                names = {member.filename for member in members if not member.is_dir()}
                if not {"prototypes.npy", "basis.npy", "metadata.npy"}.issubset(names):
                    return None
                if sum(member.file_size for member in members) > MAX_ARTIFACT_UNCOMPRESSED_BYTES:
                    return None
            return load_artifact(path)
        except (OSError, ValueError, zipfile.BadZipFile):
            return None

    @staticmethod
    def _artifact_id(path: Path) -> str:
        try:
            stat = path.stat()
            identity = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
        except OSError:
            identity = str(path)
        return "som_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def _walk_files(self, root: Path, suffixes: set[str]) -> list[Path]:
        if not root.exists():
            return []
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            return []
        if resolved_root.is_file():
            return [resolved_root] if resolved_root.suffix.lower() in suffixes else []
        paths: list[Path] = []
        stack: list[tuple[Path, int]] = [(resolved_root, 0)]
        visited_entries = 0
        while stack and len(paths) < MAX_SOURCE_FILES_PER_ROOT and visited_entries < MAX_SOURCE_ENTRIES_PER_ROOT:
            directory, depth = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        visited_entries += 1
                        if visited_entries >= MAX_SOURCE_ENTRIES_PER_ROOT:
                            break
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if depth < MAX_SOURCE_DEPTH:
                                    stack.append((Path(entry.path), depth + 1))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            candidate = Path(entry.path)
                            if candidate.suffix.lower() not in suffixes:
                                continue
                            resolved = candidate.resolve(strict=True)
                            if resolved.is_relative_to(resolved_root):
                                paths.append(resolved)
                                if len(paths) >= MAX_SOURCE_FILES_PER_ROOT:
                                    break
                        except OSError:
                            continue
            except OSError:
                continue
        return paths

    def _write_job(self, job_dir: Path, record: dict[str, Any]) -> None:
        record["updated_at"] = _utc_now()
        _json_write_atomic(job_dir / "job.json", record)

    def _recover_interrupted_jobs(self) -> None:
        """Mark persisted in-flight work interrupted after the server restarts."""
        if not self.run_root.is_dir():
            return
        for path in self.run_root.glob("*/job.json"):
            if not _JOB_ID_RE.fullmatch(path.parent.name):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(record, dict)
                or record.get("job_id") != path.parent.name
                or record.get("status") not in _ACTIVE_STATES
            ):
                continue
            record["status"] = "interrupted"
            record["stage"] = "interrupted"
            record["error"] = "Explorer restarted while this job was in progress; no live process is attached."
            record["error_code"] = "interrupted"
            record["completed_at"] = _utc_now()
            try:
                self._write_job(path.parent, record)
            except OSError:
                continue

    def _job_is_active(self, path: Path) -> bool:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(record, dict) and record.get("status") in _ACTIVE_STATES

    def _update_stage(
        self,
        job_id: str,
        job_dir: Path,
        record: dict[str, Any],
        stage: str,
        stage_index: int,
        current: int = 0,
        total: int = 0,
    ) -> None:
        record["status"] = {
            "collect": "collecting",
            "fit": "fitting",
            "evaluate": "evaluating",
            "validate": "validating",
        }.get(stage, "running")
        record["stage"] = stage
        record["progress"] = {
            "stage_index": stage_index,
            "stage_count": len(_STAGES),
            "current": int(current),
            "total": int(total),
        }
        self._write_job(job_dir, record)

    def _execute_pilot(
        self,
        job_id: str,
        prompts_path: Path,
        prompt_rows: list[dict[str, Any]],
        requested_revision: str | None,
        job_dir: Path,
        record: dict[str, Any],
        cancel_event: threading.Event,
    ) -> None:
        activations_path = job_dir / "activations.npz"
        artifact_path = job_dir / "som.npz"
        evaluation_path = job_dir / "evaluation.json"
        started = time.monotonic()
        try:
            self._check_prompt_identity(prompts_path, record)
            self._check_cancel(cancel_event)
            collect_args = [
                sys.executable,
                "-m",
                "research.som_pipeline",
                "collect",
                "--prompts",
                str(prompts_path),
                "--out",
                str(activations_path),
                "--model",
                record["model_id"],
                "--layer",
                str(record["layer"]),
                "--device",
                record["device"],
            ]
            if requested_revision:
                collect_args.extend(["--revision", requested_revision])
            self._update_stage(job_id, job_dir, record, "collect", 1, 0, len(prompt_rows))
            self._run_command(job_id, "collect", collect_args, job_dir, record, cancel_event, started)
            self._check_cancel(cancel_event)
            activation_data = load_activations(activations_path)
            validate_activation_match(prompt_rows, activation_data)
            self._validate_activation_identity(record, activation_data)
            if activation_data.metadata.get("dataset_sha256") != record["prompt_dataset_sha256"]:
                raise SOMServiceError("not_compatible", "Collected activations do not match the selected prompt source revision.", 409)
            record["revision"] = activation_data.metadata["revision"]
            record["artifacts"] = ["activations.npz"]
            self._update_stage(job_id, job_dir, record, "fit", 2, 0, 1)

            fit_args = [
                sys.executable,
                "-m",
                "research.som_pipeline",
                "fit",
                "--activations",
                str(activations_path),
                "--out",
                str(artifact_path),
                "--rows",
                str(record["fit"]["rows"]),
                "--cols",
                str(record["fit"]["cols"]),
                "--epochs",
                str(record["fit"]["epochs"]),
                "--seed",
                str(record["fit"]["seed"]),
            ]
            self._run_command(job_id, "fit", fit_args, job_dir, record, cancel_event, started)
            self._check_cancel(cancel_event)
            artifact = load_artifact(artifact_path)
            self._validate_artifact_identity(record, activation_data, artifact, activations_path)
            record["artifacts"] = ["activations.npz", "som.npz"]
            self._update_stage(job_id, job_dir, record, "evaluate", 3, 0, 0)
            self._check_prompt_identity(prompts_path, record)

            evaluation_args = [
                sys.executable,
                "-m",
                "research.som_pipeline",
                "evaluate",
                "--prompts",
                str(prompts_path),
                "--activations",
                str(activations_path),
                "--artifact",
                str(artifact_path),
                "--out",
                str(evaluation_path),
                "--split",
                record["evaluation"]["split"],
                "--model",
                record["model_id"],
                "--device",
                record["device"],
                "--max-questions",
                str(record["evaluation"]["max_questions"]),
            ]
            self._run_command(job_id, "evaluate", evaluation_args, job_dir, record, cancel_event, started)
            self._check_cancel(cancel_event)
            self._update_stage(job_id, job_dir, record, "validate", 3, 1, 1)
            self._validate_evaluation(record, evaluation_path)
            record["artifacts"] = ["activations.npz", "som.npz", "evaluation.json"]
            hashes_before = self._hash_run_outputs(job_dir)
            evaluation_result = self._validate_completed_outputs(job_dir, record)
            hashes_after = self._hash_run_outputs(job_dir)
            if hashes_before != hashes_after:
                raise SOMServiceError("integrity_failed", "A SOM output changed during final validation.", 500)
            record["artifact_sha256"] = hashes_after
            record["evaluation_summary"] = self._safe_evaluation_summary(evaluation_result)
            record["status"] = "completed"
            record["stage"] = "complete"
            record["progress"] = {"stage_index": 3, "stage_count": 3, "current": 1, "total": 1}
            record["error"] = None
            record["completed_at"] = _utc_now()
            record["scientific_interpretation"] = "plumbing_only" if record["dataset_kind"] == "bundled_example" else "descriptive_diagnostics_only"
            self._write_job(job_dir, record)
        except _JobCancelled:
            record["status"] = "cancelled"
            record["stage"] = "cancelled"
            record["error"] = "The local job was cancelled."
            record["completed_at"] = _utc_now()
            self._write_job(job_dir, record)
        except Exception as exc:  # noqa: BLE001 - persist bounded stage errors at the worker boundary
            record["status"] = "failed"
            record["stage"] = record.get("stage") or "failed"
            if isinstance(exc, SOMServiceError):
                record["error"] = exc.message
                record["error_code"] = exc.code
            else:
                stage = record.get("stage", "pipeline")
                record["error"] = f"{str(stage).capitalize()} failed. See the local run log for details."
                record["error_code"] = "stage_failed"
            record["completed_at"] = _utc_now()
            try:
                self._write_job(job_dir, record)
            except OSError:
                pass
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                self._cancel_events.pop(job_id, None)

    @staticmethod
    def _validate_activation_identity(record: dict[str, Any], data: ActivationData) -> None:
        metadata = data.metadata
        if metadata.get("model_id") != record["model_id"]:
            raise SOMServiceError("not_compatible", "Collected activations do not match the requested model.", 409)
        if int(metadata.get("layer", -1)) != record["layer"]:
            raise SOMServiceError("not_compatible", "Collected activations do not match the requested layer.", 409)
        if metadata.get("hook") != record["hook"]:
            raise SOMServiceError("not_compatible", "Collected activations do not match the requested hook.", 409)
        actual_revision = metadata.get("revision")
        if not isinstance(actual_revision, str) or not actual_revision:
            raise SOMServiceError("not_compatible", "Collected activations have no resolved model revision.", 409)
        requested = record["revision_requested"]
        if requested != "adapter default" and actual_revision != requested:
            raise SOMServiceError("not_compatible", "Collected activations do not match the requested model revision.", 409)
        record["revision"] = actual_revision

    @staticmethod
    def _check_prompt_identity(prompts_path: Path, record: dict[str, Any]) -> None:
        try:
            current_hash = prompt_dataset_sha256(load_prompt_jsonl(prompts_path))
        except (OSError, ValueError):
            raise SOMServiceError("invalid_source", "The selected prompt source is no longer readable or valid.", 400) from None
        if current_hash != record["prompt_dataset_sha256"]:
            raise SOMServiceError("not_compatible", "The selected prompt source changed after this job was created.", 409)

    @staticmethod
    def _validate_artifact_identity(
        record: dict[str, Any], data: ActivationData, artifact: SOMArtifact, activations_path: Path
    ) -> None:
        validate_artifact(artifact)
        metadata = artifact.metadata
        if metadata.get("model_id") != record["model_id"]:
            raise SOMServiceError("not_compatible", "The fitted SOM does not match the requested model.", 409)
        if metadata.get("revision") != record["revision"]:
            raise SOMServiceError("not_compatible", "The fitted SOM does not match the collected model revision.", 409)
        if metadata.get("layer") != record["layer"] or metadata.get("hook") != record["hook"]:
            raise SOMServiceError("not_compatible", "The fitted SOM does not match the collected layer and hook.", 409)
        if metadata.get("activation_sha256") != _sha256_file(activations_path):
            raise SOMServiceError("not_compatible", "The fitted SOM does not identify the collected activations.", 409)
        if metadata.get("hidden_size") != data.vectors.shape[1]:
            raise SOMServiceError("not_compatible", "The fitted SOM hidden size does not match collected activations.", 409)

    @staticmethod
    def _validate_evaluation(record: dict[str, Any], path: Path) -> dict[str, Any]:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise SOMServiceError("invalid_artifact", "The evaluation output is missing or invalid JSON.", 500) from None
        if not isinstance(result, dict) or result.get("schema_version") != 1:
            raise SOMServiceError("invalid_artifact", "The evaluation output has an unsupported schema.", 500)
        expected = {
            "model_id": record["model_id"],
            "revision": record["revision"],
            "layer": record["layer"],
            "hook": record["hook"],
            "split": record["evaluation"]["split"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise SOMServiceError("not_compatible", "The evaluation output does not match the fitted SOM identity.", 409)
        question_count = result.get("question_count")
        if isinstance(question_count, bool) or not isinstance(question_count, int) or not 1 <= question_count <= record["evaluation"]["max_questions"]:
            raise SOMServiceError("invalid_artifact", "The evaluation output has an invalid question count.", 500)
        methods = result.get("methods")
        expected_methods = {"baseline", "som_remove", "global_mean_remove", "pca_remove", "random_remove"}
        if not isinstance(methods, list) or set(methods) != expected_methods:
            raise SOMServiceError("invalid_artifact", "The evaluation output is missing required comparison methods.", 500)
        summary = result.get("score_summary")
        if not isinstance(summary, dict) or not isinstance(summary.get("conditions"), dict):
            raise SOMServiceError("invalid_artifact", "The evaluation output is missing its score summary.", 500)
        for method in expected_methods:
            conditions = summary["conditions"].get(method)
            if not isinstance(conditions, dict) or not {"neutral", "false_pressure", "true_claim", "valid_correction"}.issubset(conditions):
                raise SOMServiceError("invalid_artifact", "The evaluation summary is missing condition guardrails.", 500)
        score_records = result.get("scored_records")
        condition_names = {"neutral", "false_pressure", "true_claim", "valid_correction"}
        expected_record_count = question_count * len(condition_names) * len(expected_methods)
        if not isinstance(score_records, list) or len(score_records) != expected_record_count:
            raise SOMServiceError("invalid_artifact", "The evaluation output does not contain a complete question-by-condition comparison grid.", 500)
        for item in score_records:
            if (
                not isinstance(item, dict)
                or item.get("split") != record["evaluation"]["split"]
                or item.get("condition") not in condition_names
                or item.get("method") not in expected_methods
                or not isinstance(item.get("question_id"), str)
                or not item["question_id"]
                or any(
                    isinstance(item.get(key), bool)
                    or not isinstance(item.get(key), (int, float))
                    or not math.isfinite(float(item[key]))
                    for key in ("target_logprob", "foil_logprob", "margin")
                )
            ):
                raise SOMServiceError("invalid_artifact", "The evaluation output contains malformed score records.", 500)
        question_ids = {item["question_id"] for item in score_records}
        observed_keys = {
            (item["question_id"], item["condition"], item["method"])
            for item in score_records
        }
        expected_keys = {
            (question_id, condition, method)
            for question_id in question_ids
            for condition in condition_names
            for method in expected_methods
        }
        if len(question_ids) != question_count or observed_keys != expected_keys:
            raise SOMServiceError("invalid_artifact", "The evaluation output has duplicate or missing score records.", 500)
        score_summary = result["score_summary"]
        expected_conditions = {"neutral", "false_pressure", "true_claim", "valid_correction"}
        for key in ("false_pressure_effect", "truthful_correction_guardrails"):
            if not isinstance(score_summary.get(key), dict):
                raise SOMServiceError("invalid_artifact", "The evaluation output is missing a required score summary section.", 500)
        if set(score_summary["false_pressure_effect"]) != expected_methods:
            raise SOMServiceError("invalid_artifact", "The false-pressure summary is incomplete.", 500)
        expected_guardrails = expected_methods - {"baseline"}
        if set(score_summary["truthful_correction_guardrails"]) != expected_guardrails:
            raise SOMServiceError("invalid_artifact", "The truthful-correction guardrails are incomplete.", 500)
        basis_ranks = result.get("basis_ranks")
        expected_basis_names = {"som", "global_mean", "pca", "random"}
        if (
            not isinstance(basis_ranks, dict)
            or set(basis_ranks) != expected_basis_names
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in basis_ranks.values())
        ):
            raise SOMServiceError("invalid_artifact", "The evaluation output has invalid basis ranks.", 500)
        if result.get("score_mode") not in {"sequence_logprob", "mean_token_logprob"}:
            raise SOMServiceError("invalid_artifact", "The evaluation output has an unsupported score mode.", 500)
        for method, method_conditions in score_summary["conditions"].items():
            if not isinstance(method_conditions, dict) or not expected_conditions.issubset(method_conditions):
                raise SOMServiceError("invalid_artifact", "The condition score summary is incomplete.", 500)
        return result

    @staticmethod
    def _safe_numeric_summary(value: Any, *, depth: int = 0) -> Any:
        if depth > 8:
            raise SOMServiceError("invalid_artifact", "The evaluation score summary is too deeply nested.", 500)
        if isinstance(value, dict):
            if len(value) > 64:
                raise SOMServiceError("invalid_artifact", "The evaluation score summary is too large.", 500)
            output = {}
            for key, child in value.items():
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
                    raise SOMServiceError("invalid_artifact", "The evaluation score summary has an invalid key.", 500)
                output[key] = SOMService._safe_numeric_summary(child, depth=depth + 1)
            return output
        if isinstance(value, list):
            if len(value) > 64:
                raise SOMServiceError("invalid_artifact", "The evaluation score summary is too large.", 500)
            return [SOMService._safe_numeric_summary(child, depth=depth + 1) for child in value]
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise SOMServiceError("invalid_artifact", "The evaluation score summary contains unsupported values.", 500)

    @classmethod
    def _safe_evaluation_summary(cls, result: dict[str, Any]) -> dict[str, Any]:
        summary = result["score_summary"]
        return {
            "question_count": result["question_count"],
            "methods": list(result["methods"]),
            "basis_ranks": dict(result["basis_ranks"]),
            "score_mode": result["score_mode"],
            "score_summary": {
                key: cls._safe_numeric_summary(summary[key])
                for key in ("conditions", "false_pressure_effect", "truthful_correction_guardrails")
            },
        }

    @staticmethod
    def _hash_run_outputs(job_dir: Path) -> dict[str, str]:
        names = ("activations.npz", "som.npz", "evaluation.json")
        hashes = {}
        try:
            for name in names:
                path = job_dir / name
                if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > MAX_RUN_OUTPUT_BYTES:
                    raise SOMServiceError("integrity_failed", "A required SOM output is missing, empty, or too large.", 500)
                hashes[name] = _sha256_file(path)
        except OSError:
            raise SOMServiceError("integrity_failed", "A required SOM output could not be read.", 500) from None
        return hashes

    def _validate_completed_outputs(self, job_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
        expected_artifacts = ["activations.npz", "som.npz", "evaluation.json"]
        if record.get("artifacts") != expected_artifacts:
            raise SOMServiceError("integrity_failed", "The completed job does not identify the required outputs.", 500)
        activations_path = job_dir / "activations.npz"
        artifact_path = job_dir / "som.npz"
        evaluation_path = job_dir / "evaluation.json"
        try:
            data = load_activations(activations_path)
            self._validate_activation_identity(record, data)
            if record.get("prompt_dataset_sha256") and data.metadata.get("dataset_sha256") != record["prompt_dataset_sha256"]:
                raise SOMServiceError("integrity_failed", "The activation dataset identity does not match the job record.", 500)
            artifact = load_artifact(artifact_path)
            self._validate_artifact_identity(record, data, artifact, activations_path)
            return self._validate_evaluation(record, evaluation_path)
        except SOMServiceError:
            raise
        except Exception:  # noqa: BLE001 - malformed persisted files must produce a bounded validation error
            raise SOMServiceError("integrity_failed", "Completed SOM outputs failed structural validation.", 500) from None

    def _run_command(
        self,
        job_id: str,
        stage: str,
        argv: list[str],
        job_dir: Path,
        record: dict[str, Any],
        cancel_event: threading.Event,
        started: float,
    ) -> None:
        self._check_cancel(cancel_event)
        if time.monotonic() - started > MAX_JOB_SECONDS:
            raise SOMServiceError("timeout", "The local SOM pilot exceeded its time limit.", 408)
        log_path = job_dir / f"{stage}.log"
        try:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=self.repo_root,
                    env=self._offline_environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    close_fds=True,
                    start_new_session=(os.name != "nt"),
                )
                with self._lock:
                    self._processes[job_id] = process
                try:
                    while process.poll() is None:
                        self._check_cancel(cancel_event)
                        elapsed = time.monotonic() - started
                        if elapsed > MAX_JOB_SECONDS:
                            self._stop_process(process)
                            raise SOMServiceError("timeout", "The local SOM pilot exceeded its time limit.", 408)
                        if stage == "evaluate":
                            current, total = self._evaluation_progress(log_path)
                            if total:
                                self._update_stage(job_id, job_dir, record, stage, 3, current, total)
                        time.sleep(0.2)
                    if process.returncode != 0:
                        raise SOMServiceError("stage_failed", f"{stage.capitalize()} failed with exit code {process.returncode}. See the local run log.", 500)
                finally:
                    with self._lock:
                        self._processes.pop(job_id, None)
        except SOMServiceError:
            raise
        except OSError:
            raise SOMServiceError("unavailable", f"The local {stage} command could not be started.", 503) from None

    @staticmethod
    def _evaluation_progress(log_path: Path) -> tuple[int, int]:
        try:
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 16_384))
                tail = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return 0, 0
        matches = list(_PROGRESS_RE.finditer(tail))
        if not matches:
            return 0, 0
        return int(matches[-1].group(1)), int(matches[-1].group(2))

    @staticmethod
    def _offline_environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["HF_DATASETS_OFFLINE"] = "1"
        return environment

    @staticmethod
    def _stop_process(process: subprocess.Popen[Any]) -> None:
        try:
            if os.name != "nt" and process.pid:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                if os.name != "nt" and process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            process.wait()

    @staticmethod
    def _check_cancel(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise _JobCancelled

    def _public_job(self, record: dict[str, Any], job_dir: Path) -> dict[str, Any]:
        allowed = (
            "job_id", "kind", "status", "stage", "progress", "created_at", "updated_at", "completed_at",
            "source_id", "source_label", "dataset_kind", "model_id", "revision_requested", "revision",
            "layer", "hook", "device", "fit", "evaluation", "artifacts", "error", "error_code",
            "scientific_interpretation",
        )
        result = {key: record[key] for key in allowed if key in record}
        result["artifact_kinds"] = [Path(name).stem for name in result.get("artifacts", []) if isinstance(name, str)]
        result.pop("artifacts", None)
        if record.get("status") == "completed" and record.get("kind") == "pilot":
            try:
                before = self._hash_run_outputs(job_dir)
                evaluation_result = self._validate_completed_outputs(job_dir, record)
                after = self._hash_run_outputs(job_dir)
                if before != after:
                    raise SOMServiceError("integrity_failed", "SOM outputs changed during status validation.", 500)
                stored_hashes = record.get("artifact_sha256")
                if stored_hashes is None:
                    integrity_status = "unverified"
                    integrity_note = "Outputs passed structural validation, but creation-time SHA-256 checksums were not recorded."
                elif (
                    not isinstance(stored_hashes, dict)
                    or set(stored_hashes) != set(after)
                    or stored_hashes != after
                ):
                    raise SOMServiceError("integrity_failed", "A SOM output differs from its recorded SHA-256 checksum.", 500)
                else:
                    integrity_status = "verified"
                    integrity_note = "Output checksums and validated structure match the completed job record."
                result["integrity_status"] = integrity_status
                result["result_validity"] = "valid" if integrity_status == "verified" else "structure_valid"
                result["integrity_note"] = integrity_note
                result["evaluation_summary"] = self._safe_evaluation_summary(evaluation_result)
            except Exception:  # noqa: BLE001 - persisted job data is untrusted at the API boundary
                result["status"] = "invalid"
                result["integrity_status"] = "invalid"
                result["result_validity"] = "invalid"
                result["error"] = "Completed SOM outputs failed integrity validation."
                result["error_code"] = "integrity_failed"
                result.pop("evaluation_summary", None)
        return result

    def _external_job_records(self) -> list[dict[str, Any]]:
        payload = self._read_manifest()
        if payload is None:
            return []
        raw_jobs = payload.get("external_jobs")
        if not isinstance(raw_jobs, list):
            return []
        records: list[dict[str, Any]] = []
        for item in raw_jobs[:64]:
            if not isinstance(item, dict):
                continue
            job_id = item.get("job_id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
                continue
            label = item.get("label")
            if (
                not isinstance(label, str)
                or not label.strip()
                or len(label) > 120
                or "\\" in label
                or label.startswith("/")
                or re.match(r"^[A-Za-z]:", label)
            ):
                label = f"Recorded external SOM job {job_id}"
            status = item.get("recorded_status")
            if not isinstance(status, str) or len(status) > 64:
                status = "unknown"
            as_of = item.get("recorded_as_of") or item.get("recorded_at")
            if not isinstance(as_of, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?Z)?", as_of):
                as_of = None
            source_ref = item.get("source_ref")
            if not isinstance(source_ref, str) or len(source_ref) > 160 or source_ref.startswith("/") or "\\" in source_ref or ".." in source_ref:
                source_ref = None
            source_url = item.get("source_url")
            if isinstance(source_url, str):
                parsed_url = urlsplit(source_url)
                valid_url = (
                    parsed_url.scheme == "https"
                    and bool(parsed_url.hostname)
                    and parsed_url.username is None
                    and parsed_url.password is None
                    and not parsed_url.query
                    and not parsed_url.fragment
                    and len(source_url) <= 600
                )
            else:
                valid_url = False
            if not valid_url:
                source_url = None
            record = {
                "job_id": job_id,
                "label": label.strip(),
                "recorded_status": status,
                "recorded_as_of": as_of,
                "source_ref": source_ref,
                "source_url": source_url,
                "live_status": "unverified",
            }
            model_id = item.get("model_id")
            if isinstance(model_id, str) and model_id and len(model_id) <= 200:
                record["model_id"] = model_id
            revision = item.get("revision")
            if isinstance(revision, str) and revision and len(revision) <= 200:
                record["revision"] = revision
            layer = item.get("layer")
            if isinstance(layer, int) and not isinstance(layer, bool) and 0 <= layer < 128:
                record["layer"] = layer
            hook = item.get("hook")
            if isinstance(hook, str) and hook and len(hook) <= 256:
                record["hook"] = hook
            note = item.get("live_status_note")
            if isinstance(note, str) and len(note) <= 300:
                record["live_status_note"] = note
            artifact_status = item.get("artifact_status")
            if isinstance(artifact_status, str) and artifact_status in {
                "not_available", "not_recorded", "validation_failed", "unknown"
            }:
                record["artifact_status"] = artifact_status
            for field, limit in (
                ("experiment_schema", 120),
                ("model_label", 120),
                ("projection", 200),
                ("compatibility_note", 300),
            ):
                value = item.get(field)
                if isinstance(value, str) and value and len(value) <= limit:
                    record[field] = value
            layers = item.get("layers")
            if (
                isinstance(layers, list)
                and len(layers) <= 128
                and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 128 for value in layers)
            ):
                record["layers"] = layers
            for field in ("readouts", "views"):
                values = item.get(field)
                if isinstance(values, list) and len(values) <= 32 and all(
                    isinstance(value, str) and 0 < len(value) <= 80 for value in values
                ):
                    record[field] = values
            record["pipeline_compatible"] = False
            records.append(record)
        return records

    @staticmethod
    def _public_external_job(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": record["job_id"],
            "kind": "external_record",
            "label": record["label"],
            "status": "unverified",
            "recorded_status": record["recorded_status"],
            "recorded_as_of": record["recorded_as_of"],
            "live_status": "unverified",
            "live_status_note": record.get("live_status_note", "No live scheduler result is available."),
            "source_ref": record.get("source_ref"),
            "source_url": record.get("source_url"),
            "model_id": record.get("model_id"),
            "revision": record.get("revision"),
            "layer": record.get("layer"),
            "hook": record.get("hook"),
            "artifact_status": record.get("artifact_status", "not_available"),
            "experiment_schema": record.get("experiment_schema"),
            "model_label": record.get("model_label"),
            "layers": record.get("layers"),
            "projection": record.get("projection"),
            "readouts": record.get("readouts"),
            "views": record.get("views"),
            "pipeline_compatible": False,
            "compatibility_note": record.get(
                "compatibility_note",
                "This external teacher-forced artifact format has not been validated for the explorer SOM pipeline.",
            ),
        }
