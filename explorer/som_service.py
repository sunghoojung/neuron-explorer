"""Local, bounded SOM job service for the neuron explorer.

The service only reads prompt, activation, and SOM artifact files from the
bundled example directory or explicitly configured local source roots. Jobs
write to unique run directories below ``EXPLORER_SOM_RUNS_DIR``. It never
accepts a shell command or an arbitrary source path from a caller.
"""

from __future__ import annotations

import base64
import copy
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
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
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
_NATIVE_ID_RE = re.compile(r"^native_[a-f0-9]{24}$")
_NATIVE_MAP_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_PROGRESS_RE = re.compile(r"scoring\s+(\d+)/(\d+)\s+held-out questions", re.IGNORECASE)
_NATIVE_HOOK_TEMPLATE = "model.layers.{layer}.self_attn.o_proj"
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
MAX_NATIVE_ROOTS = 8
MAX_NATIVE_MAPS = 128
MAX_NATIVE_DIRECTION_VECTORS = 1024
MAX_NATIVE_JSON_BYTES = 16 * 1024 * 1024
MAX_NATIVE_NPZ_BYTES = 64 * 1024 * 1024
MAX_NATIVE_SOURCE_FILE_BYTES = 32 * 1024 * 1024
MAX_NATIVE_SNAPSHOT_FILES = 4096
MAX_NATIVE_SNAPSHOT_BYTES = 512 * 1024 * 1024
MAX_NATIVE_MAP_PNG_BYTES = 16 * 1024 * 1024
MAX_NATIVE_REPORT_BYTES = 3_355_443
MAX_NATIVE_STAGE_FILES = 8192
MAX_NATIVE_CAPTURE_BYTES = 4 * 1024 * 1024 * 1024
MAX_NATIVE_TREE_ENTRIES = 20000
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


class _NativeIncomplete(Exception):
    pass


class _NativeHtmlInspector(HTMLParser):
    """Collect HTML constructs that can load active or remote content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.unsafe = False
        self.references: list[tuple[str, str, str]] = []
        self._style_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._inspect(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "style" and self._style_depth:
            self._style_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._style_depth and re.search(r"@import|url\s*\(", data, flags=re.IGNORECASE):
            self.unsafe = True

    def _inspect(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower()
        if normalized_tag in {"script", "iframe", "object", "embed", "form", "base", "link"}:
            self.unsafe = True
        if normalized_tag == "style":
            self._style_depth += 1
        for key, raw_value in attrs:
            name = (key or "").lower()
            value = raw_value or ""
            if name.startswith("on"):
                self.unsafe = True
            if name in {"href", "src", "action", "poster", "xlink:href", "srcset"}:
                self.references.append((normalized_tag, name, value))
            if name == "style" and re.search(r"@import|url\s*\(", value, flags=re.IGNORECASE):
                self.unsafe = True
            if normalized_tag == "meta" and name == "http-equiv" and value.lower() == "refresh":
                self.unsafe = True


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
        native_som_roots: Sequence[str | os.PathLike[str] | dict[str, str]] | None = None,
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
        self._native_roots_warning: str | None = None
        if native_som_roots is None:
            raw_native_roots = os.environ.get("EXPLORER_SOM_NATIVE_ROOTS", "")
            if raw_native_roots.strip():
                native_som_roots = [value for value in raw_native_roots.split(os.pathsep) if value.strip()]
            else:
                configured_native_roots = manifest.get("native_som_roots", []) if manifest else []
                if not isinstance(configured_native_roots, list) or len(configured_native_roots) > MAX_NATIVE_ROOTS:
                    self._native_roots_warning = "Local native SOM roots are invalid; no native runs were loaded."
                    configured_native_roots = []
                native_som_roots = configured_native_roots
        try:
            self.native_som_roots = self._normalize_native_roots(native_som_roots)
        except (OSError, TypeError, ValueError):
            self._native_roots_warning = "Local native SOM roots are invalid; no native runs were loaded."
            self.native_som_roots = ()
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
        self._native_run_cache: dict[str, tuple[str, dict[str, Any]]] = {}
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

    @staticmethod
    def _normalize_native_roots(
        roots: Sequence[str | os.PathLike[str] | dict[str, str]] | None,
    ) -> tuple[dict[str, Path | None], ...]:
        if roots is None:
            return ()
        if isinstance(roots, (str, bytes)) or len(roots) > MAX_NATIVE_ROOTS:
            raise ValueError("invalid native roots")
        normalized: list[dict[str, Path | None]] = []
        for item in roots:
            if isinstance(item, dict):
                if set(item) - {"output_root", "snapshot_root"}:
                    raise ValueError("invalid native root fields")
                output_value = item.get("output_root")
                snapshot_value = item.get("snapshot_root")
                if not isinstance(output_value, str) or not output_value.strip():
                    raise ValueError("native output root is required")
                if snapshot_value is not None and (not isinstance(snapshot_value, str) or not snapshot_value.strip()):
                    raise ValueError("invalid native snapshot root")
                output_root = Path(output_value).expanduser().absolute()
                snapshot_root = Path(snapshot_value).expanduser().absolute() if snapshot_value is not None else None
            else:
                if not isinstance(item, (str, os.PathLike)):
                    raise TypeError("invalid native root")
                value = os.fsdecode(os.fspath(item))
                if not value.strip() or len(value) > 2048 or "\x00" in value:
                    raise ValueError("invalid native root")
                output_root = Path(value).expanduser().absolute()
                snapshot_root = None
            if len(os.fspath(output_root)) > 2048 or (
                snapshot_root is not None and len(os.fspath(snapshot_root)) > 2048
            ):
                raise ValueError("native root is too long")
            normalized.append({"output_root": output_root, "snapshot_root": snapshot_root})
        return tuple(normalized)

    @staticmethod
    def _native_id(configured: dict[str, Path | None]) -> str:
        output_root = configured["output_root"]
        snapshot_root = configured["snapshot_root"]
        assert isinstance(output_root, Path)
        identity = f"{output_root}\0{snapshot_root or ''}"
        return "native_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    def _native_root_for_id(self, native_id: str) -> dict[str, Path | None]:
        if not isinstance(native_id, str) or not _NATIVE_ID_RE.fullmatch(native_id):
            raise SOMServiceError("unknown_id", "No matching validated native SOM report was found.", 404)
        for configured in self.native_som_roots:
            if self._native_id(configured) == native_id:
                return configured
        raise SOMServiceError("unknown_id", "No matching validated native SOM report was found.", 404)

    def _scan_native_reports(self) -> list[dict[str, Any]]:
        return [self._native_catalog_summary(self._native_report_for_root(item)) for item in self.native_som_roots]

    def _native_report_for_root(self, configured: dict[str, Path | None]) -> dict[str, Any]:
        native_id = self._native_id(configured)
        output_root = configured["output_root"]
        assert isinstance(output_root, Path)
        if not output_root.is_dir():
            return {
                "native_id": native_id,
                "status": "unavailable",
                "integrity_status": "unverified",
                "capture_provenance_status": "unavailable",
                "report_available": False,
                "error_code": "native_run_unavailable",
            }
        try:
            fingerprint = self._native_tree_fingerprint(configured)
            cached = self._native_run_cache.get(native_id)
            if cached and cached[0] == fingerprint:
                return copy.deepcopy(cached[1])
            report = self._validate_native_run(native_id, configured)
            self._native_run_cache[native_id] = (fingerprint, copy.deepcopy(report))
            return report
        except _NativeIncomplete:
            return {
                "native_id": native_id,
                "status": "incomplete",
                "integrity_status": "incomplete",
                "capture_provenance_status": "pending",
                "report_available": False,
                "error_code": "native_outputs_incomplete",
            }
        except Exception:  # noqa: BLE001 - local artifacts are untrusted input at the API boundary
            return {
                "native_id": native_id,
                "status": "invalid",
                "integrity_status": "invalid",
                "capture_provenance_status": "unverified",
                "report_available": False,
                "error_code": "native_integrity_failed",
            }

    @staticmethod
    def _native_catalog_summary(report: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "native_id", "status", "integrity_status", "capture_provenance_status", "model_id",
            "revision", "layers", "hook_template", "map_count", "behavior_status", "report_available",
            "report_url", "error_code",
        )
        return {key: report[key] for key in keys if key in report}

    @staticmethod
    def _native_file(root: Path, relative_name: str, *, required: bool) -> Path | None:
        if not isinstance(relative_name, str) or "\\" in relative_name:
            raise ValueError("invalid native relative path")
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("invalid native relative path")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("native root is not a plain directory")
        resolved_root = root.resolve(strict=True)
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("native artifact path contains a symlink")
        if not candidate.exists():
            if required:
                raise _NativeIncomplete
            return None
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not candidate.is_file():
            raise ValueError("native artifact is outside its configured root")
        return resolved

    @staticmethod
    def _native_directory(root: Path, relative_name: str) -> Path:
        if "\\" in relative_name:
            raise ValueError("invalid native relative directory")
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("invalid native relative directory")
        if root.is_symlink() or not root.is_dir():
            raise ValueError("native root is not a plain directory")
        resolved_root = root.resolve(strict=True)
        candidate = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("native directory path contains a symlink")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not candidate.is_dir():
            raise ValueError("native directory is outside its configured root")
        return resolved

    @classmethod
    def _native_tree_fingerprint(cls, configured: dict[str, Path | None]) -> str:
        digest = hashlib.sha256()
        entries_seen = 0
        for name in ("output_root", "snapshot_root"):
            root = configured.get(name)
            if root is None:
                digest.update(f"{name}:none\n".encode())
                continue
            if not root.is_dir() or root.is_symlink():
                digest.update(f"{name}:missing\n".encode())
                continue
            stack = [root]
            while stack:
                directory = stack.pop()
                try:
                    entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
                except OSError:
                    raise _NativeIncomplete from None
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > MAX_NATIVE_TREE_ENTRIES:
                        raise ValueError("native run has too many files")
                    candidate = Path(entry.path)
                    if entry.is_symlink():
                        raise ValueError("native run contains a symlink")
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        raise _NativeIncomplete from None
                    relative = candidate.relative_to(root).as_posix()
                    identity = (name, relative, stat.st_mode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
                    digest.update(repr(identity).encode("utf-8"))
                    digest.update(b"\n")
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(candidate)
        return digest.hexdigest()

    @classmethod
    def _native_json(cls, root: Path, relative_name: str) -> dict[str, Any] | list[Any]:
        path = cls._native_file(root, relative_name, required=True)
        assert path is not None
        if path.stat().st_size > MAX_NATIVE_JSON_BYTES:
            raise ValueError("native metadata is too large")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("native metadata is invalid") from None
        if not isinstance(payload, (dict, list)):
            raise TypeError("native metadata has an invalid structure")
        return payload

    @staticmethod
    def _native_file_matches(path: Path, expected_sha256: Any, max_bytes: int) -> bool:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            return False
        try:
            stat_before = path.stat()
            if not path.is_file() or path.is_symlink() or stat_before.st_size > max_bytes:
                return False
            actual = _sha256_file(path)
            stat_after = path.stat()
        except OSError:
            return False
        return (
            stat_before.st_size == stat_after.st_size
            and stat_before.st_mtime_ns == stat_after.st_mtime_ns
            and actual == expected_sha256
        )

    @classmethod
    def _native_verify_file(
        cls,
        root: Path,
        relative_name: str,
        expected_sha256: Any,
        max_bytes: int = MAX_NATIVE_JSON_BYTES,
    ) -> Path:
        path = cls._native_file(root, relative_name, required=True)
        assert path is not None
        if not cls._native_file_matches(path, expected_sha256, max_bytes):
            raise ValueError("native artifact digest mismatch")
        return path

    @classmethod
    def _native_snapshot_inventory(cls, root: Path, files: Any) -> dict[str, str]:
        """Verify every bounded file entry claimed by a frozen snapshot."""
        if not isinstance(files, dict) or not files or len(files) > MAX_NATIVE_SNAPSHOT_FILES:
            raise ValueError("native snapshot file index is invalid")
        normalized: dict[str, str] = {}
        total_bytes = 0
        for relative_name, digest in files.items():
            if (
                not isinstance(relative_name, str)
                or not relative_name
                or "\\" in relative_name
                or len(relative_name) > 1024
                or PurePosixPath(relative_name).is_absolute()
                or PurePosixPath(relative_name).as_posix() != relative_name
                or any(part in {"", ".", ".."} for part in PurePosixPath(relative_name).parts)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                raise ValueError("native snapshot file entry is invalid")
            path = cls._native_file(root, relative_name, required=True)
            assert path is not None
            size = path.stat().st_size
            if size > MAX_NATIVE_SOURCE_FILE_BYTES:
                raise ValueError("native snapshot file exceeds its size bound")
            total_bytes += size
            if total_bytes > MAX_NATIVE_SNAPSHOT_BYTES:
                raise ValueError("native snapshot exceeds its total size bound")
            if not cls._native_file_matches(path, digest, MAX_NATIVE_SOURCE_FILE_BYTES):
                raise ValueError("native snapshot file digest mismatch")
            normalized[relative_name] = digest
        return normalized

    @staticmethod
    def _native_binding_matches(actual: Any, expected: dict[str, Any]) -> bool:
        base_fields = ("bundle_sha256", "snapshot_sha256", "token_audit_sha256")
        return isinstance(actual, dict) and all(actual.get(key) == expected.get(key) for key in base_fields)

    @staticmethod
    def _native_int(value: Any, *, minimum: int = 0, maximum: int = 2**31 - 1) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError("invalid native integer")
        return value

    @staticmethod
    def _native_named_hashes(value: Any, suffix: str, maximum_entries: int) -> bool:
        if not isinstance(value, dict) or not value or len(value) > maximum_entries:
            return False
        for name, digest in value.items():
            if (
                not isinstance(name, str)
                or len(name) > 160
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or not name.endswith(suffix)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[a-f0-9]{64}", digest)
            ):
                return False
        return True

    @staticmethod
    def _native_float(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("invalid native metric")
        result = float(value)
        if minimum is not None and result < minimum or maximum is not None and result > maximum:
            raise ValueError("native metric outside bounds")
        return result

    @classmethod
    def _native_geometry_metrics(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError("invalid native geometry metrics")
        vectors = cls._native_int(raw.get("vectors"), maximum=MAX_NATIVE_DIRECTION_VECTORS)
        rank90 = cls._native_int(raw.get("rank90"), maximum=MAX_NATIVE_DIRECTION_VECTORS)
        participation_rank = cls._native_float(raw.get("participation_rank"), minimum=0.0)
        energy = raw.get("energy_fractions")
        if not isinstance(energy, list) or len(energy) > MAX_NATIVE_DIRECTION_VECTORS:
            raise ValueError("invalid native energy fractions")
        energy_fractions = [cls._native_float(value, minimum=0.0, maximum=1.0) for value in energy]
        if energy_fractions and (len(energy_fractions) != vectors or not math.isclose(sum(energy_fractions), 1.0, rel_tol=1e-5, abs_tol=1e-5)):
            raise ValueError("inconsistent native energy fractions")
        if rank90 > vectors:
            raise ValueError("invalid native geometry rank")
        return {
            "vectors": vectors,
            "rank90": rank90,
            "participation_rank": participation_rank,
            "energy_fractions": energy_fractions,
        }

    @staticmethod
    def _native_html_is_safe(path: Path, map_names: set[str]) -> bool:
        try:
            if path.stat().st_size > MAX_NATIVE_REPORT_BYTES:
                return False
            html = path.read_text(encoding="utf-8")
            parser = _NativeHtmlInspector()
            parser.feed(html)
            parser.close()
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        if parser.unsafe:
            return False
        for tag, name, value in parser.references:
            if name == "srcset":
                return False
            values = [value]
            for url in values:
                if url.startswith("#"):
                    if tag == "a" and name == "href":
                        continue
                    return False
                if url.lower().startswith("data:"):
                    if tag != "img" or name != "src":
                        return False
                    image_match = re.fullmatch(
                        r"data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/]+={0,2})",
                        url,
                        flags=re.IGNORECASE,
                    )
                    if image_match is None or len(image_match.group(2)) > (MAX_NATIVE_MAP_PNG_BYTES * 4 // 3 + 8):
                        return False
                    try:
                        image_bytes = base64.b64decode(image_match.group(2), validate=True)
                    except (ValueError, base64.binascii.Error):
                        return False
                    mime = image_match.group(1).lower()
                    valid_magic = {
                        "png": image_bytes.startswith(b"\x89PNG\r\n\x1a\n"),
                        "jpeg": image_bytes.startswith(b"\xff\xd8\xff"),
                        "gif": image_bytes.startswith((b"GIF87a", b"GIF89a")),
                        "webp": len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP",
                    }[mime]
                    if not valid_magic:
                        return False
                    continue
                parts = urlsplit(url)
                if parts.scheme or parts.netloc or parts.query or parts.fragment:
                    return False
                match = re.fullmatch(r"maps/([A-Za-z0-9][A-Za-z0-9_.-]{0,119})/map\.png", parts.path)
                if match and match.group(1) in map_names and tag == "img" and name == "src":
                    continue
                if tag == "a" and name == "href" and parts.path in {
                    "behavior/blinded-review.json",
                    "behavior/strict-screen.json",
                }:
                    continue
                return False
        return True

    @classmethod
    def _native_map_summary(
        cls,
        root: Path,
        map_name: str,
        expected_binding: dict[str, Any],
        configured_views: set[str],
    ) -> dict[str, Any]:
        if not _NATIVE_MAP_NAME_RE.fullmatch(map_name):
            raise ValueError("invalid native map name")
        directory = f"maps/{map_name}"
        receipt = cls._native_json(root, f"{directory}/complete.json")
        summary = cls._native_json(root, f"{directory}/summary.json")
        if not isinstance(receipt, dict) or not isinstance(summary, dict):
            raise TypeError("invalid native map receipt")
        if not cls._native_binding_matches(receipt.get("binding"), expected_binding):
            raise ValueError("native map binding mismatch")
        if not cls._native_binding_matches(summary.get("binding"), expected_binding):
            raise ValueError("native map summary binding mismatch")
        task = receipt.get("task")
        if not isinstance(task, dict) or summary.get("task") != task or task.get("name") != map_name:
            raise ValueError("native map task mismatch")
        kind = task.get("kind")
        view = task.get("view")
        layer = cls._native_int(task.get("layer"), maximum=127)
        side = cls._native_int(task.get("side"), minimum=1, maximum=32)
        seed = cls._native_int(task.get("seed"), maximum=2**32 - 1)
        if kind not in {"som", "kmeans"} or not isinstance(view, str) or view not in configured_views:
            raise ValueError("unsupported native map task")
        if summary.get("causal_steering_tested") is not False:
            raise ValueError("native map has unsupported causal claim")
        direction_sign = summary.get("direction_sign")
        if direction_sign not in {"good minus bad", "bad minus good"}:
            raise ValueError("unsupported native map direction sign")

        files = receipt.get("files")
        expected_file_names = {"map.npz", "map.png", "summary.json"}
        if isinstance(files, dict) and "worker.lock" in files:
            expected_file_names.add("worker.lock")
        if not isinstance(files, dict) or set(files) != expected_file_names:
            raise ValueError("native map file receipt is incomplete")
        verified_paths: dict[str, Path] = {}
        for name, digest in files.items():
            if name not in expected_file_names:
                raise ValueError("unexpected native map file")
            limit = {
                "map.npz": MAX_NATIVE_NPZ_BYTES,
                "map.png": MAX_NATIVE_MAP_PNG_BYTES,
                "summary.json": MAX_NATIVE_JSON_BYTES,
                "worker.lock": 1024,
            }[name]
            verified_paths[name] = cls._native_verify_file(root, f"{directory}/{name}", digest, limit)
        png_path = verified_paths["map.png"]
        with png_path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("native map image is not PNG")

        import numpy as np

        npz_path = verified_paths["map.npz"]
        with zipfile.ZipFile(npz_path) as archive:
            members = archive.infolist()
            if len(members) > 128 or sum(member.file_size for member in members) > 256 * 1024 * 1024:
                raise ValueError("native map archive exceeds bounds")
            if any(PurePosixPath(member.filename).is_absolute() or ".." in PurePosixPath(member.filename).parts for member in members):
                raise ValueError("native map archive contains an invalid member")
        with np.load(npz_path, allow_pickle=False) as arrays:
            required_arrays = {"prototypes", "cells", "fit", "errors"}
            if not required_arrays.issubset(arrays.files):
                raise ValueError("native map arrays are incomplete")
            prototypes = arrays["prototypes"]
            cells = arrays["cells"]
            fit = arrays["fit"]
            errors = arrays["errors"]
            if (
                prototypes.ndim != 2
                or prototypes.shape[0] != side * side
                or prototypes.shape[1] <= 0
                or prototypes.shape[1] > 65536
                or not np.issubdtype(prototypes.dtype, np.floating)
                or not np.isfinite(prototypes).all()
                or cells.ndim != 1
                or not np.issubdtype(cells.dtype, np.integer)
                or fit.shape != cells.shape
                or fit.dtype != np.bool_
                or errors.shape != cells.shape
                or not np.issubdtype(errors.dtype, np.floating)
                or not np.isfinite(errors).all()
                or (errors < 0).any()
                or (cells < 0).any()
                or (cells >= side * side).any()
            ):
                raise ValueError("native map arrays failed structural validation")
            states = cls._native_int(summary.get("states"), minimum=1, maximum=MAX_NATIVE_STAGE_FILES * 100)
            fit_states = cls._native_int(summary.get("fit_states"), maximum=states)
            if states != len(cells) or fit_states != int(fit.sum()):
                raise ValueError("native map counts do not match arrays")
            hidden_size = int(prototypes.shape[1])
            occupied_cells = len(np.unique(cells))

        geometry = cls._native_geometry_metrics(summary.get("candidate_direction_geometry"))
        condition_geometry = cls._native_geometry_metrics(summary.get("condition_geometry"))
        associations_raw = summary.get("associations")
        if not isinstance(associations_raw, dict) or len(associations_raw) > 32:
            raise ValueError("invalid native map associations")
        associations = []
        for key, metrics in sorted(associations_raw.items()):
            match = re.fullmatch(r"([A-Za-z0-9_-]{1,64})/(MAP_FIT|MAP_PROJECT)", key) if isinstance(key, str) else None
            if not match or not isinstance(metrics, dict):
                raise ValueError("invalid native map association key")
            selected_metrics: dict[str, int | float] = {}
            for field in (
                "states", "occupied_cells", "quantization_error", "condition_nmi", "source_nmi",
                "quality_nmi", "answer_identity_nmi", "answer_length_nmi",
            ):
                if field not in metrics:
                    continue
                if field in {"states", "occupied_cells"}:
                    selected_metrics[field] = cls._native_int(metrics[field], maximum=MAX_NATIVE_STAGE_FILES * 100)
                else:
                    selected_metrics[field] = cls._native_float(
                        metrics[field], minimum=0.0,
                        maximum=None if field == "quantization_error" else 1.0,
                    )
            associations.append({
                "cohort": match.group(1) if match.group(1) in {"standardized", "reviewed_error_replay"} else "other",
                "split": match.group(2),
                "metrics": selected_metrics,
            })

        history = summary.get("fit_history")
        if not isinstance(history, list) or not history or len(history) > 10000:
            raise ValueError("invalid native fit history")
        history_values = [cls._native_float(value, minimum=0.0) for value in history]
        return {
            "map_name": map_name,
            "kind": kind,
            "view": view,
            "layer": layer,
            "side": side,
            "seed": seed,
            "map_shape": [side, side, hidden_size],
            "prototype_count": side * side,
            "states": states,
            "fit_states": fit_states,
            "occupied_cells": occupied_cells,
            "direction_sign": direction_sign,
            "direction_geometry": geometry,
            "condition_geometry": condition_geometry,
            "associations": associations,
            "fit_history": {
                "epochs": len(history_values),
                "initial": history_values[0],
                "final": history_values[-1],
                "best": min(history_values),
            },
            "png_sha256": files["map.png"],
        }

    @classmethod
    def _native_stage_receipts(
        cls,
        root: Path,
        expected_binding: dict[str, Any],
        world_size: int,
        expected_capture_identity: dict[str, Any],
        result_hashes: dict[str, str],
        expected_map_receipts: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], tuple[int, bool, int]]:
        stage_targets = {
            "capture": {},
            "analyze": {
                "../../behavior/banks/complete.json": ("behavior/banks/complete.json", result_hashes["banks"]),
                "../../geometry-complete.json": ("geometry-complete.json", result_hashes["geometry"]),
            },
            "generate": {
                "../../behavior/complete.json": ("behavior/complete.json", result_hashes["behavior"]),
                "../../complete.json": ("complete.json", result_hashes["complete"]),
                "../../report.html": ("report.html", result_hashes["report"]),
            },
        }
        summaries: dict[str, dict[str, Any]] = {}
        seen_map_payloads: set[str] = set()
        for stage, special_targets in stage_targets.items():
            stage_receipt = cls._native_json(root, f"stages/{stage}/complete.json")
            if not isinstance(stage_receipt, dict):
                raise TypeError("invalid native stage receipt")
            if (
                stage_receipt.get("phase") != stage
                or stage_receipt.get("mock") is not False
                or stage_receipt.get("world_size") != world_size
                or not cls._native_binding_matches(stage_receipt.get("binding"), expected_binding)
            ):
                raise ValueError("native stage binding mismatch")
            expected_names = {f"rank{rank}.json" for rank in range(world_size)} | set(special_targets)
            files = stage_receipt.get("files")
            if not isinstance(files, dict) or set(files) != expected_names:
                raise ValueError("native stage receipt is incomplete")
            for rank in range(world_size):
                rank_name = f"rank{rank}.json"
                rank_path = cls._native_verify_file(
                    root,
                    f"stages/{stage}/{rank_name}",
                    files[rank_name],
                    MAX_NATIVE_JSON_BYTES,
                )
                rank_record = cls._native_json(root, f"stages/{stage}/{rank_name}")
                if (
                    not isinstance(rank_record, dict)
                    or rank_record.get("phase") != stage
                    or rank_record.get("rank") != rank
                    or rank_record.get("world_size") != world_size
                    or rank_record.get("mock") is not False
                    or not cls._native_binding_matches(rank_record.get("binding"), expected_binding)
                    or not isinstance(rank_record.get("payloads"), dict)
                ):
                    raise ValueError("native stage rank receipt is invalid")
                if rank_path.stat().st_size > MAX_NATIVE_JSON_BYTES:
                    raise ValueError("native stage rank receipt is too large")
                payloads = rank_record["payloads"]
                if stage == "capture":
                    relative_name = f"capture/rank{rank}/complete.json"
                    receipt_path = cls._native_file(root, relative_name, required=True)
                    assert receipt_path is not None
                    expected_payloads = {relative_name: _sha256_file(receipt_path)}
                    if payloads != expected_payloads:
                        raise ValueError("native capture stage payload binding mismatch")
                elif stage == "analyze":
                    if not payloads or len(payloads) > MAX_NATIVE_MAPS:
                        raise ValueError("native analyze stage payload set is invalid")
                    for relative_name, digest in payloads.items():
                        if (
                            not isinstance(relative_name, str)
                            or relative_name not in expected_map_receipts
                            or expected_map_receipts[relative_name] != digest
                            or relative_name in seen_map_payloads
                        ):
                            raise ValueError("native analyze stage payload binding mismatch")
                        cls._native_verify_file(root, relative_name, digest, MAX_NATIVE_JSON_BYTES)
                        seen_map_payloads.add(relative_name)
                else:
                    relative_name = f"behavior/rank{rank}/complete.json"
                    receipt_path = cls._native_file(root, relative_name, required=True)
                    assert receipt_path is not None
                    receipt_digest = _sha256_file(receipt_path)
                    if payloads != {relative_name: receipt_digest}:
                        raise ValueError("native behavior stage payload binding mismatch")
                    behavior_rank = cls._native_json(root, relative_name)
                    expected_behavior_files = {"answers.jsonl", "qualification.json"}
                    behavior_files = behavior_rank.get("files") if isinstance(behavior_rank, dict) else None
                    if (
                        not isinstance(behavior_rank, dict)
                        or behavior_rank.get("rank") != rank
                        or behavior_rank.get("weights_unchanged") is not True
                        or not cls._native_binding_matches(behavior_rank.get("binding"), expected_binding)
                        or not isinstance(behavior_files, dict)
                        or set(behavior_files) != expected_behavior_files
                    ):
                        raise ValueError("native behavior rank receipt is invalid")
                    cls._native_int(behavior_rank.get("answers"), maximum=MAX_NATIVE_STAGE_FILES * 100)
                    for filename, digest in behavior_files.items():
                        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                            raise ValueError("native behavior rank file digest is invalid")
                        cls._native_verify_file(
                            root,
                            f"behavior/rank{rank}/{filename}",
                            digest,
                            MAX_NATIVE_JSON_BYTES,
                        )
            for receipt_name, (target, expected_hash) in special_targets.items():
                if files.get(receipt_name) != expected_hash:
                    raise ValueError("native stage output digest mismatch")
                # The stage receipt uses fixed relative names. Resolve only the
                # known targets above, never a path supplied by a receipt.
                cls._native_verify_file(root, target, expected_hash, MAX_NATIVE_REPORT_BYTES if target == "report.html" else MAX_NATIVE_JSON_BYTES)
            summaries[stage] = {"status": "complete", "world_size": world_size, "rank_receipts": world_size}

        if seen_map_payloads != set(expected_map_receipts):
            raise ValueError("native analyze stage does not bind every map receipt exactly once")

        capture_pairs = 0
        missing_capture_files = False
        checked_capture_bytes = 0
        for rank in range(world_size):
            receipt = cls._native_json(root, f"capture/rank{rank}/complete.json")
            if not isinstance(receipt, dict):
                raise TypeError("invalid native capture receipt")
            if (
                receipt.get("rank") != rank
                or receipt.get("weights_unchanged") is not True
                or not cls._native_binding_matches(receipt.get("binding"), expected_binding)
            ):
                raise ValueError("native capture binding mismatch")
            capture_pairs += cls._native_int(receipt.get("pairs"), maximum=MAX_NATIVE_STAGE_FILES * 100)
            chunk_hashes = receipt.get("files")
            if not isinstance(chunk_hashes, dict) or not chunk_hashes or len(chunk_hashes) > MAX_NATIVE_STAGE_FILES:
                raise ValueError("native capture chunk manifest is invalid")
            identity = cls._native_json(root, f"capture/rank{rank}/identity.json")
            qualification = cls._native_json(root, f"capture/rank{rank}/native-qualification.json")
            if not isinstance(identity, dict) or not isinstance(qualification, dict):
                raise TypeError("native capture provenance receipt is invalid")
            identity_provenance = identity.get("provenance")
            layout = identity_provenance.get("layout") if isinstance(identity_provenance, dict) else None
            loading_info = identity_provenance.get("loading_info") if isinstance(identity_provenance, dict) else None
            if (
                identity.get("rank") != rank
                or identity.get("world_size") != world_size
                or identity.get("teacher_forced") is not True
                or identity.get("zero_logit_equivalence") is not True
                or set(identity.get("layers", [])) != set(expected_capture_identity["layers"])
                or set(identity.get("readouts", [])) != set(expected_capture_identity["readouts"])
                or not isinstance(identity.get("capture_policy"), str)
                or "prompt-only" not in identity.get("capture_policy", "").lower()
                or identity.get("bundle_sha256") != expected_binding["bundle_sha256"]
                or identity.get("snapshot_sha256") != expected_binding["snapshot_sha256"]
                or identity.get("token_audit_sha256") != expected_binding["token_audit_sha256"]
                or not isinstance(identity_provenance, dict)
                or identity_provenance.get("model_id") != expected_capture_identity["model_id"]
                or identity_provenance.get("revision") != expected_capture_identity["revision"]
                or identity_provenance.get("tokenizer_revision") != expected_capture_identity["tokenizer_revision"]
                or not isinstance(layout, list)
                or not cls._native_named_hashes(identity_provenance.get("checkpoint_sha256"), ".safetensors", 16)
                or not cls._native_named_hashes(identity_provenance.get("implementation_sha256"), ".py", 64)
                or not cls._native_named_hashes(identity_provenance.get("metadata_sha256"), ".json", 128)
                or not isinstance(loading_info, dict)
                or any(loading_info.get(key) != [] for key in ("error_msgs", "mismatched_keys", "missing_keys", "unexpected_keys"))
            ):
                raise ValueError("native capture identity does not match the frozen model configuration")
            layout_by_layer = {
                item.get("layer"): item
                for item in layout
                if isinstance(item, dict) and isinstance(item.get("layer"), int)
            }
            for layer in expected_capture_identity["layers"]:
                layout_entry = layout_by_layer.get(layer)
                if not isinstance(layout_entry, dict) or layout_entry.get("projection_module") != _NATIVE_HOOK_TEMPLATE.format(layer=layer):
                    raise ValueError("native capture hook does not match the declared attention projection")
            qualification_binding = qualification.get("binding")
            router_state = qualification.get("router_state")
            records = qualification.get("records")
            if (
                not cls._native_binding_matches(qualification_binding, expected_binding)
                or qualification.get("read_only_logits_exact") is not True
                or not isinstance(router_state, str)
                or "prompt only" not in router_state.lower()
                or "no supplied continuation" not in router_state.lower()
                or not isinstance(records, list)
                or not records
                or len(records) > 64
                or not 0.0 <= cls._native_float(qualification.get("causality_tolerance"), minimum=0.0) <= 1.0
            ):
                raise ValueError("native capture qualification receipt is invalid")
            for filename, digest in chunk_hashes.items():
                if not isinstance(filename, str) or not (
                    re.fullmatch(r"chunk-[0-9]{6}\.npz", filename)
                    or filename in {"identity.json", "native-qualification.json"}
                ):
                    raise ValueError("native capture chunk name is invalid")
                chunk_path = cls._native_file(root, f"capture/rank{rank}/{filename}", required=False)
                if chunk_path is None:
                    missing_capture_files = True
                    continue
                size = chunk_path.stat().st_size
                checked_capture_bytes += size
                if checked_capture_bytes > MAX_NATIVE_CAPTURE_BYTES:
                    raise ValueError("native capture exceeds its size bound")
                max_bytes = MAX_NATIVE_JSON_BYTES if filename.endswith(".json") else MAX_NATIVE_NPZ_BYTES
                if not cls._native_file_matches(chunk_path, digest, max_bytes):
                    raise ValueError("native capture chunk digest mismatch")
        return summaries, (capture_pairs, missing_capture_files, checked_capture_bytes)

    @classmethod
    def _native_bank_summaries(
        cls,
        root: Path,
        bank_receipt: dict[str, Any],
        expected_binding: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not cls._native_binding_matches(bank_receipt.get("binding"), expected_binding):
            raise ValueError("native bank binding mismatch")
        files = bank_receipt.get("files")
        reports = bank_receipt.get("bank_reports")
        names = {"global", "kmeans", "som"}
        if not isinstance(files, dict) or set(files) != {f"{name}.npz" for name in names}:
            raise ValueError("native bank file receipt is incomplete")
        if not isinstance(reports, dict) or set(reports) != names:
            raise ValueError("native bank reports are incomplete")

        import numpy as np

        summaries = []
        for name in ("global", "kmeans", "som"):
            path = cls._native_verify_file(root, f"behavior/banks/{name}.npz", files[f"{name}.npz"], MAX_NATIVE_NPZ_BYTES)
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) > 64 or sum(member.file_size for member in members) > 256 * 1024 * 1024:
                    raise ValueError("native bank archive exceeds bounds")
                if any(PurePosixPath(member.filename).is_absolute() or ".." in PurePosixPath(member.filename).parts for member in members):
                    raise ValueError("native bank archive contains an invalid member")
            with np.load(path, allow_pickle=False) as arrays:
                required = {"center", "scale", "pca_mean", "pca_components", "coef", "bias", "vectors", "activation_rms"}
                if set(arrays.files) != required:
                    raise ValueError("native bank arrays are incomplete")
                vectors = arrays["vectors"]
                pca_components = arrays["pca_components"]
                if (
                    vectors.ndim != 2
                    or vectors.shape[0] < 1
                    or vectors.shape[0] > MAX_NATIVE_DIRECTION_VECTORS
                    or vectors.shape[1] < 1
                    or vectors.shape[1] > 65536
                    or pca_components.ndim != 2
                    or pca_components.shape[1] != vectors.shape[1]
                    or pca_components.shape[0] < 1
                    or pca_components.shape[0] > 1024
                    or any(not np.isfinite(arrays[key]).all() for key in arrays.files)
                ):
                    raise ValueError("native bank arrays failed structural validation")
                vector_count, hidden_size = (int(value) for value in vectors.shape)
                component_count = int(pca_components.shape[0])

            report = reports[name]
            if not isinstance(report, dict) or report.get("method") != name:
                raise ValueError("native bank summary identity mismatch")
            direction = report.get("direction")
            routing = report.get("routing_input")
            if (
                not isinstance(direction, str)
                or "good" not in direction.lower()
                or "bad" not in direction.lower()
                or "shared suffix" not in direction.lower()
                or not isinstance(routing, str)
                or "prompt-only" not in routing.lower()
                or "before" not in routing.lower()
                or "answer" not in routing.lower()
            ):
                raise ValueError("native bank direction or readout is unsupported")
            item: dict[str, Any] = {
                "method": name,
                "vector_count": vector_count,
                "hidden_size": hidden_size,
                "pca_component_count": component_count,
                "direction_source": "good-minus-bad target from shared-suffix readout",
                "routing_input": "independent prompt-only pre-answer readout",
                "training_families": cls._native_int(report.get("training_families"), maximum=100000),
                "training_pairs": cls._native_int(report.get("training_pairs"), maximum=MAX_NATIVE_STAGE_FILES * 100),
            }
            for key in ("cells_with_local_estimates",):
                if key in report:
                    item[key] = cls._native_int(report[key], maximum=MAX_NATIVE_DIRECTION_VECTORS)
            for source, output in (
                ("fit_route_accuracy", "fit_route_accuracy"),
                ("fit_majority_route_accuracy", "fit_majority_route_accuracy"),
                ("project_route_accuracy", "project_route_accuracy"),
                ("project_majority_route_accuracy", "project_majority_route_accuracy"),
            ):
                if source in report:
                    item[output] = cls._native_float(report[source], minimum=0.0, maximum=1.0)
            for source, output in (
                ("global_direction_norm", "global_direction_norm"),
                ("mean_individual_delta_norm", "mean_individual_delta_norm"),
            ):
                if source in report:
                    item[output] = cls._native_float(report[source], minimum=0.0)
            per_condition = report.get("project_by_condition", {})
            if per_condition:
                if not isinstance(per_condition, dict) or len(per_condition) > 32:
                    raise ValueError("native bank condition summaries are invalid")
                item["project_by_condition"] = {}
                for condition, values in sorted(per_condition.items()):
                    if not isinstance(condition, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", condition) or not isinstance(values, dict):
                        raise ValueError("native bank condition key is invalid")
                    item["project_by_condition"][condition] = {
                        "n": cls._native_int(values.get("n"), maximum=MAX_NATIVE_STAGE_FILES * 100),
                        "route_matches": cls._native_int(values.get("route_matches"), maximum=MAX_NATIVE_STAGE_FILES * 100),
                    }
                    if item["project_by_condition"][condition]["route_matches"] > item["project_by_condition"][condition]["n"]:
                        raise ValueError("native bank condition counts are inconsistent")
            summaries.append(item)
        return summaries

    @classmethod
    def _native_behavior_summary(
        cls,
        behavior: dict[str, Any],
        strict_screen: dict[str, Any],
        paired_screens: dict[str, Any],
        bank_summaries: list[dict[str, Any]],
        expected_answers: int,
    ) -> dict[str, Any]:
        cases = cls._native_int(behavior.get("cases"), minimum=1, maximum=MAX_NATIVE_STAGE_FILES * 100)
        arms = cls._native_int(behavior.get("arms"), minimum=1, maximum=128)
        answers = cls._native_int(behavior.get("answers"), minimum=1, maximum=MAX_NATIVE_STAGE_FILES * 100)
        if answers != expected_answers or answers != cases * arms:
            raise ValueError("native behavior counts are inconsistent")
        for key in ("mock", "mitigation_established", "hypothesis_confirmed", "all_strengths_prespecified"):
            if not isinstance(behavior.get(key), bool):
                raise TypeError("native behavior claims are malformed")
        review = behavior.get("semantic_review")
        if not isinstance(review, str) or len(review) > 2048:
            raise ValueError("native semantic review status is malformed")
        review_status = "pending" if "pending" in review.lower() else "recorded"

        scope = strict_screen.get("scope")
        cells = strict_screen.get("cells")
        if not isinstance(scope, str) or "lexical" not in scope.lower() or not isinstance(cells, dict) or len(cells) > 2048:
            raise ValueError("native strict screen is invalid")
        by_method: dict[str, dict[str, int]] = {}
        for key, value in cells.items():
            parts = key.split("/") if isinstance(key, str) else []
            if (
                len(parts) != 3
                or parts[0] not in {"DEVELOP", "EVALUATE"}
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", parts[1])
                or not re.fullmatch(r"[A-Za-z0-9_@.-]{1,64}", parts[2])
                or not isinstance(value, dict)
            ):
                raise ValueError("native strict-screen cell is malformed")
            n = cls._native_int(value.get("n"), maximum=MAX_NATIVE_STAGE_FILES * 100)
            strict_matches = cls._native_int(value.get("strict_matches"), maximum=n)
            needs_review = cls._native_int(value.get("needs_semantic_review"), maximum=n)
            capped = cls._native_int(value.get("capped"), maximum=n)
            agg = by_method.setdefault(parts[2], {"n": 0, "strict_matches": 0, "needs_semantic_review": 0, "capped": 0})
            for metric, count in (("n", n), ("strict_matches", strict_matches), ("needs_semantic_review", needs_review), ("capped", capped)):
                agg[metric] += count

        strict_summary = []
        for method, counts in sorted(by_method.items()):
            if counts["n"] == 0:
                continue
            strict_summary.append({
                "method": method,
                **counts,
                "strict_match_rate": counts["strict_matches"] / counts["n"],
            })

        condition_counts = paired_screens.get("condition_case_counts")
        records = paired_screens.get("records")
        if not isinstance(condition_counts, dict) or not isinstance(records, list) or len(records) != answers:
            raise ValueError("native paired-screen summary is incomplete")
        normalized_conditions = {}
        for condition, count in condition_counts.items():
            if not isinstance(condition, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", condition):
                raise ValueError("native condition label is invalid")
            normalized_conditions[condition] = cls._native_int(count, maximum=cases)
        if sum(normalized_conditions.values()) != cases:
            raise ValueError("native condition counts do not match case count")
        semantic_grade_count = 0
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("native paired-screen record is malformed")
            if record.get("semantic_grade") is not None:
                semantic_grade_count += 1

        return {
            "status": "complete",
            "cases": cases,
            "arms": arms,
            "answers": answers,
            "all_strengths_prespecified": behavior["all_strengths_prespecified"],
            "hypothesis_confirmed": behavior["hypothesis_confirmed"],
            "mitigation_established": behavior["mitigation_established"],
            "semantic_review_status": review_status,
            "semantic_grade_count": semantic_grade_count,
            "condition_case_counts": normalized_conditions,
            "strict_screen": {
                "scope": "conservative_lexical_only",
                "semantic_grades_available": semantic_grade_count > 0,
                "by_method": strict_summary,
            },
            "banks": bank_summaries,
        }

    def _validate_native_run(
        self,
        native_id: str,
        configured: dict[str, Path | None],
    ) -> dict[str, Any]:
        output_root = configured["output_root"]
        snapshot_root = configured["snapshot_root"]
        assert isinstance(output_root, Path)
        if snapshot_root is None:
            raise _NativeIncomplete
        if not output_root.is_dir() or not snapshot_root.is_dir():
            raise _NativeIncomplete

        snapshot_file = self._native_file(snapshot_root, "SNAPSHOT.json", required=True)
        assert snapshot_file is not None
        snapshot = self._native_json(snapshot_root, "SNAPSHOT.json")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != "attention-router-runtime-v1"
            or snapshot.get("experiment_schema") != "som-teacher-forced-pairs-v1"
            or snapshot.get("exact_copied_bytes_authoritative") is not True
        ):
            raise ValueError("unsupported native SOM snapshot")
        snapshot_files = snapshot.get("files")
        snapshot_hashes = self._native_snapshot_inventory(snapshot_root, snapshot_files)

        snapshot_inputs = {
            "config": "configs/som_teacher_forced_v1.json",
            "manifest": "datasets/som-teacher-forced-v1/MANIFEST.json",
            "recipe": "datasets/som-teacher-forced-v1/recipe.json",
            "tokens": "preflight/native-tokens.json",
            "behavior_config": "configs/som_teacher_behavior_v1.json",
            "behavior_plan": "datasets/som-teacher-behavior-v1/PLAN.json",
        }
        snapshot_input_hashes: dict[str, str] = {}
        for label, relative_name in snapshot_inputs.items():
            digest = snapshot_hashes.get(relative_name)
            if digest is None:
                raise ValueError("native snapshot omits a required input hash")
            snapshot_input_hashes[label] = digest

        config = self._native_json(snapshot_root, snapshot_inputs["config"])
        manifest = self._native_json(snapshot_root, snapshot_inputs["manifest"])
        recipe = self._native_json(snapshot_root, snapshot_inputs["recipe"])
        tokens = self._native_json(snapshot_root, snapshot_inputs["tokens"])
        if (
            not isinstance(config, dict)
            or config.get("schema") != "som-teacher-forced-pairs-v1"
            or not isinstance(manifest, dict)
            or manifest.get("schema") != "som-teacher-forced-inputs-v1"
            or not isinstance(recipe, dict)
            or recipe.get("schema") != "attention-router-native-recipe-v1"
            or not isinstance(tokens, dict)
            or tokens.get("schema") != "som-teacher-token-preflight-v1"
        ):
            raise ValueError("native SOM snapshot schemas do not match")
        recipe_model = recipe.get("model")
        if not isinstance(recipe_model, dict):
            raise TypeError("native model recipe is invalid")
        model_id = _valid_model_id(recipe_model.get("model_id"))
        revision = _valid_revision(recipe_model.get("revision"))
        tokenizer_revision = _valid_revision(recipe.get("tokenizer_revision"))
        if not revision or not tokenizer_revision:
            raise ValueError("native model revisions are not pinned")
        raw_layers = config.get("layers")
        raw_views = config.get("views")
        raw_readouts = config.get("readouts")
        if (
            not isinstance(raw_layers, list)
            or not raw_layers
            or len(raw_layers) > 128
            or not isinstance(raw_views, list)
            or not raw_views
            or len(raw_views) > 32
            or not isinstance(raw_readouts, list)
            or not raw_readouts
            or len(raw_readouts) > 32
        ):
            raise ValueError("native model layers or readout configuration is invalid")
        layers = sorted({_valid_layer(layer) for layer in raw_layers})
        if len(layers) != len(raw_layers):
            raise ValueError("native model layers are duplicated")
        views = sorted({self._native_label(value) for value in raw_views})
        readouts = sorted({self._native_label(value) for value in raw_readouts})
        world_size = self._native_int(config.get("world_size"), minimum=1, maximum=32)
        snapshot_source_sha = snapshot.get("source_git_sha")
        if not isinstance(snapshot_source_sha, str) or not re.fullmatch(r"[a-f0-9]{40}", snapshot_source_sha):
            raise ValueError("native source revision is invalid")

        complete = self._native_json(output_root, "complete.json")
        geometry = self._native_json(output_root, "geometry-complete.json")
        behavior = self._native_json(output_root, "behavior/complete.json")
        bank_receipt = self._native_json(output_root, "behavior/banks/complete.json")
        if not all(isinstance(value, dict) for value in (complete, geometry, behavior, bank_receipt)):
            raise ValueError("native completion receipts are malformed")
        assert isinstance(complete, dict) and isinstance(geometry, dict)
        assert isinstance(behavior, dict) and isinstance(bank_receipt, dict)

        expected_binding = {
            "bundle_sha256": snapshot_input_hashes["manifest"],
            "snapshot_sha256": _sha256_file(snapshot_file),
            "token_audit_sha256": snapshot_input_hashes["tokens"],
        }
        if (
            not self._native_binding_matches(complete.get("binding"), expected_binding)
            or not self._native_binding_matches(geometry.get("binding"), expected_binding)
            or not self._native_binding_matches(behavior.get("binding"), expected_binding)
            or not self._native_binding_matches(bank_receipt.get("binding"), expected_binding)
        ):
            raise ValueError("native run is not bound to the configured snapshot")
        if (
            complete.get("mock") is not False
            or complete.get("teacher_forced") is not True
            or complete.get("scientific_success_claim") is not False
            or geometry.get("mock") is not False
            or geometry.get("teacher_forced") is not True
            or geometry.get("base_weights_updated") is not False
            or geometry.get("scientific_success_claim") is not False
            or behavior.get("mock") is not False
        ):
            raise ValueError("native result lineage failed validation")

        complete_hash = _sha256_file(self._native_file(output_root, "complete.json", required=True))
        geometry_path = self._native_verify_file(output_root, "geometry-complete.json", complete.get("geometry_sha256"))
        behavior_path = self._native_verify_file(output_root, "behavior/complete.json", complete.get("behavior_sha256"))
        geometry_hash = _sha256_file(geometry_path)
        behavior_hash = _sha256_file(behavior_path)

        behavior_binding = behavior.get("binding")
        banks_hash = _sha256_file(self._native_file(output_root, "behavior/banks/complete.json", required=True))
        if (
            not isinstance(behavior_binding, dict)
            or behavior_binding.get("banks_sha256") != banks_hash
            or behavior_binding.get("behavior_plan_sha256") != snapshot_input_hashes["behavior_plan"]
            or bank_receipt.get("config_sha256") != snapshot_input_hashes["behavior_config"]
            or bank_receipt.get("plan_sha256") != snapshot_input_hashes["behavior_plan"]
        ):
            raise ValueError("native behavior banks do not match their frozen plan")

        report_path = self._native_verify_file(
            output_root,
            "report.html",
            geometry.get("report_sha256"),
            MAX_NATIVE_REPORT_BYTES,
        )
        report_hash = _sha256_file(report_path)
        stability_path = self._native_verify_file(output_root, "stability.json", geometry.get("stability_sha256"))
        stability = self._native_json(output_root, "stability.json")
        if not isinstance(stability, list) or len(stability) > MAX_NATIVE_MAPS * MAX_NATIVE_MAPS:
            raise ValueError("native stability summary is invalid")
        stability_values = []
        for item in stability:
            if not isinstance(item, dict):
                raise TypeError("native stability record is malformed")
            stability_values.append(self._native_float(item.get("project_adjusted_rand"), minimum=-1.0, maximum=1.0))
        stability_summary = {
            "comparisons": len(stability_values),
            "mean_project_adjusted_rand": sum(stability_values) / len(stability_values) if stability_values else None,
            "minimum_project_adjusted_rand": min(stability_values) if stability_values else None,
            "maximum_project_adjusted_rand": max(stability_values) if stability_values else None,
        }

        pairs = self._native_int(geometry.get("pairs"), minimum=1, maximum=MAX_NATIVE_STAGE_FILES * 100)
        answer_sequences = self._native_int(geometry.get("answer_sequences"), minimum=1, maximum=MAX_NATIVE_STAGE_FILES * 100)
        map_count = self._native_int(geometry.get("maps"), minimum=1, maximum=MAX_NATIVE_MAPS)
        capture_workers = self._native_int(geometry.get("capture_workers"), minimum=1, maximum=32)
        if (
            answer_sequences != pairs * 2
            or complete.get("maps") != map_count
            or geometry.get("capture_workers") != world_size
        ):
            raise ValueError("native geometry counts do not match the snapshot")
        conditions = geometry.get("conditions")
        if not isinstance(conditions, dict) or not conditions or len(conditions) > 32:
            raise ValueError("native geometry conditions are invalid")
        condition_counts = {}
        for condition, count in conditions.items():
            if not isinstance(condition, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", condition):
                raise ValueError("native geometry condition key is invalid")
            condition_counts[condition] = self._native_int(count, maximum=MAX_NATIVE_STAGE_FILES * 100)

        behavior_file_hashes = behavior.get("files")
        expected_behavior_files = {
            "blinded-review.json", "paired-screens.json", "strict-screen.json", "unblind.json",
        }
        if not isinstance(behavior_file_hashes, dict) or set(behavior_file_hashes) != expected_behavior_files:
            raise ValueError("native behavior receipt is incomplete")
        behavior_files: dict[str, Path] = {}
        for filename, digest in behavior_file_hashes.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}\.json", filename):
                raise ValueError("native behavior file name is invalid")
            behavior_files[filename] = self._native_verify_file(
                output_root,
                f"behavior/{filename}",
                digest,
            )
        strict_screen = self._native_json(output_root, "behavior/strict-screen.json")
        paired_screens = self._native_json(output_root, "behavior/paired-screens.json")
        if not isinstance(strict_screen, dict) or not isinstance(paired_screens, dict):
            raise TypeError("native behavior summaries are malformed")

        bank_summaries = self._native_bank_summaries(output_root, bank_receipt, expected_binding)
        behavior_summary = self._native_behavior_summary(
            behavior,
            strict_screen,
            paired_screens,
            bank_summaries,
            self._native_int(complete.get("free_generation_answers"), maximum=MAX_NATIVE_STAGE_FILES * 100),
        )
        if behavior_summary["cases"] != self._native_int(behavior.get("cases"), minimum=1):
            raise ValueError("native behavior case count is inconsistent")

        maps_root = self._native_directory(output_root, "maps")
        try:
            with os.scandir(maps_root) as entries_iter:
                entries = list(entries_iter)
        except OSError:
            raise _NativeIncomplete from None
        map_names = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                raise ValueError("native map directory contains an unexpected entry")
            if not _NATIVE_MAP_NAME_RE.fullmatch(entry.name):
                raise ValueError("native map name is invalid")
            map_names.append(entry.name)
        map_names.sort()
        if len(map_names) != map_count or len(map_names) > MAX_NATIVE_MAPS:
            raise _NativeIncomplete
        maps = [self._native_map_summary(output_root, name, expected_binding, set(views)) for name in map_names]
        if any(item["layer"] not in layers for item in maps):
            raise ValueError("native map layer is not in the frozen configuration")
        hidden_sizes = {item["map_shape"][2] for item in maps}
        if len(hidden_sizes) != 1 or any(item["hidden_size"] != next(iter(hidden_sizes)) for item in bank_summaries):
            raise ValueError("native maps and banks use different feature dimensions")
        expected_map_receipts: dict[str, str] = {}
        for map_name in map_names:
            relative_name = f"maps/{map_name}/complete.json"
            receipt_path = self._native_file(output_root, relative_name, required=True)
            assert receipt_path is not None
            expected_map_receipts[relative_name] = _sha256_file(receipt_path)

        stage_summaries, capture_info = self._native_stage_receipts(
            output_root,
            expected_binding,
            world_size,
            {
                "model_id": model_id,
                "revision": revision,
                "tokenizer_revision": tokenizer_revision,
                "layers": layers,
                "readouts": readouts,
            },
            {
                "banks": banks_hash,
                "geometry": geometry_hash,
                "behavior": behavior_hash,
                "complete": complete_hash,
                "report": report_hash,
            },
            expected_map_receipts,
        )
        capture_pairs, missing_capture_files, capture_bytes = capture_info
        if capture_pairs != pairs:
            raise ValueError("native capture pair count does not match geometry")
        capture_status = "pending" if missing_capture_files else "verified"
        report_available = self._native_html_is_safe(report_path, set(map_names))

        model_revision_hook_match = (
            model_id == self.default_model
            and self.default_revision is not None
            and revision == self.default_revision
            and _NATIVE_HOOK_TEMPLATE == DEFAULT_HOOK
        )
        integrity_status = "verified" if capture_status == "verified" else "outputs_verified"
        return {
            "native_id": native_id,
            "status": "completed",
            "integrity_status": integrity_status,
            "result_validity": "verified" if capture_status == "verified" else "outputs_verified_capture_pending",
            "capture_provenance_status": capture_status,
            "report_available": report_available,
            "report_url": f"/api/agent/som/native-reports/{native_id}/report" if report_available else None,
            "model_id": model_id,
            "revision": revision,
            "layers": layers,
            "map_count": map_count,
            "behavior_status": "complete",
            "hook_template": _NATIVE_HOOK_TEMPLATE,
            "feature_space": "attention_projection_output",
            "model_revision_hook_match": model_revision_hook_match,
            "reuse_compatibility": "not_assessed",
            "compatibility_note": "This native format uses attention projection outputs; the local pilot uses decoder block outputs. No cross-format reuse has been assessed.",
            "provenance": {
                "model_id": model_id,
                "revision": revision,
                "hook_template": _NATIVE_HOOK_TEMPLATE,
                "experiment_schema": snapshot["experiment_schema"],
                "dataset_schema": manifest["schema"],
                "source_git_sha": snapshot_source_sha,
                "snapshot_sha256": expected_binding["snapshot_sha256"],
                "bundle_sha256": expected_binding["bundle_sha256"],
                "token_audit_sha256": expected_binding["token_audit_sha256"],
                "tokenizer_revision": tokenizer_revision,
                "readouts": readouts,
                "views": views,
                "direction_source": "good-minus-bad shared-suffix state",
                "router_input": "independent prompt-only pre-answer readout",
            },
            "lineage": {
                "teacher_forced": True,
                "mock": False,
                "base_weights_updated": False,
                "scientific_success_claim": False,
                "zero_logit_equivalence_verified": True,
            },
            "stages": stage_summaries,
            "geometry": {
                "pairs": pairs,
                "answer_sequences": answer_sequences,
                "condition_counts": condition_counts,
                "map_count": map_count,
                "capture_workers": capture_workers,
                "stability": stability_summary,
                "report_sha256": report_hash,
                "stability_sha256": _sha256_file(stability_path),
                "capture_chunks_verified": not missing_capture_files,
                "capture_bytes_verified": capture_bytes,
            },
            "maps": maps,
            "behavior": behavior_summary,
            "scientific_interpretation": "Exploratory teacher-forced geometry and behavioral screening; no causal mitigation claim is established.",
            "error_code": None,
        }

    @staticmethod
    def _native_label(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
            raise ValueError("invalid native label")
        return value

    def status(self) -> dict[str, Any]:
        prompt_count = len(self._scan_prompt_sources())
        artifact_count = len(self._scan_artifacts())
        external_jobs = self._external_job_records()
        native_reports = self._scan_native_reports()
        config_warning = self._source_roots_warning or self._native_roots_warning or self._manifest_warning
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
            "native_reports": native_reports,
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

    def list_native_reports(self) -> dict[str, Any]:
        """List validated native teacher-forced runs without exposing local paths."""
        reports = []
        for configured in self.native_som_roots:
            report = self._native_report_for_root(configured)
            reports.append(self._native_catalog_summary(report))
        return {"native_reports": reports}

    def get_native_report(self, native_id: str) -> dict[str, Any]:
        """Return receipt-bound aggregate summaries for one native SOM run."""
        configured = self._native_root_for_id(native_id)
        report = self._native_report_for_root(configured)
        return report

    def get_native_report_file(self, native_id: str) -> Path | None:
        """Return a validated local report path for the server-only HTML route."""
        report = self.get_native_report(native_id)
        if not report.get("report_available") or report.get("status") != "completed":
            return None
        configured = self._native_root_for_id(native_id)
        root = configured["output_root"]
        assert isinstance(root, Path)
        try:
            path = self._native_file(root, "report.html", required=True)
            expected = report.get("geometry", {}).get("report_sha256")
            if path is None or not self._native_file_matches(path, expected, MAX_NATIVE_REPORT_BYTES):
                return None
            if not self._native_html_is_safe(path, {item["map_name"] for item in report.get("maps", [])}):
                return None
        except (OSError, ValueError, _NativeIncomplete):
            return None
        return path

    def get_native_report_asset(self, native_id: str, relative_name: str) -> Path | None:
        """Return one validated map PNG referenced by the native HTML report."""
        report = self.get_native_report(native_id)
        if report.get("status") != "completed":
            return None
        if not isinstance(relative_name, str):
            return None
        match = re.fullmatch(r"maps/([A-Za-z0-9][A-Za-z0-9_.-]{0,119})/map\.png", relative_name)
        if not match:
            return None
        map_name = match.group(1)
        map_record = next((item for item in report.get("maps", []) if item.get("map_name") == map_name), None)
        if map_record is None:
            return None
        configured = self._native_root_for_id(native_id)
        root = configured["output_root"]
        assert isinstance(root, Path)
        try:
            path = self._native_file(root, relative_name, required=True)
            if path is None or not self._native_file_matches(path, map_record.get("png_sha256"), MAX_NATIVE_MAP_PNG_BYTES):
                return None
        except (OSError, ValueError, _NativeIncomplete):
            return None
        try:
            with path.open("rb") as handle:
                return path if handle.read(8) == b"\x89PNG\r\n\x1a\n" else None
        except OSError:
            return None

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
