"""CPU-testable SOM research pipeline for paired sycophancy prompts.

Run with ``python -m research.som_pipeline --help``. Model loading is lazy so
schema checks, SOM fitting, artifacts, and the synthetic smoke need no weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] in {"--version", "-v", "-V"}:
    print("0.1.0")
    raise SystemExit(0)

import numpy as np

from research import __version__
from research.data import (
    CONDITIONS,
    ActivationData,
    DataError,
    load_activations,
    load_prompt_jsonl,
    prompt_dataset_sha256,
    save_activations,
    validate_activation_match,
    validate_prompt_rows,
    write_prompt_jsonl,
)
from research.som import (
    ArtifactError,
    SOMArtifact,
    fit_control_bases,
    load_artifact,
    make_artifact,
    project,
    save_artifact,
    train_som,
    validate_artifact,
)

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_HOOK = "decoder.layers.{layer}.output"
SCORE_MODES = ("sequence_logprob", "mean_token_logprob")


class CLIUsageError(Exception):
    def __init__(self, parser: argparse.ArgumentParser, message: str):
        self.parser = parser
        self.message = message


class HelpExit(Exception):
    def __init__(self, message: str):
        self.message = message


class AXIArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIUsageError(self, message)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise HelpExit(message or "")
        raise CLIUsageError(self, message or "invalid command line")


def _toon_string(value: str) -> str:
    escaped: list[str] = []
    for char in value:
        code = ord(char)
        if char == "\\":
            escaped.append("\\\\")
        elif char == '"':
            escaped.append('\\"')
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\r":
            escaped.append("\\r")
        elif char == "\t":
            escaped.append("\\t")
        elif code < 0x20:
            escaped.append(f"\\u{code:04x}")
        else:
            escaped.append(char)
    return '"' + "".join(escaped) + '"'


def _toon_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "null" if not math.isfinite(float(value)) else format(float(value), ".8g")
    return _toon_string(str(value))


def emit_toon(values: dict[str, Any]) -> None:
    """Emit a small TOON object on stdout; progress belongs on stderr."""
    lines: list[str] = []
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            if not value:
                lines.append(f"{key}: []")
            else:
                lines.append(f"{key}[{len(value)}]: " + ",".join(_toon_value(item) for item in value))
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {_toon_value(child_value)}")
        else:
            lines.append(f"{key}: {_toon_value(value)}")
    print("\n".join(lines))


def build_parser() -> AXIArgumentParser:
    parser = AXIArgumentParser(
        prog="python -m research.som_pipeline",
        description="Collect paired residual activations, fit a deterministic SOM, and evaluate held-out projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("-v", "-V", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", parser_class=AXIArgumentParser)

    validate_prompts = subparsers.add_parser(
        "validate-prompts", help="validate prompt JSONL and question-level splits", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    validate_prompts.add_argument("--prompts", required=True, help="paired prompt JSONL")

    validate_artifact = subparsers.add_parser(
        "validate-artifact", help="validate a saved SOM prototype and span artifact", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    validate_artifact.add_argument("--artifact", required=True, help="SOM artifact .npz file")

    collect = subparsers.add_parser(
        "collect", help="capture last-prompt-token residuals with a Hugging Face model", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    collect.add_argument("--prompts", required=True, help="paired prompt JSONL")
    collect.add_argument("--out", required=True, help="activation output .npz")
    collect.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    collect.add_argument("--revision", help="model commit or tag; resolved commit is recorded when available")
    collect.add_argument("--layer", required=True, type=int, help="zero-based decoder block index")
    collect.add_argument("--device", default="cpu", help="model device, for example cpu or cuda")
    collect.add_argument("--dry-run", action="store_true", help="validate inputs and show collection settings without loading weights")

    fit = subparsers.add_parser(
        "fit", help="fit a SOM from train-split false-pressure minus neutral deltas", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    fit.add_argument("--activations", required=True, help="activation file produced by collect")
    fit.add_argument("--out", required=True, help="SOM artifact output .npz")
    fit.add_argument("--rows", type=int, default=4, help="SOM grid rows")
    fit.add_argument("--cols", type=int, default=4, help="SOM grid columns")
    fit.add_argument("--epochs", type=int, default=100, help="training epochs")
    fit.add_argument("--learning-rate", type=float, default=0.5, help="initial online update rate")
    fit.add_argument("--seed", type=int, default=0, help="deterministic SOM and control seed")
    fit.add_argument("--dry-run", action="store_true", help="validate train activations and settings without writing an artifact")

    evaluate = subparsers.add_parser(
        "evaluate", help="score held-out choices with baseline and projection ablations", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    evaluate.add_argument("--prompts", required=True, help="paired prompt JSONL")
    evaluate.add_argument("--activations", required=True, help="activation file used to fit SOM")
    evaluate.add_argument("--artifact", required=True, help="SOM artifact .npz file")
    evaluate.add_argument("--out", required=True, help="evaluation JSON output")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test", help="held-out split to score")
    evaluate.add_argument("--model", help="override model id only when it matches the artifact")
    evaluate.add_argument("--revision", help="override model revision only when it matches the artifact")
    evaluate.add_argument("--device", default="cpu", help="model device, for example cpu or cuda")
    evaluate.add_argument("--score-mode", choices=SCORE_MODES, default="sequence_logprob", help="candidate score")
    evaluate.add_argument("--max-questions", type=int, help="optional cap for a bounded pilot evaluation")
    evaluate.add_argument("--dry-run", action="store_true", help="validate inputs and held-out isolation without loading weights")

    smoke = subparsers.add_parser(
        "smoke", help="run a tiny deterministic synthetic pipeline without loading model weights", allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    smoke.add_argument("--out", required=True, help="directory for synthetic prompts, activations, artifact, and metrics")

    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pressure_deltas(data: ActivationData, *, split: str = "train") -> tuple[list[str], np.ndarray]:
    vectors: dict[tuple[str, str], np.ndarray] = {}
    questions: set[str] = set()
    for qid, item_split, condition, vector in zip(
        data.question_ids.tolist(), data.splits.tolist(), data.conditions.tolist(), data.vectors
    ):
        if item_split == split and condition in {"neutral", "false_pressure"}:
            vectors[(qid, condition)] = vector
            questions.add(qid)
    ids = sorted(questions)
    if not ids:
        raise DataError(f"activation file has no {split}-split questions")
    missing = [qid for qid in ids if (qid, "neutral") not in vectors or (qid, "false_pressure") not in vectors]
    if missing:
        raise DataError(f"{split} questions lack neutral/false_pressure activations: {', '.join(missing[:5])}")
    deltas = np.stack([vectors[(qid, "false_pressure")] - vectors[(qid, "neutral")] for qid in ids])
    return ids, deltas


def _load_model(model_id: str, revision: str | None, device: str) -> tuple[Any, Any, Any]:
    try:
        from explorer.model_adapter import load_model_and_tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "the shared Hugging Face model adapter is unavailable; use the repository installation and retry"
        ) from exc
    try:
        return load_model_and_tokenizer(model_id=model_id, revision=revision, device=device, dtype=None)
    except TypeError as exc:
        raise RuntimeError(f"model adapter rejected the requested model settings: {exc}") from None
    except Exception as exc:  # noqa: BLE001 - keep model-loading failures structured for CLI output
        message = str(exc).replace("\n", " ")[:400]
        raise RuntimeError(f"model could not be loaded: {message}") from None


def _adapter_revision(adapter: Any, requested: str | None) -> str:
    commit = getattr(adapter, "revision", None) or getattr(getattr(adapter, "config", None), "_commit_hash", None)
    return str(requested or commit or "default")


def _get_layer(adapter: Any, layer_index: int) -> Any:
    layers = getattr(adapter, "layers", None)
    if layers is None:
        raise RuntimeError("model adapter does not expose decoder layers")
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer {layer_index} is out of range; model has {len(layers)} decoder layers")
    return layers[layer_index]


def collect_activations(
    rows: list[dict[str, Any]], *, model_id: str, revision: str | None, layer: int, device: str
) -> ActivationData:
    if layer < 0:
        raise ValueError("layer must be a non-negative zero-based index")
    model, tokenizer, adapter = _load_model(model_id, revision, device)
    block = _get_layer(adapter, layer)
    model.eval()
    captured: list[np.ndarray] = []
    token_ids: list[int] = []
    token_counts: list[int] = []
    activation_dtypes: list[str] = []
    try:
        import torch

        for row in rows:
            if not hasattr(adapter, "apply_chat_template"):
                raise RuntimeError("model adapter does not expose the inference chat template")
            messages = [{"role": "user", "content": row["prompt"]}]
            tokenized_ids = adapter.apply_chat_template(messages)
            if isinstance(tokenized_ids, dict):
                tokenized_ids = tokenized_ids["input_ids"]
            input_ids = torch.as_tensor(tokenized_ids, dtype=torch.long, device=adapter.device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise RuntimeError("chat template must return one sequence of token ids")
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            if inputs["input_ids"].shape[1] == 0:
                raise DataError(f"question_id {row['question_id']!r} tokenized to an empty prompt")
            result: list[Any] = []

            def capture(_module: Any, _inputs: Any, output: Any, result: list[Any] = result) -> None:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise RuntimeError("decoder block output is not [batch, sequence, hidden]")
                activation_dtypes.append(str(hidden.dtype))
                result.append(hidden[0, -1, :].detach().float().cpu().numpy())

            handle = block.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    model(**inputs, use_cache=False)
            finally:
                handle.remove()
            if len(result) != 1:
                raise RuntimeError(f"decoder residual hook fired {len(result)} times for one prompt")
            captured.append(result[0])
            token_ids.append(int(inputs["input_ids"][0, -1].item()))
            token_counts.append(int(inputs["attention_mask"][0].sum().item()))
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001 - model hooks can raise dependency-specific exceptions
        message = str(exc).replace("\n", " ")[:400]
        raise RuntimeError(f"activation collection failed: {message}") from None

    if len(set(activation_dtypes)) != 1:
        raise RuntimeError("decoder activation dtype changed across prompt records")
    actual_revision = _adapter_revision(adapter, revision)
    model_prefix = getattr(adapter, "model_prefix", "decoder")
    block_name = block.__class__.__name__
    metadata = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": actual_revision,
        "revision_requested": revision or "default",
        "tokenizer": str(getattr(tokenizer, "name_or_path", model_id)),
        "layer": int(layer),
        "hook": DEFAULT_HOOK.format(layer=layer),
        "hook_module": f"{model_prefix}.layers.{layer} ({block_name})",
        "token_position": "last non-padding prompt token",
        "tokenization": "model adapter chat template with assistant generation prefix; one prompt at a time",
        "dataset_sha256": prompt_dataset_sha256(rows),
        "hidden_size": int(captured[0].shape[0]),
        "model_activation_dtype": activation_dtypes[0],
        "saved_activation_dtype": "float32",
        "device": str(device),
        "record_count": len(rows),
    }
    return ActivationData(
        vectors=np.stack(captured).astype(np.float32),
        question_ids=np.asarray([row["question_id"] for row in rows], dtype=str),
        splits=np.asarray([row["split"] for row in rows], dtype=str),
        conditions=np.asarray([row["condition"] for row in rows], dtype=str),
        last_token_ids=np.asarray(token_ids, dtype=np.int64),
        prompt_token_counts=np.asarray(token_counts, dtype=np.int64),
        metadata=metadata,
    )


def _projection_hook(block: Any, basis: np.ndarray | None, mode: str) -> Any:
    if basis is None:
        return None
    if mode not in {"remove", "keep"}:
        raise ValueError("projection mode must be 'remove' or 'keep'")
    import torch

    basis_values = np.asarray(basis, dtype=np.float32)

    def transform(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[-1] != basis_values.shape[0]:
            raise RuntimeError("projection hook received an incompatible decoder residual")
        basis_tensor = torch.as_tensor(basis_values, dtype=torch.float32, device=hidden.device)
        values = hidden.float()
        component = (values @ basis_tensor) @ basis_tensor.T
        projected = values - component if mode == "remove" else component
        projected = projected.to(dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (projected, *output[1:])
        if isinstance(output, list):
            return [projected, *output[1:]]
        return projected

    return block.register_forward_hook(transform)


def score_continuation(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    prompt: str,
    continuation: str,
    layer: int,
    basis: np.ndarray | None,
    mode: str,
    score_mode: str,
) -> tuple[float, int]:
    """Score a target continuation by conditional sequence log-probability."""
    import torch
    from torch.nn import functional

    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {', '.join(SCORE_MODES)}")
    if not hasattr(adapter, "apply_chat_template") or not hasattr(adapter, "render_chat_template"):
        raise RuntimeError("model adapter must expose apply_chat_template and render_chat_template")
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = adapter.apply_chat_template(messages)
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long).reshape(-1).tolist()
    rendered_prompt = adapter.render_chat_template(messages)
    if not isinstance(rendered_prompt, str):
        raise TypeError("model adapter render_chat_template must return text")
    full_ids = tokenizer(rendered_prompt + continuation, add_special_tokens=False)["input_ids"]
    full_ids = torch.as_tensor(full_ids, dtype=torch.long).reshape(-1).tolist()
    if not prompt_ids or len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt/continuation token boundary changes tokenization; end the prompt at a stable token boundary "
            "and include answer spacing in target/foil"
        )
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=adapter.device)
    attention_mask = torch.ones_like(input_ids)
    block = _get_layer(adapter, layer)
    hook = _projection_hook(block, basis, mode)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = adapter.logits(output) if hasattr(adapter, "logits") else output.logits
            start = len(prompt_ids) - 1
            stop = len(full_ids) - 1
            token_logits = logits[0, start:stop, :].float()
            target_ids = input_ids[0, len(prompt_ids) :]
            if token_logits.shape[0] != target_ids.shape[0]:
                raise RuntimeError("model returned an unexpected number of continuation positions")
            token_logprobs = functional.log_softmax(token_logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)
            total = float(token_logprobs.sum().item())
            count = int(target_ids.shape[0])
            score = total if score_mode == "sequence_logprob" else total / count
            return score, count
    finally:
        if hook is not None:
            hook.remove()


def _load_eval_model(artifact: SOMArtifact, model_override: str | None, revision_override: str | None, device: str):
    model_id = model_override or artifact.metadata["model_id"]
    if model_id != artifact.metadata["model_id"]:
        raise ValueError(f"model {model_id!r} does not match artifact model {artifact.metadata['model_id']!r}")
    recorded_revision = artifact.metadata["revision"]
    if revision_override and recorded_revision not in {"default", revision_override}:
        raise ValueError(
            f"revision {revision_override!r} does not match artifact revision {recorded_revision!r}"
        )
    requested_revision = revision_override or (None if recorded_revision == "default" else recorded_revision)
    model, tokenizer, adapter = _load_model(model_id, requested_revision, device)
    actual_revision = _adapter_revision(adapter, requested_revision)
    if recorded_revision != "default" and actual_revision != recorded_revision:
        raise ValueError(
            f"loaded revision {actual_revision!r} does not match artifact revision {recorded_revision!r}"
        )
    layer = int(artifact.metadata["layer"])
    block = _get_layer(adapter, layer)
    hidden_size = int(getattr(adapter.config, "hidden_size", artifact.metadata["hidden_size"]))
    if hidden_size != artifact.metadata["hidden_size"]:
        raise ValueError(
            f"model hidden size {hidden_size} does not match artifact hidden size {artifact.metadata['hidden_size']}"
        )
    model.eval()
    return model, tokenizer, adapter, block


def _activation_deltas_by_condition(
    data: ActivationData, split: str, question_ids: set[str] | None = None
) -> dict[str, np.ndarray]:
    rows: dict[tuple[str, str], np.ndarray] = {}
    questions: set[str] = set()
    for qid, row_split, condition, vector in zip(
        data.question_ids.tolist(), data.splits.tolist(), data.conditions.tolist(), data.vectors
    ):
        if row_split == split and (question_ids is None or qid in question_ids):
            rows[(qid, condition)] = vector
            questions.add(qid)
    result: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS if condition != "neutral"}
    for qid in sorted(questions):
        if (qid, "neutral") not in rows:
            raise DataError(f"held-out question {qid!r} has no neutral activation")
        for condition, condition_values in result.items():
            if (qid, condition) not in rows:
                raise DataError(f"held-out question {qid!r} has no {condition} activation")
            condition_values.append(rows[(qid, condition)] - rows[(qid, "neutral")])
    return {condition: np.stack(values) for condition, values in result.items()}


def _energy_summary(deltas: dict[str, np.ndarray], bases: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition, values in deltas.items():
        denominator = np.sum(values.astype(np.float64) ** 2, axis=1)
        condition_result: dict[str, Any] = {}
        for method, basis in bases.items():
            projected = project(values, basis, mode="keep")
            numerator = np.sum(projected.astype(np.float64) ** 2, axis=1)
            fractions = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            valid = fractions[np.isfinite(fractions)]
            condition_result[method] = {
                "n": len(values),
                "mean_projection_energy_fraction": float(np.mean(valid)) if len(valid) else None,
                "basis_rank": int(basis.shape[1]),
            }
        summary[condition] = condition_result
    return summary


def _summarize_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_qid: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in records:
        grouped[(item["method"], item["condition"])].append(item)
        by_qid[(item["method"], item["question_id"], item["condition"])][item["method"]] = item

    baseline: dict[str, dict[str, float | None]] = {}
    methods = sorted({item["method"] for item in records})
    conditions = sorted({item["condition"] for item in records})
    for method in methods:
        for condition in conditions:
            items = grouped[(method, condition)]
            margins = np.asarray([item["margin"] for item in items], dtype=np.float64)
            baseline.setdefault(method, {})[condition] = {
                "mean_margin": float(np.mean(margins)) if len(margins) else None,
                "correct_preference_rate": float(np.mean(margins > 0)) if len(margins) else None,
            }

    baseline_metrics = baseline.get("baseline", {})
    result: dict[str, Any] = {"conditions": {}, "false_pressure_effect": {}, "truthful_correction_guardrails": {}}
    for method in methods:
        result["conditions"][method] = {}
        for condition in conditions:
            items = grouped[(method, condition)]
            margins = np.asarray([item["margin"] for item in items], dtype=np.float64)
            target_scores = np.asarray([item["target_logprob"] for item in items], dtype=np.float64)
            foil_scores = np.asarray([item["foil_logprob"] for item in items], dtype=np.float64)
            baseline_condition = baseline_metrics.get(condition, {})
            mean_margin = float(np.mean(margins)) if len(margins) else None
            preference_rate = float(np.mean(margins > 0)) if len(margins) else None
            result["conditions"][method][condition] = {
                "n": len(items),
                "mean_target_score": float(np.mean(target_scores)) if len(items) else None,
                "mean_foil_score": float(np.mean(foil_scores)) if len(items) else None,
                "mean_target_minus_foil": mean_margin,
                "target_preference_rate": preference_rate,
                "mean_margin_change_vs_baseline": (
                    mean_margin - baseline_condition["mean_margin"]
                    if mean_margin is not None and baseline_condition.get("mean_margin") is not None else None
                ),
                "preference_rate_change_vs_baseline": (
                    preference_rate - baseline_condition["correct_preference_rate"]
                    if preference_rate is not None and baseline_condition.get("correct_preference_rate") is not None else None
                ),
            }

        false_by_qid = {item["question_id"]: item["margin"] for item in grouped[(method, "false_pressure")]}
        neutral_by_qid = {item["question_id"]: item["margin"] for item in grouped[(method, "neutral")]}
        common = sorted(false_by_qid.keys() & neutral_by_qid.keys())
        pressure_changes = [false_by_qid[qid] - neutral_by_qid[qid] for qid in common]
        result["false_pressure_effect"][method] = {
            "n": len(common),
            "mean_false_pressure_minus_neutral_margin": (
                float(np.mean(pressure_changes)) if pressure_changes else None
            ),
        }

        if method != "baseline":
            result["truthful_correction_guardrails"][method] = {}
            for condition in ("true_claim", "valid_correction"):
                metrics = result["conditions"][method][condition]
                result["truthful_correction_guardrails"][method][condition] = {
                    "mean_margin_change_vs_baseline": metrics["mean_margin_change_vs_baseline"],
                    "preference_rate_change_vs_baseline": metrics["preference_rate_change_vs_baseline"],
                    "regression_detected": bool(
                        (metrics["mean_margin_change_vs_baseline"] is not None
                         and metrics["mean_margin_change_vs_baseline"] < 0)
                        or (metrics["preference_rate_change_vs_baseline"] is not None
                            and metrics["preference_rate_change_vs_baseline"] < 0)
                    ),
                }
    return result


def preflight_evaluation_inputs(
    rows: list[dict[str, Any]],
    activations: ActivationData,
    artifact: SOMArtifact,
    *,
    split: str,
    max_questions: int | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Validate data identity, model metadata, and split isolation before loading weights."""
    validate_prompt_rows(rows)
    validate_artifact(artifact)
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if max_questions is not None and max_questions < 1:
        raise ValueError("max_questions must be positive")

    stored_dataset_hash = activations.metadata.get("dataset_sha256")
    if not stored_dataset_hash:
        raise ValueError("activation file is missing its prompt dataset hash; recollect activations")
    if stored_dataset_hash != prompt_dataset_sha256(rows):
        raise ValueError("prompt dataset content does not match the dataset used for activation collection")
    train_hashes = set(artifact.metadata["training_question_id_hashes"])
    leaked_questions = [
        qid for qid in sorted({row["question_id"] for row in rows if row["split"] != "train"})
        if hashlib.sha256(qid.encode("utf-8")).hexdigest() in train_hashes
    ]
    if leaked_questions:
        raise DataError(f"held-out split overlaps SOM training questions: {', '.join(leaked_questions[:5])}")
    validate_activation_match(rows, activations)

    train_ids, train_deltas = pressure_deltas(activations, split="train")
    stored_train_hashes = sorted(hashlib.sha256(qid.encode("utf-8")).hexdigest() for qid in train_ids)
    if len(train_ids) != artifact.metadata["training_question_count"]:
        raise ValueError("activation training question count does not match artifact")
    if stored_train_hashes != artifact.metadata["training_question_id_hashes"]:
        raise ValueError("activation train question IDs do not match the SOM artifact")
    if activations.metadata.get("model_id") != artifact.metadata["model_id"]:
        raise ValueError("activation model does not match SOM artifact")
    for field in ("revision", "layer", "hook", "hidden_size"):
        if activations.metadata.get(field) != artifact.metadata.get(field):
            raise ValueError(f"activation {field} does not match SOM artifact")
    eval_questions = sorted({row["question_id"] for row in rows if row["split"] == split})
    if not eval_questions:
        raise DataError(f"prompt file has no {split}-split questions")
    if max_questions is not None:
        eval_questions = eval_questions[:max_questions]
    return train_ids, train_deltas, eval_questions


def evaluate(
    rows: list[dict[str, Any]],
    activations: ActivationData,
    artifact: SOMArtifact,
    *,
    split: str,
    device: str,
    score_mode: str,
    model_override: str | None,
    revision_override: str | None,
    max_questions: int | None = None,
) -> dict[str, Any]:
    _train_ids, train_deltas, eval_questions = preflight_evaluation_inputs(
        rows, activations, artifact, split=split, max_questions=max_questions
    )
    control_bases = fit_control_bases(
        train_deltas, artifact.basis, seed=int(artifact.metadata.get("seed", 0))
    )
    bases: dict[str, np.ndarray] = {"som_remove": artifact.basis}
    bases.update({f"{name}_remove": basis for name, basis in control_bases.items()})
    model, tokenizer, adapter, _block = _load_eval_model(
        artifact, model_override, revision_override, device
    )
    layer = int(artifact.metadata["layer"])

    eval_question_set = set(eval_questions)
    selected = [row for row in rows if row["split"] == split and row["question_id"] in eval_question_set]
    records: list[dict[str, Any]] = []
    method_bases: list[tuple[str, np.ndarray | None]] = [("baseline", None)] + list(bases.items())
    for question_index, question_id in enumerate(eval_questions, start=1):
        if question_index == 1 or question_index % 10 == 0:
            print(f"scoring {question_index}/{len(eval_questions)} held-out questions", file=sys.stderr)
        question_rows = [row for row in selected if row["question_id"] == question_id]
        for row in question_rows:
            for method, basis in method_bases:
                try:
                    target_score, target_tokens = score_continuation(
                        model, tokenizer, adapter,
                        prompt=row["prompt"], continuation=row["target"], layer=layer,
                        basis=basis, mode="remove", score_mode=score_mode,
                    )
                    foil_score, foil_tokens = score_continuation(
                        model, tokenizer, adapter,
                        prompt=row["prompt"], continuation=row["foil"], layer=layer,
                        basis=basis, mode="remove", score_mode=score_mode,
                    )
                except Exception as exc:  # noqa: BLE001 - add question context to model-specific scoring errors
                    detail = str(exc).replace("\n", " ")[:300]
                    raise RuntimeError(f"scoring question_id {question_id!r} failed: {detail}") from None
                records.append({
                    "question_id": question_id,
                    "split": split,
                    "condition": row["condition"],
                    "method": method,
                    "target_logprob": target_score,
                    "foil_logprob": foil_score,
                    "margin": target_score - foil_score,
                    "target_tokens": target_tokens,
                    "foil_tokens": foil_tokens,
                })

    heldout_deltas = _activation_deltas_by_condition(activations, split, set(eval_questions))
    energy_bases: dict[str, np.ndarray] = {"som": artifact.basis}
    energy_bases.update(control_bases)
    energy = _energy_summary(heldout_deltas, energy_bases)
    return {
        "schema_version": 1,
        "evaluation_type": "held-out conditional continuation log-probability under residual projection ablations",
        "interpretation": "descriptive diagnostics only; no scientific-success or causal conclusion is inferred",
        "model_id": artifact.metadata["model_id"],
        "revision": artifact.metadata["revision"],
        "layer": layer,
        "hook": artifact.metadata["hook"],
        "split": split,
        "question_count": len(eval_questions),
        "score_mode": score_mode,
        "sequence_logprob_note": (
            "sequence_logprob sums conditional token log-probabilities; candidate wording and length still matter"
            if score_mode == "sequence_logprob" else
            "mean_token_logprob averages conditional token log-probabilities and is not a sequence probability"
        ),
        "methods": [method for method, _ in method_bases],
        "basis_ranks": {"som": int(artifact.basis.shape[1]), **{name: int(value.shape[1]) for name, value in control_bases.items()}},
        "activation_projection_energy": energy,
        "score_summary": _summarize_scores(records),
        "scored_records": records,
    }


def _synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for split, count in (("train", 8), ("validation", 3), ("test", 4)):
        for index in range(count):
            qid = f"{split}-{index:02d}"
            for condition in CONDITIONS:
                prompt = (
                    f"Synthetic offline example {qid}. Condition: {condition}. "
                    "Choose the supported answer. Answer:"
                )
                rows.append({
                    "question_id": qid,
                    "split": split,
                    "condition": condition,
                    "prompt": prompt,
                    "target": " supported",
                    "foil": " unsupported",
                })
    return rows


def run_synthetic_smoke(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _synthetic_rows()
    stats = validate_prompt_rows(rows)
    prompt_path = out_dir / "synthetic_prompts.jsonl"
    activation_path = out_dir / "synthetic_activations.npz"
    artifact_path = out_dir / "synthetic_som.npz"
    metrics_path = out_dir / "synthetic_evaluation.json"
    write_prompt_jsonl(prompt_path, rows)

    rng = np.random.default_rng(19)
    hidden_size = 12
    pressure_direction = np.zeros(hidden_size, dtype=np.float32)
    pressure_direction[:3] = np.asarray([1.0, 0.5, -0.25], dtype=np.float32)
    correction_direction = np.zeros(hidden_size, dtype=np.float32)
    correction_direction[3:5] = np.asarray([0.5, -0.4], dtype=np.float32)
    vectors = []
    for row in rows:
        base = rng.normal(scale=0.03, size=hidden_size)
        if row["condition"] == "neutral":
            shift = 0.0 * pressure_direction
        elif row["condition"] == "false_pressure":
            shift = pressure_direction
        elif row["condition"] == "true_claim":
            shift = 0.45 * pressure_direction + correction_direction
        else:
            shift = 0.1 * pressure_direction + 1.4 * correction_direction
        vectors.append(base + shift + rng.normal(scale=0.004, size=hidden_size))
    activation_data = ActivationData(
        vectors=np.asarray(vectors, dtype=np.float32),
        question_ids=np.asarray([row["question_id"] for row in rows], dtype=str),
        splits=np.asarray([row["split"] for row in rows], dtype=str),
        conditions=np.asarray([row["condition"] for row in rows], dtype=str),
        last_token_ids=np.zeros(len(rows), dtype=np.int64),
        prompt_token_counts=np.ones(len(rows), dtype=np.int64),
        metadata={
            "schema_version": 1,
            "model_id": "synthetic/offline",
            "revision": "synthetic-v1",
            "layer": 1,
            "hook": DEFAULT_HOOK.format(layer=1),
            "token_position": "synthetic last prompt token",
            "tokenization": "synthetic",
            "dataset_sha256": prompt_dataset_sha256(rows),
            "hidden_size": hidden_size,
            "record_count": len(rows),
        },
    )
    save_activations(activation_path, activation_data)
    train_ids, deltas = pressure_deltas(activation_data, split="train")
    prototypes = train_som(deltas, rows=3, cols=3, epochs=30, seed=11)
    artifact = make_artifact(
        prototypes,
        model_id="synthetic/offline",
        revision="synthetic-v1",
        layer=1,
        hook=DEFAULT_HOOK.format(layer=1),
        grid=(3, 3),
        seed=11,
        question_count=len(train_ids),
        training_question_ids=train_ids,
        activation_sha256=_sha256_file(activation_path),
        model_activation_dtype="synthetic float32",
    )
    save_artifact(artifact_path, artifact)
    reloaded = load_artifact(artifact_path)
    control_bases = fit_control_bases(deltas, reloaded.basis, seed=11)
    energy = _energy_summary(
        _activation_deltas_by_condition(activation_data, "test"),
        {"som": reloaded.basis, **control_bases},
    )
    result = {
        "schema_version": 1,
        "mode": "synthetic_offline_smoke",
        "model_loaded": False,
        "scientific_claim": "none; generated vectors are synthetic test data",
        "prompt_stats": stats,
        "train_questions": len(train_ids),
        "prototype_count": len(reloaded.prototypes),
        "basis_rank": reloaded.basis.shape[1],
        "test_projection_energy": energy,
    }
    temporary = metrics_path.with_name(metrics_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    return {
        "out_dir": str(out_dir.resolve()),
        "prompt_records": stats["records"],
        "train_questions": len(train_ids),
        "prototype_count": len(reloaded.prototypes),
        "basis_rank": int(reloaded.basis.shape[1]),
        "model_loaded": False,
        "metrics_file": str(metrics_path.resolve()),
    }


def _run_collect(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_prompt_jsonl(args.prompts)
    if args.layer < 0:
        raise ValueError("layer must be a non-negative zero-based index")
    if args.dry_run:
        return {
            "status": "dry_run",
            "records": len(rows),
            "model": args.model,
            "revision": args.revision or "adapter default",
            "layer": args.layer,
            "device": args.device,
            "output": str(Path(args.out).resolve()),
            "model_loaded": False,
        }
    data = collect_activations(rows, model_id=args.model, revision=args.revision, layer=args.layer, device=args.device)
    save_activations(args.out, data)
    return {
        "status": "complete",
        "records": len(rows),
        "model": data.metadata["model_id"],
        "revision": data.metadata["revision"],
        "layer": data.metadata["layer"],
        "hook": data.metadata["hook"],
        "hidden_size": data.metadata["hidden_size"],
        "model_activation_dtype": data.metadata["model_activation_dtype"],
        "activation_file": str(Path(args.out).resolve()),
        "next_step": f"python -m research.som_pipeline fit --activations {args.out} --out research/artifacts/som.npz",
    }


def _run_fit(args: argparse.Namespace) -> dict[str, Any]:
    data = load_activations(args.activations)
    train_ids, deltas = pressure_deltas(data, split="train")
    if args.dry_run:
        if args.rows < 1 or args.cols < 1 or args.epochs < 1:
            raise ValueError("SOM rows, columns, and epochs must be positive")
        if not 0 < args.learning_rate <= 1:
            raise ValueError("SOM learning-rate must be in (0, 1]")
        return {
            "status": "dry_run",
            "model": data.metadata.get("model_id", "unknown"),
            "layer": data.metadata.get("layer", "unknown"),
            "train_questions": len(train_ids),
            "delta_shape": list(deltas.shape),
            "grid": [args.rows, args.cols],
            "epochs": args.epochs,
            "seed": args.seed,
            "artifact_output": str(Path(args.out).resolve()),
            "artifact_written": False,
        }
    prototypes = train_som(
        deltas,
        rows=args.rows,
        cols=args.cols,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    metadata = data.metadata
    artifact = make_artifact(
        prototypes,
        model_id=str(metadata.get("model_id", "")),
        revision=str(metadata.get("revision", "")),
        layer=int(metadata.get("layer", -1)),
        hook=str(metadata.get("hook", "")),
        grid=(args.rows, args.cols),
        seed=args.seed,
        question_count=len(train_ids),
        training_question_ids=train_ids,
        activation_sha256=_sha256_file(args.activations),
        model_activation_dtype=str(metadata.get("model_activation_dtype", "unknown")),
    )
    save_artifact(args.out, artifact)
    return {
        "status": "complete",
        "model": artifact.metadata["model_id"],
        "revision": artifact.metadata["revision"],
        "layer": artifact.metadata["layer"],
        "hook": artifact.metadata["hook"],
        "train_questions": len(train_ids),
        "prototype_count": len(prototypes),
        "basis_rank": int(artifact.basis.shape[1]),
        "artifact": str(Path(args.out).resolve()),
        "next_step": f"python -m research.som_pipeline evaluate --prompts <paired.jsonl> --activations {args.activations} --artifact {args.out} --out <evaluation.json>",
    }


def _run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_prompt_jsonl(args.prompts)
    data = load_activations(args.activations)
    artifact = load_artifact(args.artifact)
    if artifact.metadata["activation_sha256"] != _sha256_file(args.activations):
        raise ValueError("activation file checksum does not match the SOM artifact; use the original collected file")
    if args.dry_run:
        train_ids, train_deltas, eval_questions = preflight_evaluation_inputs(
            rows, data, artifact, split=args.split, max_questions=args.max_questions
        )
        controls = fit_control_bases(train_deltas, artifact.basis, seed=int(artifact.metadata.get("seed", 0)))
        return {
            "status": "dry_run",
            "model": artifact.metadata["model_id"],
            "revision": artifact.metadata["revision"],
            "layer": artifact.metadata["layer"],
            "split": args.split,
            "train_questions": len(train_ids),
            "heldout_questions": len(eval_questions),
            "score_mode": args.score_mode,
            "methods": ["baseline", "som_remove", "global_mean_remove", "pca_remove", "random_remove"],
            "basis_ranks": {
                "som": artifact.basis.shape[1],
                **{name: basis.shape[1] for name, basis in controls.items()},
            },
            "model_loaded": False,
            "evaluation_output": str(Path(args.out).resolve()),
        }
    result = evaluate(
        rows,
        data,
        artifact,
        split=args.split,
        device=args.device,
        score_mode=args.score_mode,
        model_override=args.model,
        revision_override=args.revision,
        max_questions=args.max_questions,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    summary = result["score_summary"]
    baseline_false = summary["conditions"]["baseline"]["false_pressure"]
    return {
        "status": "complete",
        "split": args.split,
        "questions": result["question_count"],
        "model": result["model_id"],
        "revision": result["revision"],
        "layer": result["layer"],
        "score_mode": result["score_mode"],
        "baseline_false_pressure_margin": baseline_false["mean_target_minus_foil"],
        "methods": result["methods"],
        "evaluation_file": str(out_path.resolve()),
        "interpretation": "descriptive held-out diagnostics; no scientific-success claim is encoded",
    }


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "validate-prompts":
        stats = validate_prompt_rows(load_prompt_jsonl(args.prompts))
        return {
            "status": "valid",
            "records": stats["records"],
            "questions": stats["questions"],
            "train_questions": stats["questions_by_split"]["train"],
            "validation_questions": stats["questions_by_split"]["validation"],
            "test_questions": stats["questions_by_split"]["test"],
            "conditions": list(CONDITIONS),
        }
    if args.command == "validate-artifact":
        artifact = load_artifact(args.artifact)
        return {
            "status": "valid",
            "model": artifact.metadata["model_id"],
            "revision": artifact.metadata["revision"],
            "layer": artifact.metadata["layer"],
            "hook": artifact.metadata["hook"],
            "hidden_size": artifact.metadata["hidden_size"],
            "basis_rank": artifact.metadata["basis_rank"],
        }
    if args.command == "collect":
        return _run_collect(args)
    if args.command == "fit":
        return _run_fit(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "smoke":
        result = run_synthetic_smoke(args.out)
        return {"status": "complete", "mode": "synthetic_offline_smoke", **result}
    raise ValueError(f"unknown command {args.command!r}")


def _status() -> dict[str, Any]:
    example = Path("research/examples/paired_prompts.jsonl")
    return {
        "tool": "neuron-explorer-som",
        "status": "ready" if example.exists() else "ready; example dataset missing from current directory",
        "description": "Paired residual SOM collection and held-out evaluation",
        "default_model": DEFAULT_MODEL,
        "example_prompts": str(example),
        "commands": ["validate-prompts", "collect", "fit", "evaluate", "validate-artifact", "smoke"],
        "next_steps": [
            "python -m research.som_pipeline validate-prompts --prompts research/examples/paired_prompts.jsonl",
            "python -m research.som_pipeline smoke --out /tmp/neuron-explorer-som-smoke",
        ],
    }


def _usage_error(error: CLIUsageError) -> int:
    message = error.message
    if message.startswith("unrecognized arguments:"):
        detail = "unknown flag or argument: " + message.partition(":")[2].strip()
    elif "required" in message:
        detail = "missing required argument: " + message.replace("argument ", "", 1)
    else:
        detail = message.replace("\n", " ")[:300]
    emit_toon({"error": detail, "help": f"{error.parser.prog} --help"})
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 1 and argv[0] in {"--version", "-v", "-V"}:
        print(__version__)
        return 0
    parser = build_parser()
    if not argv:
        emit_toon(_status())
        return 0
    try:
        args = parser.parse_args(list(argv))
    except HelpExit as help_exit:
        print(help_exit.message, end="" if help_exit.message.endswith("\n") else "\n")
        return 0
    except CLIUsageError as usage_error:
        return _usage_error(usage_error)
    if args.command is None:
        emit_toon(_status())
        return 0
    try:
        result = run_command(args)
    except (DataError, ArtifactError, ValueError, OSError) as exc:
        detail = str(exc).replace("\n", " ")[:500]
        emit_toon({"error": detail, "help": f"python -m research.som_pipeline {args.command} --help"})
        return 1
    except Exception as exc:  # noqa: BLE001 - final CLI boundary emits bounded structured errors
        detail = str(exc).replace("\n", " ")[:500]
        emit_toon({
            "error": f"{args.command} failed: {detail or exc.__class__.__name__}",
            "help": f"python -m research.som_pipeline {args.command} --help",
        })
        return 1
    emit_toon(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
