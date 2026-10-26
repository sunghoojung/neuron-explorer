"""Bounded residual-stream capture and candidate-subspace scoring helpers.

These helpers borrow the explorer's already-loaded runtime and add temporary
forward hooks for one serialized pass at a time. They do not load a model,
change runtime capture flags, or persist inputs or activations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

MAX_INPUT_TOKENS = 512
MAX_SAMPLE_COUNT = 512
MAX_SAMPLE_MESSAGES = 64
MAX_SAMPLE_CHARACTERS = 65536
MAX_CAPTURE_WINDOW = 128
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_BASIS_RANK = 64
_CAPTURE_POINTS = ("resid_pre", "resid_mid", "resid_post", "attn_write", "mlp_write")
_PROJECTION_POINTS = ("resid_pre", "resid_mid", "resid_post")
_SCORE_MODES = ("sequence_logprob", "mean_token_logprob")
_VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}


class ResidualCaptureCancelled(RuntimeError):
    """Raised when a caller cancels between bounded local model passes."""


def _check_cancelled(cancel_event: Any | None) -> None:
    if cancel_event is not None and callable(getattr(cancel_event, "is_set", None)) and cancel_event.is_set():
        raise ResidualCaptureCancelled("Residual capture was cancelled.")


def _messages(sample: Any) -> list[dict[str, str]]:
    if not isinstance(sample, Mapping):
        raise TypeError("Each sample must be an object containing messages.")
    raw_messages = sample.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages or len(raw_messages) > MAX_SAMPLE_MESSAGES:
        raise ValueError(f"Each sample must contain 1 to {MAX_SAMPLE_MESSAGES} messages.")
    messages: list[dict[str, str]] = []
    total_characters = 0
    for message in raw_messages:
        if not isinstance(message, Mapping):
            raise TypeError("Message entries must be objects.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in _VALID_ROLES or not isinstance(content, str):
            raise TypeError("Messages need a supported role and text content.")
        total_characters += len(content)
        if total_characters > MAX_SAMPLE_CHARACTERS:
            raise ValueError("A sample exceeds the local text-size limit.")
        messages.append({"role": str(role), "content": content})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("Each sample needs at least one user message.")
    if messages[-1]["role"] == "assistant":
        raise ValueError("The final message must precede the assistant response.")
    return messages


def _runtime_parts(rt: Any) -> tuple[Any, Any, torch.device]:
    ensure_loaded = getattr(rt, "ensure_loaded", None)
    if not callable(ensure_loaded):
        raise TypeError("The shared explorer runtime cannot load its configured model.")
    ensure_loaded()
    adapter = getattr(rt, "adapter", None)
    decoder = getattr(rt, "lm", None)
    if adapter is None or decoder is None:
        raise RuntimeError("The shared explorer runtime has no loaded decoder adapter.")
    if getattr(adapter, "family", None) != "llama":
        family = getattr(adapter, "family", "unknown")
        raise ValueError(f"Residual capture currently supports Llama decoders; the loaded family is {family!r}.")
    if not hasattr(rt, "lock") or not hasattr(adapter, "layers"):
        raise RuntimeError("The shared explorer runtime is missing its decoder lock or layer adapter.")
    return adapter, decoder, torch.device(getattr(rt, "dev", getattr(adapter, "device", "cpu")))


def _hidden(value: Any, *, source: str) -> torch.Tensor:
    candidate = value
    if not torch.is_tensor(candidate) and isinstance(candidate, (tuple, list)) and candidate:
        candidate = candidate[0]
    if not torch.is_tensor(candidate) or candidate.ndim != 3:
        raise RuntimeError(f"The {source} hook did not receive a [batch, sequence, hidden] tensor.")
    return candidate


def _config_hidden_size(adapter: Any) -> int:
    configs = (getattr(adapter, "text_config", None), getattr(adapter, "config", None))
    for config in configs:
        value = getattr(config, "hidden_size", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    metadata = getattr(adapter, "config_metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("hidden_size")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise RuntimeError("The loaded model does not report a valid hidden size.")


def _validate_layers(adapter: Any, layers: Sequence[int]) -> list[int]:
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence) or not layers:
        raise ValueError("layers must be a non-empty sequence of zero-based layer indices.")
    count = len(adapter.layers)
    result: list[int] = []
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < count:
            raise ValueError(f"Every layer must be an integer from 0 to {count - 1}.")
        if layer in result:
            raise ValueError("layers must not contain duplicates.")
        result.append(layer)
    return result


def _validate_points(points: Sequence[str]) -> list[str]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence) or not points:
        raise ValueError("points must be a non-empty sequence of supported residual points.")
    result: list[str] = []
    for point in points:
        if not isinstance(point, str) or point not in _CAPTURE_POINTS:
            raise ValueError(f"Unsupported residual point {point!r}; choose from {', '.join(_CAPTURE_POINTS)}.")
        if point in result:
            raise ValueError("points must not contain duplicates.")
        result.append(point)
    return result


def _template_sha256(adapter: Any) -> str | None:
    tokenizer = getattr(adapter, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return None
    try:
        serialized = json.dumps(template, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        serialized = repr(template)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _token_ids(adapter: Any, messages: list[dict[str, str]]) -> list[int]:
    apply_template = getattr(adapter, "apply_chat_template", None)
    if not callable(apply_template):
        raise TypeError("The model adapter does not expose chat-template tokenization.")
    encoded = apply_template(messages, add_generation_prompt=True)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if torch.is_tensor(encoded):
        encoded = encoded.detach().cpu().tolist()
    if not isinstance(encoded, (tuple, list)):
        raise TypeError("The model chat template did not return token IDs.")
    if encoded and isinstance(encoded[0], (tuple, list)):
        if len(encoded) != 1:
            raise RuntimeError("The model chat template returned more than one sequence.")
        encoded = encoded[0]
    token_ids = []
    for token_id in encoded:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("The model chat template returned an invalid token ID.")
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("The model chat template returned a negative token ID.")
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError("A sample tokenized to an empty prompt.")
    if len(token_ids) > MAX_INPUT_TOKENS:
        raise ValueError(f"A prompt exceeds the {MAX_INPUT_TOKENS}-token model-input limit.")
    return token_ids


def _decode_token(tokenizer: Any, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return str(token_id)
    try:
        return str(decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    except TypeError:
        return str(decode([token_id], skip_special_tokens=False))


def _point_module(block: Any, point: str) -> tuple[Any, str]:
    if point == "resid_pre":
        return block, "pre"
    if point == "resid_post":
        return block, "post"
    if point == "resid_mid":
        module = getattr(block, "post_attention_layernorm", None)
        if module is None:
            module = getattr(block, "pre_feedforward_layernorm", None)
        if module is None:
            raise ValueError("The selected Llama layer does not expose its post-attention norm input.")
        return module, "pre"
    if point == "attn_write":
        attention = getattr(block, "self_attn", None)
        module = getattr(attention, "o_proj", None) if attention is not None else None
        if module is None:
            raise ValueError("The selected Llama layer does not expose self_attn.o_proj.")
        return module, "post"
    if point == "mlp_write":
        mlp = getattr(block, "mlp", None)
        module = getattr(mlp, "down_proj", None) if mlp is not None else None
        if module is None:
            raise ValueError("The selected Llama layer does not expose mlp.down_proj.")
        return module, "post"
    raise ValueError(f"Unsupported residual point {point!r}.")


def _install_capture_hooks(
    adapter: Any,
    layers: list[int],
    points: list[str],
    window: int,
    current: dict[str, Any],
    dtypes: set[str],
) -> list[Any]:
    handles: list[Any] = []

    def save(layer: int, point: str, value: Any, source: str) -> None:
        key = (layer, point)
        values = current["values"]
        if key in values:
            raise RuntimeError(f"The {point} hook fired more than once for layer {layer}.")
        hidden = _hidden(value, source=source)
        if hidden.shape[0] != 1:
            raise RuntimeError("Residual capture requires a single prompt per model pass.")
        dtypes.add(str(hidden.dtype))
        captured = hidden[0, -window:, :].detach().to(device="cpu", dtype=torch.float32).contiguous()
        values[key] = captured.numpy().copy()

    try:
        for layer_index in layers:
            block = adapter.layers[layer_index]
            for point in points:
                module, location = _point_module(block, point)
                if location == "pre":
                    def pre_hook(_module: Any, inputs: tuple[Any, ...], layer: int = layer_index, name: str = point) -> None:
                        if not inputs:
                            raise RuntimeError(f"The {name} hook received no inputs.")
                        save(layer, name, inputs[0], name)

                    handles.append(module.register_forward_pre_hook(pre_hook))
                else:
                    def post_hook(
                        _module: Any,
                        _inputs: tuple[Any, ...],
                        output: Any,
                        layer: int = layer_index,
                        name: str = point,
                    ) -> None:
                        save(layer, name, output, name)

                    handles.append(module.register_forward_hook(post_hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles


def _assert_runtime_idle(rt: Any) -> None:
    active = ("_push", "_heads", "_read", "_capture_all", "_capture_full")
    if any(getattr(rt, name, None) is not None for name in active):
        raise RuntimeError("The shared runtime has an active capture or intervention state.")


def capture_residuals(
    rt: Any,
    samples: Sequence[Mapping[str, Any]],
    layers: Sequence[int],
    points: Sequence[str],
    window: int = 8,
    progress: Callable[[str, int, int], Any] | None = None,
    cancel_event: Any | None = None,
) -> dict[str, Any]:
    """Capture right-aligned final prompt-token windows at selected Llama points.

    Prompt tokenization uses the shared adapter with ``add_generation_prompt=True``.
    The input contains the assistant generation prefix but no continuation; every
    captured position is therefore causal and sees no future answer tokens.
    """
    _check_cancelled(cancel_event)
    adapter, decoder, device = _runtime_parts(rt)
    selected_layers = _validate_layers(adapter, layers)
    selected_points = _validate_points(points)
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= MAX_CAPTURE_WINDOW:
        raise ValueError(f"window must be from 1 to {MAX_CAPTURE_WINDOW} tokens.")
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise ValueError("samples must be a non-empty sequence.")
    if len(samples) > MAX_SAMPLE_COUNT:
        raise ValueError(f"Capture is limited to {MAX_SAMPLE_COUNT} samples per job.")
    hidden_size = _config_hidden_size(adapter)
    estimated_bytes = len(samples) * len(selected_layers) * len(selected_points) * window * hidden_size * 4
    if estimated_bytes > MAX_CAPTURE_BYTES:
        raise ValueError("The requested residual capture exceeds the 256 MiB output limit.")

    messages_by_sample: list[list[dict[str, str]]] = []
    token_ids_by_sample: list[list[int]] = []
    for index, sample in enumerate(samples):
        _check_cancelled(cancel_event)
        messages = _messages(sample)
        token_ids = _token_ids(adapter, messages)
        messages_by_sample.append(messages)
        token_ids_by_sample.append(token_ids)
        if progress is not None:
            progress("tokenizing", index + 1, len(samples))

    vectors = np.zeros(
        (len(samples), len(selected_layers), len(selected_points), window, hidden_size),
        dtype=np.float32,
    )
    token_matrix = np.full((len(samples), window), -1, dtype=np.int64)
    position_matrix = np.full((len(samples), window), -1, dtype=np.int64)
    valid = np.zeros((len(samples), window), dtype=np.bool_)
    token_text: list[list[str | None]] = [[None] * window for _ in samples]
    lengths: list[int] = []
    observed_dtypes: set[str] = set()
    current: dict[str, Any] = {"values": {}}
    handles: list[Any] = []

    with rt.lock:
        _assert_runtime_idle(rt)
        try:
            handles = _install_capture_hooks(
                adapter,
                selected_layers,
                selected_points,
                window,
                current,
                observed_dtypes,
            )
            with torch.inference_mode():
                for sample_index, token_ids in enumerate(token_ids_by_sample):
                    _check_cancelled(cancel_event)
                    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    current["values"] = {}
                    decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    values = current["values"]
                    expected_keys = {
                        (layer_index, point)
                        for layer_index in selected_layers
                        for point in selected_points
                    }
                    if set(values) != expected_keys:
                        missing = len(expected_keys - set(values))
                        raise RuntimeError(f"Residual hooks did not capture all requested points ({missing} missing).")
                    token_count = len(token_ids)
                    visible_count = min(window, token_count)
                    left_pad = window - visible_count
                    lengths.append(token_count)
                    token_matrix[sample_index, left_pad:] = token_ids[-visible_count:]
                    position_matrix[sample_index, left_pad:] = np.arange(token_count - visible_count, token_count)
                    valid[sample_index, left_pad:] = True
                    tokenizer = getattr(adapter, "tokenizer", getattr(rt, "tok", None))
                    token_text[sample_index][left_pad:] = [
                        _decode_token(tokenizer, token_id) for token_id in token_ids[-visible_count:]
                    ]
                    for layer_position, layer_index in enumerate(selected_layers):
                        for point_position, point in enumerate(selected_points):
                            value = values[(layer_index, point)]
                            if value.shape != (visible_count, hidden_size):
                                raise RuntimeError(
                                    f"Captured {point} shape {value.shape} does not match the configured hidden size."
                                )
                            vectors[sample_index, layer_position, point_position, left_pad:] = value
                    if progress is not None:
                        progress("capturing", sample_index + 1, len(samples))
        finally:
            for handle in handles:
                handle.remove()

    config = getattr(adapter, "config", None)
    model_id = getattr(adapter, "model_id", getattr(rt, "model_id", None))
    revision = getattr(adapter, "revision", None) or getattr(config, "_commit_hash", None)
    return {
        "vectors": vectors,
        "token_ids": token_matrix,
        "positions": position_matrix,
        "valid": valid,
        "tokens": token_text,
        "lengths": lengths,
        "metadata": {
            "schema_version": 1,
            "model_id": str(model_id) if model_id is not None else None,
            "revision": str(revision) if revision is not None else None,
            "family": adapter.family,
            "device": str(device),
            "hidden_size": hidden_size,
            "layers": selected_layers,
            "points": selected_points,
            "window": window,
            "capture_dtype": "float32",
            "model_activation_dtypes": sorted(observed_dtypes),
            "template_sha256": _template_sha256(adapter),
            "tokenization": "shared adapter chat template with assistant generation prefix",
            "token_position": "right-aligned final prompt tokens; positions are zero-based in the full prompt",
            "causal_context": "prompt only with assistant generation prefix; no continuation is supplied",
            "record_count": len(samples),
            "estimated_capture_bytes": estimated_bytes,
        },
    }


def _basis_array(basis: Any, hidden_size: int) -> np.ndarray:
    if torch.is_tensor(basis):
        basis = basis.detach().to(device="cpu", dtype=torch.float32).numpy()
    values = np.asarray(basis, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or values.shape[0] != hidden_size
        or not 1 <= values.shape[1] <= min(MAX_BASIS_RANK, hidden_size)
        or not np.isfinite(values).all()
    ):
        raise ValueError("basis must be a finite [hidden_size, rank] matrix with a bounded rank.")
    gram = values.T @ values
    if not np.allclose(gram, np.eye(values.shape[1], dtype=np.float32), rtol=1e-3, atol=1e-3):
        raise ValueError("basis columns must be orthonormal.")
    return values


def _project_hidden(hidden: torch.Tensor, basis: torch.Tensor, alpha: float, boundary: int) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise RuntimeError("Residual projection requires a single [batch, sequence, hidden] tensor.")
    if basis.shape[0] != hidden.shape[-1]:
        raise RuntimeError("Projection basis hidden size does not match the residual tensor.")
    values = hidden.float()
    candidate_component = (values @ basis) @ basis.T
    projected = values + alpha * candidate_component
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    answer_path = positions >= boundary
    result = torch.where(answer_path[None, :, None], projected, values)
    return result.to(dtype=hidden.dtype)


def _replace_hidden(container: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(container):
        return replacement
    if isinstance(container, tuple):
        return (replacement, *container[1:])
    if isinstance(container, list):
        return [replacement, *container[1:]]
    raise RuntimeError("The residual hook output has an unsupported structure.")


def score_residual_continuation(
    rt: Any,
    sample: Mapping[str, Any],
    layer: int,
    point: str,
    basis: Any | None = None,
    alpha: float = 0.0,
    score_mode: str = "sequence_logprob",
    continuation: str | None = None,
) -> dict[str, Any]:
    """Score a continuation with an optional projection on the answer path.

    The stable chat-template prefix is required. Projection changes only the
    selected residual at positions beginning with the final prompt token, the
    state used to predict the first continuation token. Use ``alpha=-1`` to
    remove the candidate component, ``0`` for the deterministic baseline, and
    ``1`` to double it.
    """
    adapter, decoder, device = _runtime_parts(rt)
    selected_layers = _validate_layers(adapter, [layer])
    if point not in _PROJECTION_POINTS:
        raise ValueError(f"Residual scoring supports points {', '.join(_PROJECTION_POINTS)}.")
    if score_mode not in _SCORE_MODES:
        raise ValueError(f"score_mode must be one of {', '.join(_SCORE_MODES)}.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be a finite number from -1 to 1.")
    alpha = float(alpha)
    if not -1.0 <= alpha <= 1.0:
        raise ValueError("alpha must be a finite number from -1 to 1.")
    messages = _messages(sample)
    prompt_ids = _token_ids(adapter, messages)
    if continuation is None:
        continuation = sample.get("target")
    if not isinstance(continuation, str) or not continuation:
        raise ValueError("A non-empty continuation is required, either explicitly or as sample.target.")
    if len(continuation) > MAX_SAMPLE_CHARACTERS:
        raise ValueError("The continuation exceeds the local text-size limit.")
    render_template = getattr(adapter, "render_chat_template", None)
    if not callable(render_template):
        raise TypeError("The model adapter does not expose chat-template rendering.")
    rendered_prompt = render_template(messages, add_generation_prompt=True)
    if not isinstance(rendered_prompt, str):
        raise TypeError("The model chat template did not render text.")
    tokenizer = getattr(adapter, "tokenizer", getattr(rt, "tok", None))
    if not callable(tokenizer):
        raise TypeError("The shared runtime tokenizer cannot encode continuations.")
    encoded_full = tokenizer(rendered_prompt + continuation, add_special_tokens=False)
    full_ids = encoded_full.get("input_ids") if isinstance(encoded_full, Mapping) else encoded_full
    if torch.is_tensor(full_ids):
        full_ids = full_ids.detach().cpu().tolist()
    if full_ids and isinstance(full_ids[0], (tuple, list)):
        if len(full_ids) != 1:
            raise RuntimeError("The model tokenizer returned more than one continuation sequence.")
        full_ids = full_ids[0]
    if not isinstance(full_ids, (tuple, list)):
        raise TypeError("The model tokenizer did not return continuation token IDs.")
    normalized_full_ids = []
    for token_id in full_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise TypeError("The model tokenizer returned an invalid continuation token ID.")
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("The model tokenizer returned a negative continuation token ID.")
        normalized_full_ids.append(token_id)
    full_ids = normalized_full_ids
    if not prompt_ids or len(full_ids) <= len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt and continuation tokenization is not prefix-stable; adjust the continuation spacing or prompt boundary."
        )
    if len(full_ids) > MAX_INPUT_TOKENS:
        raise ValueError(f"Prompt and continuation exceed the {MAX_INPUT_TOKENS}-token model-input limit.")
    target_count = len(full_ids) - len(prompt_ids)
    if target_count > MAX_INPUT_TOKENS:
        raise ValueError("The continuation exceeds the local token limit.")
    hidden_size = _config_hidden_size(adapter)
    basis_values: np.ndarray | None = None
    if basis is not None:
        basis_values = _basis_array(basis, hidden_size)
    elif alpha != 0.0:
        raise ValueError("A non-zero alpha requires a candidate basis.")
    block = adapter.layers[selected_layers[0]]
    module, location = _point_module(block, point)
    boundary = len(prompt_ids) - 1
    basis_tensor = (
        torch.as_tensor(basis_values, dtype=torch.float32, device=device)
        if basis_values is not None and alpha != 0.0
        else None
    )

    def transform(hidden_value: Any) -> Any:
        hidden = _hidden(hidden_value, source=point)
        if hidden.shape[-1] != hidden_size or hidden.shape[1] != len(full_ids):
            raise RuntimeError("Residual projection received an unexpected hidden shape.")
        if basis_tensor is None:
            return hidden_value
        return _replace_hidden(hidden_value, _project_hidden(hidden, basis_tensor, alpha, boundary))

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    start = boundary
    stop = len(full_ids) - 1
    target_ids = input_ids[0, len(prompt_ids) :]
    handle = None
    try:
        with rt.lock:
            _assert_runtime_idle(rt)
            if location == "pre":
                def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                    if not inputs:
                        raise RuntimeError(f"The {point} projection hook received no inputs.")
                    return (transform(inputs[0]), *inputs[1:])

                handle = module.register_forward_pre_hook(pre_hook)
            else:
                def post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                    return transform(output)

                handle = module.register_forward_hook(post_hook)
            with torch.inference_mode():
                output = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None and isinstance(output, (tuple, list)) and output:
                    hidden = output[0]
                if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape != (1, len(full_ids), hidden_size):
                    raise RuntimeError("The shared decoder did not return its final hidden states.")
                lm_head = getattr(adapter, "lm_head", None)
                if lm_head is None:
                    raise RuntimeError("The model adapter does not expose its language-model head.")
                token_logits = lm_head(hidden[0, start:stop])
                if not torch.is_tensor(token_logits) or token_logits.ndim != 2 or token_logits.shape[0] != target_count:
                    raise RuntimeError("The model returned an unexpected number of continuation positions.")
                token_logits = token_logits.float()
                if not torch.isfinite(token_logits).all().item():
                    raise FloatingPointError("The model produced non-finite logits while scoring the continuation.")
                token_logprobs = F.log_softmax(token_logits, dim=-1).gather(
                    1, target_ids[:, None]
                ).squeeze(1)
                if not torch.isfinite(token_logprobs).all().item():
                    raise FloatingPointError("The model produced non-finite token log-probabilities while scoring the continuation.")
                total_logprob = float(token_logprobs.sum().item())
                score = total_logprob if score_mode == "sequence_logprob" else total_logprob / target_count
                if not math.isfinite(score):
                    raise FloatingPointError("The model produced a non-finite continuation score.")
    finally:
        if handle is not None:
            handle.remove()

    return {
        "score": score,
        "token_count": target_count,
        "score_mode": score_mode,
        "layer": selected_layers[0],
        "point": point,
        "alpha": alpha,
        "basis_rank": int(basis_values.shape[1]) if basis_values is not None else 0,
        "boundary_token_index": boundary,
        "boundary_policy": "apply at the final prompt token and later answer-path positions",
    }
