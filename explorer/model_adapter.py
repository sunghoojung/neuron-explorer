"""Shared model loading and architecture access for the neuron explorer.

The explorer currently supports Llama causal language models and Qwen3.5's
conditional-generation wrapper. The adapter keeps the decoder-facing API
consistent while preserving each model family's tokenizer and normalization
behavior.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"


def _resolved_model_id(model_id: str | None = None) -> str:
    return str(model_id or os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL)).strip()


def resolve_model_revision(model_id: str | None = None, revision: str | None = None) -> str | None:
    """Resolve the explicit revision, environment override, or pinned default."""
    if revision:
        return str(revision).strip()
    env_revision = os.environ.get("EXPLORER_REVISION") or os.environ.get("EXPLORER_MODEL_REVISION")
    if env_revision:
        return env_revision.strip()
    if _resolved_model_id(model_id) == DEFAULT_MODEL:
        return DEFAULT_REVISION
    return None


def model_cache_namespace(model_id: str | None = None, revision: str | None = None) -> str:
    """Return a filesystem-safe, stable namespace for one model and revision.

    This function does not inspect the Hugging Face cache or download config or
    weights, so builders can use it before the model is loaded.
    """
    resolved_id = _resolved_model_id(model_id)
    resolved_revision = resolve_model_revision(resolved_id, revision) or "default"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved_id).strip("-._").lower() or "model"
    digest = hashlib.sha256(f"{resolved_id}\0{resolved_revision}".encode()).hexdigest()[:12]
    return f"{slug[:48]}-{digest}"


def infer_model_family(model_id: str, config: Any | None = None) -> str:
    """Identify supported model families from config first, then the model ID."""
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [str(a).lower() for a in (getattr(config, "architectures", None) or [])]
    descriptor = " ".join([model_type, *architectures, model_id.lower()])
    if "qwen3_5" in descriptor or "qwen3.5" in descriptor:
        return "qwen3_5"
    if "llama" in descriptor:
        return "llama"
    return model_type or "causal_lm"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Choose an explicit device or available Apple MPS, with CPU as fallback."""
    requested = device or os.environ.get("EXPLORER_DEVICE")
    if requested:
        resolved = torch.device(requested)
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable; choose EXPLORER_DEVICE=cpu or use an MPS-capable Mac")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; choose EXPLORER_DEVICE=cpu or run in a CUDA allocation")
        return resolved
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except (AttributeError, RuntimeError):
            pass
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _normalized_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    values = value if isinstance(value, (tuple, list, set)) else [value]
    ids = set()
    for item in values:
        if item is None:
            continue
        try:
            ids.add(int(item))
        except (TypeError, ValueError):
            continue
    return ids


def _model_config(config: Any) -> Any:
    """Use text config for multimodal wrappers when it is available."""
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config()
        except (TypeError, ValueError):
            pass
    text_config = getattr(config, "text_config", None)
    return text_config or config


def _config_summary(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    text_config = _model_config(config)
    summary: dict[str, Any] = {}
    for name in (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "layer_types",
    ):
        value = getattr(text_config, name, None)
        if value is not None:
            if isinstance(value, tuple):
                value = list(value)
            summary[name] = value
    return summary


def cached_config_metadata(model_id: str | None = None, revision: str | None = None) -> dict[str, Any] | None:
    """Read config metadata only from the local Transformers cache."""
    resolved_id = _resolved_model_id(model_id)
    resolved_revision = resolve_model_revision(resolved_id, revision)
    kwargs: dict[str, Any] = {"local_files_only": True}
    if resolved_revision:
        kwargs["revision"] = resolved_revision
    try:
        config = AutoConfig.from_pretrained(resolved_id, **kwargs)
    except (OSError, ValueError, RuntimeError):
        return None
    return _config_summary(config)


class ModelAdapter:
    """Expose decoder blocks and model-specific behavior through one interface."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        revision: str | None = None,
        config: Any | None = None,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = _resolved_model_id(model_id)
        self.revision = resolve_model_revision(self.model_id, revision)
        self.config = config or getattr(model, "config", None)
        self.text_config = _model_config(self.config) if self.config is not None else None
        self.family = infer_model_family(self.model_id, self.config)
        self.cache_namespace = model_cache_namespace(self.model_id, self.revision)

        self.language_model, self.model_prefix = self._find_language_model(model)
        self.layers = getattr(self.language_model, "layers", None)
        if self.layers is None:
            raise ValueError(f"Could not find decoder layers for model family {self.family!r}")
        self.norm = getattr(self.language_model, "norm", None)
        if self.norm is None:
            raise ValueError(f"Could not find the final decoder norm for model family {self.family!r}")
        self.lm_head = getattr(model, "lm_head", None)
        if self.lm_head is None:
            raise ValueError(f"Could not find lm_head for model family {self.family!r}")

        self.device = torch.device(device) if device is not None else self._model_device()
        self.eos_token_ids = _normalized_token_ids(getattr(tokenizer, "eos_token_id", None))
        self.eos_token_ids.update(_normalized_token_ids(getattr(self.config, "eos_token_id", None)))
        self.eos_token_ids.update(_normalized_token_ids(getattr(self.text_config, "eos_token_id", None)))
        generation_config = getattr(model, "generation_config", None)
        self.eos_token_ids.update(_normalized_token_ids(getattr(generation_config, "eos_token_id", None)))
        self.pad_token_id = self._pad_token_id()
        self.config_metadata = _config_summary(self.config)

        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

    @staticmethod
    def _find_language_model(model: Any) -> tuple[Any, str]:
        """Return the decoder and its model-relative attribute path."""
        candidates = (
            ("model.language_model", lambda: model.model.language_model),
            ("language_model", lambda: model.language_model),
            ("model", lambda: model.model),
            ("", lambda: model),
        )
        for prefix, getter in candidates:
            try:
                candidate = getter()
            except (AttributeError, TypeError):
                continue
            if getattr(candidate, "layers", None) is not None and getattr(candidate, "norm", None) is not None:
                return candidate, prefix
        raise ValueError("Could not resolve a decoder with layers and final norm")

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            return torch.device("cpu")

    def _pad_token_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.config, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.text_config, "pad_token_id", None)
        if pad is None:
            pad = getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)
        if pad is None and self.eos_token_ids:
            pad = min(self.eos_token_ids)
        return int(pad) if pad is not None else 0

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "family": self.family,
            "model_prefix": self.model_prefix,
            "cache_namespace": self.cache_namespace,
            "config": self.config_metadata,
            "device": str(self.device),
        }

    def down_col(self, layer: int, neuron: int) -> torch.Tensor:
        return self.layers[layer].mlp.down_proj.weight[:, neuron]

    def layer_norm_scale(self, layer: int) -> torch.Tensor:
        """Return the actual multiplicative scale used before the layer MLP.

        Llama RMSNorm stores its scale directly in ``weight``. Qwen3.5's
        offset RMSNorm stores a delta from one, so its effective scale is
        ``1 + weight``.
        """
        block = self.layers[layer]
        norm = getattr(block, "post_attention_layernorm", None)
        if norm is None:
            norm = getattr(block, "pre_feedforward_layernorm", None)
        if norm is None or getattr(norm, "weight", None) is None:
            raise ValueError(f"Could not find the MLP input norm for layer {layer}")
        weight = norm.weight.float()
        return weight + 1.0 if self.family == "qwen3_5" else weight

    def attention_projection(self, layer: int) -> tuple[Any, int] | None:
        """Return (output projection, number of input heads) if present."""
        block = self.layers[layer]
        attention = getattr(block, "self_attn", None)
        if attention is not None:
            projection = getattr(attention, "o_proj", None)
            head_dim = getattr(attention, "head_dim", None)
            if projection is not None and head_dim:
                return projection, projection.in_features // int(head_dim)
        linear_attention = getattr(block, "linear_attn", None)
        if linear_attention is not None:
            projection = getattr(linear_attention, "out_proj", None)
            heads = getattr(linear_attention, "num_v_heads", None)
            if projection is not None and heads:
                return projection, int(heads)
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            **self._chat_template_kwargs(tokenize=True, add_generation_prompt=add_generation_prompt),
        )
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    def render_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the model's chat template, optionally adding the reply prefix."""
        rendered = self.tokenizer.apply_chat_template(
            messages,
            **self._chat_template_kwargs(
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            ),
        )
        if isinstance(rendered, str):
            return rendered
        raise TypeError("Tokenizer chat template did not return rendered text")

    def _chat_template_kwargs(self, tokenize: bool, add_generation_prompt: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "tokenize": tokenize,
        }
        if self.family == "qwen3_5":
            kwargs["enable_thinking"] = False
        return kwargs

    @staticmethod
    def logits(output: Any) -> torch.Tensor:
        logits = getattr(output, "logits", None)
        if logits is not None:
            return logits
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError("Model output does not contain logits")

    def logits_to_keep_kwargs(self) -> dict[str, int]:
        """Use the model's supported last-logit optimization when explicit."""
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (AttributeError, TypeError, ValueError):
            return {}
        if "logits_to_keep" in parameters:
            return {"logits_to_keep": 1}
        if "num_logits_to_keep" in parameters:
            return {"num_logits_to_keep": 1}
        return {}

    def is_eos_token(self, token_id: int) -> bool:
        return int(token_id) in self.eos_token_ids


def load_model_and_tokenizer(
    model_id: str | None = None,
    revision: str | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    local_files_only: bool = True,
) -> tuple[Any, Any, ModelAdapter]:
    """Load the supported model/tokenizer pair and return its decoder adapter.

    Loading is local-cache-only by default to avoid implicit checkpoint downloads.
    """
    resolved_id = _resolved_model_id(model_id)
    resolved_revision = resolve_model_revision(resolved_id, revision)
    resolved_device = resolve_device(device)
    load_dtype = dtype or _default_dtype(resolved_device)

    load_kwargs: dict[str, Any] = {}
    if resolved_revision:
        load_kwargs["revision"] = resolved_revision
    load_kwargs["local_files_only"] = local_files_only
    config = AutoConfig.from_pretrained(resolved_id, **load_kwargs)
    family = infer_model_family(resolved_id, config)
    tokenizer = AutoTokenizer.from_pretrained(resolved_id, **load_kwargs)

    if family == "qwen3_5":
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "This Qwen3.5 checkpoint requires a Transformers version with "
                "Qwen3_5ForConditionalGeneration"
            ) from exc
        model_class = Qwen3_5ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        resolved_id,
        config=config,
        dtype=load_dtype,
        **load_kwargs,
    ).to(resolved_device).eval()
    adapter = ModelAdapter(
        model,
        tokenizer,
        resolved_id,
        revision=resolved_revision,
        config=config,
        device=resolved_device,
    )
    return model, tokenizer, adapter
