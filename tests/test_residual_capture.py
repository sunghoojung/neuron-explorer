"""Offline residual capture and projection coverage using tiny random Llama models."""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from explorer.model_adapter import ModelAdapter
from explorer.residual_capture import (
    MAX_INPUT_TOKENS,
    _project_hidden,
    capture_residuals,
    score_residual_continuation,
)


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"
    chat_template = "fixture-chat-template-v1"

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [(ord(character) % 29) + 1 for character in text]

    def _render(self, messages, add_generation_prompt: bool) -> str:
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def apply_chat_template(self, messages, **kwargs):
        rendered = self._render(messages, kwargs.get("add_generation_prompt", True))
        if kwargs.get("tokenize", True):
            return self._encode(rendered)
        return rendered

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        return {"input_ids": self._encode(text)}

    @staticmethod
    def decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(f"<{int(token_id)}>" for token_id in token_ids)


class TinyRuntime:
    def __init__(self, device: str = "cpu", family: str = "llama"):
        torch.manual_seed(19)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=MAX_INPUT_TOKENS + 32,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval().to(device)
        tokenizer = TinyTokenizer()
        adapter = ModelAdapter(
            model,
            tokenizer,
            "fixture/Llama-tiny",
            revision="fixture-revision",
            config=config,
            device=device,
        )
        adapter.family = family
        self.model = model
        self.lm = adapter.language_model
        self.tok = tokenizer
        self.adapter = adapter
        self.dev = torch.device(device)
        self.lock = threading.Lock()
        self._push = None
        self._heads = None
        self._read = None
        self._capture_all = None
        self._capture_full = None
        self._hook_handles = []

    def ensure_loaded(self):
        return None


def _sample(text: str = "A short factual question.") -> dict:
    return {
        "id": "fixture-sample",
        "pair_id": "fixture-pair",
        "family_id": "fixture-family",
        "split": "test",
        "group": "fixture-group",
        "messages": [{"role": "user", "content": text}],
        "target": " answer",
        "foil": " alternative",
    }


def _devices() -> list[str]:
    return ["cpu", *(["mps"] if torch.backends.mps.is_available() else [])]


def _hook_counts(runtime: TinyRuntime) -> dict[int, tuple[int, int]]:
    return {
        id(module): (len(module._forward_pre_hooks), len(module._forward_hooks))
        for module in runtime.lm.modules()
    }


@pytest.mark.parametrize("device", _devices())
def test_capture_conserves_llama_residual_updates_and_right_aligns_tokens(device):
    runtime = TinyRuntime(device)
    baseline_hooks = _hook_counts(runtime)
    points = ("resid_pre", "resid_mid", "resid_post", "attn_write", "mlp_write")
    result = capture_residuals(runtime, [_sample()], [0], points, window=32)

    assert result["vectors"].shape == (1, 1, len(points), 32, 16)
    assert result["vectors"].dtype == np.float32
    assert result["token_ids"].shape == result["positions"].shape == result["valid"].shape == (1, 32)
    assert result["valid"].dtype == np.bool_
    assert result["lengths"][0] > 0
    visible = int(result["valid"][0].sum())
    left_pad = 32 - visible
    assert visible == min(32, result["lengths"][0])
    assert np.all(result["token_ids"][0, :left_pad] == -1)
    assert np.all(result["positions"][0, :left_pad] == -1)
    assert result["tokens"][0][:left_pad] == [None] * left_pad
    np.testing.assert_array_equal(
        result["positions"][0, left_pad:],
        np.arange(result["lengths"][0] - visible, result["lengths"][0]),
    )
    assert result["tokens"][0][left_pad:][-1]
    assert result["metadata"]["model_id"] == "fixture/Llama-tiny"
    assert result["metadata"]["revision"] == "fixture-revision"
    assert result["metadata"]["hidden_size"] == 16
    assert result["metadata"]["template_sha256"]
    assert result["metadata"]["causal_context"].startswith("prompt only")
    assert baseline_hooks == _hook_counts(runtime)
    assert runtime._hook_handles == []

    by_point = {
        point: torch.from_numpy(result["vectors"][0, 0, index, left_pad:])
        for index, point in enumerate(points)
    }
    torch.testing.assert_close(
        by_point["resid_pre"] + by_point["attn_write"],
        by_point["resid_mid"],
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        by_point["resid_mid"] + by_point["mlp_write"],
        by_point["resid_post"],
        rtol=2e-4,
        atol=2e-4,
    )
    assert all(value.shape[-1] == 16 for value in by_point.values())
    assert all(value.shape[-1] != 24 for value in by_point.values())


def test_capture_rejects_oversize_prompt_before_decoder_forward():
    runtime = TinyRuntime()
    calls = 0
    original_forward = runtime.lm.forward

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(*args, **kwargs)

    runtime.lm.forward = counted_forward
    with pytest.raises(ValueError, match="512-token"):
        capture_residuals(runtime, [_sample("x" * (MAX_INPUT_TOKENS + 1))], [0], ["resid_pre"])
    assert calls == 0


def test_capture_removes_temporary_hooks_after_model_failure():
    runtime = TinyRuntime()
    baseline_hooks = _hook_counts(runtime)
    original_forward = runtime.lm.forward

    def failed_forward(*_args, **_kwargs):
        raise RuntimeError("fixture forward failure")

    runtime.lm.forward = failed_forward
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        capture_residuals(runtime, [_sample()], [0, 1], ["resid_pre", "resid_post"])
    assert baseline_hooks == _hook_counts(runtime)
    runtime.lm.forward = original_forward


def test_capture_rejects_unsupported_family_and_point():
    runtime = TinyRuntime(family="qwen3_5")
    with pytest.raises(ValueError, match="supports Llama"):
        capture_residuals(runtime, [_sample()], [0], ["resid_pre"])
    runtime.adapter.family = "llama"
    with pytest.raises(ValueError, match="Unsupported residual point"):
        capture_residuals(runtime, [_sample()], [0], ["unknown"])


def test_projection_is_limited_to_final_prompt_token_and_answer_path():
    hidden = torch.arange(1, 25, dtype=torch.float32).reshape(1, 3, 8)
    basis = torch.zeros((8, 1), dtype=torch.float32)
    basis[0, 0] = 1.0

    projected = _project_hidden(hidden, basis, alpha=-1.0, boundary=2)
    torch.testing.assert_close(projected[:, :2], hidden[:, :2])
    torch.testing.assert_close(projected[:, 2, 0], torch.zeros(1))
    torch.testing.assert_close(projected[:, 2, 1:], hidden[:, 2, 1:])


@pytest.mark.parametrize("device", _devices())
def test_continuation_probe_baseline_zero_alpha_and_projection_cleanup(device):
    runtime = TinyRuntime(device)
    baseline_hooks = _hook_counts(runtime)
    basis = np.zeros((16, 1), dtype=np.float32)
    basis[0, 0] = 1.0
    sample = _sample("A prompt with stable tokenization.")

    baseline = score_residual_continuation(runtime, sample, 0, "resid_pre", score_mode="sequence_logprob")
    zero_alpha = score_residual_continuation(
        runtime,
        sample,
        0,
        "resid_pre",
        basis=basis,
        alpha=0.0,
        score_mode="sequence_logprob",
    )
    removed = score_residual_continuation(
        runtime,
        sample,
        0,
        "resid_pre",
        basis=basis,
        alpha=-1.0,
        score_mode="mean_token_logprob",
    )

    assert baseline["token_count"] > 0
    assert baseline["boundary_token_index"] > 0
    assert baseline["boundary_policy"].startswith("apply at the final prompt token")
    assert baseline["score"] == zero_alpha["score"]
    assert removed["token_count"] == baseline["token_count"]
    assert np.isfinite([baseline["score"], zero_alpha["score"], removed["score"]]).all()
    assert baseline_hooks == _hook_counts(runtime)


def test_continuation_probe_waits_for_runtime_lock_before_installing_hook():
    runtime = TinyRuntime()
    baseline_hooks = _hook_counts(runtime)
    original_forward = runtime.lm.forward
    forward_observed_projection_hook = False

    class HookCheckingLock:
        def __enter__(self):
            assert _hook_counts(runtime) == baseline_hooks

        def __exit__(self, exc_type, exc_value, traceback):
            del exc_type, exc_value, traceback
            return False

    def checked_forward(*args, **kwargs):
        nonlocal forward_observed_projection_hook
        forward_observed_projection_hook = _hook_counts(runtime) != baseline_hooks
        assert forward_observed_projection_hook
        return original_forward(*args, **kwargs)

    runtime.lock = HookCheckingLock()
    runtime.lm.forward = checked_forward
    basis = np.zeros((16, 1), dtype=np.float32)
    basis[0, 0] = 1.0
    result = score_residual_continuation(runtime, _sample(), 0, "resid_pre", basis=basis, alpha=-1.0)

    assert np.isfinite(result["score"])
    assert forward_observed_projection_hook
    assert baseline_hooks == _hook_counts(runtime)


def test_continuation_probe_scores_match_full_model_logits_for_each_residual_point():
    runtime = TinyRuntime()
    sample = _sample("A controlled continuation prompt.")
    continuation = sample["target"]
    messages = sample["messages"]
    prompt_ids = runtime.adapter.apply_chat_template(messages, add_generation_prompt=True)
    rendered_prompt = runtime.adapter.render_chat_template(messages, add_generation_prompt=True)
    full_ids = runtime.tok(rendered_prompt + continuation, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    target_ids = input_ids[0, len(prompt_ids) :]
    start = len(prompt_ids) - 1
    stop = len(full_ids) - 1
    with torch.inference_mode():
        direct_logits = runtime.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        reference_token_logprobs = F.log_softmax(direct_logits[0, start:stop].float(), dim=-1).gather(
            1, target_ids[:, None]
        ).squeeze(1)
    expected_score = float(reference_token_logprobs.sum().item())
    baseline_hooks = _hook_counts(runtime)
    basis = np.zeros((16, 1), dtype=np.float32)
    basis[0, 0] = 1.0

    for point in ("resid_pre", "resid_mid", "resid_post"):
        result = score_residual_continuation(
            runtime,
            sample,
            0,
            point,
            basis=basis,
            alpha=0.0,
            score_mode="sequence_logprob",
        )
        assert result["score"] == pytest.approx(expected_score, rel=1e-6, abs=1e-6)
        assert result["token_count"] == len(target_ids)
        assert baseline_hooks == _hook_counts(runtime)


def test_continuation_probe_replaces_tuple_layer_output_and_cleans_hook():
    runtime = TinyRuntime()
    baseline_hooks = _hook_counts(runtime)
    basis = np.zeros((16, 1), dtype=np.float32)
    basis[0, 0] = 1.0

    result = score_residual_continuation(
        runtime,
        _sample(),
        0,
        "resid_post",
        basis=basis,
        alpha=-1.0,
    )

    assert np.isfinite(result["score"])
    assert baseline_hooks == _hook_counts(runtime)


def test_continuation_probe_rejects_nonfinite_logits_and_cleans_hook():
    runtime = TinyRuntime()
    baseline_hooks = _hook_counts(runtime)
    original_forward = runtime.adapter.lm_head.forward

    def nonfinite_forward(*args, **kwargs):
        return torch.full_like(original_forward(*args, **kwargs), float("nan"))

    runtime.adapter.lm_head.forward = nonfinite_forward
    try:
        with pytest.raises(FloatingPointError, match="non-finite logits"):
            score_residual_continuation(
                runtime,
                _sample(),
                0,
                "resid_pre",
                basis=np.eye(16, 1, dtype=np.float32),
                alpha=-1.0,
            )
    finally:
        runtime.adapter.lm_head.forward = original_forward

    assert baseline_hooks == _hook_counts(runtime)


def test_continuation_probe_requires_stable_prefix_and_valid_basis():
    runtime = TinyRuntime()
    sample = _sample()

    class UnstableTokenizer(TinyTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            ids = self._encode(text)
            if text.endswith(sample["target"]):
                ids[0] = (ids[0] + 1) % 32
            return {"input_ids": ids}

    runtime.adapter.tokenizer = UnstableTokenizer()
    runtime.tok = runtime.adapter.tokenizer
    with pytest.raises(ValueError, match="prefix-stable"):
        score_residual_continuation(runtime, sample, 0, "resid_pre")
    runtime.adapter.tokenizer = TinyTokenizer()
    runtime.tok = runtime.adapter.tokenizer

    non_orthogonal = np.ones((16, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="orthonormal"):
        score_residual_continuation(runtime, sample, 0, "resid_pre", basis=non_orthogonal, alpha=-1.0)
    with pytest.raises(ValueError, match="A non-zero alpha requires"):
        score_residual_continuation(runtime, sample, 0, "resid_pre", basis=None, alpha=0.5)
