"""Offline CPU coverage for decoder adapters and runtime hooks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "explorer"))

import model_adapter
import runtime as runtime_module
from model_adapter import ModelAdapter
from runtime import Runtime


class TinyTokenizer:
    def __init__(self, eos_token_id=3, pad_token_id=None):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.padding_side = "right"
        self.template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        if kwargs.get("tokenize", True):
            return [1, 4, 2] if kwargs.get("add_generation_prompt", True) else [1, 4]
        return "<user>hello<assistant>" if kwargs.get("add_generation_prompt", True) else "<user>hello</user>"

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(65 + int(token_id)) for token_id in token_ids)


def tiny_llama(eos_ids=(5, 7)):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=list(eos_ids),
    )
    torch.manual_seed(11)
    return LlamaForCausalLM(config).eval()


def make_adapter(model=None, tokenizer=None, model_id=model_adapter.DEFAULT_MODEL):
    model = model or tiny_llama()
    tokenizer = tokenizer or TinyTokenizer()
    return ModelAdapter(model, tokenizer, model_id, config=model.config, device="cpu")


def test_cache_namespace_is_stable_and_revision_specific():
    default_namespace = model_adapter.model_cache_namespace()
    assert default_namespace == model_adapter.model_cache_namespace(
        model_adapter.DEFAULT_MODEL,
        model_adapter.DEFAULT_REVISION,
    )
    assert default_namespace != model_adapter.model_cache_namespace(
        model_adapter.DEFAULT_MODEL,
        "another-revision",
    )
    assert default_namespace != model_adapter.model_cache_namespace("Qwen/Qwen3.5-4B")


def test_device_resolution_prefers_mps_when_available_and_honors_cpu(monkeypatch):
    monkeypatch.delenv("EXPLORER_DEVICE", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert model_adapter.resolve_device().type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert model_adapter.resolve_device().type == "mps"
    assert model_adapter.resolve_device("cpu").type == "cpu"
    assert model_adapter._default_dtype(torch.device("mps")) == torch.float16


def test_llama_adapter_uses_decoder_norm_and_tokenizer_ids():
    tokenizer = TinyTokenizer(eos_token_id=3)
    model = tiny_llama(eos_ids=(5, 7))
    model.config.pad_token_id = None
    adapter = make_adapter(model, tokenizer)

    assert adapter.family == "llama"
    assert adapter.model_prefix == "model"
    assert adapter.layers is model.model.layers
    assert adapter.eos_token_ids == {3, 5, 7}
    assert adapter.pad_token_id == 3
    assert tokenizer.padding_side == "left"
    assert adapter.is_eos_token(7)
    assert not adapter.is_eos_token(4)

    expected = model.model.layers[0].post_attention_layernorm.weight.detach().float()
    torch.testing.assert_close(adapter.layer_norm_scale(0), expected)
    assert adapter.apply_chat_template([{"role": "user", "content": "hello"}]) == [1, 4, 2]
    assert "enable_thinking" not in tokenizer.template_kwargs
    assert adapter.render_chat_template([{"role": "user", "content": "hello"}]) == "<user>hello<assistant>"
    assert adapter.render_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=False,
    ) == "<user>hello</user>"


def test_qwen_adapter_applies_offset_norm_and_template_flag():
    block = SimpleNamespace(
        mlp=SimpleNamespace(down_proj=torch.nn.Linear(3, 2, bias=False)),
        post_attention_layernorm=SimpleNamespace(
            weight=torch.nn.Parameter(torch.tensor([-0.25, 0.0, 0.5]))
        ),
    )
    decoder = SimpleNamespace(layers=[block], norm=torch.nn.Identity())
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=decoder),
        lm_head=torch.nn.Identity(),
        config=SimpleNamespace(model_type="qwen3_5", eos_token_id=[7, 8]),
    )
    tokenizer = TinyTokenizer(eos_token_id=6)
    adapter = ModelAdapter(model, tokenizer, "Qwen/Qwen3.5-4B", config=model.config)

    torch.testing.assert_close(adapter.layer_norm_scale(0), torch.tensor([0.75, 1.0, 1.5]))
    assert adapter.model_prefix == "model.language_model"
    assert adapter.eos_token_ids == {6, 7, 8}
    assert adapter.apply_chat_template([{"role": "user", "content": "hello"}]) == [1, 4, 2]
    assert tokenizer.template_kwargs["enable_thinking"] is False


def test_runtime_hook_changes_down_projection_input_and_close_removes_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "MODEL", model_adapter.DEFAULT_MODEL)
    monkeypatch.setattr(runtime_module, "MODEL_REVISION", model_adapter.DEFAULT_REVISION)
    monkeypatch.setattr(runtime_module, "NO_MODEL", False)
    model = tiny_llama()
    tokenizer = TinyTokenizer()
    adapter = ModelAdapter(model, tokenizer, model_adapter.DEFAULT_MODEL, device="cpu")
    rt = Runtime(tmp_path / "stats", cache_root=tmp_path / "cache")
    rt._attach_model(model, tokenizer, adapter, "cpu")

    ids = torch.tensor([[1, 2, 4]])
    projection_inputs = []
    observer = adapter.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda _module, args: projection_inputs.append(args[0].detach().clone())
    )
    with torch.inference_mode():
        baseline = model(input_ids=ids, use_cache=False).logits
        rt._push = [(0, 5, "add", torch.tensor([0.25]))]
        acts = rt.full_activations(ids.flatten().tolist())
        steered = model(input_ids=ids, use_cache=False).logits

    assert acts.shape == (2, 3, 24)
    assert rt._capture_full is None
    expected = torch.from_numpy(acts[0])
    observed = projection_inputs[1][0].cpu()
    torch.testing.assert_close(observed[:, 5], expected[:, 5] + 0.25)
    torch.testing.assert_close(observed[:, :5], expected[:, :5])
    torch.testing.assert_close(observed[:, 6:], expected[:, 6:])
    assert not torch.allclose(baseline, steered)

    observer.remove()
    rt.close()
    assert rt._hook_handles == []
    with torch.inference_mode():
        after_close = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(after_close, baseline)


class EosOnlyModel:
    def __init__(self, eos_id, vocab_size=32):
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        self.calls += 1
        logits = torch.zeros((*input_ids.shape, self.vocab_size), device=input_ids.device)
        logits[..., self.eos_id] = 1
        return SimpleNamespace(logits=logits, past_key_values=None)

    __call__ = forward


def test_runtime_stops_on_configured_eos_token_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "MODEL", model_adapter.DEFAULT_MODEL)
    monkeypatch.setattr(runtime_module, "MODEL_REVISION", model_adapter.DEFAULT_REVISION)
    monkeypatch.setattr(runtime_module, "NO_MODEL", False)
    tokenizer = TinyTokenizer(eos_token_id=3)
    model = tiny_llama(eos_ids=(5, 7))
    adapter = ModelAdapter(model, tokenizer, model_adapter.DEFAULT_MODEL, device="cpu")
    rt = Runtime(tmp_path / "stats", cache_root=tmp_path / "cache")
    eos_model = EosOnlyModel(7)
    rt.model, rt.tok, rt.adapter, rt.dev = eos_model, tokenizer, adapter, torch.device("cpu")

    messages = [[{"role": "user", "content": f"row {i}"}] for i in range(3)]
    generated = list(rt.steer_multi([], messages, max_new=2))

    assert generated == [(None, None)]
    assert eos_model.calls == 1
    assert rt._push is None
    rt.close()


def test_llama_cache_path_does_not_use_legacy_qwen_files(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "MODEL", model_adapter.DEFAULT_MODEL)
    monkeypatch.setattr(runtime_module, "MODEL_REVISION", model_adapter.DEFAULT_REVISION)
    llama = Runtime(tmp_path / "stats", cache_root=tmp_path / "cache")

    assert llama.cache_dir.endswith(llama.cache_namespace)
    assert llama.cache_namespace == model_adapter.model_cache_namespace(
        model_adapter.DEFAULT_MODEL,
        model_adapter.DEFAULT_REVISION,
    )
    assert llama._cache_path("lens_index.npz") == str(Path(llama.cache_dir) / "lens_index.npz")
    llama.close()


def test_chat_ids_lazily_loads_the_tokenizer_before_first_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "MODEL", model_adapter.DEFAULT_MODEL)
    monkeypatch.setattr(runtime_module, "MODEL_REVISION", model_adapter.DEFAULT_REVISION)
    monkeypatch.setattr(runtime_module, "NO_MODEL", False)
    model = tiny_llama()
    tokenizer = TinyTokenizer()
    adapter = ModelAdapter(model, tokenizer, model_adapter.DEFAULT_MODEL, device="cpu")
    monkeypatch.setattr(
        runtime_module,
        "load_model_and_tokenizer",
        lambda *_args, **_kwargs: (model, tokenizer, adapter),
    )
    rt = Runtime(tmp_path / "stats", cache_root=tmp_path / "cache")

    assert rt.model is None
    assert rt.chat_ids("hello") == [1, 4, 2]
    assert rt.status == "ready"
    rt.close()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable on this host")
def test_mps_model_inputs_and_steering_tensors_stay_on_mps(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "MODEL", model_adapter.DEFAULT_MODEL)
    monkeypatch.setattr(runtime_module, "MODEL_REVISION", model_adapter.DEFAULT_REVISION)
    monkeypatch.setattr(runtime_module, "NO_MODEL", False)
    model = tiny_llama().to("mps")
    tokenizer = TinyTokenizer()
    adapter = ModelAdapter(model, tokenizer, model_adapter.DEFAULT_MODEL, device="mps")
    rt = Runtime(tmp_path / "stats", cache_root=tmp_path / "cache")
    rt._attach_model(model, tokenizer, adapter, "mps")
    values = rt._push_entry(0, 5, 0.5, "mul", (1,))[3]
    assert values.device.type == "mps"
    assert adapter.down_col(0, 5).device.type == "mps"
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2, 4]], device="mps"), use_cache=False)
    assert adapter.logits(output).device.type == "mps"
    assert rt.model_info["device"] == "mps"
    rt.close()
