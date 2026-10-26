"""Offline checks for model-derived explorer geometry and run isolation."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "explorer"))

import server


@pytest.fixture
def llama_config():
    return {
        "model_type": "llama",
        "num_hidden_layers": 28,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
    }


def test_llama_geometry_comes_from_config_and_supports_non_square_width(llama_config):
    arch = server.architecture_from_config(llama_config)

    assert arch["n_layers"] == llama_config["num_hidden_layers"]
    assert arch["hidden"] == llama_config["hidden_size"]
    assert arch["inter"] == 8192
    assert arch["grid_side"] == 91
    assert arch["attn_heads"] == 24
    assert arch["kv_heads"] == 8
    assert arch["layer_types"] == ["full_attention"] * 28


def test_nested_text_config_is_supported(llama_config):
    arch = server.architecture_from_config({"text_config": llama_config})

    assert arch["n_layers"] == 28
    assert arch["inter"] == 8192


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"num_hidden_layers": 28, "hidden_size": 3072},
        {
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_attention_heads": 4,
            "layer_types": ["full_attention"],
        },
    ],
)
def test_incomplete_or_inconsistent_config_has_no_guessed_geometry(config):
    assert server.architecture_from_config(config) is None


def test_runs_are_discovered_only_with_matching_model_and_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS", str(tmp_path))
    monkeypatch.setattr(server, "MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    monkeypatch.setattr(server, "MODEL_REVISION", "0cb88a4f764b7a12671c53f0838cd831a0843b95")

    (tmp_path / "other.meta.json").write_text(json.dumps({
        "model": "Qwen/Qwen3.5-4B", "layer": 0,
    }))
    (tmp_path / "other.jsonl").write_text("{}\n")
    assert server.layer_runs() == {}

    (tmp_path / "llama.meta.json").write_text(json.dumps({
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "layer": 27,
    }))
    (tmp_path / "llama.jsonl").write_text("{}\n")

    assert server.layer_runs() == {27: str(tmp_path / "llama.jsonl")}


def test_explicit_revision_mismatch_is_rejected():
    assert not server.run_matches_model({
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "another-revision",
    })


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


@pytest.mark.asyncio
async def test_chat_rejects_empty_message():
    response = await server.agent_chat(JsonRequest({"message": "  "}))

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_chat_reservation_conflict_does_not_replace_active_ui_queue(monkeypatch):
    queue = object()
    state = {"active": False, "queue": queue}

    class Backend:
        class History:
            @staticmethod
            def load(_chat_id):
                return {}

        history = History()

        @staticmethod
        def reserve_chat(_chat_id):
            return False

    monkeypatch.setattr(server, "chat_backend", Backend())
    monkeypatch.setitem(server._chats, "test-chat", state)
    response = await server.agent_chat(JsonRequest({"chat": "test-chat", "message": "hello"}))

    assert response.status_code == 409
    assert state["queue"] is queue


@pytest.mark.asyncio
async def test_chat_rejects_expired_attachment_id(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "UPLOADS", str(tmp_path))

    class Backend:
        class History:
            @staticmethod
            def load(_chat_id):
                return {}

        history = History()

        @staticmethod
        def reserve_chat(_chat_id):
            pytest.fail("expired attachment should be rejected before reserving chat")

    monkeypatch.setattr(server, "chat_backend", Backend())
    response = await server.agent_chat(JsonRequest({
        "chat": "test-expired-attachment",
        "message": "Please inspect this file.",
        "attachments": ["expired"],
    }))

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_som_pilot_uses_dynamic_layer_bound_and_rejects_out_of_range(monkeypatch):
    monkeypatch.setattr(server, "ARCH", {"n_layers": 31})

    class SOMService:
        @staticmethod
        def start_pilot(**_kwargs):
            pytest.fail("out-of-range layer must be rejected before queuing a job")

    monkeypatch.setattr(server, "som_service", SOMService())
    response = await server.agent_som_pilot(JsonRequest({"source_id": "source-opaque", "layer": 31}))

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "layer must be between 0 and 30 for the active model."


@pytest.mark.asyncio
async def test_som_pilot_requires_an_exact_active_model_revision(monkeypatch):
    monkeypatch.setattr(server, "ARCH", {"n_layers": 31})
    monkeypatch.setattr(server, "MODEL_REVISION", None)
    monkeypatch.setattr(server, "current_model_info", lambda: {"model_id": "example/model", "revision": None})

    class SOMService:
        @staticmethod
        def start_pilot(**_kwargs):
            pytest.fail("a pilot without an exact model revision must not be queued")

    monkeypatch.setattr(server, "som_service", SOMService())
    response = await server.agent_som_pilot(JsonRequest({"source_id": "source-opaque", "layer": 30}))

    assert response.status_code == 409
    assert "exact revision" in json.loads(response.body)["error"]
