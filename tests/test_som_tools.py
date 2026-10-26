from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "explorer"))

import agent_mcp
import server


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class FakeSOMService:
    def __init__(self):
        self.pilot_args = None

    @staticmethod
    def status():
        return {"status": "ready", "pipeline": "paired residual SOM"}

    @staticmethod
    def list_sources():
        return {"sources": [{"source_id": "private-opaque", "kind": "configured", "status": "ready"}], "errors": []}

    @staticmethod
    def list_artifacts():
        return {"artifacts": [{"artifact_id": "artifact-opaque", "status": "validated"}], "errors": []}

    @staticmethod
    def list_jobs():
        return {"jobs": [{"job_id": "historical-private-job", "status": "completed", "kind": "existing"}]}

    @staticmethod
    def get_artifact(artifact_id):
        return {"artifact_id": artifact_id, "status": "validated", "metadata": {"layer": 10}}

    @staticmethod
    def get_job(job_id):
        return {"job_id": job_id, "status": "evaluating", "progress": {"completed": 2, "total": 8}}

    def start_pilot(self, **kwargs):
        self.pilot_args = kwargs
        return {"job_id": "new-pilot-job", "status": "queued", "max_questions": kwargs["max_questions"]}


def _tool(name):
    return next(tool for tool in agent_mcp.TOOLS if tool["name"] == name)


def _json_body(response):
    return json.loads(response.body)


def test_som_mcp_tools_are_scoped_and_explain_the_scientific_stages():
    expected = {"som_status", "som_read_artifact", "som_run_pilot", "som_job_status"}
    registered = {tool["name"] for tool in agent_mcp.TOOLS}
    assert expected <= registered

    pilot = _tool("som_run_pilot")
    schema = pilot["inputSchema"]
    assert set(schema["properties"]) == {
        "source_id", "layer", "rows", "cols", "epochs", "seed", "split", "max_questions",
    }
    assert schema["additionalProperties"] is False
    assert "path" not in schema["properties"]
    assert "command" not in schema["properties"]
    description = pilot["description"].lower()
    assert "collection captures" in description
    assert "train-split" in description
    assert "held-out" in description
    assert "not research proof" in description
    assert "exact revision" in description

    status_schema = _tool("som_job_status")["inputSchema"]
    wait_schema = status_schema["properties"]["wait_seconds"]
    assert wait_schema == {"type": "integer", "minimum": 0, "maximum": 20, "default": 10}


def test_mcp_protocol_advertises_the_som_tools(monkeypatch):
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    output = io.StringIO()
    monkeypatch.setattr(agent_mcp.sys, "stdin", io.StringIO(request))
    monkeypatch.setattr(agent_mcp.sys, "stdout", output)

    agent_mcp.main()

    response = json.loads(output.getvalue())
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert {"som_status", "som_read_artifact", "som_run_pilot", "som_job_status"} <= names


def test_som_mcp_calls_use_only_scoped_explorer_endpoints(monkeypatch):
    calls = []

    def fake_get(path, **query):
        calls.append(("get", path, query))
        return {"ok": True}

    def fake_post(path, body, timeout=60):
        calls.append(("post", path, body, timeout))
        return {"queued": True}

    monkeypatch.setattr(agent_mcp, "get", fake_get)
    monkeypatch.setattr(agent_mcp, "post", fake_post)

    assert agent_mcp.call("som_status", {}) == {"ok": True}
    assert agent_mcp.call("som_read_artifact", {"artifact_id": "artifact id/one"}) == {"ok": True}
    assert agent_mcp.call("som_run_pilot", {"source_id": "source-one", "layer": 10}) == {"queued": True}
    assert agent_mcp.call("som_job_status", {"job_id": "job/one?x"}) == {"ok": True}

    assert calls == [
        ("get", "/api/agent/som/status", {}),
        ("get", "/api/agent/som/artifact", {"artifact_id": "artifact id/one"}),
        ("post", "/api/agent/som/pilot", {"source_id": "source-one", "layer": 10}, 60),
        ("get", "/api/agent/som/jobs/job%2Fone%3Fx", {}),
    ]


def test_som_mcp_surfaces_structured_http_errors(monkeypatch):
    error = HTTPError(
        "http://127.0.0.1:8791/api/agent/som/artifact",
        404,
        "Not Found",
        {},
        io.BytesIO(b'{"error":"No matching validated SOM artifact was found.","code":"unknown_id"}'),
    )

    def raise_http_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(agent_mcp.urllib.request, "urlopen", raise_http_error)

    with pytest.raises(RuntimeError, match="No matching validated SOM artifact was found"):
        agent_mcp.get("/api/agent/som/artifact", artifact_id="missing")


def test_som_job_status_waits_for_local_transition_without_real_sleep(monkeypatch):
    responses = iter([
        {"kind": "pilot", "status": "collecting"},
        {"kind": "pilot", "status": "completed", "result_validity": "verified"},
    ])
    calls = []
    clock = {"now": 0.0}
    sleeps = []

    def fake_get(path):
        calls.append(path)
        return next(responses)

    def fake_sleep(seconds):
        assert 0 < seconds <= 1
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(agent_mcp, "get", fake_get)
    monkeypatch.setattr(agent_mcp.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(agent_mcp.time, "sleep", fake_sleep)

    result = agent_mcp.som_job_status("local-job", wait_seconds=5)

    assert result["status"] == "completed"
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_som_job_status_returns_historical_external_record_immediately(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_mcp, "get", lambda path: calls.append(path) or {
        "kind": "external_record", "status": "unverified", "recorded_status": "FAILED",
    })
    monkeypatch.setattr(agent_mcp.time, "sleep", lambda *_args: pytest.fail("external history must not be polled"))

    result = agent_mcp.som_job_status("123456", wait_seconds=20)

    assert result["status"] == "unverified"
    assert result["recorded_status"] == "FAILED"
    assert len(calls) == 1


def test_som_job_status_caps_wait_at_twenty_seconds_without_real_sleep(monkeypatch):
    calls = []
    clock = {"now": 0.0}
    sleeps = []
    monkeypatch.setattr(agent_mcp, "get", lambda path: calls.append(path) or {"kind": "pilot", "status": "queued"})
    monkeypatch.setattr(agent_mcp.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds):
        assert 0 < seconds <= 1
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(agent_mcp.time, "sleep", fake_sleep)

    result = agent_mcp.som_job_status("local-job", wait_seconds=20)

    assert result["status"] == "queued"
    assert len(calls) == 21
    assert sleeps == [1.0] * 20


def test_som_job_status_rejects_wait_above_cap_before_request(monkeypatch):
    monkeypatch.setattr(agent_mcp, "get", lambda *_args: pytest.fail("invalid wait must not make a request"))

    with pytest.raises(ValueError, match="0 through 20"):
        agent_mcp.som_job_status("local-job", wait_seconds=21)


def _configure_server(monkeypatch, service, *, n_layers=28):
    monkeypatch.setattr(server, "som_service", service)
    monkeypatch.setattr(server, "ARCH", {"n_layers": n_layers})
    monkeypatch.setattr(server, "MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    monkeypatch.setattr(server, "MODEL_REVISION", "active-revision-123")
    monkeypatch.setattr(server, "current_model_info", lambda: {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "active-revision-123",
    })
    monkeypatch.setattr(server, "rt", SimpleNamespace(dev="mps"))


def test_som_status_distinguishes_existing_work_from_new_jobs(monkeypatch):
    service = FakeSOMService()
    _configure_server(monkeypatch, service)

    result = server.agent_som_status()

    assert result["active_model"] == {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "active-revision-123",
        "n_layers": 28,
        "device": "mps",
    }
    assert result["sources"]["sources"][0]["source_id"] == "private-opaque"
    assert result["artifacts"]["artifacts"][0]["status"] == "validated"
    assert result["jobs"]["jobs"][0]["kind"] == "existing"


def test_som_read_artifact_and_job_status_return_safe_service_summaries(monkeypatch):
    _configure_server(monkeypatch, FakeSOMService())

    artifact = server.agent_som_artifact("artifact-opaque")
    job = server.agent_som_job("job-opaque")

    assert artifact["metadata"] == {"layer": 10}
    assert "prototypes" not in artifact
    assert job["status"] == "evaluating"
    assert job["progress"] == {"completed": 2, "total": 8}


def test_som_endpoint_preserves_structured_not_found_errors(monkeypatch):
    class MissingArtifactService(FakeSOMService):
        @staticmethod
        def get_artifact(_artifact_id):
            raise server.SOMServiceError("unknown_id", "No matching validated SOM artifact was found.", 404)

    _configure_server(monkeypatch, MissingArtifactService())

    response = server.agent_som_artifact("missing-artifact")

    assert response.status_code == 404
    assert _json_body(response) == {
        "error": "No matching validated SOM artifact was found.",
        "code": "unknown_id",
    }


@pytest.mark.asyncio
async def test_som_pilot_uses_active_revision_runtime_device_and_bounded_options(monkeypatch):
    service = FakeSOMService()
    _configure_server(monkeypatch, service, n_layers=28)

    response = await server.agent_som_pilot(JsonRequest({
        "source_id": "source-opaque", "layer": 27, "rows": 3, "cols": 5,
        "epochs": 80, "seed": 19, "split": "validation", "max_questions": 6,
    }))

    assert response.status_code == 202
    assert _json_body(response)["status"] == "queued"
    assert service.pilot_args == {
        "source_id": "source-opaque",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "active-revision-123",
        "layer": 27,
        "device": "mps",
        "split": "validation",
        "rows": 3,
        "cols": 5,
        "epochs": 80,
        "seed": 19,
        "max_questions": 6,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"source_id": "source-opaque", "layer": True},
        {"source_id": "source-opaque", "layer": 1, "max_questions": 13},
        {"source_id": "source-opaque", "layer": 1, "output": "/tmp/result.npz"},
        {"source_id": "source-opaque", "layer": 1, "split": "train"},
    ],
)
async def test_som_pilot_rejects_unbounded_or_unscoped_options(monkeypatch, body):
    service = FakeSOMService()
    _configure_server(monkeypatch, service)

    response = await server.agent_som_pilot(JsonRequest(body))

    assert response.status_code == 400
    assert service.pilot_args is None
