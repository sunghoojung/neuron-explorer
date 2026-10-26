from __future__ import annotations

import hashlib
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

_FIXTURE_PNG = b"\x89PNG\r\n\x1a\nfixture-image-bytes"
_FIXTURE_PNG_SHA256 = hashlib.sha256(_FIXTURE_PNG).hexdigest()


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class FakeSOMService:
    def __init__(self, native_report_path=None, native_assets=None):
        self.pilot_args = None
        self.native_report_path = native_report_path
        self.native_asset_request = None
        self.native_assets = native_assets or {}

    @staticmethod
    def status():
        return {
            "status": "ready",
            "pipeline": "paired residual SOM",
            "native_reports": [{
                "native_id": "native_0123456789abcdef01234567",
                "integrity_status": "verified",
                "reuse_compatibility": "not_assessed",
                "report_available": True,
            }],
        }

    @staticmethod
    def list_sources():
        return {"sources": [{"source_id": "private-opaque", "kind": "configured", "status": "ready"}], "errors": []}

    @staticmethod
    def list_artifacts():
        return {"artifacts": [{"artifact_id": "artifact-opaque", "status": "validated"}], "errors": []}

    @staticmethod
    def list_native_reports():
        return {"native_reports": [{
            "native_id": "native_0123456789abcdef01234567",
            "integrity_status": "verified",
            "reuse_compatibility": "not_assessed",
            "report_available": True,
        }]}

    @staticmethod
    def list_jobs():
        return {"jobs": [{"job_id": "historical-private-job", "status": "completed", "kind": "existing"}]}

    @staticmethod
    def get_artifact(artifact_id):
        return {"artifact_id": artifact_id, "status": "validated", "metadata": {"layer": 10}}

    def get_native_report(self, native_id):
        report_sha256 = None
        if self.native_report_path and self.native_report_path.is_file():
            report_sha256 = hashlib.sha256(self.native_report_path.read_bytes()).hexdigest()
        return {
            "native_id": native_id,
            "status": "completed",
            "integrity_status": "verified",
            "capture_provenance_status": "verified",
            "reuse_compatibility": "not_assessed",
            "report_available": True,
            "provenance": {"model": "test/model", "layers": [1, 3], "views": ["prompt_token"]},
            "geometry": {"map_count": 2, "pairs": 5, "report_sha256": report_sha256},
            "maps": [
                {
                    "map_name": "fixture-map-a",
                    "kind": "fixture",
                    "view": "prompt_token",
                    "layer": 3,
                    "side": "positive",
                    "seed": 7,
                    "prototype_count": 16,
                    "png_sha256": _FIXTURE_PNG_SHA256,
                    "fit_history": [{"epoch": 1, "loss": 0.4}],
                    "associations": [{"label": "fixture", "count": 2}],
                    "direction_geometry": {"rank": 1},
                    "condition_geometry": {"distance": 0.2},
                },
                {
                    "map_name": "fixture-map-b",
                    "kind": "fixture",
                    "view": "answer_token",
                    "layer": 3,
                    "side": "negative",
                    "seed": 8,
                    "prototype_count": 16,
                    "png_sha256": _FIXTURE_PNG_SHA256,
                },
            ],
            "behavior": {
                "status": "complete",
                "cases": 3,
                "arms": 2,
                "answers": 6,
                "strict_screen": {
                    "scope": "conservative_lexical_only",
                    "semantic_grades_available": False,
                    "by_method": [],
                },
                "semantic_review_status": "pending",
                "semantic_grade_count": 0,
                "blind_records": [{"private": "fixture-only"}],
            },
        }

    def get_native_report_file(self, _native_id):
        return self.native_report_path

    def get_native_report_asset(self, native_id, relative_name):
        self.native_asset_request = (native_id, relative_name)
        return self.native_assets.get(relative_name)

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
    expected = {"som_status", "som_read_artifact", "som_read_native_report", "som_run_pilot", "som_job_status"}
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

    native = _tool("som_read_native_report")
    assert set(native["inputSchema"]["properties"]) == {"native_id", "map_name"}
    assert native["inputSchema"]["properties"]["map_name"]["maxLength"] == 120
    assert native["inputSchema"]["additionalProperties"] is False
    assert "not applied" in native["description"]
    assert "raw direction arrays" in native["description"]
    assert "brief records for each map" in native["description"]

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
    assert {"som_status", "som_read_artifact", "som_read_native_report", "som_run_pilot", "som_job_status"} <= names


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
    native_report = {
        "native_id": "native id/one",
        "maps": [{"map_name": "fixture-map", "layer": 2, "fit_history": [1]}],
    }

    def fake_get(path, **query):
        calls.append(("get", path, query))
        if path == "/api/agent/som/native-report":
            return native_report
        return {"ok": True}

    monkeypatch.setattr(agent_mcp, "get", fake_get)
    compact = agent_mcp.call("som_read_native_report", {"native_id": "native id/one"})
    assert compact["maps"] == [{"map_name": "fixture-map", "layer": 2}]
    assert "fit_history" not in compact["maps"][0]
    assert agent_mcp.call("som_run_pilot", {"source_id": "source-one", "layer": 10}) == {"queued": True}
    assert agent_mcp.call("som_job_status", {"job_id": "job/one?x"}) == {"ok": True}

    assert calls == [
        ("get", "/api/agent/som/status", {}),
        ("get", "/api/agent/som/artifact", {"artifact_id": "artifact id/one"}),
        ("get", "/api/agent/som/native-report", {"native_id": "native id/one"}),
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
        "kind": "external_record", "status": "unverified", "recorded_status": "failed",
    })
    monkeypatch.setattr(agent_mcp.time, "sleep", lambda *_args: pytest.fail("external history must not be polled"))

    result = agent_mcp.som_job_status("historical-job-42", wait_seconds=20)

    assert result["status"] == "unverified"
    assert result["recorded_status"] == "failed"
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


def test_som_status_distinguishes_native_reports_and_existing_jobs(monkeypatch, tmp_path):
    service = FakeSOMService(tmp_path / "native-report.html")
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
    native = result["native_reports"]["native_reports"][0]
    assert native["integrity_status"] == "verified"
    assert native["report_url"].endswith("/api/agent/som/native-reports/native_0123456789abcdef01234567/report")
    assert result["jobs"]["jobs"][0]["kind"] == "existing"


def test_som_read_artifact_and_job_status_return_safe_service_summaries(monkeypatch):
    _configure_server(monkeypatch, FakeSOMService())

    artifact = server.agent_som_artifact("artifact-opaque")
    job = server.agent_som_job("job-opaque")

    assert artifact["metadata"] == {"layer": 10}
    assert "prototypes" not in artifact
    assert job["status"] == "evaluating"
    assert job["progress"] == {"completed": 2, "total": 8}


def test_som_native_report_returns_safe_aggregate_and_atlas_link(monkeypatch, tmp_path):
    native_id = "native_0123456789abcdef01234567"
    service = FakeSOMService(tmp_path / "native-report.html")
    _configure_server(monkeypatch, service)

    report = server.agent_som_native_report(native_id)

    assert report["integrity_status"] == "verified"
    assert report["reuse_compatibility"] == "not_assessed"
    assert report["geometry"]["map_count"] == 2
    assert report["report_url"].endswith(f"/api/agent/som/native-reports/{native_id}/report")
    assert "direction_arrays" not in report
    assert "local_path" not in report


def test_native_mcp_report_defaults_to_compact_map_records():
    report = FakeSOMService().get_native_report("native-fixture")

    compact = agent_mcp._compact_native_report(report)

    assert compact["maps"] == [
        {
            "map_name": "fixture-map-a",
            "kind": "fixture",
            "view": "prompt_token",
            "layer": 3,
            "side": "positive",
            "seed": 7,
            "prototype_count": 16,
        },
        {
            "map_name": "fixture-map-b",
            "kind": "fixture",
            "view": "answer_token",
            "layer": 3,
            "side": "negative",
            "seed": 8,
            "prototype_count": 16,
        },
    ]
    for item in compact["maps"]:
        assert not {"fit_history", "associations", "direction_geometry", "condition_geometry"} & item.keys()


def test_native_mcp_report_selects_one_exact_map_detail():
    report = FakeSOMService().get_native_report("native-fixture")

    compact = agent_mcp._compact_native_report(report, "fixture-map-a")

    assert compact["selected_map_detail"] == report["maps"][0]
    assert len(compact["maps"]) == 2
    assert all("fit_history" not in item for item in compact["maps"])


def test_native_mcp_report_rejects_unknown_map_name():
    report = FakeSOMService().get_native_report("native-fixture")

    with pytest.raises(ValueError, match="Unknown map_name"):
        agent_mcp._compact_native_report(report, "../not-a-map")


def test_native_mcp_call_passes_exact_map_selector(monkeypatch):
    calls = []
    report = FakeSOMService().get_native_report("native-fixture")
    monkeypatch.setattr(agent_mcp, "get", lambda path, **query: calls.append((path, query)) or report)

    result = agent_mcp.call(
        "som_read_native_report",
        {"native_id": "native-fixture", "map_name": "fixture-map-b"},
    )

    assert calls == [(
        "/api/agent/som/native-report",
        {"native_id": "native-fixture"},
    )]
    assert result["selected_map_detail"] == report["maps"][1]


def test_native_mcp_call_returns_clear_error_for_unknown_map(monkeypatch):
    monkeypatch.setattr(agent_mcp, "get", lambda *_args, **_kwargs: FakeSOMService().get_native_report("native-fixture"))

    with pytest.raises(ValueError, match="Unknown map_name"):
        agent_mcp.call(
            "som_read_native_report",
            {"native_id": "native-fixture", "map_name": "missing-map"},
        )


def test_som_native_report_view_has_restrictive_headers_and_inline_disposition(monkeypatch, tmp_path):
    report_path = tmp_path / "native-report.html"
    report_path.write_text(
        '<meta charset="utf-8"><title>Fixture atlas</title><style>h1{color:#123}</style>'
        '<h1>Fixture atlas</h1><pre>{"fixture":"metadata"}</pre><article>'
        '<img src="maps/fixture-map-a/map.png"><img src="maps/fixture-map-b/map.png">'
        '<a href="behavior/strict-screen.json">Strict screen</a>'
        '<a href="behavior/blinded-review.json">Full answers</a>'
        '</article>',
        encoding="utf-8",
    )
    service = FakeSOMService(report_path, {
        "maps/fixture-map-a/map.png": report_path.with_name("map-a.png"),
        "maps/fixture-map-b/map.png": report_path.with_name("map-b.png"),
    })
    service.native_assets["maps/fixture-map-a/map.png"].write_bytes(_FIXTURE_PNG)
    service.native_assets["maps/fixture-map-b/map.png"].write_bytes(_FIXTURE_PNG)
    _configure_server(monkeypatch, service)

    response = server.agent_som_native_report_view("native_0123456789abcdef01234567")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["content-security-policy"] == (
        "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
        "base-uri 'none'; form-action 'none'; connect-src 'none'"
    )
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    body = response.body.decode("utf-8")
    assert body.startswith("<!doctype html><html><head>")
    assert "</head><body>" in body
    assert body.count('src="data:image/png;base64,') == 2
    assert "src=\"maps/" not in body
    assert "href=\"behavior/" not in body
    assert 'href="#strict-screen-summary"' in body
    assert '<a href="#blinded-review-summary">Semantic review status</a>' in body
    assert "Full answers" not in body
    assert '<summary>Data and provenance</summary><pre>' in body
    assert "Integrity and capture provenance verified" in body
    assert "5 pairs" in body and "3 cases" in body and "2 arms" in body
    assert "Semantic grading: pending" in body
    assert '<section id="strict-screen-summary"' in body
    assert '<section id="blinded-review-summary"' in body
    assert "blind_records" not in body
    assert "<script" not in body.lower()


def test_som_native_report_view_rejects_missing_report(monkeypatch, tmp_path):
    _configure_server(monkeypatch, FakeSOMService())

    response = server.agent_som_native_report_view("missing-native-report")

    assert response.status_code == 404
    assert _json_body(response)["code"] == "unknown_id"


def test_som_native_report_view_rejects_unlisted_file_references(monkeypatch, tmp_path):
    report_path = tmp_path / "native-report.html"
    report_path.write_text(
        '<html><head></head><body><pre>fixture</pre><img src="../../private.png">'
        '<a href="behavior/strict-screen.json">Strict</a>'
        '<a href="behavior/blinded-review.json">Review</a></body></html>',
        encoding="utf-8",
    )
    asset_paths = {}
    for map_name in ("fixture-map-a", "fixture-map-b"):
        path = tmp_path / f"{map_name}.png"
        path.write_bytes(_FIXTURE_PNG)
        asset_paths[f"maps/{map_name}/map.png"] = path
    _configure_server(monkeypatch, FakeSOMService(report_path, asset_paths))

    response = server.agent_som_native_report_view("native-fixture")

    assert response.status_code == 409
    assert _json_body(response)["code"] == "unavailable"


def test_som_native_report_view_rejects_map_bytes_that_changed_after_validation(monkeypatch, tmp_path):
    report_path = tmp_path / "native-report.html"
    report_path.write_text(
        '<html><head></head><body><pre>fixture</pre>'
        '<img src="maps/fixture-map-a/map.png">'
        '<a href="behavior/strict-screen.json">Strict</a>'
        '<a href="behavior/blinded-review.json">Review</a></body></html>',
        encoding="utf-8",
    )
    asset_paths = {}
    for map_name in ("fixture-map-a", "fixture-map-b"):
        path = tmp_path / f"{map_name}.png"
        path.write_bytes(_FIXTURE_PNG if map_name == "fixture-map-a" else b"\x89PNG\r\n\x1a\nchanged")
        asset_paths[f"maps/{map_name}/map.png"] = path
    _configure_server(monkeypatch, FakeSOMService(report_path, asset_paths))

    response = server.agent_som_native_report_view("native-fixture")

    assert response.status_code == 409
    assert _json_body(response)["code"] == "unavailable"


def test_som_native_report_asset_uses_only_the_map_route_and_nosniff(monkeypatch, tmp_path):
    map_path = tmp_path / "map.png"
    map_path.write_bytes(b"\x89PNG\r\n\x1a\nmock")
    service = FakeSOMService(native_assets={"maps/layer13-view-a/map.png": map_path})
    _configure_server(monkeypatch, service)

    response = server.agent_som_native_report_asset("native_0123456789abcdef01234567", "layer13-view-a")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert service.native_asset_request == (
        "native_0123456789abcdef01234567",
        "maps/layer13-view-a/map.png",
    )


def test_native_report_behavior_links_return_only_validated_aggregates(monkeypatch):
    service = FakeSOMService()
    _configure_server(monkeypatch, service)

    strict = server.agent_som_native_strict_screen("native-fixture")
    review = server.agent_som_native_blinded_review("native-fixture")

    assert _json_body(strict) == {
        "scope": "conservative_lexical_only",
        "semantic_grades_available": False,
        "by_method": [],
    }
    assert _json_body(review) == {
        "semantic_review_status": "pending",
        "semantic_grade_count": 0,
    }
    assert strict.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert review.headers["content-security-policy"] == "default-src 'none'; sandbox"
    assert strict.headers["x-content-type-options"] == "nosniff"
    assert review.headers["x-content-type-options"] == "nosniff"
    assert strict.headers["cache-control"] == "no-store"
    assert review.headers["cache-control"] == "no-store"
    assert "blind_records" not in _json_body(review)


def test_native_report_asset_and_behavior_link_paths_are_registered():
    paths = {route.path for route in server.app.routes}

    assert "/api/agent/som/native-reports/{native_id}/report" in paths
    assert "/api/agent/som/native-reports/{native_id}/maps/{map_name}/map.png" in paths
    assert "/api/agent/som/native-reports/{native_id}/behavior/strict-screen.json" in paths
    assert "/api/agent/som/native-reports/{native_id}/behavior/blinded-review.json" in paths


def test_native_report_behavior_links_withhold_unverified_summaries(monkeypatch):
    class UnverifiedService(FakeSOMService):
        def get_native_report(self, native_id):
            report = super().get_native_report(native_id)
            report["integrity_status"] = "outputs_verified"
            return report

    _configure_server(monkeypatch, UnverifiedService())

    response = server.agent_som_native_blinded_review("native-fixture")

    assert response.status_code == 409
    assert _json_body(response) == {
        "error": "No validated native SOM behavior summary was found.",
        "code": "unavailable",
    }


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
