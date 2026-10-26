"""HTTP boundaries for residual jobs without loading model weights."""

from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from explorer.residual_api import MAX_REQUEST_BYTES, create_residual_router

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(tmp_path):
    runtime = SimpleNamespace(
        model_id="fixture/model", revision="fixture-revision", model=None,
        adapter=None, dev="cpu", cache_namespace="fixture-model",
        config_metadata={"num_hidden_layers": 28, "hidden_size": 16},
    )
    router, service = create_residual_router(runtime, tmp_path)
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as session:
        yield session
    service.close()


async def test_config_and_demo_do_not_require_loaded_weights(client):
    config = (await client.get("/api/residuals/config")).json()
    assert config["loaded"] is False
    assert config["hidden_size"] == 16
    assert config["default_layers"] == [6, 13, 20]
    demo = (await client.get("/api/residuals/demo")).json()
    assert demo["synthetic"] is True
    result = await client.post("/api/residuals/datasets/validate", json=demo["dataset"])
    assert result.status_code == 200
    assert result.json()["valid"] is True


@pytest.mark.parametrize("body", [b"[1]", b"null", b"{bad", b"\xff"])
async def test_bad_json_returns_safe_error(client, body):
    response = await client.post("/api/residuals/captures", content=body)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"
    assert response.headers["cache-control"] == "no-store"


async def test_body_size_and_encoding_are_checked_before_job_start(client):
    oversized = await client.post("/api/residuals/captures", content=b" " * (MAX_REQUEST_BYTES + 1))
    assert oversized.status_code == 413
    compressed = await client.post("/api/residuals/captures", content=b"{}", headers={"Content-Encoding": "gzip"})
    assert compressed.status_code == 415
    assert (await client.get("/api/residuals/jobs")).json()["jobs"] == []


async def test_invalid_ids_and_partial_group_filters_are_rejected(client):
    assert (await client.get("/api/residuals/jobs/invalid-id")).status_code == 404
    response = await client.get("/api/residuals/captures/" + "a" * 32 + "/summary?group_a=one")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


async def test_nested_invalid_values_do_not_escape_as_server_errors(client):
    dataset = (await client.get("/api/residuals/demo")).json()["dataset"]
    response = await client.post("/api/residuals/captures", json={"dataset": dataset, "points": [{}]})
    assert response.status_code == 400
    dataset["samples"][0]["messages"][0]["role"] = []
    response = await client.post("/api/residuals/datasets/validate", json=dataset)
    assert response.status_code == 400


async def test_integrated_routes_keep_local_request_protection():
    from explorer import server

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url=f"http://127.0.0.1:{server.PORT}",
    ) as session:
        assert (await session.get("/api/residuals/config")).status_code == 200
        response = await session.post(
            "/api/residuals/captures", json={},
            headers={"Origin": "https://example.org", "Sec-Fetch-Site": "cross-site"},
        )
        assert response.status_code == 403
        assert (await session.get("/residuals.html")).status_code == 200
