import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from explorer.som_study_api import STUDY_SCHEMA, create_som_study_router


def client(root):
    app = FastAPI()
    app.include_router(create_som_study_router(root))
    return TestClient(app)


def write_study(root, schema=STUDY_SCHEMA):
    path = root / "research_outputs" / "som_study" / "som-study-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": schema, "layers": []}))


def test_missing_study_is_404(tmp_path):
    response = client(tmp_path).get("/api/som-study")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "study_missing"


def test_study_served_with_no_store(tmp_path):
    write_study(tmp_path)
    response = client(tmp_path).get("/api/som-study")
    assert response.status_code == 200
    assert response.json()["schema"] == STUDY_SCHEMA
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("schema", ["other-v1", None])
def test_wrong_schema_rejected(tmp_path, schema):
    write_study(tmp_path, schema)
    response = client(tmp_path).get("/api/som-study")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "study_invalid"


def test_neuron_alignment_route(tmp_path):
    from explorer.som_study_api import ALIGNMENT_SCHEMA

    assert client(tmp_path).get("/api/som-study/neurons").status_code == 404
    path = tmp_path / "research_outputs" / "som_study" / "neuron-alignment-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": ALIGNMENT_SCHEMA, "directions": {}}))
    response = client(tmp_path).get("/api/som-study/neurons")
    assert response.status_code == 200 and response.json()["schema"] == ALIGNMENT_SCHEMA
