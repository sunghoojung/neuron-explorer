"""Read-only route for the SOM sycophancy steering study summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

STUDY_SCHEMA = "som-steering-study-v1"
ALIGNMENT_SCHEMA = "som-neuron-write-alignment-v1"
MATCHED_PAIR_SCHEMA = "som-matched-pair-study-v1"
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def load_study(repo_root: str | Path) -> dict:
    path = Path(repo_root) / "research_outputs" / "som_study" / "som-study-v1.json"
    study = json.loads(path.read_bytes())
    if study.get("schema") != STUDY_SCHEMA:
        raise ValueError("unexpected study schema")
    return study


def create_som_study_router(repo_root: str | Path) -> APIRouter:
    router = APIRouter(prefix="/api/som-study", tags=["som-study"])

    @router.get("")
    def get_study() -> Any:
        try:
            return JSONResponse(load_study(repo_root), headers=HEADERS)
        except FileNotFoundError:
            return JSONResponse(
                {"error": {"code": "study_missing", "message": "Run research/build_som_study.py first."}},
                status_code=404, headers=HEADERS,
            )
        except ValueError as error:
            return JSONResponse(
                {"error": {"code": "study_invalid", "message": str(error)}},
                status_code=500, headers=HEADERS,
            )

    @router.get("/neurons")
    def get_neurons() -> Any:
        path = Path(repo_root) / "research_outputs" / "som_study" / "neuron-alignment-v1.json"
        try:
            data = json.loads(path.read_bytes())
        except FileNotFoundError:
            return JSONResponse(
                {"error": {"code": "alignment_missing", "message": "Run research/build_som_neuron_alignment.py first."}},
                status_code=404, headers=HEADERS,
            )
        if data.get("schema") != ALIGNMENT_SCHEMA:
            return JSONResponse({"error": {"code": "alignment_invalid", "message": "unexpected alignment schema"}},
                                status_code=500, headers=HEADERS)
        return JSONResponse(data, headers=HEADERS)

    @router.get("/matched-pair")
    def get_matched_pair() -> Any:
        path = Path(repo_root) / "research_outputs" / "som_study" / "matched-pair-v1.json"
        try:
            data = json.loads(path.read_bytes())
        except FileNotFoundError:
            return JSONResponse(
                {"error": {"code": "matched_pair_missing", "message": "Run research/build_som_matched_pair.py first."}},
                status_code=404, headers=HEADERS,
            )
        if data.get("schema") != MATCHED_PAIR_SCHEMA:
            return JSONResponse({"error": {"code": "matched_pair_invalid", "message": "unexpected matched-pair schema"}},
                                status_code=500, headers=HEADERS)
        return JSONResponse(data, headers=HEADERS)

    return router
