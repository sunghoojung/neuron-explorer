"""Read-only route for the SOM sycophancy steering study summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

STUDY_SCHEMA = "som-steering-study-v1"
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

    return router
