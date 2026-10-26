"""Read-only HTTP routes for receipt-verified taxonomy SOM atlases."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

try:
    from .atlas_service import AtlasService, AtlasServiceError
except ImportError:  # server.py places explorer/ on sys.path
    from atlas_service import AtlasService, AtlasServiceError


def _call(function, *args: Any) -> Any:
    try:
        return function(*args)
    except AtlasServiceError as error:
        return JSONResponse(
            {"error": {"code": error.code, "message": error.message}},
            status_code=error.http_status,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )


def create_atlas_router(repo_root: str | Path) -> APIRouter:
    service = AtlasService(repo_root)
    router = APIRouter(prefix="/api/atlases", tags=["sycophancy-atlases"])

    @router.get("")
    def list_atlases() -> Any:
        return _call(service.list_atlases)

    @router.get("/{atlas_id}")
    def get_atlas(atlas_id: str) -> Any:
        return _call(service.get_atlas, atlas_id)

    return router
