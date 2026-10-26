"""FastAPI routes for the local residual-groups workspace."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

try:
    from .residual_service import ResidualService, ResidualServiceError
except ImportError:  # server.py places explorer/ on sys.path
    from residual_service import ResidualService, ResidualServiceError


MAX_REQUEST_BYTES = 1024 * 1024


def _error(error: ResidualServiceError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message}},
        status_code=error.http_status,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except ResidualServiceError as error:
        return _error(error)


async def _json_body(request: Request) -> dict[str, Any] | JSONResponse:
    encoding = request.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in {"", "identity"}:
        return JSONResponse(
            {"error": {"code": "unsupported_encoding", "message": "Compressed request bodies are not accepted"}},
            status_code=415,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > MAX_REQUEST_BYTES:
                raise ValueError
        except ValueError:
            return JSONResponse(
                {"error": {"code": "request_too_large", "message": "Request body exceeds the 1 MiB limit"}},
                status_code=413,
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            return JSONResponse(
                {"error": {"code": "request_too_large", "message": "Request body exceeds the 1 MiB limit"}},
                status_code=413,
                headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            )
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return JSONResponse(
            {"error": {"code": "invalid_json", "message": "Request body must be valid JSON"}},
            status_code=400,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    if not isinstance(value, dict):
        return JSONResponse(
            {"error": {"code": "invalid_json", "message": "Request body must be a JSON object"}},
            status_code=400,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )
    return value


def create_residual_router(rt: Any, repo_root: str | Path) -> tuple[APIRouter, ResidualService]:
    """Create the router and its shared-runtime service.

    The service is returned separately so the app lifespan can call close().
    """
    service = ResidualService(rt, repo_root)
    router = APIRouter(prefix="/api/residuals", tags=["residual-groups"])

    @router.get("/config")
    def config() -> Any:
        return _call(service.config, rt)

    @router.get("/demo")
    def demo() -> Any:
        return _call(service.demo)

    @router.post("/datasets/validate")
    async def validate_dataset(request: Request) -> Any:
        payload = await _json_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        dataset = payload.get("dataset", payload)
        return await run_in_threadpool(_call, service.validate, dataset)

    @router.get("/jobs")
    def list_jobs() -> Any:
        return _call(service.list_jobs)

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> Any:
        return _call(service.get_job, job_id)

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Any:
        return _call(service.cancel_job, job_id)

    @router.post("/captures")
    async def start_capture(request: Request) -> Any:
        payload = await _json_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        return await run_in_threadpool(_call, service.start_capture, payload)

    @router.get("/captures")
    def list_captures() -> Any:
        return _call(service.list_captures)

    @router.get("/captures/{capture_id}/summary")
    def capture_summary(
        capture_id: str,
        layer: int | None = Query(default=None),
        point: str | None = Query(default=None),
        group_a: str | None = Query(default=None),
        group_b: str | None = Query(default=None),
    ) -> Any:
        if (group_a is None) != (group_b is None):
            return _error(ResidualServiceError("invalid_input", "group_a and group_b must be provided together"))
        return _call(service.get_capture_summary, capture_id, layer=layer, point=point, group_a=group_a, group_b=group_b)

    @router.get("/captures/{capture_id}/map")
    def capture_map(
        capture_id: str,
        layer: int = Query(...),
        point: str = Query(...),
        group_a: str = Query(...),
        group_b: str = Query(...),
        split: str = Query(default="all"),
    ) -> Any:
        return _call(
            service.get_capture_map,
            capture_id,
            layer=layer,
            point=point,
            group_a=group_a,
            group_b=group_b,
            split=split,
        )

    @router.get("/captures/{capture_id}/sample")
    def capture_sample(capture_id: str, sample_id: str = Query(...)) -> Any:
        return _call(service.get_capture_sample, capture_id, sample_id)

    @router.get("/captures/{capture_id}/export")
    def export_capture(capture_id: str) -> Any:
        try:
            path = service.export_capture(capture_id)
        except ResidualServiceError as error:
            return _error(error)
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"residual-capture-{capture_id}.npz",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @router.post("/fits")
    async def start_fit(request: Request) -> Any:
        payload = await _json_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        return await run_in_threadpool(_call, service.start_fit, payload)

    @router.get("/fits")
    def list_fits() -> Any:
        return _call(service.list_fits)

    @router.get("/fits/{fit_id}")
    def get_fit(fit_id: str) -> Any:
        return _call(service.get_fit, fit_id)

    @router.get("/fits/{fit_id}/cells/{cell_id}")
    def get_fit_cell(fit_id: str, cell_id: int, split: str = Query(default="all")) -> Any:
        return _call(service.get_fit_cell, fit_id, cell_id, split)

    @router.post("/probes")
    async def start_probe(request: Request) -> Any:
        payload = await _json_body(request)
        if isinstance(payload, JSONResponse):
            return payload
        return await run_in_threadpool(_call, service.start_probe, payload)

    @router.get("/probes/{probe_id}")
    def get_probe(probe_id: str) -> Any:
        return _call(service.get_probe, probe_id)

    return router, service
