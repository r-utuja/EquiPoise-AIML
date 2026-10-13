"""FastAPI service: `/health`, `/predict`, `/breakeven`, `/early-response`, `/metadata`.

Thin HTTP wrapper over `Service`: request bodies are validated by the pydantic schemas
generated from the feature schema (ranges, category levels, no unknown fields), and every
answer is the same object the app shows. The bundle is loaded once at startup.
Research prototype, not for clinical use.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from equipoise.decision.recommend import DISCLAIMER
from equipoise.serving.schemas import (
    BreakEvenRequest,
    BreakEvenResponse,
    EarlyResponse,
    EarlyResponseRequest,
    Health,
    Prediction,
    PredictRequest,
)
from equipoise.serving.service import ATTRIBUTION, Service, ServiceUnavailable


def create_app(service: Service | None = None) -> FastAPI:
    """Build the app; pass a `Service` in tests, otherwise it is loaded from the environment
    (see `service.resolve_bundle_path`) at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or Service.from_env()
        yield

    app = FastAPI(title="Equipoise API", version="0.1.0", lifespan=lifespan,
                  description=f"{DISCLAIMER} {ATTRIBUTION}")

    def svc(request: Request) -> Service:
        return request.app.state.service

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        return svc(request).health()

    @app.get("/metadata")
    def metadata(request: Request) -> dict[str, Any]:
        return svc(request).metadata()

    @app.post("/predict", response_model=Prediction)
    def predict(req: PredictRequest, request: Request) -> Prediction:
        try:
            return svc(request).predict(req)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/breakeven", response_model=BreakEvenResponse)
    def breakeven(req: BreakEvenRequest, request: Request) -> BreakEvenResponse:
        try:
            return svc(request).breakeven(req)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.post("/early-response", response_model=EarlyResponse)
    def early_response(req: EarlyResponseRequest, request: Request) -> EarlyResponse:
        try:
            return svc(request).early_response(req)
        except ServiceUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app
