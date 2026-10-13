"""App-side client: the same five calls whether the app talks to the API or to the bundle.

`ApiClient` posts JSON to the FastAPI service (used when `API_URL` is set, e.g. in
docker-compose). `LocalClient` validates the same payloads with the same pydantic schemas
and calls `Service` in-process, so the app can run as a single container. Both return
plain dicts (the API's JSON), so the app code is identical in either mode.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from equipoise.config import Config, load_config
from equipoise.serving.schemas import BreakEvenRequest, EarlyResponseRequest, PredictRequest
from equipoise.serving.service import Service


class Client(Protocol):
    mode: str

    def health(self) -> dict[str, Any]: ...
    def metadata(self) -> dict[str, Any]: ...
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def breakeven(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def early_response(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ClientError(RuntimeError):
    """A request was rejected (validation) or the capability is unavailable."""


class LocalClient:
    mode = "local bundle"

    def __init__(self, service: Service) -> None:
        self.service = service

    def _call(self, fn, model, payload: dict[str, Any]) -> dict[str, Any]:
        from pydantic import ValidationError

        from equipoise.serving.service import ServiceUnavailable
        try:
            return fn(model.model_validate(payload)).model_dump(mode="json")
        except (ValidationError, ValueError, ServiceUnavailable) as e:
            raise ClientError(str(e)) from e

    def health(self) -> dict[str, Any]:
        return self.service.health().model_dump(mode="json")

    def metadata(self) -> dict[str, Any]:
        return self.service.metadata()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call(self.service.predict, PredictRequest, payload)

    def breakeven(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call(self.service.breakeven, BreakEvenRequest, payload)

    def early_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call(self.service.early_response, EarlyResponseRequest, payload)


class ApiClient:
    mode = "API"

    def __init__(self, base_url: str, timeout: float) -> None:
        import requests

        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.http = requests.Session()

    def _get(self, path: str) -> dict[str, Any]:
        r = self.http.get(self.base + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post(self.base + path, json=payload, timeout=self.timeout)
        if r.status_code in (422, 503):
            raise ClientError(str(r.json().get("detail")))
        r.raise_for_status()
        return r.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def metadata(self) -> dict[str, Any]:
        return self._get("/metadata")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/predict", payload)

    def breakeven(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/breakeven", payload)

    def early_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/early-response", payload)


def make_client(cfg: Config | None = None) -> Client:
    """`ApiClient` when `$API_URL` is set, else `LocalClient` on the bundle."""
    cfg = cfg or load_config()
    s = cfg.raw["serving"]
    url = os.environ.get(s["api_url_env_var"])
    if url:
        return ApiClient(url, float(s["request_timeout_s"]))
    return LocalClient(Service.from_env(cfg))
