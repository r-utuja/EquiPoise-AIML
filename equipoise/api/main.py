"""ASGI entry point: `uvicorn api.main:app --reload` (docs at http://localhost:8000/docs).

The application itself lives in `equipoise.serving.api` so it is importable and tested as
part of the package. Set EQUIPOISE_SYNTHETIC=1 to serve the synthetic demo bundle.
"""

from equipoise.serving.api import create_app

app = create_app()
