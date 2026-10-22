"""AfterSight API entry point.

Run (from backend/):  uvicorn app.main:app --reload --port 8020
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import health, routes
from app.core.config import get_settings
from app.core.errors import install_error_handlers


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AfterSight API",
        version=__version__,
        description="Explainable building damage triage — classical ML. Damage classification on building footprints derived from xBD annotations.",
    )

    # Only the Vite dev server may call the API from a browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    install_error_handlers(app)  # structured errors only: no tracebacks, no internal paths
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(routes.router, prefix="/api")
    return app


app = create_app()
