"""AfterSight API entry point.

Run (from backend/):  uvicorn app.main:app --reload --port 8020
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import health
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AfterSight API",
        version=__version__,
        description="Explainable building damage triage — classical ML on provided xBD building footprints.",
    )

    # Only the Vite dev server may call the API from a browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Request, _exc: Exception) -> JSONResponse:
        # Never leak stack traces or internal paths to the client; uvicorn still logs the traceback.
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Unexpected server error."}},
        )

    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
