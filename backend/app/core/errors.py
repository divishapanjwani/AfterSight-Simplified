"""Structured API errors: {"error": {"code", "message", "fix"?}} — never tracebacks or internal paths."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, fix: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message, self.fix = message, fix
        if code:
            self.code = code


class NotFound(AppError):
    status, code = 404, "not_found"


class InvalidRequest(AppError):
    status, code = 422, "invalid_request"


class NotReady(AppError):
    """Data or artifacts the request needs have not been built yet (503 + the command that fixes it)."""

    status, code = 503, "not_ready"


class ResultsInvalid(AppError):
    """A saved results file exists but cannot be used (bad JSON, missing keys). The file is at fault, not the request."""

    status, code = 503, "results_invalid"


class ModelNotTrained(AppError):
    status, code = 409, "model_not_trained"


class Conflict(AppError):
    status, code = 409, "conflict"


def _body(code: str, message: str, fix: str | None = None, details: list | None = None) -> dict:
    error: dict = {"code": code, "message": message}
    if fix:
        error["fix"] = fix
    if details:
        error["details"] = details
    return {"error": error}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(_r: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=_body(exc.code, exc.message, exc.fix))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_r: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only; never echo the rejected input back.
        details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content=_body("invalid_request", "The request is not valid.", details=details))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_r: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        message = "No such endpoint or resource." if exc.status_code == 404 else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_body(code, message))

    @app.exception_handler(Exception)
    async def unhandled(_r: Request, _exc: Exception) -> JSONResponse:
        # uvicorn still logs the traceback server-side
        return JSONResponse(status_code=500, content=_body("internal_error", "Unexpected server error."))
