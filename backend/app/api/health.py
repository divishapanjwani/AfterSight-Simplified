from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.core.config import SCOPE_LABEL

router = APIRouter()



class HealthResponse(BaseModel):
    status: str
    version: str
    scope: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, scope=SCOPE_LABEL)
