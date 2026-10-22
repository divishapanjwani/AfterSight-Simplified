"""FastAPI dependencies: one Store and one JobManager per process (tests override get_store)."""

from functools import lru_cache

from app.core.config import get_settings
from app.core.jobs import JobManager
from app.core.store import Store, StorePaths


@lru_cache
def get_store() -> Store:
    return Store(StorePaths.from_settings(get_settings()))


@lru_cache
def get_jobs() -> JobManager:
    store = get_store()
    return JobManager(store.paths.artifacts / "jobs", on_success=store.forget_models)
