"""Background jobs (training) run as subprocesses and are polled by id. One training job at a time.

Commands are built from validated enum values only — no request text ever reaches the command line.
"""

import subprocess
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import REPO_ROOT
from app.core.errors import Conflict, NotFound


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self, log_dir: Path, on_success: Callable[[], None] | None = None):
        self.log_dir = log_dir
        self.on_success = on_success
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def training_command(self, designs: list[str] | None, schemes: list[str] | None, include_optional: bool) -> list[str]:
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py")]
        for d in designs or []:
            cmd += ["--design", d]
        for s in schemes or []:
            cmd += ["--scheme", s]
        if not include_optional:
            cmd.append("--no-optional")
        return cmd

    def start(self, kind: str, command: list[str]) -> dict[str, Any]:
        with self._lock:
            if any(j["kind"] == kind and j["status"] in ("queued", "running") for j in self._jobs.values()):
                raise Conflict(f"A {kind} job is already running.", code="job_running")
            job_id = uuid.uuid4().hex[:16]
            job = {"job_id": job_id, "kind": kind, "status": "queued", "created_at": _now(), "started_at": None,
                   "finished_at": None, "return_code": None, "_tail": deque(maxlen=40)}
            self._jobs[job_id] = job
        threading.Thread(target=self._run, args=(job, command), daemon=True).start()
        return self.public(job)

    def _run(self, job: dict[str, Any], command: list[str]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job["status"], job["started_at"] = "running", _now()
        with (self.log_dir / f"{job['job_id']}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:  # type: ignore[union-attr]
                log.write(line)
                job["_tail"].append(line.rstrip().replace(str(REPO_ROOT), "<repo>"))  # no internal paths in the API
            code = proc.wait()
        job["return_code"], job["finished_at"] = code, _now()
        job["status"] = "succeeded" if code == 0 else "failed"
        if code == 0 and self.on_success:
            self.on_success()

    def get(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFound("Unknown job id.", code="job_not_found")
        return self.public(job)

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        return {**{k: v for k, v in job.items() if not k.startswith("_")}, "log_tail": list(job["_tail"])}
