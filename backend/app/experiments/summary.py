"""Reading the saved Phase 3b summary (artifacts/experiments/phase3b/summary.json) for the Model Arena.

The file is written by scripts/run_phase3b.py (app/experiments/phase3b.py). This module only checks that it
has what the Arena shows; the saved values are passed through exactly as written, never replaced or filled in.
"""

import json
from typing import Any

from app.core.errors import NotFound, ResultsInvalid
from app.core.store import FIX, Store

SUMMARY = "experiments/phase3b/summary.json"
# every row needs these (event_prediction_accuracy_within_event may be null, but the key must be there)
ROW_NUMBERS = ("n_features", "test_macro_f1", "test_recall_major", "test_recall_destroyed")
ROW_TEXT = ("variant", "design", "model")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summary_problem(doc: Any) -> str | None:
    """What is wrong with a parsed summary, in one sentence (None when it is complete)."""
    if not isinstance(doc, dict):
        return "the summary is not a JSON object"
    for key, kind in (("label", str), ("variants", dict), ("rows", list)):
        if not isinstance(doc.get(key), kind):
            return f"the summary has no valid '{key}'"
    if not doc["rows"]:
        return "the summary has no result rows"
    for i, row in enumerate(doc["rows"], start=1):
        if not isinstance(row, dict):
            return f"row {i} is not an object"
        for key in ROW_TEXT:
            if not isinstance(row.get(key), str):
                return f"row {i} is missing '{key}'"
        for key in ROW_NUMBERS:
            if not _is_number(row.get(key)):
                return f"row {i} ({row['variant']}, {row['model']}) is missing '{key}'"
        event_accuracy = row.get("event_prediction_accuracy_within_event", "missing")
        if event_accuracy is not None and not _is_number(event_accuracy):
            return f"row {i} ({row['variant']}, {row['model']}) is missing 'event_prediction_accuracy_within_event'"
        if row["variant"] not in doc["variants"]:
            return f"row {i} names variant '{row['variant']}', which is not described in 'variants'"
    return None


def summary_state(store: Store) -> tuple[str, dict[str, Any] | None, str | None]:
    """('available', doc, None), ('missing', None, None) or ('invalid', None, reason)."""
    path = store.paths.artifacts / SUMMARY
    if not path.is_file():
        return "missing", None, None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "invalid", None, "the summary file is not valid JSON"
    problem = summary_problem(doc)
    if problem:
        return "invalid", None, problem
    return "available", doc, None


def load_summary(store: Store) -> dict[str, Any]:
    """The saved summary, or a structured error the Arena can show (with the command that rebuilds it)."""
    state, doc, problem = summary_state(store)
    if state == "missing":
        raise NotFound("No experiments have been run.", code="results_not_found", fix=FIX["experiments"])
    if state == "invalid":
        raise ResultsInvalid(f"The saved Phase 3b summary can't be used: {problem}.", fix=FIX["experiments"])
    return doc
