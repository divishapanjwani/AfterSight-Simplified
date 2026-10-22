"""CSV and PDF exports for one analysis (written once to outputs/reports/, then served from disk)."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from app.core.config import SCOPE_LABEL
from app.evaluation.figures import heatmap

def report_context(store: Any) -> dict[str, Any]:
    """Numbers quoted in the limitations, read from saved artifacts (never typed in)."""
    model = store.training_config.default_model
    ctx: dict[str, Any] = {"model": model}
    for design in ("within_event", "cross_event"):
        doc = store.read_json(f"metrics/{design}/four_class/comparison.json") or {}
        row = next((m for m in doc.get("models", []) if m["model"] == model), None)
        ctx[design] = row["test_macro_f1"] if row else None
    return ctx


def limitations(context: dict[str, Any] | None) -> list[str]:
    context = context or {}
    scores = ""
    if context.get("within_event") is not None and context.get("cross_event") is not None:
        scores = (f" For the default model, test macro-F1 is {context['within_event']:.2f} on new scenes of known disasters "
                  f"but {context['cross_event']:.2f} on an unseen disaster (cross-event).")
    return [
        "Footprints are derived from rasterised xBD annotation masks, not original polygons; touching buildings may be merged.",
        "Classical ML on hand-crafted colour/texture features. Within-event scores are optimistic." + scores,
        "Confidence values are model scores (see 'confidence meaning'), not calibrated probabilities of being correct.",
        "Not for operational rescue or routing decisions. Pixel coordinates only — no geographic location or area.",
    ]


def write_csv(analysis: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(analysis["counts"]["ground_truth"])
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["building_uid", "scene_id", "split", "model", "predicted", "true_label", "correct",
                         "confidence", "confidence_kind", *[f"score_{c}" for c in classes], "needs_review",
                         "area_px", "mixed_label", "touches_border"])
        for b in analysis["buildings"]:
            writer.writerow([b["building_uid"], analysis["scene_id"], analysis["provenance"]["split"], analysis["model"]["name"],
                             b["predicted"], b["true_label"], b["correct"], b["confidence"], analysis["confidence_kind"],
                             *[b["scores"].get(c) for c in classes], b["needs_review"], b["area_px"], b["mixed_label"], b["touches_border"]])
    return path


def write_pdf(analysis: dict[str, Any], path: Path, context: dict[str, Any] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    p = lambda text, style="BodyText": Paragraph(text, styles[style])  # noqa: E731
    prov, model, counts = analysis["provenance"], analysis["model"], analysis["counts"]
    classes = list(counts["ground_truth"])
    story = [
        p("AfterSight — scene analysis report", "Title"),
        p(f"<i>{SCOPE_LABEL}</i>"),
        Spacer(1, 4 * mm),
        p(f"<b>Scene</b> {analysis['scene_id']} · event {analysis['event']} · disaster type {analysis['disaster_type'] or 'unknown'} (from config)"),
        p(f"<b>Model</b> {model['label']} · trained {model['trained_at']} · {model['design_label']} · held-out TEST macro-F1 {model['test_macro_f1']}"),
        p(f"<b>Split of this scene</b>: {prov['split'].upper()} — {prov['note']}"),
    ]
    if prov["warning"]:
        story.append(p(f"<font color='#b3400a'><b>Warning:</b> {prov['warning']}</font>"))
    story += [Spacer(1, 4 * mm), p("Buildings per class", "Heading2")]
    rows = [["class", "GROUND TRUTH", "PREDICTED"]] + [[c, counts["ground_truth"][c], counts["predicted"][c]] for c in classes]
    table = Table(rows, hAlign="LEFT")
    table.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("ALIGN", (1, 1), (-1, -1), "RIGHT")]))
    story.append(table)
    m = analysis["scene_metrics"]
    story += [Spacer(1, 4 * mm), p("Metrics on this scene", "Heading2")]
    if m:
        story.append(p(f"accuracy {m['accuracy']} · macro-F1 {m['macro_f1']} · weighted-F1 {m['weighted_f1']} · n={m['n']} "
                       f"<i>(computed on a {prov['split']} scene)</i>"))
        for note in m["notes"]:
            story.append(p(f"<i>{note}</i>"))
        fig = heatmap(np.array(m["confusion_matrix"]["rows_true_cols_pred"]), classes, classes,
                      f"Confusion matrix · {analysis['scene_id']}", f"{model['label']} · scene split: {prov['split']}",
                      "predicted", "ground truth", path.with_suffix(".confusion.png"))
        story.append(RLImage(str(fig), width=110 * mm, height=95 * mm))
    else:
        story.append(p("This scene has no labelled buildings."))
    reviewed = sum(b["needs_review"] for b in analysis["buildings"])
    story += [p(f"Suggested for human review: {reviewed} buildings (confidence below {analysis['review_threshold']}, "
                f"threshold chosen on validation). Confidence meaning: {analysis['confidence_kind']}."),
              Spacer(1, 4 * mm), p("Limitations", "Heading2"), *[p(f"• {t}") for t in limitations(context)],
              Spacer(1, 2 * mm), p(f"<font size=8>analysis {analysis['analysis_id']} · generated {analysis['created_at']}</font>")]
    SimpleDocTemplate(str(path), pagesize=A4, title=f"AfterSight {analysis['scene_id']}",
                      leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm).build(story)
    return path
