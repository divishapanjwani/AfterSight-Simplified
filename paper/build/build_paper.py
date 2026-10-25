"""Build paper/AfterSight_KNN_vs_DT_IEEE.docx (provisional IEEE-style two-column layout).

Every number in the text and tables is read from saved artifacts at build time:
  artifacts/models/<design>/four_class/{knn,decision_tree}.json, artifacts/metrics/<design>/four_class/comparison.json,
  data/splits/<design>_v1.json, artifacts/inspection/xbd/inspection_report.json,
  artifacts/analyses/confounding/within_event.json, paper/build/stats.json (scripts/paper_stats.py).

Usage (repo root): paper/build/.venv/bin/python paper/build/build_paper.py
The official IEEE template could not be downloaded (ieee.org returned an empty response to scripted requests), so
this is a provisional layout: move the text into the official IEEE conference template before submission.
"""

import json
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "AfterSight_Final_Research_Paper.docx"
FIG = ROOT / "paper" / "figures"
CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
EVENT_NAME = {"hurricane-harvey": "Hurricane Harvey", "hurricane-matthew": "Hurricane Matthew",
              "hurricane-michael": "Hurricane Michael", "palu-tsunami": "Palu tsunami",
              "santa-rosa-wildfire": "Santa Rosa wildfire"}
CLASS_NAME = {"no-damage": "No damage", "minor-damage": "Minor", "major-damage": "Major", "destroyed": "Destroyed"}


# ---------------------------------------------------------------- evidence
def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


M = {d: {m: load(f"artifacts/models/{d}/four_class/{m}.json") for m in ("knn", "decision_tree")}
     for d in ("within_event", "cross_event")}
CMP = {d: {r["model"]: r for r in load(f"artifacts/metrics/{d}/four_class/comparison.json")["models"]}
       for d in ("within_event", "cross_event")}
SPLIT = {d: load(f"data/splits/{d}_v1.json") for d in ("within_event", "cross_event")}
INSP = load("artifacts/inspection/xbd/inspection_report.json")
CONF = load("artifacts/analyses/confounding/within_event.json")
ST = load("paper/build/stats.json")

EVENTS = SPLIT["within_event"]["feature_table_fingerprint"]["events"]

# Scene accounting from the per-scene inspection table: why 1,239 cached pairs become 1,179 in every split.
import csv  # noqa: E402

with (ROOT / "artifacts" / "inspection" / "xbd" / "scenes.csv").open(encoding="utf-8") as fh:
    SCENES = [r for r in csv.DictReader(fh) if r["event"] in EVENTS]
SC_NO_BLOB = [r for r in SCENES if int(r["n_raw_footprints"]) == 0]
SC_ALL_DROPPED = [r for r in SCENES if int(r["n_raw_footprints"]) > 0 and int(r["n_buildings"]) == 0]
SC_WITH = [r for r in SCENES if int(r["n_buildings"]) > 0]
assert len(SC_NO_BLOB) + len(SC_ALL_DROPPED) + len(SC_WITH) == len(SCENES)
assert set(r["scene_id"] for r in SC_WITH) == set(SPLIT["within_event"]["scenes"]) == set(SPLIT["cross_event"]["scenes"])
ALL_DROPPED_BLOBS = sum(int(r["n_raw_footprints"]) for r in SC_ALL_DROPPED)
MICHAEL_CACHED = sum(1 for r in SCENES if r["event"] == "hurricane-michael")
MICHAEL_WITH = sum(1 for r in SC_WITH if r["event"] == "hurricane-michael")
TOT = INSP["totals"]
CC = INSP["class_counts"]
N_BUILD = sum(CC["overall"].values())


def t(design: str, model: str) -> dict:
    return M[design][model]["metrics"]["test"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(design: str, model: str, key: str = "macro_f1_ci95") -> str:
    low, high = ST["designs"][design]["models"][model][key]
    return f"{low:.3f}–{high:.3f}"


def diff(design: str) -> dict:
    return ST["designs"][design]["paired_difference_knn_minus_dt"]


def n(x: int) -> str:
    return f"{x:,}"


michael_minor = CC["by_event"]["hurricane-michael"]["minor-damage"] / sum(CC["by_event"]["hurricane-michael"].values())
other = {c: sum(CC["by_event"][e][c] for e in EVENTS if e != "hurricane-michael") for c in CLASSES}
other_minor = other["minor-damage"] / sum(other.values())
log1p = M["within_event"]["knn"]["preprocessing"]["log1p_features"]
per_event = {m: M["within_event"][m]["metrics"]["test_per_event"] for m in ("knn", "decision_tree")}

# ---------------------------------------------------------------- document helpers
doc = Document()
styles = doc.styles


def set_font(style, size: float, bold: bool = False, italic: bool = False) -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), "Times New Roman")
    # heading styles carry theme fonts (e.g. Calibri Light) that override the explicit font; remove them
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if fonts.get(qn(attr)) is not None:
            del fonts.attrib[qn(attr)]


normal = styles["Normal"]
set_font(normal, 10)
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.line_spacing = 1.0

for name, size in (("Heading 1", 10), ("Heading 2", 10)):
    st = styles[name]
    set_font(st, size, italic=(name == "Heading 2"))
    st.font.small_caps = name == "Heading 1"
    st.paragraph_format.space_before = Pt(8 if name == "Heading 1" else 4)
    st.paragraph_format.space_after = Pt(3 if name == "Heading 1" else 2)
    st.paragraph_format.keep_with_next = True
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if name == "Heading 1" else WD_ALIGN_PARAGRAPH.LEFT
    st.paragraph_format.first_line_indent = Pt(0)


def set_columns(section, num: int) -> None:
    sect = section._sectPr
    cols = sect.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), "360")  # 0.25 in between columns


def page_setup(section) -> None:
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(0.75), Inches(1.0)
    section.left_margin = section.right_margin = Inches(0.625)
    section.header_distance = Inches(0.3)


def rich(p, text: str, size: float | None = None) -> None:
    """Add text with *italic*, **bold**, _{subscript} and ^{superscript} markup (no nesting)."""
    text = text.replace("ᵖʳᵉ", "^{pre}").replace("ᵖᵒˢᵗ", "^{post}")
    text = re.sub(r"_(?!\{)([A-Za-z0-9])", r"_{\1}", text)
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})", text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("_{"):
            r = p.add_run(part[2:-1])
            r.font.subscript = True
        elif part.startswith("^{"):
            r = p.add_run(part[2:-1])
            r.font.superscript = True
        else:
            r = p.add_run(part)
        if size:
            r.font.size = Pt(size)


def para(text: str, indent: bool = True, size: float | None = None, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = align
    p.paragraph_format.first_line_indent = Inches(0.14) if indent else Inches(0)
    rich(p, text, size)
    return p


def h1(text: str) -> None:
    doc.add_paragraph(text, style="Heading 1")


def h2(text: str) -> None:
    doc.add_paragraph(text, style="Heading 2")


EQ = [0]


def equation(expr: str) -> None:
    EQ[0] += 1
    number = EQ[0]
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(1.72), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.48), WD_TAB_ALIGNMENT.RIGHT)
    p.add_run("\t")
    rich(p, expr)
    p.add_run(f"\t({number})")


def figure(path: Path, caption: str, width: float = 3.45) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    c.paragraph_format.space_after = Pt(6)
    rich(c, caption, 8)


def table_caption(number: str, title: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"TABLE {number}")
    r.font.size = Pt(8)
    p2 = doc.add_paragraph()
    p2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(2)
    p2.paragraph_format.keep_with_next = True
    r2 = p2.add_run(title)
    r2.font.size = Pt(8)
    r2.font.small_caps = True


def border(cell, edges: dict[str, str]) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, size in edges.items():
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), size)
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tcpr.append(borders)


def table(rows: list[list[str]], widths: list[float], header_rows: int = 1, bold_rows: tuple = (), size: float = 7.5,
          note: str | None = None) -> None:
    tb = doc.add_table(rows=len(rows), cols=len(rows[0]))
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    tb.autofit = False
    # fixed layout + total width + grid columns, so Word and LibreOffice keep the table inside the column
    tblpr = tb._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblpr.append(layout)
    total = OxmlElement("w:tblW")
    total.set(qn("w:w"), str(int(sum(widths) * 1440)))
    total.set(qn("w:type"), "dxa")
    for old in tblpr.findall(qn("w:tblW")):
        tblpr.remove(old)
    tblpr.append(total)
    for j, gc in enumerate(tb._tbl.tblGrid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(widths[j] * 1440)))
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = tb.cell(i, j)
            cell.width = Inches(widths[j])
            p = cell.paragraphs[0]
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            rich(p, value, size)
            if i in bold_rows:
                for r in p.runs:
                    r.bold = True
            edges = {}
            if i == 0:
                edges["top"] = "8"
            if i == header_rows - 1:
                edges["bottom"] = "4"
            if i == len(rows) - 1:
                edges["bottom"] = "8"
            if edges:
                border(cell, edges)
            tcpr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement("w:tcMar")
            for side in ("left", "right"):
                e = OxmlElement(f"w:{side}")
                e.set(qn("w:w"), "40")
                e.set(qn("w:type"), "dxa")
                mar.append(e)
            tcpr.append(mar)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        rich(p, note, 7)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


# ---------------------------------------------------------------- front matter (one column)
first = doc.sections[0]
page_setup(first)
header = first.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.CENTER
hr = header.add_run("PROVISIONAL IEEE-STYLE LAYOUT (not the official template). Venue not yet chosen; "
                    "move this text into the official IEEE conference template before submission.")
hr.font.size, hr.italic = Pt(7), True
hr.font.color.rgb = RGBColor(0x80, 0, 0)

title = doc.add_paragraph()
title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(10)
tr = title.add_run("K-Nearest Neighbours versus Decision Trees for Building Damage Classification from "
                   "Satellite Imagery: Within-Event and Cross-Event Evaluation")
tr.font.size = Pt(22)

authors = [("Author Name", "Dept. of Computer Engineering", "Institution, City, Country", "email@example.com")] * 4 + \
          [("Guide Name", "Dept. of Computer Engineering", "Institution, City, Country", "email@example.com")]
ab = doc.add_table(rows=2, cols=3)
ab.alignment = WD_TABLE_ALIGNMENT.CENTER
cells = [ab.cell(0, 0), ab.cell(0, 1), ab.cell(0, 2), ab.cell(1, 0), ab.cell(1, 1)]
for cell, (name, dept, inst, mail) in zip(cells, authors):
    cell.width = Inches(2.4)
    lines = [(name, 10, False), (dept, 9, True), (inst, 9, True), (mail, 9, False)]
    p = cell.paragraphs[0]
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for k, (text, size, italic) in enumerate(lines):
        r = p.add_run(text)
        r.font.size, r.italic = Pt(size), italic
        if k < len(lines) - 1:
            r.add_break()
guide_note = ab.cell(1, 2).paragraphs[0]
guide_note.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
gn = guide_note.add_run("[Author block placeholders: replace with the four authors and the project guide, in the agreed order.]")
gn.font.size, gn.italic = Pt(7), True
gn.font.color.rgb = RGBColor(0x80, 0, 0)

body = doc.add_section(WD_SECTION.CONTINUOUS)
page_setup(body)
set_columns(body, 2)

# ---------------------------------------------------------------- abstract
w, x = "within_event", "cross_event"
abstract = (
    "Rapid building-level damage estimates help prioritise field surveys after a disaster. This study "
    "compares two classical, interpretable classifiers, k-nearest neighbours (KNN) and a decision tree (DT), for "
    "four-class building damage classification when each building is described by 37 engineered pre- and post-event "
    f"image features covering colour, change, texture, surrounding context and shape. Building regions are derived "
    f"from the annotation masks of the xBD training split; five disaster events provide {n(N_BUILD)} labelled buildings "
    f"from {n(len(SC_WITH))} image pairs. Both classifiers share one preprocessing pipeline fitted on training "
    "data, are tuned on validation macro-F1 and are evaluated once on held-out test scenes under two designs: a "
    "scene-grouped within-event split and a cross-event split that holds out Hurricane Michael. Within events, KNN "
    f"reaches a test macro-F1 of {f3(t(w, 'knn')['macro_f1'])} (95% scene-level bootstrap interval {ci(w, 'knn')}) "
    f"against {f3(t(w, 'decision_tree')['macro_f1'])} ({ci(w, 'decision_tree')}) for the DT. On the held-out event both "
    f"models degrade and the ranking reverses ({f3(t(x, 'knn')['macro_f1'])} versus {f3(t(x, 'decision_tree')['macro_f1'])}; "
    f"paired difference {diff(x)['macro_f1']:+.3f}, interval {diff(x)['ci95'][0]:+.3f} to {diff(x)['ci95'][1]:+.3f}), while "
    "KNN keeps the higher accuracy. The DT used balanced class weights and KNN did not, which limits causal "
    "interpretation. The results show that conclusions depend on the evaluation design and metric, and that "
    "transfer to a new event is the main weakness; a single held-out event does not support general claims."
)
ap = doc.add_paragraph()
ap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
lead = ap.add_run("Abstract—")
lead.bold, lead.italic, lead.font.size = True, True, Pt(9)
rest = ap.add_run(abstract)
rest.bold, rest.font.size = True, Pt(9)
kw = doc.add_paragraph()
kw.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
kw.paragraph_format.space_before = Pt(4)
kw.paragraph_format.space_after = Pt(6)
lead = kw.add_run("Index Terms—")
lead.bold, lead.italic, lead.font.size = True, True, Pt(9)
rest = kw.add_run("building damage assessment, xBD, k-nearest neighbours, decision tree, change features, "
                  "cross-event generalisation, remote sensing.")
rest.bold, rest.font.size = True, Pt(9)

# ---------------------------------------------------------------- I. Introduction
h1("I. Introduction")
para("After a hurricane, flood, wildfire or tsunami, responders need an early picture of which buildings are damaged "
     "and how severely. Paired satellite images acquired before and after an event cover a whole affected area, but "
     "manual inspection of thousands of buildings is slow. The xBD dataset [1], [2] made this a standard machine "
     "learning task by providing pre- and post-event image pairs with building annotations graded on a four-level "
     "damage scale. Much of the subsequent work uses deep convolutional networks [5], [6], which learn features "
     "directly from pixels.")
para("Classical classifiers on engineered features remain attractive for teaching and for low-resource settings, and "
     "each of their predictions can be traced to measurable quantities. Two such classifiers explain "
     "themselves in different ways: k-nearest neighbours (KNN) [11] justifies a label with similar labelled examples, "
     "and a decision tree (DT) [12] with an explicit sequence of threshold tests. How these two compare on building "
     "damage, and how their comparison behaves when a model meets a disaster it has not seen, is the subject of this "
     "paper. We address three research questions. **RQ1:** How do KNN and a DT compare on four-class damage "
     "classification of new scenes from disasters represented in training? **RQ2:** How does the comparison change "
     "when an entire event is held out? **RQ3:** Which damage classes account for the errors under each design?")
para("The study is built on AfterSight, an explainable damage-triage application developed by the authors, whose "
     "pipeline fixes the data, features and evaluation protocol. The contributions are: (i) a leakage-controlled "
     "protocol with scene-grouped and event-held-out splits, preprocessing fitted on training data only and a single "
     "test evaluation per model; (ii) a KNN–DT comparison reporting per-class results and scene-level bootstrap "
     "intervals rather than single scores; and (iii) an explicit account of the protocol differences between the two "
     "models and of how the conclusion depends on the metric. This is an applied comparative study; it proposes no new "
     "algorithm or dataset and makes no state-of-the-art claim. Fig. 1 summarises the pipeline.")
figure(FIG / "fig1_pipeline.png", "Fig. 1. Implemented pipeline used for both classifiers. Building regions come from "
       "annotation masks, not from a detector.")

# ---------------------------------------------------------------- II. Related Work
h1("II. Related Work")
para("Remote sensing has long been used to map building damage. Reviews of earthquake damage detection [3] distinguish "
     "post-event-only from pre/post change-based approaches, and classical change detection compares co-registered "
     "images through differencing, ratios and transformed features [4]. Texture and structure descriptors are "
     "particularly useful at high spatial resolution: Cooner *et al.* [7] classified damage after the 2010 Haiti "
     "earthquake with random forests and neural networks and reported that texture and structure variables mattered "
     "more than spectral ones. Our features follow this tradition, using grey-level co-occurrence statistics [9] and "
     "Canny edges [10].")
para("xBD [1], [2] provides imagery for many disasters, targeted below 0.8 m ground sample distance, with four damage "
     "levels. Deep models such as RescueNet [5] and BDANet [6] jointly segment buildings and grade damage on this data. "
     "Generalisation across events is a central concern: Valentijn *et al.* [8] tested a convolutional network on "
     "13 xBD disasters and found that performance, measured by the area under the ROC curve, differed strongly "
     "between disasters yet remained high without labelled data from the test disaster. We study the same question "
     "for classical classifiers on engineered features, measured by four-class macro-F1.")
para("KNN [11] and classification trees [12] are standard methods implemented in scikit-learn [13]. With imbalanced "
     "classes, accuracy can be dominated by the majority class [14], so macro-averaged F1 is commonly preferred [15]. "
     "Because the buildings of one image share acquisition conditions, we quantify uncertainty with a bootstrap [16] "
     "that resamples whole scenes, as recommended for clustered data [17].")

# ---------------------------------------------------------------- III. Data
h1("III. Dataset and Problem Formulation")
h2("A. Source and Event Selection")
n_mirror_events = len(INSP["all_events"])
n_mirror_scenes = sum(v["scenes"] for v in INSP["all_events"].values())
para(f"We use a third-party Parquet repackaging of the xBD *training* split ({n_mirror_events} events, "
     f"{n(n_mirror_scenes)} pre/post image pairs of 1024 × 1024 RGB pixels). The official xBD test and hold-out splits "
     "are not part of this copy, so all evaluation partitions are carved from the training split. Each record holds the "
     "pre- and post-event images, a binary building-footprint mask and a damage mask with values 1–4 for no, minor, "
     "major and destroyed damage. Five events were chosen in an earlier project phase to cover four hazard types "
     "with substantial numbers of severely damaged buildings: Hurricanes Harvey, Matthew and Michael, the Palu "
     "tsunami and the Santa Rosa wildfire. The choice was a design decision rather than a formal rule (for example, "
     "Santa Rosa contributes many destroyed but only 55 major-damage buildings), and it is treated as a threat to "
     "validity in Section VII. Table I lists the resulting counts.")
rows = [["Event", "Scenes", "No dmg.", "Minor", "Major", "Destr."]]
for e in EVENTS:
    c = CC["by_event"][e]
    rows.append([EVENT_NAME[e], n(INSP["all_events"][e]["scenes"])] + [n(c[k]) for k in CLASSES])
rows.append(["Total", n(TOT["scenes_discovered"])] + [n(CC["overall"][k]) for k in CLASSES])
table_caption("I", "Labelled buildings per event and damage class")
table(rows, [1.05, 0.48, 0.5, 0.45, 0.45, 0.45], bold_rows=(len(rows) - 1,),
      note=f"Scenes = cached image pairs; {n(len(SC_WITH))} of them contain at least one labelled building (Sec. III-B).")
h2("B. Building Regions from Annotation Masks")
dr = INSP["drop_reasons"]
para("The copy contains no building polygons, so building regions are reconstructed from the masks rather than "
     "detected: each 4-connected component of the footprint mask becomes one building, and its label is the most "
     f"frequent damage value inside it. Components smaller than 16 px ({n(dr['too_small'])}), without any damage value "
     f"({n(dr['label_missing'])}) or with a tied vote ({n(dr['label_tie'])}) are excluded and logged, leaving "
     f"{n(N_BUILD)} buildings. Touching buildings can merge into one component; {n(TOT['mixed_label_blobs'])} components "
     "contain more than one damage value and receive the majority label. Original building identifiers and geographic "
     "coordinates are unavailable, so all geometry is in image pixels. "
     f"Of the {n(len(SCENES))} image pairs, {len(SC_NO_BLOB)} contain no building component and {len(SC_ALL_DROPPED)} "
     f"contain only excluded components (all {ALL_DROPPED_BLOBS} lacked a damage value), so {n(len(SC_WITH))} pairs "
     "contribute buildings; these are exactly the scenes assigned in both split designs.")
h2("C. Task")
para("Each building *i* is represented by a feature vector **x**ᵢ ∈ ℝ³⁷ and a label *y*ᵢ ∈ {no damage, minor, major, "
     f"destroyed}}. The classes are imbalanced: {CC['overall']['no-damage'] / N_BUILD:.0%} of buildings are undamaged.")

# ---------------------------------------------------------------- IV. Features and classifiers
h1("IV. Features and Classifiers")
h2("A. Engineered Features")
para("Features are computed on the building's pixels in the pre-event and post-event images (Table II). Grey levels "
     "use *g* = 0.299*R* + 0.587*G* + 0.114*B*. Writing Ω for the set of footprint pixels and *g*ᵖʳᵉ, *g*ᵖᵒˢᵗ for the "
     "two grey images, the mean absolute grey change and the brightness ratio are")
equation("*D* = (1/|Ω|) Σ_{p∈Ω} |*g*ᵖᵒˢᵗ(*p*) − *g*ᵖʳᵉ(*p*)|,")
equation("*B* = *ḡ*ᵖᵒˢᵗ / max(*ḡ*ᵖʳᵉ, 1),")
para("where *ḡ* denotes the mean grey level over Ω. Texture uses the fraction of footprint pixels on Canny edges "
     "(σ = 1.5) and two grey-level co-occurrence statistics [9] computed from horizontally and vertically adjacent "
     "pixel pairs (*i*, *j*) inside the footprint, after quantising grey levels to 32 bins:", indent=False)
equation("*C* = Σ *P*(*i*,*j*)(*i* − *j*)²,    *H* = Σ *P*(*i*,*j*) / (1 + (*i* − *j*)²)")
para("where *P* is the normalised co-occurrence matrix. Each texture statistic is reported before, after and as "
     "post − pre. Context features compare the building with a ring of ground pixels within 8 px of it (excluding all "
     "buildings); with ring mean *r̄*, the relative brightening is", indent=False)
equation("*K* = (*ḡ*ᵖᵒˢᵗ − *r̄*ᵖᵒˢᵗ) − (*ḡ*ᵖʳᵉ − *r̄*ᵖʳᵉ)")
para("Shape is described by area in pixels and compactness 4π*A*/*P*², with polygon area *A* and perimeter *P*. "
     "No feature selection is performed, and both classifiers receive all 37 features in the same order.", indent=False)
table_caption("II", "The 37 engineered features")
table([["Group", "No.", "Features (pre, post and change)"],
       ["Colour", "12", "per-channel mean and std. dev., pre and post"],
       ["Change", "8", "|Δ mean| and Δ std. dev. per channel; *D*; *B*"],
       ["Texture", "12", "grey std., Canny edge density, GLCM *C* and *H*: pre, post, Δ"],
       ["Context", "3", "ring mean change, ring mean abs. change, *K*"],
       ["Shape", "2", "area (px), compactness"]], [0.6, 0.3, 2.5])
h2("B. Shared Preprocessing")
para("Both classifiers use one scikit-learn [13] pipeline fitted on the training split only: log(1 + *x*) for the "
     f"{len(log1p)} features whose training skewness exceeds 2 (absolute colour change, *D*, GLCM contrast and area), "
     "median imputation and standardisation to zero mean and unit variance. The feature table contains no missing "
     "values, so imputation has no effect on these data.")
h2("C. Classifiers")
kn = M[w]["knn"]["hyperparameters"]
dt = M[w]["decision_tree"]["hyperparameters"]
para("KNN predicts the majority label among the *k* training buildings nearest to a query **x** in the standardised "
     "feature space under the Euclidean distance", indent=True)
equation("*d*(**x**, **x**_{i}) = ( Σ_{j=1}^{37} (*x*_{j} − *x*_{ij})^{2} )^{1/2}")
para(f"with uniform votes; the selected value is *k* = {kn['n_neighbors']} in both designs. Its confidence is the vote "
     "fraction. The DT is a CART-style tree [12] grown with the Gini impurity of a node with class proportions *p*_c,",
     indent=False)
equation("*G* = 1 − Σ_{c} *p*_{c}^{2},")
para("using balanced class weights *w*_c = *N* / (4*N*_c), where *N* is the number of training buildings and *N*_c the "
     f"number in class *c*. The selected tree has maximum depth {dt['max_depth']} and at least "
     f"{dt['min_samples_leaf']} training buildings per leaf.", indent=False)

# ---------------------------------------------------------------- V. Setup
h1("V. Experimental Setup")
h2("A. Evaluation Designs")
sw, sx = SPLIT[w]["counts"], SPLIT[x]["counts"]
para("Buildings in one image share acquisition conditions and often damage, so every split assigns whole scenes. The "
     "**within-event** design partitions scenes of all five events with stratified group folds (seed 42; requested "
     "70/15/15). The **cross-event** design uses every Hurricane Michael scene that contains labelled buildings "
     f"({MICHAEL_WITH} of {MICHAEL_CACHED}) as the test set and splits the other four "
     "events into training and validation scenes (requested 85/15). Table III gives the realised sizes. Split manifests "
     "are hashed and checked automatically for scene overlap. The cross-event test distribution differs from training: "
     f"minor damage is {michael_minor:.1%} of Hurricane Michael buildings but {other_minor:.1%} of the other events'.")
rows = [["Design", "Split", "Scenes", "Buildings"]]
for d, label in ((w, "Within-event"), (x, "Cross-event")):
    for s in ("train", "val", "test"):
        c = SPLIT[d]["counts"][s]
        rows.append([label if s == "train" else "", s, n(c["scenes"]), n(c["buildings"])])
table_caption("III", "Scene-grouped splits")
table(rows, [0.95, 0.55, 0.6, 0.75])
h2("B. Model Selection and Protocol Differences")
gk = M[w]["knn"]["search_on_validation"]
gd = M[w]["decision_tree"]["search_on_validation"]
para(f"KNN is tuned over *k* ∈ {{5, 15, 31}} and uniform or distance weighting ({len(gk)} settings); the DT over maximum "
     f"depth ∈ {{4, 6, 8, 12}} and minimum leaf size ∈ {{20, 100}} ({len(gd)} settings). For each design and model, every "
     "setting is fitted on the training split, the one with the highest validation macro-F1 is kept, and that fitted "
     "model is evaluated on the test split exactly once; a guard object in the code rejects a second test evaluation. "
     "Both models use the same splits, labels, features, preprocessing and metric code.")
para("Three protocol differences remain and are reported rather than removed. First, the DT uses balanced class "
     "weights, whereas scikit-learn's KNN classifier has no class-weight option and was trained unweighted. Second, the "
     f"search spaces differ in size ({len(gk)} versus {len(gd)} settings). Third, standardisation affects KNN distances "
     "but not the DT's choice of splits, which is invariant to monotone rescaling of a feature; the step is present in "
     "both pipelines.")
h2("C. Metrics and Uncertainty")
para("For each class *c*, precision *P*_c, recall *R*_c and *F*1_c = 2*P*_c*R*_c/(*P*_c + *R*_c) are computed from the "
     "test confusion matrix, and the primary metric is their unweighted mean,")
equation("macro-F1 = (1/4) Σ_{c} *F*1_{c},")
para(f"with accuracy reported alongside. Each model was trained once (seed 42), so we quantify only test-set "
     f"sampling variability: {ST['resamples']:,} bootstrap replicates draw test scenes with replacement (seed "
     f"{ST['seed']}) and recompute the metric on all buildings of the drawn scenes; 2.5th and 97.5th percentiles form "
     "95% intervals [16], [17]. KNN and the DT are evaluated on the same draws, giving a paired interval for their "
     "difference. The intervals use the saved test predictions; no model was retrained for this analysis. Experiments "
     "ran with Python 3.11 and scikit-learn 1.9.1 on a laptop with an Apple M4 processor.", indent=False)

# ---------------------------------------------------------------- VI. Results
h1("VI. Results and Discussion")
h2("A. Overall Comparison")
dw, dx = diff(w), diff(x)
para(f"Table IV and Fig. 2 summarise the test results. Within events, KNN reaches a macro-F1 of "
     f"{f3(t(w, 'knn')['macro_f1'])} and the DT {f3(t(w, 'decision_tree')['macro_f1'])}; the paired difference of "
     f"{dw['macro_f1']:+.3f} has a 95% interval of {dw['ci95'][0]:+.3f} to {dw['ci95'][1]:+.3f}, which excludes zero "
     f"(RQ1). With Hurricane Michael held out, both scores fall, to {f3(t(x, 'knn')['macro_f1'])} and "
     f"{f3(t(x, 'decision_tree')['macro_f1'])}, and the ranking reverses: the difference is {dx['macro_f1']:+.3f} "
     f"({dx['ci95'][0]:+.3f} to {dx['ci95'][1]:+.3f}) (RQ2). Accuracy does not follow macro-F1: KNN is more accurate "
     f"in both designs ({f3(t(w, 'knn')['accuracy'])} versus {f3(t(w, 'decision_tree')['accuracy'])} within events and "
     f"{f3(t(x, 'knn')['accuracy'])} versus {f3(t(x, 'decision_tree')['accuracy'])} across events), because it "
     "classifies the majority no-damage class well. Which model is preferable therefore depends on both the design "
     "and the metric.")
para(f"Validation scores did not anticipate the cross-event drop: the selected models scored "
     f"{f3(M[x]['knn']['metrics']['val']['macro_f1'])} (KNN) and {f3(M[x]['decision_tree']['metrics']['val']['macro_f1'])} "
     "(DT) on validation scenes from the four training events, roughly twice their test scores on the held-out event. "
     "Validation drawn from the training events is thus an optimistic guide to performance on a new event.")
rows = [["Model", "Within-event macro-F1 [95% CI]", "Acc.", "Cross-event macro-F1 [95% CI]", "Acc."]]
for mname, mkey in (("KNN", "knn"), ("Decision tree", "decision_tree")):
    rows.append([mname, f"{f3(t(w, mkey)['macro_f1'])} [{ci(w, mkey)}]", f3(t(w, mkey)["accuracy"]),
                 f"{f3(t(x, mkey)['macro_f1'])} [{ci(x, mkey)}]", f3(t(x, mkey)["accuracy"])])
rows.append(["KNN − DT", f"{dw['macro_f1']:+.3f} [{dw['ci95'][0]:+.3f}, {dw['ci95'][1]:+.3f}]", "",
             f"{dx['macro_f1']:+.3f} [{dx['ci95'][0]:+.3f}, {dx['ci95'][1]:+.3f}]", ""])
ctx = ", ".join(f"{name} {f3(CMP[w][k]['test_macro_f1'])}/{f3(CMP[x][k]['test_macro_f1'])}"
                for name, k in (("logistic regression", "logistic_regression"), ("random forest", "random_forest"),
                                ("one-feature threshold rule", "baseline")))
table_caption("IV", "Test results (single training run; scene-bootstrap intervals)")
table(rows, [0.72, 1.05, 0.38, 1.0, 0.38],
      note=f"For context only, other models trained in the same pipeline score (within/cross macro-F1): {ctx}.")
figure(FIG / "fig2_macro_f1.png", "Fig. 2. Test macro-F1 of KNN and the DT under both designs. Error bars: 95% "
       "scene-level bootstrap intervals for fixed trained models.")
h2("B. Per-Class Behaviour")
kw_, dw_ = t(w, "knn")["per_class"], t(w, "decision_tree")["per_class"]
kx_, dx_ = t(x, "knn")["per_class"], t(x, "decision_tree")["per_class"]
para(f"Table V and Fig. 3 show where the errors arise (RQ3). Within events, KNN recovers {kw_['no-damage']['recall']:.0%} "
     f"of undamaged buildings but only {kw_['minor-damage']['recall']:.0%} of minor and {kw_['major-damage']['recall']:.0%} "
     f"of major damage, labelling most of them undamaged. The DT spreads recall more evenly "
     f"({dw_['no-damage']['recall']:.0%}–{dw_['destroyed']['recall']:.0%}), at the cost of no-damage recall and of "
     f"major-damage precision ({dw_['major-damage']['precision']:.2f}). Among the damage classes, destroyed buildings are "
     f"recognised best by both models (F1 {kw_['destroyed']['f1']:.2f} and {dw_['destroyed']['f1']:.2f}). On the held-out event, major damage "
     f"is almost never identified (F1 {kx_['major-damage']['f1']:.2f} and {dx_['major-damage']['f1']:.2f}), destroyed "
     f"recall falls to {kx_['destroyed']['recall']:.0%} and {dx_['destroyed']['recall']:.0%}, and KNN recovers only "
     f"{kx_['minor-damage']['recall']:.0%} of minor-damage buildings against {dx_['minor-damage']['recall']:.0%} for the DT. "
     f"The DT's higher cross-event macro-F1 comes mainly from minor damage (F1 {dx_['minor-damage']['f1']:.2f} versus "
     f"{kx_['minor-damage']['f1']:.2f}), whereas KNN is better on the large no-damage class (F1 "
     f"{kx_['no-damage']['f1']:.2f} versus {dx_['no-damage']['f1']:.2f}), which also explains its higher accuracy.")
rows = [["Class", "KNN P", "R", "F1", "DT P", "R", "F1"]]
for label, k_, d_ in (("Within-event", kw_, dw_), ("Cross-event", kx_, dx_)):
    rows.append([label, "", "", "", "", "", ""])
    for c in CLASSES:
        rows.append(["  " + CLASS_NAME[c]] + [f"{k_[c][m]:.2f}" for m in ("precision", "recall", "f1")]
                    + [f"{d_[c][m]:.2f}" for m in ("precision", "recall", "f1")])
table_caption("V", "Per-class test precision (P), recall (R) and F1")
table(rows, [0.9] + [0.42] * 6)
figure(FIG / "fig3_confusion.png", "Fig. 3. Row-normalised test confusion matrices (percent of each reference class). "
       "The diagonal gives per-class recall.")
h2("C. Interpreting the Differences")
para("The opposite recall profiles are consistent with the protocol difference in Section V-B: the class-weighted DT "
     "is pushed towards minority classes, whereas unweighted KNN votes reflect the class frequencies among neighbours. "
     "Because weighting was not varied, this remains an explanation to be tested, not an established cause. Two "
     "measured facts bear on the cross-event drop. First, the held-out event has a different class mix (Section V-A). "
     f"Second, the features carry a strong event signature: a logistic-regression classifier predicts the source event "
     f"from the same 37 features with test accuracy {CONF['test_metrics']['accuracy']:.3f}, against "
     f"{CONF['majority_class_accuracy']:.3f} for always predicting the largest event. When Hurricane Michael scenes were "
     f"available in training (within-event design), its test scenes scored {per_event['knn']['hurricane-michael']['macro_f1']:.3f} "
     f"(KNN) and {per_event['decision_tree']['hurricane-michael']['macro_f1']:.3f} (DT) macro-F1, compared with "
     f"{f3(t(x, 'knn')['macro_f1'])} and {f3(t(x, 'decision_tree')['macro_f1'])} when the event was unseen. A plausible "
     "reading, not tested here, is that event-specific appearance (sensor, season, region and building stock) accounts "
     "for part of the within-event performance, and KNN, which relies on similarity to stored examples, loses more "
     "when that similarity is no longer available.")
para(f"The models also differ in cost and explanation. On our hardware, test inference took "
     f"{M[w]['knn']['inference_seconds_per_1000_test'] * 1000:.1f} ms per 1,000 buildings for KNN, which compares each "
     f"query with all {n(sw['train']['buildings'])} training buildings, versus "
     f"{M[w]['decision_tree']['inference_seconds_per_1000_test'] * 1000:.2f} ms for the DT (single measurements). In the "
     "AfterSight application a KNN prediction is explained by its nearest labelled training buildings and a DT "
     "prediction by its root-to-leaf path in raw feature units.")

# ---------------------------------------------------------------- VII. Threats
h1("VII. Limitations and Threats to Validity")
para("*Internal validity.* The comparison is not fully controlled: class weighting and search-space size differ "
     "between the models, and each model was trained once. The bootstrap intervals describe variability over test "
     "scenes for fixed models; they do not include training variability or variability across events.", indent=False)
para(f"*Construct validity.* Labels come from mask-derived regions: touching buildings may merge "
     f"({n(TOT['mixed_label_blobs'])} mixed components), {n(dr['label_missing'])} unlabelled components are excluded, "
     f"and an alignment check flags {INSP['alignment']['scenes_flagged'] if isinstance(INSP['alignment']['scenes_flagged'], int) else len(INSP['alignment']['scenes_flagged'])} "
     "image pairs whose pre/post registration may be imperfect. These pairs were kept, and their effect was not measured.",
     indent=False)
para("*External validity.* Only one event is held out, so the cross-event result describes Hurricane Michael under "
     "this protocol, not unseen disasters in general. Five of the ten available events were chosen by design decision, "
     "and other selections could change the results; only xBD's training split is available, through a third-party "
     "copy, so official test partitions were not used.", indent=False)
para("*Measurement.* Macro-F1 gives rare classes equal weight and is sensitive to small changes in them; scene "
     "resampling assumes independence between scenes of one event. Model confidences are not calibrated, and the "
     "system has not been validated in operational disaster response.", indent=False)

# ---------------------------------------------------------------- VIII. Conclusion
h1("VIII. Conclusion and Future Work")
drop = {m: 1 - t(x, m)["macro_f1"] / t(w, m)["macro_f1"] for m in ("knn", "decision_tree")}
para(f"On {n(N_BUILD)} buildings from five xBD events, KNN outperformed a class-weighted decision tree in macro-F1 on "
     "new scenes of known events, but the tree was slightly better on the held-out Hurricane Michael, where the models "
     f"lost {drop['knn']:.0%} (KNN) and {drop['decision_tree']:.0%} (DT) of their within-event macro-F1; KNN remained "
     "more accurate in both designs because of the majority class. The ranking therefore depends on the evaluation design and the metric, and the main limitation of "
     "engineered-feature classifiers here is transfer to a new event rather than the choice between these two models. "
     "Future work should equalise class handling (for example by resampling for KNN), repeat training with several "
     "seeds, hold out each event in turn, include more events, and compare against learned features under the same "
     "protocol.")

# ---------------------------------------------------------------- acknowledgement + references
h1("Acknowledgment")
ack = para("[PLACEHOLDER: acknowledgements, to be completed by the authors.] "
           "[PLACEHOLDER: AI-assisted writing disclosure, to be completed according to the institution's and the venue's "
           "policy, stating which tools were used and for what.]", indent=False)
for r in ack.runs:
    r.font.color.rgb = RGBColor(0x80, 0, 0)
    r.italic = True

h1("References")
REFS = [
    "R. Gupta, B. Goodman, N. Patel, R. Hosfelt, S. Sajeev, E. Heim, J. Doshi, K. Lucas, H. Choset, and M. Gaston, "
    "“Creating xBD: A dataset for assessing building damage from satellite imagery,” in *Proc. IEEE/CVF Conf. Computer "
    "Vision and Pattern Recognition Workshops*, 2019, pp. 10–17.",
    "R. Gupta *et al.*, “xBD: A dataset for assessing building damage from satellite imagery,” arXiv:1911.09296, 2019.",
    "L. Dong and J. Shan, “A comprehensive review of earthquake-induced building damage detection with remote sensing "
    "techniques,” *ISPRS J. Photogramm. Remote Sens.*, vol. 84, pp. 85–99, 2013, doi: 10.1016/j.isprsjprs.2013.06.011.",
    "A. Singh, “Digital change detection techniques using remotely-sensed data,” *Int. J. Remote Sens.*, vol. 10, no. 6, "
    "pp. 989–1003, 1989, doi: 10.1080/01431168908903939.",
    "R. Gupta and M. Shah, “RescueNet: Joint building segmentation and damage assessment from satellite imagery,” in "
    "*Proc. 25th Int. Conf. Pattern Recognition (ICPR)*, 2021, pp. 4405–4411, doi: 10.1109/ICPR48806.2021.9412295.",
    "Y. Shen, S. Zhu, T. Yang, C. Chen, D. Pan, J. Chen, L. Xiao, and Q. Du, “BDANet: Multiscale convolutional neural "
    "network with cross-directional attention for building damage assessment from satellite images,” *IEEE Trans. "
    "Geosci. Remote Sens.*, vol. 60, pp. 1–14, 2022, doi: 10.1109/TGRS.2021.3080580.",
    "A. Cooner, Y. Shao, and J. Campbell, “Detection of urban damage using remote sensing and machine learning "
    "algorithms: Revisiting the 2010 Haiti earthquake,” *Remote Sens.*, vol. 8, no. 10, Art. no. 868, 2016, "
    "doi: 10.3390/rs8100868.",
    "T. Valentijn, J. Margutti, M. van den Homberg, and J. Laaksonen, “Multi-hazard and spatial transferability of a CNN "
    "for automated building damage assessment,” *Remote Sens.*, vol. 12, no. 17, Art. no. 2839, 2020, "
    "doi: 10.3390/rs12172839.",
    "R. M. Haralick, K. Shanmugam, and I. Dinstein, “Textural features for image classification,” *IEEE Trans. Syst., "
    "Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, 1973, doi: 10.1109/TSMC.1973.4309314.",
    "J. Canny, “A computational approach to edge detection,” *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. PAMI-8, "
    "no. 6, pp. 679–698, 1986, doi: 10.1109/TPAMI.1986.4767851.",
    "T. Cover and P. Hart, “Nearest neighbor pattern classification,” *IEEE Trans. Inf. Theory*, vol. 13, no. 1, "
    "pp. 21–27, 1967, doi: 10.1109/TIT.1967.1053964.",
    "L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, *Classification and Regression Trees*. Belmont, CA, USA: "
    "Wadsworth, 1984 (reprinted by CRC Press, doi: 10.1201/9781315139470).",
    "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, "
    "pp. 2825–2830, 2011.",
    "H. He and E. A. Garcia, “Learning from imbalanced data,” *IEEE Trans. Knowl. Data Eng.*, vol. 21, no. 9, "
    "pp. 1263–1284, 2009, doi: 10.1109/TKDE.2008.239.",
    "M. Sokolova and G. Lapalme, “A systematic analysis of performance measures for classification tasks,” *Inf. "
    "Process. Manage.*, vol. 45, no. 4, pp. 427–437, 2009, doi: 10.1016/j.ipm.2009.03.002.",
    "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall/CRC, 1994, "
    "doi: 10.1201/9780429246593.",
    "C. A. Field and A. H. Welsh, “Bootstrapping clustered data,” *J. Roy. Statist. Soc. B*, vol. 69, no. 3, "
    "pp. 369–390, 2007, doi: 10.1111/j.1467-9868.2007.00593.x.",
]
# IEEE numbers references in order of first citation: renumber the [n] markers in document order.
order: list[int] = []
cite = re.compile(r"\[(\d+)\]")
body_paragraphs = list(doc.paragraphs)
for p_ in body_paragraphs:
    for r_ in p_.runs:
        for num in cite.findall(r_.text):
            if int(num) not in order:
                order.append(int(num))
missing = [k for k in range(1, len(REFS) + 1) if k not in order]
assert not missing, f"references never cited: {missing}"
new_number = {old: new for new, old in enumerate(order, start=1)}
for p_ in body_paragraphs:
    for r_ in p_.runs:
        if cite.search(r_.text):
            r_.text = cite.sub(lambda m_: f"[{new_number[int(m_.group(1))]}]", r_.text)
REFS = [REFS[old - 1] for old in order]
for i, ref in enumerate(REFS, start=1):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.25))
    r = p.add_run(f"[{i}]\t")
    r.font.size = Pt(8)
    rich(p, ref, 8)

doc.save(OUT)
print("wrote", OUT.relative_to(ROOT))
